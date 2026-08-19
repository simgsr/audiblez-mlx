# -*- coding: utf-8 -*-
"""TTS backends.

Kokoro can run through torch (portable, works everywhere) or through MLX (Apple Silicon
only, measured ~3.8x faster on an M5 Max). Both produce the same voices from the same
weights, so the choice is purely about speed and platform.

The MLX pipeline is adapted to Kokoro's own call signature, so `gen_audio_segments` --
and everything downstream of it -- does not need to know which engine is running.
"""
import os
import platform
import re
import time
from glob import glob
from pathlib import Path

from audiblez.voices import is_edge_voice, lang_code_for

BACKENDS = ('auto', 'torch', 'mlx', 'edge')

DEFAULT_REPOS = {
    'torch': 'hexgrad/Kokoro-82M',
    'mlx': 'mlx-community/Kokoro-82M-bf16',
}

# Starting guesses for the time estimate, in characters of text per second. These only seed
# the estimate: it is recalibrated from real throughput once synthesis is under way.
CHARS_PER_SEC_GUESS = {
    'mlx': 900,        # measured ~906 on an M5 Max
    'torch_cuda': 500,
    'torch_cpu': 50,
    'edge': 150,       # network-bound; conservative, recalibrates within the first chapter
}

# Per-language overrides. Characters are not equal units of speech: a CJK character carries
# far more phonemes than a Latin one, so Kokoro measures ~150 chars/sec on Chinese against
# ~666 on English on the same machine. Without this the first ETA shown for a Chinese book
# is optimistic by roughly 4-6x, which is worst precisely when the run is longest.
CHARS_PER_SEC_BY_LANG = {
    'mlx': {'z': 150},   # measured on an M5 Max
}


def is_apple_silicon():
    return platform.system() == 'Darwin' and platform.machine() == 'arm64'


def mlx_available():
    """True when this machine can run the MLX backend."""
    if not is_apple_silicon():
        return False
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        return False
    return True


def torch_available():
    """True when the fallback torch backend is installed.

    This fork installs MLX by default; torch and kokoro are an optional extra, so their
    absence is normal rather than an error.
    """
    try:
        import kokoro  # noqa: F401
    except ImportError:
        return False
    return True


def edge_available():
    """True when the edge-tts package is installed.

    Edge is an optional extra like torch: it needs network at synthesis time, so its
    absence is normal rather than an error.
    """
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend='auto'):
    """Turn 'auto' into a concrete backend name."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Choose one of: {', '.join(BACKENDS)}")
    if backend != 'auto':
        return backend
    if mlx_available():
        return 'mlx'
    if torch_available():
        return 'torch'
    # Nothing installed: name the one that suits this machine so the error tells the truth.
    return 'mlx' if is_apple_silicon() else 'torch'


def initial_chars_per_sec(backend, lang_code=None):
    """Seed value for the time estimate, before real throughput is known."""
    resolved = resolve_backend(backend)
    if resolved == 'mlx':
        key = 'mlx'
    elif resolved == 'edge':
        key = 'edge'
    else:
        key = 'torch_cpu'
        try:
            import torch
            if torch.cuda.is_available():
                key = 'torch_cuda'
        except ImportError:
            pass
    by_lang = CHARS_PER_SEC_BY_LANG.get(key, {})
    return by_lang.get(str(lang_code or '')[:1], CHARS_PER_SEC_GUESS[key])


def find_espeak_data_path():
    """Locate espeak-ng's data directory.

    misaki's espeakng-loader ships a hardcoded path from its own build machine
    (/Users/runner/work/...), so on any other machine it fails with
    "Error processing file '.../phontab'". Pointing ESPEAK_DATA_PATH at the real
    install is the fix.
    """
    if os.environ.get('ESPEAK_DATA_PATH'):
        return os.environ['ESPEAK_DATA_PATH']
    candidates = []
    if platform.system() == 'Darwin':
        from subprocess import check_output, CalledProcessError
        try:
            cellar = Path(check_output(['brew', '--cellar'], text=True).strip())
            candidates += sorted(glob(str(cellar / 'espeak-ng' / '*' / 'share' / 'espeak-ng-data')))
        except (CalledProcessError, FileNotFoundError):
            pass
    elif platform.system() == 'Linux':
        candidates += ['/usr/share/espeak-ng-data', '/usr/lib/espeak-ng-data']
        candidates += sorted(glob('/usr/lib/*/espeak-ng-data'))
    elif platform.system() == 'Windows':
        candidates += sorted(glob('C:\\Program Files*\\eSpeak NG\\espeak-ng-data'))
    for candidate in candidates:
        if Path(candidate, 'phontab').exists():
            return candidate
    return None


def set_espeak_data_path():
    """Export ESPEAK_DATA_PATH if it is not already set. Returns the path used, if any."""
    if os.environ.get('ESPEAK_DATA_PATH'):
        return os.environ['ESPEAK_DATA_PATH']
    path = find_espeak_data_path()
    if path:
        os.environ['ESPEAK_DATA_PATH'] = path
        print('Using espeak data:', path)
    else:
        print('Could not locate espeak-ng data; MLX phonemization may fail. '
              'Set ESPEAK_DATA_PATH manually if so.')
    return path


class MlxKokoroPipeline:
    """mlx-audio's Kokoro, wearing Kokoro's KPipeline signature.

    Yields (graphemes, phonemes, audio) like KPipeline does. mlx-audio does not hand back
    grapheme/phoneme strings, so those are None -- `gen_audio_segments` discards them.
    Audio is converted to numpy so np.concatenate downstream is unaffected.
    """

    def __init__(self, lang_code, repo_id=None):
        from mlx_audio.tts.utils import load_model
        set_espeak_data_path()
        self.lang_code = lang_code
        self.repo_id = repo_id or DEFAULT_REPOS['mlx']
        self.model = load_model(self.repo_id)

    def __call__(self, text, voice, speed=1.0, split_pattern=r'\n\n\n'):
        import numpy as np
        for result in self.model.generate(text=text, voice=voice, speed=speed,
                                          lang_code=self.lang_code, split_pattern=split_pattern):
            yield None, None, np.asarray(result.audio)


# Text with nothing to pronounce. Sentence splitting hands back bare newlines (the last
# "sentence" of a chapter is routinely just '\n') and punctuation-only fragments such as
# '「」', and edge-tts splits those into *zero* chunks: it then streams nothing and raises
# nothing, so an empty payload reached soundfile, which reported the unhelpful
# "Format not recognised" and killed the whole book. `\w` covers Han, kana and Latin alike.
_SPEAKABLE_RE = re.compile(r'\w', re.UNICODE)

# Edge is a network service, and a single dropped or empty response should not cost a
# multi-hour book. Measured: valid text occasionally comes back with zero audio bytes when
# requests are issued back to back, and succeeds on a retry moments later.
EDGE_ATTEMPTS = 3
EDGE_RETRY_WAIT = 2.0


class EdgePipeline:
    """Microsoft Edge's online TTS, wearing the pipeline callable signature.

    Yields (graphemes, phonemes, audio) like the Kokoro pipelines do; graphemes and
    phonemes are None. Audio is decoded from the MP3 edge-tts returns (24kHz, the same
    rate audiblez writes) to numpy so np.concatenate downstream is unaffected.

    Each __call__ is one network request, run with asyncio.run() -- audiblez's core is
    synchronous and runs in a plain thread, so there is no running event loop to clash
    with. split_pattern is accepted and ignored: edge-tts needs no sentence-level split
    pattern.
    """

    def __init__(self, lang_code):
        self.lang_code = lang_code

    def __call__(self, text, voice, speed=1.0, split_pattern=None):
        import asyncio
        import io
        import soundfile
        import edge_tts

        # Nothing to narrate: yield no segment rather than ask the service to read silence.
        if not _SPEAKABLE_RE.search(text or ''):
            return

        rate = f"{int(round((speed - 1) * 100)):+d}%"

        async def synth():
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            chunks = []
            async for chunk in communicate.stream():
                if chunk['type'] == 'audio':
                    chunks.append(chunk['data'])
            return b''.join(chunks)

        excerpt = text.strip()[:40]
        mp3 = b''
        for attempt in range(1, EDGE_ATTEMPTS + 1):
            try:
                mp3 = asyncio.run(synth())
            except Exception as e:
                # A chapter's .wav is only written once the chapter finishes, and finished
                # chapters are skipped on the next run, so raising costs this chapter and
                # not the book.
                if attempt == EDGE_ATTEMPTS:
                    raise RuntimeError(
                        f'Edge TTS failed after {EDGE_ATTEMPTS} attempts on {excerpt!r}: '
                        f'{type(e).__name__}: {e}') from e
                print(f'Edge TTS attempt {attempt} failed ({type(e).__name__}); retrying...')
                time.sleep(EDGE_RETRY_WAIT * attempt)
                continue
            if mp3:
                break
            if attempt < EDGE_ATTEMPTS:
                print(f'Edge TTS returned no audio (attempt {attempt}); retrying...')
                time.sleep(EDGE_RETRY_WAIT * attempt)

        if not mp3:
            # The service took the text and sent nothing back, repeatedly. Dropping one
            # sentence beats losing every finished chapter to it.
            print(f'Warning: Edge TTS returned no audio for {excerpt!r}; skipping it.')
            return

        audio, _ = soundfile.read(io.BytesIO(mp3), dtype='float32')
        yield None, None, audio


def get_pipeline(voice, lang_code=None, backend='auto', repo_id=None):
    """Build a TTS pipeline callable for `voice`.

    lang_code defaults to the first letter of the voice name, which is Kokoro's own
    convention ('af_sky' -> 'a'); an Edge voice carries its locale instead
    ('zh-TW-HsiaoChenNeural' -> 'zh-TW'). Pass it explicitly for a voice whose name does
    not carry the language, such as a path to a custom .pt voice pack.
    """
    resolved = resolve_backend(backend)
    lang_code = lang_code_for(voice, lang_code)
    if resolved == 'mlx':
        if not mlx_available():
            hint = ('Install it with: pip install mlx-audio "misaki[en]"' if is_apple_silicon()
                    else 'It needs Apple Silicon; use --backend torch on this machine.')
            raise RuntimeError(f'The mlx backend is not available. {hint}')
        return MlxKokoroPipeline(lang_code, repo_id)
    if resolved == 'edge':
        if not edge_available():
            raise RuntimeError(
                'The edge backend is not installed. Add it with: pip install ".[edge]"')
        if not is_edge_voice(voice):
            raise RuntimeError(
                f'{voice!r} is not an Edge TTS voice. Edge voices look like '
                "'zh-TW-HsiaoChenNeural'; pick one from the voice list or type a full name.")
        return EdgePipeline(lang_code)
    if not torch_available():
        raise RuntimeError(
            'The torch backend is not installed. This build ships MLX by default; '
            'add the fallback with: pip install ".[torch]"')
    from kokoro import KPipeline
    return KPipeline(lang_code=lang_code, repo_id=repo_id or DEFAULT_REPOS['torch'])
