# -*- coding: utf-8 -*-
"""TTS backends.

Kokoro can run through torch (portable, works everywhere) or through MLX (Apple Silicon
only, measured ~3.8x faster on an M5 Max). Both produce the same voices from the same
weights, so the choice is purely about speed and platform.

The MLX pipeline is adapted to Kokoro's own call signature, so `gen_audio_segments` --
and everything downstream of it -- does not need to know which engine is running.
"""
import json
import os
import platform
import re
import subprocess
import tempfile
import threading
import time
from glob import glob
from pathlib import Path

from audiblez.voices import is_edge_voice, lang_code_for

BACKENDS = ('auto', 'torch', 'mlx', 'edge', 'cosyvoice')

DEFAULT_REPOS = {
    'torch': 'hexgrad/Kokoro-82M',
    'mlx': 'mlx-community/Kokoro-82M-bf16',
}

# CosyVoice2 assets. This machine already has the CosyVoice repo, model and the
# reference-voice prompt dir (shared with the yue reader); the worker subprocess
# script lives in the yue checkout. All overridable via the YUE_COSYVOICE_* env
# vars for portability.
COSYVOICE_REPO = os.environ.get(
    'YUE_COSYVOICE_REPO',
    '/Users/randallsim/Documents/python_project/tts-training/CosyVoice')
COSYVOICE_MODEL_DIR = os.environ.get(
    'YUE_COSYVOICE_MODEL_DIR',
    '/Users/randallsim/Documents/python_project/tts-training/pretrained_models/CosyVoice2-0.5B')
COSYVOICE_PROMPT_DIR = os.environ.get(
    'YUE_COSYVOICE_PROMPT_DIR',
    '/Users/randallsim/Documents/python_project/tts-training/cosyvoice_prompts')
COSYVOICE_WORKER_PYTHON = os.environ.get(
    'YUE_COSYVOICE_WORKER_PYTHON',
    '/Users/randallsim/Documents/python_project/tts-training/.venv-cosyvoice/bin/python')
COSYVOICE_WORKER_SCRIPT = os.environ.get(
    'YUE_COSYVOICE_WORKER_SCRIPT',
    '/Users/randallsim/Documents/interesting_git_project/yue/yue/tts/cosyvoice_tts_worker.py')

# Any non-whitespace character means there is something to narrate.
_SPEAKABLE_RE = re.compile(r'\S')

# Starting guesses for the time estimate, in characters of text per second. These only seed
# the estimate: it is recalibrated from real throughput once synthesis is under way.
CHARS_PER_SEC_GUESS = {
    'mlx': 900,        # measured ~906 on an M5 Max
    'torch_cuda': 500,
    'torch_cpu': 50,
    'edge': 150,       # network-bound; conservative, recalibrates within the first chapter
    'cosyvoice': 20,   # ~0.5-0.7x realtime LLM+flow on MPS; slow but honest
}

# Per-language overrides. Characters are not equal units of speech: a CJK character carries
# far more phonemes than a Latin one, so Kokoro measures ~150 chars/sec on Chinese against
# ~666 on English on the same machine. Without this the first ETA shown for a Chinese book
# is optimistic by roughly 4-6x, which is worst precisely when the run is longest.
CHARS_PER_SEC_BY_LANG = {
    'mlx': {'z': 150},   # measured on an M5 Max
    'cosyvoice': {'z': 8},   # Chinese is slower than English for CosyVoice2 too
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


def cosyvoice_available():
    """True when the CosyVoice2 worker assets are present.

    CosyVoice2 is reference-voice (zero-shot): it needs the model, the worker venv,
    and the prompt (reference) dir. Absence is normal rather than an error -- this
    machine has them via the yue/tts-training setup.
    """
    return (
        os.path.exists(COSYVOICE_WORKER_PYTHON)
        and os.path.exists(COSYVOICE_WORKER_SCRIPT)
        and os.path.exists(COSYVOICE_PROMPT_DIR)
        and os.path.exists(COSYVOICE_MODEL_DIR)
    )


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
    elif resolved == 'cosyvoice':
        key = 'cosyvoice'
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

# Edge is a network service, and a dropped response must not cost a multi-hour book.
# Measured: perfectly speakable text comes back with no audio when requests are issued back
# to back -- the service throttles -- and reads normally again after a pause. '②"。' failed
# three times running mid-book and synthesized fine (1.78s, peak 0.48) moments later, so
# the waits are sized to outlast throttling rather than a network round trip.
EDGE_ATTEMPTS = 4
EDGE_RETRY_WAITS = (3.0, 8.0, 20.0)


def _retry_wait(attempt):
    """Seconds to wait before the retry following `attempt`, holding at the last step."""
    return EDGE_RETRY_WAITS[min(attempt, len(EDGE_RETRY_WAITS)) - 1]


class EdgeNoAudio(RuntimeError):
    """Edge accepted the text and returned no audio, repeatedly.

    Its own class so core.main can fail the one chapter rather than the whole book: a
    fragment the service truly has nothing to say for would otherwise wedge every
    re-run at the same sentence.
    """


def _is_no_audio(exc):
    """True when the service reported no speech, as opposed to failing to answer at all.

    edge-tts raises NoAudioReceived both for text it finds nothing to say in and, as
    observed, when it is simply throttling. Neither is a reason to abandon the book, so
    this is treated as an empty payload rather than as a fatal error.
    """
    try:
        from edge_tts.exceptions import NoAudioReceived
    except ImportError:
        return False
    return isinstance(exc, NoAudioReceived)


class _AdaptiveConcurrencyGate:
    """Bounds how many EdgePipeline calls actually hit the network at once, adaptively.

    Edge throttles -- returns empty audio -- when requests are issued back to back (see
    EDGE_ATTEMPTS/EDGE_RETRY_WAITS above), so this is not a fixed-size pool: it starts at
    1 (fully serial), and only a run of `ramp_streak` clean, first-attempt successes earns
    it one more slot, up to `ceiling`. Any call that needed a retry -- whether from a
    transport error or a throttled empty response, the two are not distinguished here --
    drops it straight back to 1. gen_audio_segments can submit more calls than the current
    limit to a thread pool; the extras simply block in acquire() until a slot opens, so the
    gate is the single source of truth for how much concurrency is actually happening.
    """

    def __init__(self, ceiling, ramp_streak=5):
        self._ceiling = ceiling
        self._ramp_streak = ramp_streak
        self._limit = 1
        self._in_flight = 0
        self._streak = 0
        self._cond = threading.Condition()

    def acquire(self):
        with self._cond:
            while self._in_flight >= self._limit:
                self._cond.wait()
            self._in_flight += 1

    def release(self, needed_retry):
        with self._cond:
            self._in_flight -= 1
            if needed_retry:
                self._limit = 1
                self._streak = 0
            else:
                self._streak += 1
                if self._streak >= self._ramp_streak and self._limit < self._ceiling:
                    self._limit += 1
                    self._streak = 0
            self._cond.notify_all()


class EdgePipeline:
    """Microsoft Edge's online TTS, wearing the pipeline callable signature.

    Yields (graphemes, phonemes, audio) like the Kokoro pipelines do; graphemes and
    phonemes are None. Audio is decoded from the MP3 edge-tts returns (24kHz, the same
    rate audiblez writes) to numpy so np.concatenate downstream is unaffected.

    Each __call__ is one network request, run with asyncio.run() in whichever thread calls
    it -- audiblez's core is synchronous, so there is no running event loop in any given
    thread for asyncio.run() to clash with, and a fresh event loop per thread is fine.
    split_pattern is accepted and ignored: edge-tts needs no sentence-level split pattern.
    """

    # gen_audio_segments groups consecutive sentences into chunks up to this many
    # characters per call, to amortize Edge's per-call network round trip. It reads this
    # attribute rather than special-casing EdgePipeline by isinstance, so any other backend
    # that wants the same amortization can opt in the same way.
    batch_chars = 1500

    # gen_audio_segments may dispatch up to this many chunks to a thread pool at once; the
    # instance's own _AdaptiveConcurrencyGate decides how many of them are actually allowed
    # to call the network concurrently at any given moment (starting at 1, see above).
    max_concurrency = 3

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self._gate = _AdaptiveConcurrencyGate(self.max_concurrency)

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
        needed_retry = False
        self._gate.acquire()
        try:
            for attempt in range(1, EDGE_ATTEMPTS + 1):
                if attempt > 1:
                    needed_retry = True
                try:
                    mp3 = asyncio.run(synth())
                except Exception as e:
                    if not _is_no_audio(e):
                        # Could not reach the service at all: that breaks the run, not just
                        # this fragment. A chapter's .wav is only written once the chapter
                        # finishes, and finished chapters are skipped on the next run, so
                        # raising costs this chapter and not the book.
                        if attempt == EDGE_ATTEMPTS:
                            raise RuntimeError(
                                f'Edge TTS failed after {EDGE_ATTEMPTS} attempts on {excerpt!r}: '
                                f'{type(e).__name__}: {e}') from e
                        print(f'Edge TTS attempt {attempt} failed ({type(e).__name__}); retrying...')
                        time.sleep(_retry_wait(attempt))
                        continue
                    mp3 = b''   # answered, but with no speech: identical to an empty payload
                if mp3:
                    break
                if attempt < EDGE_ATTEMPTS:
                    print(f'Edge TTS returned no audio (attempt {attempt}); retrying...')
                    time.sleep(_retry_wait(attempt))
        finally:
            # Runs even when the loop above raised: a call that failed outright is at least
            # as strong a throttling signal as one that merely needed a retry, so it must
            # still drop the gate back to 1 rather than leaving a stale higher limit in
            # place (and, either way, in_flight must be released for waiters to proceed).
            self._gate.release(needed_retry)

        if not mp3:
            # The service took the text and sent nothing back, repeatedly. Yielding
            # nothing would be silent and permanent: gen_audio_segments would return a
            # short list, the chapter's .wav would be written from it, and the next run
            # skips chapters whose .wav already exists -- so the missing speech could
            # never be recovered. Raise instead; core.main drops this chapter without
            # writing it, so a re-run redoes it.
            raise EdgeNoAudio(
                f'Edge TTS returned no audio for {excerpt!r} after {EDGE_ATTEMPTS} '
                'attempts.')

        audio, _ = soundfile.read(io.BytesIO(mp3), dtype='float32')
        yield None, None, audio


class CosyVoicePipeline:
    """CosyVoice2 reference-voice TTS, wearing the pipeline callable signature.

    CosyVoice2 is zero-shot: the voice is a reference-voice prompt id (a ``<id>.wav`` +
    ``<id>.txt`` pair in the prompt dir, e.g. ``en_amanda`` or ``zh_ava``) that the model
    clones live for every chunk. It runs as a subprocess under the cosyvoice venv,
    talking JSON-lines over stdin/stdout (the same worker the yue reader uses).

    Yields ``(None, None, audio)`` per call like the Kokoro/Edge pipelines. Serial only:
    a single worker drives MPS, so ``max_concurrency`` stays 1 and each call is serialized
    on a lock. The worker is torn down on process exit via ``atexit``.
    """

    max_concurrency = 1

    def __init__(self, lang_code=None):
        import atexit
        self.lang_code = lang_code
        self._proc = None
        self._lock = threading.Lock()
        self._out = tempfile.mktemp(suffix='.wav', prefix='audiblez_cosy_')
        atexit.register(self.close)

    def _ensure_worker(self):
        if self._proc is not None:
            return
        env = dict(os.environ)
        env['YUE_COSYVOICE_REPO'] = COSYVOICE_REPO
        env['YUE_COSYVOICE_MODEL_DIR'] = COSYVOICE_MODEL_DIR
        env['YUE_COSYVOICE_PROMPT_DIR'] = COSYVOICE_PROMPT_DIR
        self._proc = subprocess.Popen(
            [COSYVOICE_WORKER_PYTHON, COSYVOICE_WORKER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, text=True)
        try:
            line = self._proc.stdout.readline()
        except Exception:
            self._proc = None
            raise
        resp = json.loads(line)
        if 'error' in resp:
            self.close()
            raise RuntimeError(f"CosyVoice worker failed to start: {resp['error']}")

    def __call__(self, text, voice, speed=1.0, split_pattern=None):
        import numpy as np
        import soundfile
        # Nothing to narrate: yield no segment rather than ask the model to read silence.
        if not _SPEAKABLE_RE.search(text or ''):
            return
        self._ensure_worker()
        with self._lock:
            self._proc.stdin.write(
                json.dumps({'text': text, 'voice': voice, 'out': self._out}) + '\n')
            self._proc.stdin.flush()
            try:
                resp = json.loads(self._proc.stdout.readline())
            except (ValueError, json.JSONDecodeError) as e:
                raise RuntimeError(
                    f'CosyVoice worker sent a bad reply: {e}') from e
            if 'error' in resp:
                raise RuntimeError(f"CosyVoice synthesis failed: {resp['error']}")
            audio, _ = soundfile.read(resp['path'], dtype='float32')
        yield None, None, audio

    def close(self):
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=3)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        try:
            if os.path.exists(self._out):
                os.remove(self._out)
        except Exception:  # noqa: BLE001
            pass


def get_pipeline(voice, lang_code=None, backend='auto', repo_id=None):
    """Build a TTS pipeline callable for `voice`.

    lang_code defaults to the first letter of the voice name, which is Kokoro's own
    convention ('af_sky' -> 'a'); an Edge voice carries its locale instead
    ('zh-TW-HsiaoChenNeural' -> 'zh-TW'). Pass it explicitly for a voice whose name does
    not carry the language, such as a path to a custom .pt voice pack.
    """
    resolved = resolve_backend(backend)
    lang_code = lang_code_for(voice, lang_code)
    if resolved != 'edge' and is_edge_voice(voice):
        # Otherwise this fails late and obscurely: mlx only after load_model has pulled
        # a 339 MB repo, torch on an assertion inside KPipeline about a 'zh-TW' lang_code.
        raise RuntimeError(
            f'{voice!r} is an Edge TTS voice, but the {resolved!r} backend was selected. '
            'Add --backend edge to use it, or pick a Kokoro voice such as af_heart.')
    if resolved == 'mlx':
        if not mlx_available():
            hint = ('Install it with: pip install mlx-audio "misaki[en]"' if is_apple_silicon()
                    else 'It needs Apple Silicon; use --backend torch on this machine.')
            raise RuntimeError(f'The mlx backend is not available. {hint}')
        return MlxKokoroPipeline(lang_code, repo_id)
    if resolved == 'cosyvoice':
        if not cosyvoice_available():
            raise RuntimeError(
                'The cosyvoice backend is not available on this machine (missing the '
                'CosyVoice model, worker venv, or prompt dir). See the COSYVOICE_* paths '
                'in audiblez/backends.py.')
        return CosyVoicePipeline(lang_code)
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
