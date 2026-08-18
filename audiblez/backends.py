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
from glob import glob
from pathlib import Path

BACKENDS = ('auto', 'torch', 'mlx')

DEFAULT_REPOS = {
    'torch': 'hexgrad/Kokoro-82M',
    'mlx': 'mlx-community/Kokoro-82M-bf16',
}


def mlx_available():
    """True when this machine can run the MLX backend."""
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        return False
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend='auto'):
    """Turn 'auto' into a concrete backend name."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Choose one of: {', '.join(BACKENDS)}")
    if backend != 'auto':
        return backend
    return 'mlx' if mlx_available() else 'torch'


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


def get_pipeline(voice, lang_code=None, backend='auto', repo_id=None):
    """Build a TTS pipeline callable for `voice`.

    lang_code defaults to the first letter of the voice name, which is Kokoro's own
    convention ('af_sky' -> 'a'). Pass it explicitly for a voice whose name does not
    carry the language, such as a path to a custom .pt voice pack.
    """
    resolved = resolve_backend(backend)
    lang_code = lang_code or voice[0]
    if resolved == 'mlx':
        if not mlx_available():
            raise RuntimeError(
                'The mlx backend needs Apple Silicon and mlx-audio. '
                'Install it with: pip install "audiblez[mlx]"')
        return MlxKokoroPipeline(lang_code, repo_id)
    from kokoro import KPipeline
    return KPipeline(lang_code=lang_code, repo_id=repo_id or DEFAULT_REPOS['torch'])
