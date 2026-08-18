# -*- coding: utf-8 -*-
"""TTS backends and models.

Two independent choices:

- **backend** -- the runtime. Kokoro can run through torch (portable, works everywhere)
  or through MLX (Apple Silicon only, measured ~3.8x faster on an M5 Max).
- **model** -- the weights. Kokoro by default; Qwen3-TTS CustomVoice as an opt-in
  alternative that runs only on the MLX runtime.

Both models are driven through the same adapter, because mlx-audio's unified
`generate()` takes Kokoro's own call signature. So `gen_audio_segments` -- and
everything downstream of it -- does not need to know which engine or model is running.

Qwen is never selected automatically: it is roughly 10x slower than Kokoro in English
and 5x in Chinese. See docs/plans/2026-08-18_qwen3-tts-backend.md for the measurements.
"""
import os
import platform
import warnings
from glob import glob
from pathlib import Path

BACKENDS = ('auto', 'torch', 'mlx')

DEFAULT_MODEL = 'kokoro'

# Per-model capabilities and defaults. What differs between Kokoro and Qwen -- voices,
# language codes, speed support, throughput -- differs by *model*, not by runtime, which
# is why this is a separate dimension from BACKENDS.
#
# chars_per_sec is keyed by language because CJK characters carry far more phonemes each:
# Kokoro measures ~666 c/s on English but ~150 c/s on Chinese. A single per-model number
# makes the first ETA on a Chinese book wrong by ~4x, which is worst exactly when the run
# is longest. Values are seeds only; core.measured_chars_per_sec recalibrates from real
# throughput once synthesis is under way.
MODELS = {
    'kokoro': dict(
        runtimes=('mlx', 'torch'),
        repos={'mlx': 'mlx-community/Kokoro-82M-bf16', 'torch': 'hexgrad/Kokoro-82M'},
        chars_per_sec={
            'mlx': {'default': 900, 'z': 150},      # measured ~906 / ~150 on an M5 Max
            'torch_cuda': {'default': 500},
            'torch_cpu': {'default': 50},
        },
        supports_speed=True,
        deterministic=True,
        lang_from_voice=True,                       # 'af_sky' -> 'a'
    ),
    'qwen3-tts': dict(
        runtimes=('mlx',),
        repos={'mlx': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit'},
        chars_per_sec={'mlx': {'default': 60, 'z': 27}},   # measured 62.3 / 27
        supports_speed=False,
        deterministic=False,
        lang_from_voice=False,                      # 'ryan' has no language prefix
    ),
}

# Kept for backwards compatibility and because these read better at call sites than
# reaching into MODELS. Derived, so there is still one source of truth.
DEFAULT_REPOS = MODELS['kokoro']['repos']
CHARS_PER_SEC_GUESS = {k: v['default'] for k, v in MODELS['kokoro']['chars_per_sec'].items()}

# Qwen names its languages in full where Kokoro uses a single letter. Kokoro's 'h' (Hindi)
# has no Qwen equivalent; german/korean/russian are Qwen-only and are passed through as-is.
KOKORO_TO_QWEN_LANG = {
    'a': 'english', 'b': 'english', 'z': 'chinese', 'j': 'japanese',
    'e': 'spanish', 'f': 'french', 'i': 'italian', 'p': 'portuguese',
}
QWEN_ONLY_LANGUAGES = ('german', 'korean', 'russian')

# Below the library default of 0.9, to reduce the run-to-run variance that makes a
# re-synthesized chapter sound unlike its neighbours. Deliberately not near-zero: very low
# temperatures can push autoregressive TTS into repetition loops, and we have not measured
# where that starts. See open question 5 in the plan.
QWEN_DEFAULT_TEMPERATURE = 0.7


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


def model_spec(model=DEFAULT_MODEL):
    """Look up a model's capabilities, rejecting unknown names."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}. Choose one of: {', '.join(MODELS)}")
    return MODELS[model]


def resolve_lang_code(lang_code, model=DEFAULT_MODEL):
    """Translate a Kokoro language letter into whatever `model` expects.

    Kokoro uses single letters ('a', 'z'); Qwen uses full names ('english', 'chinese').
    Passing a Kokoro letter straight to Qwen is silently wrong -- it is not a key in the
    model's language table, so it falls through to auto-detect and the language argument
    quietly does nothing.
    """
    if model == DEFAULT_MODEL or lang_code is None:
        return lang_code
    lowered = str(lang_code).lower()
    if lowered in KOKORO_TO_QWEN_LANG:
        return KOKORO_TO_QWEN_LANG[lowered]
    if lowered in KOKORO_TO_QWEN_LANG.values() or lowered in QWEN_ONLY_LANGUAGES:
        return lowered
    if lowered == 'auto':
        return 'auto'
    if lowered == 'h':
        raise ValueError(
            "Hindi is not supported by qwen3-tts. Use --model kokoro for Hindi, or pick "
            f"one of: {', '.join(sorted(set(KOKORO_TO_QWEN_LANG.values()) | set(QWEN_ONLY_LANGUAGES)))}")
    raise ValueError(
        f"Unknown language {lang_code!r} for model qwen3-tts. Choose one of: "
        f"{', '.join(sorted(set(KOKORO_TO_QWEN_LANG.values()) | set(QWEN_ONLY_LANGUAGES)))}")


def default_lang_code(voice, model=DEFAULT_MODEL):
    """The language to use when the caller did not name one.

    Kokoro encodes it in the voice name ('af_sky' -> 'a'). Qwen speaker names carry no
    language ('ryan' -> 'r', which is meaningless), so fall back to auto-detect.
    """
    if model_spec(model)['lang_from_voice']:
        return voice[:1]
    return 'auto'


def initial_chars_per_sec(backend, lang_code=None, model=DEFAULT_MODEL):
    """Seed value for the time estimate, before real throughput is known."""
    table = model_spec(model)['chars_per_sec']
    if resolve_backend(backend) == 'mlx':
        key = 'mlx'
    else:
        key = 'torch_cpu'
        try:
            import torch
            if torch.cuda.is_available():
                key = 'torch_cuda'
        except ImportError:
            pass
    by_lang = table.get(key) or table['mlx']
    return by_lang.get(str(lang_code or '')[:1], by_lang['default'])


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


class MlxPipeline:
    """An mlx-audio model, wearing Kokoro's KPipeline signature.

    Yields (graphemes, phonemes, audio) like KPipeline does. mlx-audio does not hand back
    grapheme/phoneme strings, so those are None -- `gen_audio_segments` discards them.
    Audio is converted to numpy so np.concatenate downstream is unaffected.
    """

    def __init__(self, lang_code, repo_id=None, model=DEFAULT_MODEL, temperature=None):
        from mlx_audio.tts.utils import load_model
        set_espeak_data_path()
        self.model_name = model
        self.spec = model_spec(model)
        self.lang_code = resolve_lang_code(lang_code, model)
        self.repo_id = repo_id or self.spec['repos']['mlx']
        self.temperature = temperature
        self._warned_about_speed = False
        self.model = load_model(self.repo_id)

    def _extra_kwargs(self, speed):
        """Model-specific arguments, kept out of the Kokoro path entirely."""
        extra = {}
        if not self.spec['supports_speed'] and speed not in (None, 1.0):
            if not self._warned_about_speed:
                # The model accepts `speed` and silently discards it, which would hand back
                # a full-length audiobook at the wrong pace with no indication of a problem.
                warnings.warn(
                    f"{self.model_name} ignores speed={speed}; audio will play at 1.0. "
                    "Use --model kokoro if you need speed control.",
                    stacklevel=3)
                self._warned_about_speed = True
        if not self.spec['deterministic']:
            temperature = self.temperature
            if temperature is None:
                temperature = QWEN_DEFAULT_TEMPERATURE
            extra['temperature'] = temperature
        return extra

    def __call__(self, text, voice, speed=1.0, split_pattern=r'\n\n\n'):
        import numpy as np
        for result in self.model.generate(text=text, voice=voice, speed=speed,
                                          lang_code=self.lang_code, split_pattern=split_pattern,
                                          **self._extra_kwargs(speed)):
            yield None, None, np.asarray(result.audio)


# The class was never Kokoro-specific -- only its name was. Kept as an alias so any
# out-of-tree caller keeps working.
MlxKokoroPipeline = MlxPipeline


def get_pipeline(voice, lang_code=None, backend='auto', repo_id=None, model=DEFAULT_MODEL,
                 temperature=None):
    """Build a TTS pipeline callable for `voice`.

    lang_code defaults to the first letter of the voice name for Kokoro, which is its own
    convention ('af_sky' -> 'a'). Pass it explicitly for a voice whose name does not carry
    the language, such as a path to a custom .pt voice pack. Qwen speaker names never
    carry a language, so it defaults to auto-detect there.
    """
    spec = model_spec(model)
    resolved = resolve_backend(backend)
    if resolved not in spec['runtimes']:
        raise RuntimeError(
            f"The {model} model does not run on the {resolved} backend "
            f"(supported: {', '.join(spec['runtimes'])}). "
            f"Use --backend {spec['runtimes'][0]}, or --model {DEFAULT_MODEL}.")
    lang_code = lang_code or default_lang_code(voice, model)
    if resolved == 'mlx':
        if not mlx_available():
            hint = ('Install it with: pip install mlx-audio "misaki[en]"' if is_apple_silicon()
                    else 'It needs Apple Silicon; use --backend torch on this machine.')
            raise RuntimeError(f'The mlx backend is not available. {hint}')
        return MlxPipeline(lang_code, repo_id, model=model, temperature=temperature)
    if not torch_available():
        raise RuntimeError(
            'The torch backend is not installed. This build ships MLX by default; '
            'add the fallback with: pip install ".[torch]"')
    from kokoro import KPipeline
    return KPipeline(lang_code=lang_code, repo_id=repo_id or spec['repos']['torch'])
