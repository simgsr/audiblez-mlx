import sys
import types
import unittest
import warnings
from unittest import mock

import numpy as np

from audiblez import backends
from audiblez.backends import get_pipeline, mlx_available, resolve_backend
from audiblez.core import safe_filename_part


class FakeResult:
    def __init__(self, audio):
        self.audio = audio


class FakeMlxModel:
    """Stands in for mlx-audio's Kokoro model, recording how it was called."""

    def __init__(self):
        self.calls = []

    def generate(self, text, voice, speed, lang_code, split_pattern=None):
        self.calls.append(dict(text=text, voice=voice, speed=speed,
                               lang_code=lang_code, split_pattern=split_pattern))
        yield FakeResult(np.zeros(120, dtype=np.float32))
        yield FakeResult(np.ones(80, dtype=np.float32))


def fake_mlx_audio(model):
    """Build a stub mlx_audio package tree whose load_model returns `model`."""
    pkg = types.ModuleType('mlx_audio')
    tts = types.ModuleType('mlx_audio.tts')
    utils = types.ModuleType('mlx_audio.tts.utils')
    utils.load_model = lambda repo_id: model
    pkg.tts = tts
    tts.utils = utils
    return {'mlx_audio': pkg, 'mlx_audio.tts': tts, 'mlx_audio.tts.utils': utils}


class ResolveBackendTest(unittest.TestCase):
    def test_explicit_backends_pass_through(self):
        self.assertEqual(resolve_backend('torch'), 'torch')
        self.assertEqual(resolve_backend('mlx'), 'mlx')

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            resolve_backend('festival')
        self.assertIn('festival', str(ctx.exception))

    def test_auto_prefers_mlx_when_available(self):
        with mock.patch.object(backends, 'mlx_available', return_value=True):
            self.assertEqual(resolve_backend('auto'), 'mlx')

    def test_mlx_unavailable_off_apple_silicon(self):
        with mock.patch('platform.system', return_value='Linux'):
            self.assertFalse(mlx_available())
        with mock.patch('platform.system', return_value='Darwin'), \
             mock.patch('platform.machine', return_value='x86_64'):
            self.assertFalse(mlx_available())


class AutoBackendFallbackTest(unittest.TestCase):
    def test_auto_uses_torch_when_only_torch_is_installed(self):
        with mock.patch.object(backends, 'mlx_available', return_value=False), \
             mock.patch.object(backends, 'torch_available', return_value=True):
            self.assertEqual(resolve_backend('auto'), 'torch')

    def test_auto_names_mlx_on_apple_silicon_when_nothing_is_installed(self):
        # The error should point at the backend that suits the machine, not the other one.
        with mock.patch.object(backends, 'mlx_available', return_value=False), \
             mock.patch.object(backends, 'torch_available', return_value=False), \
             mock.patch.object(backends, 'is_apple_silicon', return_value=True):
            self.assertEqual(resolve_backend('auto'), 'mlx')

    def test_auto_names_torch_elsewhere_when_nothing_is_installed(self):
        with mock.patch.object(backends, 'mlx_available', return_value=False), \
             mock.patch.object(backends, 'torch_available', return_value=False), \
             mock.patch.object(backends, 'is_apple_silicon', return_value=False):
            self.assertEqual(resolve_backend('auto'), 'torch')

    def test_torch_backend_without_kokoro_explains_the_extra(self):
        with mock.patch.object(backends, 'torch_available', return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                get_pipeline('af_sky', backend='torch')
        self.assertIn('.[torch]', str(ctx.exception))

    def test_mlx_requested_off_apple_silicon_says_so(self):
        with mock.patch.object(backends, 'mlx_available', return_value=False), \
             mock.patch.object(backends, 'is_apple_silicon', return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                get_pipeline('af_sky', backend='mlx')
        self.assertIn('Apple Silicon', str(ctx.exception))


class InitialChoresPerSecTest(unittest.TestCase):
    def test_mlx_seed_is_the_measured_figure(self):
        with mock.patch.object(backends, 'resolve_backend', return_value='mlx'):
            self.assertEqual(backends.initial_chars_per_sec('auto'),
                             backends.CHARS_PER_SEC_GUESS['mlx'])

    def test_torch_without_cuda_uses_the_cpu_seed(self):
        with mock.patch.object(backends, 'resolve_backend', return_value='torch'), \
             mock.patch.dict(sys.modules, {'torch': None}):
            self.assertEqual(backends.initial_chars_per_sec('torch'),
                             backends.CHARS_PER_SEC_GUESS['torch_cpu'])

    def test_chinese_gets_its_own_seed(self):
        # A CJK character is worth far more speech than a Latin one, so the flat seed is
        # ~6x optimistic for Chinese and the first ETA is badly wrong without this.
        with mock.patch.object(backends, 'resolve_backend', return_value='mlx'):
            self.assertEqual(backends.initial_chars_per_sec('mlx', 'z'), 150)
            self.assertEqual(backends.initial_chars_per_sec('mlx', 'zf_xiaoxiao'), 150)

    def test_other_languages_keep_the_default_seed(self):
        with mock.patch.object(backends, 'resolve_backend', return_value='mlx'):
            for lang in ('a', 'b', 'f', None):
                self.assertEqual(backends.initial_chars_per_sec('mlx', lang),
                                 backends.CHARS_PER_SEC_GUESS['mlx'])


class MlxAdapterTest(unittest.TestCase):
    def build(self, lang_code='a', repo_id=None):
        model = FakeMlxModel()
        with mock.patch.dict(sys.modules, fake_mlx_audio(model)), \
             mock.patch.object(backends, 'set_espeak_data_path', return_value='/data'):
            pipeline = backends.MlxKokoroPipeline(lang_code, repo_id)
        return pipeline, model

    def test_yields_kokoro_shaped_triples_with_numpy_audio(self):
        pipeline, _ = self.build()
        out = list(pipeline('Hello.', voice='af_sky', speed=1.0))
        self.assertEqual(len(out), 2)
        for graphemes, phonemes, audio in out:
            self.assertIsNone(graphemes)
            self.assertIsNone(phonemes)
            self.assertIsInstance(audio, np.ndarray)
        # gen_audio_segments concatenates these, so they must survive np.concatenate
        self.assertEqual(len(np.concatenate([a for _, _, a in out])), 200)

    def test_passes_voice_speed_and_lang_through(self):
        pipeline, model = self.build(lang_code='z')
        list(pipeline('你好。', voice='zf_xiaobei', speed=1.5, split_pattern=r'\n\n\n'))
        self.assertEqual(model.calls, [dict(text='你好。', voice='zf_xiaobei', speed=1.5,
                                            lang_code='z', split_pattern=r'\n\n\n')])

    def test_defaults_to_the_mlx_repo(self):
        pipeline, _ = self.build()
        self.assertEqual(pipeline.repo_id, backends.DEFAULT_REPOS['mlx'])

    def test_explicit_repo_id_wins(self):
        pipeline, _ = self.build(repo_id='mlx-community/Kokoro-82M-4bit')
        self.assertEqual(pipeline.repo_id, 'mlx-community/Kokoro-82M-4bit')


class GetPipelineTest(unittest.TestCase):
    def test_mlx_requested_but_unavailable_raises_actionable_error(self):
        with mock.patch.object(backends, 'mlx_available', return_value=False), \
             mock.patch.object(backends, 'is_apple_silicon', return_value=True):
            with self.assertRaises(RuntimeError) as ctx:
                get_pipeline('af_sky', backend='mlx')
        self.assertIn('pip install mlx-audio', str(ctx.exception))

    def test_lang_code_defaults_to_first_letter_of_voice(self):
        with mock.patch.object(backends, 'mlx_available', return_value=True):
            model = FakeMlxModel()
            with mock.patch.dict(sys.modules, fake_mlx_audio(model)), \
                 mock.patch.object(backends, 'set_espeak_data_path', return_value='/data'):
                pipeline = get_pipeline('zf_xiaobei', backend='mlx')
        self.assertEqual(pipeline.lang_code, 'z')

    def test_explicit_lang_code_overrides_voice_prefix(self):
        # A custom .pt voice has no language prefix: 'voice[0]' would be '/'.
        with mock.patch.object(backends, 'mlx_available', return_value=True):
            model = FakeMlxModel()
            with mock.patch.dict(sys.modules, fake_mlx_audio(model)), \
                 mock.patch.object(backends, 'set_espeak_data_path', return_value='/data'):
                pipeline = get_pipeline('/voices/custom.pt', lang_code='b', backend='mlx')
        self.assertEqual(pipeline.lang_code, 'b')


class FakeQwenModel:
    """Stands in for mlx-audio's Qwen3-TTS, which also accepts temperature and top_p."""

    def __init__(self):
        self.calls = []

    def generate(self, text, voice, speed, lang_code, split_pattern=None, temperature=None,
                 top_p=None):
        self.calls.append(dict(text=text, voice=voice, speed=speed, lang_code=lang_code,
                               split_pattern=split_pattern, temperature=temperature,
                               top_p=top_p))
        yield FakeResult(np.zeros(64, dtype=np.float32))


def fake_mlx_core(seeds):
    """Stub mlx.core that records every seed the adapter sets.

    Real mlx is installed here, so seeding would silently succeed and prove nothing;
    capturing the calls is what shows the adapter seeds per call rather than per pipeline.
    """
    core = types.ModuleType('mlx.core')
    random = types.ModuleType('mlx.core.random')
    random.seed = seeds.append
    core.random = random
    pkg = types.ModuleType('mlx')
    pkg.core = core
    return {'mlx': pkg, 'mlx.core': core, 'mlx.core.random': random}


class ModelRegistryTest(unittest.TestCase):
    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            backends.model_spec('festival-tts')
        self.assertIn('festival-tts', str(ctx.exception))

    def test_qwen_on_torch_names_the_supported_runtime(self):
        with mock.patch.object(backends, 'torch_available', return_value=True):
            with self.assertRaises(RuntimeError) as ctx:
                get_pipeline('ryan', backend='torch', model='qwen3-tts')
        message = str(ctx.exception)
        self.assertIn('mlx', message)
        self.assertIn('kokoro', message)

    def test_default_model_is_unchanged(self):
        # Regression guard for every existing call site: no model= must behave as before.
        with mock.patch.object(backends, 'mlx_available', return_value=True):
            model = FakeMlxModel()
            with mock.patch.dict(sys.modules, fake_mlx_audio(model)), \
                 mock.patch.object(backends, 'set_espeak_data_path', return_value='/data'):
                pipeline = get_pipeline('af_sky', backend='mlx')
        self.assertEqual(pipeline.repo_id, backends.MODELS['kokoro']['repos']['mlx'])
        self.assertEqual(pipeline.lang_code, 'a')


class LanguageMappingTest(unittest.TestCase):
    def test_kokoro_codes_map_to_qwen_names(self):
        self.assertEqual(backends.resolve_lang_code('a', 'qwen3-tts'), 'english')
        self.assertEqual(backends.resolve_lang_code('b', 'qwen3-tts'), 'english')
        self.assertEqual(backends.resolve_lang_code('z', 'qwen3-tts'), 'chinese')

    def test_qwen_names_pass_through(self):
        self.assertEqual(backends.resolve_lang_code('chinese', 'qwen3-tts'), 'chinese')
        self.assertEqual(backends.resolve_lang_code('german', 'qwen3-tts'), 'german')

    def test_kokoro_is_left_alone(self):
        self.assertEqual(backends.resolve_lang_code('a', 'kokoro'), 'a')

    def test_hindi_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            backends.resolve_lang_code('h', 'qwen3-tts')
        self.assertIn('Hindi', str(ctx.exception))

    def test_qwen_voice_does_not_derive_language_from_its_name(self):
        # 'ryan'[0] == 'r', which is not a language; it must fall back to auto-detect.
        self.assertEqual(backends.default_lang_code('ryan', 'qwen3-tts'), 'auto')
        self.assertEqual(backends.default_lang_code('af_sky', 'kokoro'), 'a')


class ThroughputSeedTest(unittest.TestCase):
    def test_chinese_seed_differs_from_the_default(self):
        with mock.patch.object(backends, 'resolve_backend', return_value='mlx'):
            self.assertEqual(backends.initial_chars_per_sec('mlx', 'a', 'kokoro'), 900)
            self.assertEqual(backends.initial_chars_per_sec('mlx', 'z', 'kokoro'), 150)
            self.assertEqual(backends.initial_chars_per_sec('mlx', 'a', 'qwen3-tts'), 60)
            self.assertEqual(backends.initial_chars_per_sec('mlx', 'z', 'qwen3-tts'), 27)

    def test_unknown_language_falls_back_to_the_default(self):
        with mock.patch.object(backends, 'resolve_backend', return_value='mlx'):
            self.assertEqual(backends.initial_chars_per_sec('mlx', None, 'kokoro'), 900)
            self.assertEqual(backends.initial_chars_per_sec('mlx', 'f', 'kokoro'), 900)


class QwenAdapterTest(unittest.TestCase):
    def setUp(self):
        # The adapter seeds MLX's PRNG on every call, so exercising it needs an importable
        # mlx. Production never lacks one -- MlxPipeline only loads where mlx_audio does --
        # but the Linux CI job has neither, so the fake environment has to supply it or
        # every call here dies on the import rather than on anything under test. Tests that
        # assert on seeding layer their own recording stub over this one.
        patcher = mock.patch.dict(sys.modules, fake_mlx_core([]))
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, lang_code='english', temperature=None, top_p=None, seed=None):
        model = FakeQwenModel()
        with mock.patch.dict(sys.modules, fake_mlx_audio(model)), \
             mock.patch.object(backends, 'set_espeak_data_path', return_value='/data'):
            pipeline = backends.MlxPipeline(lang_code, model='qwen3-tts',
                                            temperature=temperature, top_p=top_p, seed=seed)
        return pipeline, model

    def test_forwards_a_low_temperature_by_default(self):
        pipeline, model = self.build()
        list(pipeline('Hello.', voice='ryan'))
        self.assertEqual(model.calls[0]['temperature'], backends.QWEN_DEFAULT_TEMPERATURE)
        self.assertLess(backends.QWEN_DEFAULT_TEMPERATURE, 0.9)  # below the library default

    def test_explicit_temperature_wins(self):
        pipeline, model = self.build(temperature=0.1)
        list(pipeline('Hello.', voice='ryan'))
        self.assertEqual(model.calls[0]['temperature'], 0.1)

    def test_zero_temperature_is_forwarded_but_warned_about(self):
        # mlx-audio branches to argmax at temperature <= 0, so 0 is a different algorithm
        # rather than the bottom of the scale -- and a measured runaway on this model.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipeline, model = self.build(temperature=0)
        list(pipeline('Hello.', voice='ryan'))
        self.assertEqual(model.calls[0]['temperature'], 0)
        self.assertEqual(len(caught), 1)
        self.assertIn('greedy', str(caught[0].message))

    def test_a_normal_temperature_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.build(temperature=0.1)
        self.assertEqual(caught, [])

    def test_kokoro_is_not_warned_about_a_zero_temperature(self):
        # Kokoro never receives the value, so there is nothing to warn about.
        model = FakeMlxModel()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with mock.patch.dict(sys.modules, fake_mlx_audio(model)), \
                 mock.patch.object(backends, 'set_espeak_data_path', return_value='/data'):
                backends.MlxPipeline('a', model='kokoro', temperature=0)
        self.assertEqual(caught, [])

    def test_negative_temperature_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.build(temperature=-0.5)
        self.assertIn('temperature', str(ctx.exception))

    def test_forwards_a_narrowed_top_p_by_default(self):
        pipeline, model = self.build()
        list(pipeline('Hello.', voice='ryan'))
        self.assertEqual(model.calls[0]['top_p'], backends.QWEN_DEFAULT_TOP_P)
        # mlx-audio only filters inside 0 < top_p < 1; at its own default of 1.0 the
        # distribution tail is untouched, which is the drift this setting exists to curb.
        self.assertLess(backends.QWEN_DEFAULT_TOP_P, 1.0)

    def test_explicit_top_p_wins(self):
        pipeline, model = self.build(top_p=0.5)
        list(pipeline('Hello.', voice='ryan'))
        self.assertEqual(model.calls[0]['top_p'], 0.5)

    def test_top_p_outside_the_usable_range_is_rejected(self):
        # mlx-audio would accept and ignore these, narrating a whole book unfiltered.
        for bad in (8, 0, -0.5, 1.5):
            with self.assertRaises(ValueError, msg=f'top_p={bad} should be rejected') as ctx:
                self.build(top_p=bad)
            self.assertIn('top_p', str(ctx.exception))

    def test_seeds_the_rng_before_every_call(self):
        pipeline, _ = self.build()
        seeds = []
        with mock.patch.dict(sys.modules, fake_mlx_core(seeds)):
            list(pipeline('One.', voice='ryan'))
            list(pipeline('Two.', voice='ryan'))
        # Per call, not per pipeline: re-running one chapter must reproduce it exactly.
        self.assertEqual(seeds, [backends.QWEN_DEFAULT_SEED, backends.QWEN_DEFAULT_SEED])

    def test_explicit_seed_wins(self):
        pipeline, _ = self.build(seed=1234)
        seeds = []
        with mock.patch.dict(sys.modules, fake_mlx_core(seeds)):
            list(pipeline('Hello.', voice='ryan'))
        self.assertEqual(seeds, [1234])

    def test_a_negative_seed_opts_out_of_seeding(self):
        pipeline, _ = self.build(seed=-1)
        seeds = []
        with mock.patch.dict(sys.modules, fake_mlx_core(seeds)):
            list(pipeline('Hello.', voice='ryan'))
        self.assertEqual(seeds, [], 'a negative seed should leave the RNG alone')

    def test_kokoro_is_never_seeded(self):
        model = FakeMlxModel()
        with mock.patch.dict(sys.modules, fake_mlx_audio(model)), \
             mock.patch.object(backends, 'set_espeak_data_path', return_value='/data'):
            pipeline = backends.MlxPipeline('a', model='kokoro')
        seeds = []
        with mock.patch.dict(sys.modules, fake_mlx_core(seeds)):
            list(pipeline('Hello.', voice='af_sky'))
        self.assertEqual(seeds, [], 'Kokoro does not sample; seeding it would mislead')

    def test_kokoro_is_never_given_a_temperature(self):
        model = FakeMlxModel()  # its generate() has no temperature parameter at all
        with mock.patch.dict(sys.modules, fake_mlx_audio(model)), \
             mock.patch.object(backends, 'set_espeak_data_path', return_value='/data'):
            pipeline = backends.MlxPipeline('a', model='kokoro')
        list(pipeline('Hello.', voice='af_sky'))
        self.assertEqual(len(model.calls), 1)

    def test_unsupported_speed_warns_once_but_does_not_raise(self):
        pipeline, model = self.build()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            list(pipeline('One.', voice='ryan', speed=1.5))
            list(pipeline('Two.', voice='ryan', speed=1.5))
        self.assertEqual(len(caught), 1, 'the warning should not repeat per sentence')
        self.assertIn('speed', str(caught[0].message))
        self.assertEqual(len(model.calls), 2)

    def test_speed_of_one_is_silent(self):
        pipeline, _ = self.build()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            list(pipeline('Hello.', voice='ryan', speed=1.0))
        self.assertEqual(caught, [])

    def test_lang_code_is_translated_at_construction(self):
        pipeline, _ = self.build(lang_code='z')
        self.assertEqual(pipeline.lang_code, 'chinese')


class VoiceRegistryTest(unittest.TestCase):
    def test_qwen_speakers_are_listed(self):
        from audiblez.voices import flat_voices
        speakers = flat_voices('qwen3-tts')
        self.assertEqual(len(speakers), 9)
        self.assertIn('ryan', speakers)
        self.assertIn('serena', speakers)

    def test_kokoro_table_is_unchanged(self):
        from audiblez.voices import flat_voices
        self.assertEqual(len(flat_voices('kokoro')), 54)

    def test_voice_language_lookup(self):
        from audiblez.voices import voice_language
        self.assertEqual(voice_language('ryan', 'qwen3-tts'), 'a')
        self.assertEqual(voice_language('serena', 'qwen3-tts'), 'z')
        self.assertIsNone(voice_language('af_sky', 'qwen3-tts'))

    def test_every_qwen_language_has_a_flag(self):
        # The GUI prefixes a flag and get_selected_voice strips it again; a missing flag
        # would leave the emoji in the voice name and reach the model as an unknown speaker.
        from audiblez.voices import flags, voices_for
        for lang in voices_for('qwen3-tts'):
            self.assertIn(lang, flags)

    def test_unknown_model_is_rejected(self):
        from audiblez.voices import voices_for
        with self.assertRaises(ValueError):
            voices_for('festival-tts')


class MeasuredEtaTest(unittest.TestCase):
    def stats(self, processed, elapsed, seed=900):
        import time
        from types import SimpleNamespace
        return SimpleNamespace(processed_chars=processed, chars_per_sec=seed,
                               start_time=time.time() - elapsed, total_chars=100000)

    def test_keeps_the_seed_until_there_is_enough_data(self):
        from audiblez.core import measured_chars_per_sec
        self.assertEqual(measured_chars_per_sec(self.stats(processed=100, elapsed=60)), 900)
        self.assertEqual(measured_chars_per_sec(self.stats(processed=50000, elapsed=1)), 900)

    def test_switches_to_real_throughput(self):
        from audiblez.core import measured_chars_per_sec
        rate = measured_chars_per_sec(self.stats(processed=20000, elapsed=100))
        self.assertAlmostEqual(rate, 200, delta=5)

    def test_missing_start_time_does_not_explode(self):
        from types import SimpleNamespace
        from audiblez.core import measured_chars_per_sec
        stats = SimpleNamespace(processed_chars=50000, chars_per_sec=900, total_chars=100000)
        self.assertEqual(measured_chars_per_sec(stats), 900)

    def test_resumed_chapters_do_not_inflate_the_rate(self):
        # Re-running a half-finished book credits existing chapters to processed_chars
        # instantly; only characters actually synthesized may drive the rate.
        import time
        from types import SimpleNamespace
        from audiblez.core import measured_chars_per_sec
        stats = SimpleNamespace(processed_chars=800000, synthesized_chars=0,
                                chars_per_sec=900, total_chars=850000,
                                start_time=time.time() - 20)
        self.assertEqual(measured_chars_per_sec(stats), 900)
        stats.synthesized_chars = 4000
        self.assertAlmostEqual(measured_chars_per_sec(stats), 200, delta=5)


class ConcatListTest(unittest.TestCase):
    """The ffmpeg concat list resolves relative entries against its own directory."""

    def build_list(self, tmpdir, names):
        import os
        from pathlib import Path
        from audiblez.core import concat_wavs_with_ffmpeg
        from unittest import mock
        out = Path(tmpdir) / 'audiobooks'
        out.mkdir()
        files = []
        for n in names:
            p = out / n
            p.write_bytes(b'')
            files.append(p)
        written = {}

        def fake_run(cmd, *a, **k):
            list_path = [c for c in cmd if str(c).endswith('_wav_list.txt')][0]
            written['content'] = Path(list_path).read_text()
            Path(cmd[-1]).write_bytes(b'x')  # pretend ffmpeg produced the concat file
            return mock.Mock(returncode=0)

        with mock.patch('audiblez.core.subprocess.run', side_effect=fake_run), \
             mock.patch('audiblez.core.find_aac_encoder', return_value='aac'):
            concat_wavs_with_ffmpeg(files, str(out), 'book.epub')
        return written['content']

    def test_entries_are_absolute(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            content = self.build_list(tmp, ['a.wav', 'b.wav'])
        for line in content.strip().splitlines():
            self.assertRegex(line, r"^file '/", f'relative entry would be resolved twice: {line}')

    def test_single_quotes_are_escaped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            content = self.build_list(tmp, ["it's.wav"])
        self.assertIn(r"'\''", content)


class SafeFilenameTest(unittest.TestCase):
    def test_blended_voice_is_safe(self):
        self.assertEqual(safe_filename_part('af_heart,af_bella'), 'af_heart_af_bella')

    def test_voice_path_is_safe(self):
        self.assertEqual(safe_filename_part('/voices/my custom.pt'), 'voices_my_custom.pt')

    def test_plain_voice_is_untouched(self):
        self.assertEqual(safe_filename_part('af_sky'), 'af_sky')


if __name__ == '__main__':
    unittest.main()
