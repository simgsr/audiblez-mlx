import sys
import types
import unittest
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
