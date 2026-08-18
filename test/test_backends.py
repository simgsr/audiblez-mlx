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

    def test_auto_falls_back_to_torch(self):
        with mock.patch.object(backends, 'mlx_available', return_value=False):
            self.assertEqual(resolve_backend('auto'), 'torch')

    def test_mlx_unavailable_off_apple_silicon(self):
        with mock.patch('platform.system', return_value='Linux'):
            self.assertFalse(mlx_available())
        with mock.patch('platform.system', return_value='Darwin'), \
             mock.patch('platform.machine', return_value='x86_64'):
            self.assertFalse(mlx_available())


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
        with mock.patch.object(backends, 'mlx_available', return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                get_pipeline('af_sky', backend='mlx')
        self.assertIn('audiblez[mlx]', str(ctx.exception))

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


class SafeFilenameTest(unittest.TestCase):
    def test_blended_voice_is_safe(self):
        self.assertEqual(safe_filename_part('af_heart,af_bella'), 'af_heart_af_bella')

    def test_voice_path_is_safe(self):
        self.assertEqual(safe_filename_part('/voices/my custom.pt'), 'voices_my_custom.pt')

    def test_plain_voice_is_untouched(self):
        self.assertEqual(safe_filename_part('af_sky'), 'af_sky')


if __name__ == '__main__':
    unittest.main()
