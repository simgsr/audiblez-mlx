import sys
import types
import unittest
from unittest import mock

import numpy as np

from audiblez import backends
from audiblez.backends import get_pipeline, mlx_available, resolve_backend
from audiblez.core import safe_filename_part
from audiblez.voices import lang_code_for


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
        self.assertEqual(resolve_backend('edge'), 'edge')

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


class FakeEdgeTTS:
    """Stands in for the edge_tts package, recording how Communicate was called.

    stream() yields one audio chunk holding `mp3_bytes` -- a real MP3, so the pipeline's
    soundfile decode is exercised rather than mocked away.
    """

    def __init__(self, mp3_bytes):
        self.mp3_bytes = mp3_bytes
        self.calls = []

    def Communicate(self, text, voice, rate=None, **kwargs):
        self.calls.append(dict(text=text, voice=voice, rate=rate, **kwargs))
        return self

    async def stream(self):
        yield {'type': 'audio', 'data': self.mp3_bytes}


def fake_edge_tts_module(fake):
    pkg = types.ModuleType('edge_tts')
    pkg.Communicate = fake.Communicate
    return {'edge_tts': pkg}


def make_mp3():
    import io
    import numpy as np
    import soundfile
    buf = io.BytesIO()
    soundfile.write(buf, np.zeros(1000, dtype=np.float32), 24000, format='MP3')
    return buf.getvalue()


class EdgePipelineTest(unittest.TestCase):
    def build(self, lang_code='zh-TW'):
        fake = FakeEdgeTTS(make_mp3())
        pipeline = backends.EdgePipeline(lang_code)
        return pipeline, fake

    def run_pipeline(self, pipeline, fake, *args, **kwargs):
        # The pipeline imports edge_tts inside __call__, so the fake must be in
        # sys.modules for the call itself, not just for construction.
        with mock.patch.dict(sys.modules, fake_edge_tts_module(fake)):
            return list(pipeline(*args, **kwargs))

    def test_yields_kokoro_shaped_triples_with_numpy_audio(self):
        pipeline, fake = self.build()
        out = self.run_pipeline(pipeline, fake, '你好。', voice='zh-TW-HsiaoChenNeural', speed=1.0)
        self.assertEqual(len(out), 1)
        for graphemes, phonemes, audio in out:
            self.assertIsNone(graphemes)
            self.assertIsNone(phonemes)
            self.assertIsInstance(audio, np.ndarray)
        # gen_audio_segments concatenates these, so they must survive np.concatenate
        self.assertGreater(len(np.concatenate([a for _, _, a in out])), 0)

    def test_maps_speed_to_rate(self):
        pipeline, fake = self.build()
        self.run_pipeline(pipeline, fake, 'Hello.', voice='en-US-AriaNeural', speed=1.5)
        self.run_pipeline(pipeline, fake, 'Hello.', voice='en-US-AriaNeural', speed=0.5)
        self.run_pipeline(pipeline, fake, 'Hello.', voice='en-US-AriaNeural', speed=1.0)
        self.assertEqual([c['rate'] for c in fake.calls], ['+50%', '-50%', '+0%'])

    def test_passes_text_and_voice_through(self):
        pipeline, fake = self.build()
        self.run_pipeline(pipeline, fake, 'Hello.', voice='en-US-AriaNeural', speed=1.0)
        self.assertEqual(fake.calls[0]['text'], 'Hello.')
        self.assertEqual(fake.calls[0]['voice'], 'en-US-AriaNeural')

    def test_ignores_split_pattern(self):
        # edge-tts needs no sentence-level split pattern; the argument must not explode.
        pipeline, fake = self.build()
        out = self.run_pipeline(pipeline, fake, 'Hello.', voice='en-US-AriaNeural',
                                speed=1.0, split_pattern=r'\n\n\n')
        self.assertEqual(len(out), 1)


class ScriptedEdgeTTS:
    """edge_tts stand-in whose outcome is scripted, one entry per Communicate call.

    An entry is either the bytes to stream back or an exception to raise. Empty bytes
    reproduce what the real service does for text it finds nothing to say in: it streams no
    chunk at all and raises nothing.
    """

    def __init__(self, script=()):
        self.script = list(script)
        self.calls = []
        self._next = b''

    def Communicate(self, text, voice, rate=None, **kwargs):
        self.calls.append(dict(text=text, voice=voice, rate=rate))
        self._next = self.script.pop(0) if self.script else b''
        return self

    async def stream(self):
        if isinstance(self._next, Exception):
            raise self._next
        if self._next:
            yield {'type': 'audio', 'data': self._next}


class EdgeUnspeakableTextTest(unittest.TestCase):
    """The bug that killed a 15-chapter book 8% in.

    Sentence splitting hands back a bare '\\n' as the last "sentence" of a chapter. edge-tts
    splits that into zero chunks, then streams nothing and raises nothing, so empty bytes
    reached soundfile and surfaced as 'Format not recognised'.
    """

    def run_text(self, text, script=()):
        fake = ScriptedEdgeTTS(script)
        pipeline = backends.EdgePipeline('zh-CN')
        with mock.patch.dict(sys.modules, fake_edge_tts_module(fake)):
            return list(pipeline(text, voice='zh-CN-XiaoxiaoNeural', speed=1.0)), fake

    def test_bare_newline_is_skipped_without_a_request(self):
        out, fake = self.run_text('\n')
        self.assertEqual(out, [])
        self.assertEqual(fake.calls, [], 'nothing to say means nothing to ask the service')

    def test_punctuation_and_whitespace_only_is_skipped(self):
        for text in ['「」', '……', '   ', '', '—', '　　']:
            out, fake = self.run_text(text)
            self.assertEqual(out, [], f'{text!r} should produce no audio')
            self.assertEqual(fake.calls, [], f'{text!r} should not be sent')

    def test_real_text_is_still_synthesized(self):
        out, fake = self.run_text('一九六五年春于美爱荷华城。', script=[make_mp3()])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(fake.calls), 1)

    def test_speakable_covers_han_kana_latin_and_digits(self):
        for text in ['你', 'カナ', 'a', '7']:
            out, fake = self.run_text(text, script=[make_mp3()])
            self.assertEqual(len(fake.calls), 1, f'{text!r} should be sent')


class EdgeRetryTest(unittest.TestCase):
    """Edge is a network service; one dropped response must not cost a multi-hour book."""

    def call(self, script, text='你好。'):
        fake = ScriptedEdgeTTS(script)
        pipeline = backends.EdgePipeline('zh-CN')
        with mock.patch.dict(sys.modules, fake_edge_tts_module(fake)), \
             mock.patch.object(backends, 'time'):  # no real backoff in tests
            return list(pipeline(text, voice='zh-CN-XiaoxiaoNeural', speed=1.0)), fake

    def test_empty_response_is_retried_then_succeeds(self):
        out, fake = self.call([b'', make_mp3()])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(fake.calls), 2)

    def test_persistent_silence_skips_the_sentence_rather_than_crashing(self):
        out, fake = self.call([b''] * backends.EDGE_ATTEMPTS)
        self.assertEqual(out, [], 'one silent sentence must not lose the finished chapters')
        self.assertEqual(len(fake.calls), backends.EDGE_ATTEMPTS)

    def test_transient_exception_is_retried(self):
        out, fake = self.call([RuntimeError('websocket dropped'), make_mp3()])
        self.assertEqual(len(out), 1)
        self.assertEqual(len(fake.calls), 2)

    def test_no_audio_received_is_retried_then_succeeds(self):
        # The reported crash: '②"。' raised NoAudioReceived three times mid-book and read
        # fine moments later. It is speakable -- the service was throttling.
        from edge_tts.exceptions import NoAudioReceived
        out, fake = self.call([NoAudioReceived('no audio'), make_mp3()], text='②"。')
        self.assertEqual(len(out), 1)
        self.assertEqual(len(fake.calls), 2)

    def test_persistent_no_audio_skips_rather_than_killing_the_book(self):
        # However often the service says "no audio", it is never a reason to lose the
        # finished chapters -- that is what turned a throttled fragment into a dead run.
        from edge_tts.exceptions import NoAudioReceived
        out, fake = self.call([NoAudioReceived('no audio')] * backends.EDGE_ATTEMPTS,
                              text='②"。')
        self.assertEqual(out, [])
        self.assertEqual(len(fake.calls), backends.EDGE_ATTEMPTS)

    def test_transport_failure_still_raises(self):
        # A service that cannot be reached is a broken run, not a skippable fragment, so
        # this must stay fatal rather than silently producing a book full of gaps.
        with self.assertRaises(RuntimeError):
            self.call([ConnectionError('unreachable')] * backends.EDGE_ATTEMPTS)

    def test_backoff_outlasts_throttling_and_holds_at_the_last_step(self):
        waits = [backends._retry_wait(a) for a in range(1, backends.EDGE_ATTEMPTS + 1)]
        self.assertEqual(waits[:3], [3.0, 8.0, 20.0])
        self.assertEqual(waits[-1], 20.0, 'later attempts hold at the longest wait')
        self.assertGreaterEqual(sum(waits[:backends.EDGE_ATTEMPTS - 1]), 30,
                                'total backoff must outlast throttling, not a round trip')

    def test_persistent_failure_raises_naming_the_text(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.call([RuntimeError('boom')] * backends.EDGE_ATTEMPTS)
        self.assertIn(f'after {backends.EDGE_ATTEMPTS} attempts', str(ctx.exception))
        self.assertIn('你好。', str(ctx.exception))


class EdgeAvailableTest(unittest.TestCase):
    def test_true_when_edge_tts_installed(self):
        with mock.patch.dict(sys.modules, {'edge_tts': types.ModuleType('edge_tts')}):
            self.assertTrue(backends.edge_available())

    def test_false_when_not_installed(self):
        with mock.patch.dict(sys.modules, {'edge_tts': None}):
            self.assertFalse(backends.edge_available())


class LangCodeForTest(unittest.TestCase):
    def test_edge_voice_uses_its_locale(self):
        self.assertEqual(lang_code_for('zh-TW-HsiaoChenNeural'), 'zh-TW')
        self.assertEqual(lang_code_for('zh-HK-HiuMaanNeural'), 'zh-HK')
        self.assertEqual(lang_code_for('en-US-AriaNeural'), 'en-US')

    def test_kokoro_voice_uses_first_letter(self):
        self.assertEqual(lang_code_for('af_sky'), 'a')
        self.assertEqual(lang_code_for('zf_xiaobei'), 'z')

    def test_explicit_code_wins(self):
        self.assertEqual(lang_code_for('af_sky', 'b'), 'b')
        self.assertEqual(lang_code_for('zh-TW-HsiaoChenNeural', 'z'), 'z')


class EdgeGetPipelineTest(unittest.TestCase):
    def test_edge_backend_rejects_a_kokoro_voice(self):
        with mock.patch.object(backends, 'edge_available', return_value=True):
            with self.assertRaises(RuntimeError) as ctx:
                get_pipeline('af_sky', backend='edge')
        self.assertIn('not an Edge TTS voice', str(ctx.exception))

    def test_edge_backend_without_package_explains_the_extra(self):
        with mock.patch.object(backends, 'edge_available', return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                get_pipeline('zh-TW-HsiaoChenNeural', backend='edge')
        self.assertIn('.[edge]', str(ctx.exception))

    def test_edge_backend_returns_an_edge_pipeline(self):
        with mock.patch.object(backends, 'edge_available', return_value=True):
            pipeline = get_pipeline('zh-TW-HsiaoChenNeural', backend='edge')
        self.assertIsInstance(pipeline, backends.EdgePipeline)
        self.assertEqual(pipeline.lang_code, 'zh-TW')


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


class GenTextLangCodeTest(unittest.TestCase):
    """gen_text has to resolve the language the same way main() does.

    It used to take voice[:1], which reads 'zh-TW-HsiaoChenNeural' as Kokoro's 'z' -- so a
    Taiwan voice, which reads traditional script natively, got its text simplified first.
    """

    def gen_text_lang_code(self, voice, lang_code=None):
        """The lang_code gen_text hands to the pipeline and to gen_audio_segments."""
        from audiblez import core
        seen = {}

        def fake_get_pipeline(voice, lang_code=None, backend='auto', repo_id=None):
            seen['pipeline'] = lang_code
            return mock.Mock()

        def fake_gen_audio_segments(pipeline, text, voice, speed, **kwargs):
            seen['segments'] = kwargs.get('lang_code')
            return [np.zeros(4, dtype='float32')]

        with mock.patch.object(core, 'get_pipeline', side_effect=fake_get_pipeline), \
             mock.patch.object(core, 'gen_audio_segments', side_effect=fake_gen_audio_segments), \
             mock.patch.object(core, 'set_espeak_library'), \
             mock.patch.object(core, 'load_spacy'), \
             mock.patch.object(core.soundfile, 'write'):
            core.gen_text('hello', voice=voice, lang_code=lang_code, output_file='x.wav')
        self.assertEqual(seen['pipeline'], seen['segments'],
                         'the pipeline and the text normalizer must agree on the language')
        return seen['pipeline']

    def test_edge_voice_keeps_its_full_locale(self):
        self.assertEqual(self.gen_text_lang_code('zh-TW-HsiaoChenNeural'), 'zh-TW')
        self.assertEqual(self.gen_text_lang_code('zh-HK-HiuMaanNeural'), 'zh-HK')
        self.assertEqual(self.gen_text_lang_code('en-US-AriaNeural'), 'en-US')

    def test_kokoro_voice_still_uses_its_first_letter(self):
        self.assertEqual(self.gen_text_lang_code('af_sky'), 'a')
        self.assertEqual(self.gen_text_lang_code('zf_xiaoxiao'), 'z')

    def test_explicit_lang_code_wins(self):
        self.assertEqual(self.gen_text_lang_code('/voices/custom.pt', lang_code='j'), 'j')


class SafeFilenameTest(unittest.TestCase):
    def test_blended_voice_is_safe(self):
        self.assertEqual(safe_filename_part('af_heart,af_bella'), 'af_heart_af_bella')

    def test_voice_path_is_safe(self):
        self.assertEqual(safe_filename_part('/voices/my custom.pt'), 'voices_my_custom.pt')

    def test_plain_voice_is_untouched(self):
        self.assertEqual(safe_filename_part('af_sky'), 'af_sky')


if __name__ == '__main__':
    unittest.main()
