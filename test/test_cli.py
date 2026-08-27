import os
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import mock

from audiblez.cli import check_backend_installed


class CliTest(unittest.TestCase):
    def cli(self, args):
        cmd = f'cd .. && python -m audiblez.cli {args}'
        return os.popen(cmd).read()

    def test_help(self):
        out = self.cli('--help')
        self.assertIn('af_sky', out)
        self.assertIn('usage:', out)
        self.assertIn('edge', out)
        self.assertIn('zh-TW-HsiaoChenNeural', out)

    def test_epub(self):
        out = self.cli('epub/mini.epub')
        self.assertIn('Found cover image', out)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path('../mini.m4b').exists())
        self.assertTrue(Path('../mini.m4b').stat().st_size > 256 * 1024)

    def test_epub_voice_and_output_folder(self):
        out = self.cli('epub/mini.epub -v af_sky -o test/prova')
        self.assertIn('Found cover image', out)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path('./prova/mini.m4b').exists())
        self.assertTrue(Path('./prova/mini.m4b').stat().st_size > 256 * 1024)

    @unittest.skip('Not implemented yet')
    def test_md(self):
        content = (
            '## Italy\n'
            'Italy, officially the Italian Republic, is a country in '
            '(Southern)[https://en.wikipedia.org/wiki/Southern_Europe] and Western Europe. '
            'It consists of a peninsula that extends into the Mediterranean Sea, '
            'with the Alps on its northern land border, '
            'as well as nearly 800 islands, notably Sicily and Sardinia.')
        file_name = NamedTemporaryFile('w', suffix='.txt', delete=False).write(content)
        out = self.cli(file_name)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path(file_name).exists())
        self.assertTrue(Path('file_name').stat().st_size > 256 * 1024)

    @unittest.skip('Not implemented yet')
    def test_txt(self):
        content = (
            'Italy, officially the Italian Republic, is a country in Southern and Western Europe. '
            'It consists of a peninsula that extends into the Mediterranean Sea, '
            'with the Alps on its northern land border, '
            'as well as nearly 800 islands, notably Sicily and Sardinia.')
        file_name = NamedTemporaryFile('w', suffix='.txt', delete=False).write(content)
        out = self.cli(file_name)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path('text.mp4').exists())
        self.assertTrue(Path('text.mp4').stat().st_size > 256 * 1024)


class CheckBackendInstalledTest(unittest.TestCase):
    """check_backend_installed fails fast with an actionable message when a backend
    can't run, and passes silently when it can."""

    def run_check(self, backend, mlx=False, torch=False, edge=False, apple=False):
        with mock.patch('audiblez.cli.mlx_available', return_value=mlx), \
             mock.patch('audiblez.cli.torch_available', return_value=torch), \
             mock.patch('audiblez.cli.edge_available', return_value=edge), \
             mock.patch('audiblez.cli.is_apple_silicon', return_value=apple), \
             mock.patch('audiblez.cli.sys.exit', side_effect=SystemExit) as exit_:
            try:
                check_backend_installed(backend)
            except SystemExit:
                pass
            return exit_.call_count

    def test_explicit_backend_present_passes(self):
        self.assertEqual(self.run_check('edge', edge=True), 0)
        self.assertEqual(self.run_check('torch', torch=True), 0)
        self.assertEqual(self.run_check('mlx', mlx=True), 0)

    def test_explicit_edge_missing_fails(self):
        self.assertEqual(self.run_check('edge', edge=False), 1)

    def test_explicit_torch_missing_fails(self):
        self.assertEqual(self.run_check('torch', torch=False), 1)

    def test_auto_with_local_backend_passes(self):
        self.assertEqual(self.run_check('auto', mlx=True), 0)
        self.assertEqual(self.run_check('auto', torch=True), 0)

    def test_auto_nothing_installed_fails(self):
        self.assertEqual(self.run_check('auto', apple=False), 1)
        self.assertEqual(self.run_check('auto', apple=True), 1)

    def test_auto_only_edge_installed_fails(self):
        # auto never picks edge, so it must tell the user to pass --backend edge.
        self.assertEqual(self.run_check('auto', edge=True), 1)
