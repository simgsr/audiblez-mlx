# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez import DEFAULT_OUTPUT_FOLDER
from audiblez.backends import edge_available, is_apple_silicon, mlx_available, torch_available
from audiblez.voices import voices, available_voices_str, edge_voices_str


def check_backend_installed(backend):
    """Fail fast with an actionable message when the chosen backend can't run.

    Without this, a bare `pip install .` on a non-Apple machine (where mlx-audio does
    not install) leaves you with no speech engine at all, and the failure only surfaces
    later and less clearly. This catches it up front and names the exact install command.
    """
    if backend == 'edge':
        if not edge_available():
            print('The edge backend is not installed. Add it with: pip install ".[edge]"')
            sys.exit(1)
        return
    if backend == 'mlx':
        if not mlx_available():
            print('The mlx backend is not available. ' + (
                'Install it with: pip install mlx-audio "misaki[en]"' if is_apple_silicon()
                else 'It needs Apple Silicon; use --backend torch on this machine.'))
            sys.exit(1)
        return
    if backend == 'torch':
        if not torch_available():
            print('The torch backend is not installed. Add it with: pip install ".[torch]"')
            sys.exit(1)
        return
    # auto: mlx when available, else torch. Edge is never chosen by auto.
    if mlx_available() or torch_available():
        return
    if edge_available():
        print('No local TTS backend is installed, but the Edge backend is. '
              'Run with --backend edge, or install a local backend with: '
              'pip install ".[torch]"')
        sys.exit(1)
    if is_apple_silicon():
        print('No TTS backend is installed. Run: pip install .  '
              '(or pip install ".[edge]" for Edge voices)')
    else:
        print('No TTS backend is installed. On this machine you need the torch backend:\n'
              '  pip install ".[torch,edge]"')
    sys.exit(1)


def cli_main():
    voices_str = ', '.join(voices)
    epilog = ('example:\n' +
              '  audiblez book.epub -v af_sky\n\n' +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available Kokoro voices:\n' +
              available_voices_str + '\n\n' +
              'Edge voices (online, needs network; use --backend edge):\n' +
              edge_voices_str)
    default_voice = 'af_sky'
    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', help='Path to the epub file')
    parser.add_argument('-v', '--voice', default=default_voice, help=f'Choose narrating voice: {voices_str}')
    parser.add_argument('-p', '--pick', default=False, help=f'Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-s', '--speed', default=1.0, help=f'Set speed from 0.5 to 2.0', type=float)
    parser.add_argument('-c', '--cuda', default=False, action='store_true',
                        help='Use an Nvidia GPU via Torch. Ignored on Apple Silicon, where the '
                             'mlx backend already runs on the GPU')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_FOLDER, metavar='FOLDER',
                        help=f'Output folder for the audiobook and intermediate files '
                             f'(default: {DEFAULT_OUTPUT_FOLDER}/, created if missing)')
    parser.add_argument('-b', '--backend', default='auto', choices=['auto', 'torch', 'mlx', 'edge'],
                        help='TTS engine: mlx is Apple-Silicon only and faster, auto picks it when available; '
                             'edge is Microsoft\'s online TTS (needs network)')
    parser.add_argument('--lang', default=None, metavar='CODE',
                        help='Language code: Kokoro (a, b, e, f, h, i, j, p, z) or an Edge locale '
                             '(en-US, zh-TW, zh-HK, ...). Defaults to the first letter of the voice name, '
                             'or the locale for an Edge voice; set it when using a custom .pt voice')
    parser.add_argument('--repo-id', default=None, metavar='REPO',
                        help='Hugging Face model repo to use instead of the backend default')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    check_backend_installed(args.backend)

    if args.cuda:
        try:
            import torch
        except ImportError:
            print('--cuda needs the torch backend: pip install ".[torch]"')
            sys.exit(1)
        if torch.cuda.is_available():
            print('CUDA GPU available')
            torch.set_default_device('cuda')
        else:
            print('CUDA GPU not available. Defaulting to CPU')

    from audiblez.core import main
    main(args.epub_file_path, args.voice, args.pick, args.speed, args.output,
         backend=args.backend, lang_code=args.lang, repo_id=args.repo_id)


if __name__ == '__main__':
    cli_main()
