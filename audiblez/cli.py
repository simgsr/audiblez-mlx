# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez import DEFAULT_OUTPUT_FOLDER
from audiblez.voices import voices, available_voices_str


def cli_main():
    voices_str = ', '.join(voices)
    epilog = ('example:\n' +
              '  audiblez book.epub -l en-us -v af_sky\n\n' +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available voices:\n' +
              available_voices_str)
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
    parser.add_argument('-b', '--backend', default='auto', choices=['auto', 'torch', 'mlx'],
                        help='TTS engine: mlx is Apple-Silicon only and faster, auto picks it when available')
    parser.add_argument('--lang', default=None, metavar='CODE',
                        help='Kokoro language code (a, b, e, f, h, i, j, p, z). Defaults to the first letter '
                             'of the voice name; set it when using a custom .pt voice')
    parser.add_argument('--repo-id', default=None, metavar='REPO',
                        help='Hugging Face model repo to use instead of the backend default')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

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
