# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez import DEFAULT_OUTPUT_FOLDER
from audiblez.backends import (DEFAULT_MODEL, MODELS, QWEN_DEFAULT_SEED,
                               QWEN_DEFAULT_TEMPERATURE, QWEN_DEFAULT_TOP_P)
from audiblez.voices import voices, describe_voices


def cli_main():
    voices_str = ', '.join(voices)
    # The epilog is built before parse_args, so it cannot show only the chosen model's
    # voices -- list both, grouped, and let the reader pick.
    epilog = ('example:\n' +
              '  audiblez book.epub -l en-us -v af_sky\n\n' +
              'to run GUI just run:\n'
              '  audiblez-ui\n\n' +
              'available voices (--model kokoro, the default):\n' +
              describe_voices('kokoro') + '\n\n' +
              'available voices (--model qwen3-tts):\n' +
              describe_voices('qwen3-tts'))
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
                        help='TTS runtime: mlx is Apple-Silicon only and faster, auto picks it when available')
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL, choices=sorted(MODELS),
                        help='TTS model. qwen3-tts is Apple-Silicon only, ~10x slower than kokoro in '
                             'English (~5x in Chinese), has 9 fixed voices and ignores --speed; it is '
                             'never chosen automatically')
    parser.add_argument('--lang', default=None, metavar='CODE',
                        help='Language: a Kokoro code (a, b, e, f, h, i, j, p, z) or, for --model '
                             'qwen3-tts, a name such as english/chinese/german. Defaults to the first '
                             'letter of the voice name for kokoro; set it when using a custom .pt voice')
    parser.add_argument('--repo-id', default=None, metavar='REPO',
                        help='Hugging Face model repo to use instead of the backend default')
    parser.add_argument('--temperature', default=None, metavar='T', type=float,
                        help=f'Sampling temperature for --model qwen3-tts (default: '
                             f'{QWEN_DEFAULT_TEMPERATURE}). Lower is steadier; measured '
                             f'sane down to 0.1. Avoid 0, which switches to greedy '
                             f'decoding: it runs to the token cap and bypasses --top-p '
                             f'and --seed, and audiblez warns if you ask for it. Ignored '
                             f'by kokoro, which does not sample')
    parser.add_argument('--top-p', default=None, metavar='P', type=float,
                        help=f'Nucleus sampling cutoff for --model qwen3-tts (default: '
                             f'{QWEN_DEFAULT_TOP_P}). Lower narrows the tone and pacing '
                             f'drift between chapters; must be above 0 and at most 1, '
                             f'where 1.0 filters nothing. Ignored by kokoro, which does '
                             f'not sample')
    parser.add_argument('--seed', default=None, metavar='N', type=int,
                        help=f'Random seed for --model qwen3-tts (default: '
                             f'{QWEN_DEFAULT_SEED}). The same text and seed always give '
                             f'the same audio, so a re-run chapter matches its '
                             f'neighbours; pass a negative value for fresh randomness on '
                             f'every run. Ignored by kokoro, which is already '
                             f'deterministic')

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
         backend=args.backend, lang_code=args.lang, repo_id=args.repo_id, model=args.model,
         temperature=args.temperature, top_p=args.top_p, seed=args.seed)


if __name__ == '__main__':
    cli_main()
