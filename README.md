# audiblez_mlx: Generate audiobooks from e-books, on Apple Silicon

![Audiblez GUI on MacOSX](./imgs/mac.png)

**This is a fork of [audiblez](https://github.com/santinic/audiblez) built for Apple Silicon and MLX.**

Upstream audiblez runs Kokoro through PyTorch, which on a Mac means the CPU: the original
README states "on my M2 MacBook Pro, on CPU, it takes about 1 hour... about 60 characters
per second". This fork runs Kokoro through [MLX](https://github.com/Blaizzy/mlx-audio)
instead, on the Apple GPU.

### What this fork is for

- **MLX is the default and the only required backend.** PyTorch is not installed unless you
  ask for it, which removes a ~2GB dependency that did nothing useful on a Mac.
- **Measured 906 characters/second on an M5 Max**, against 238 for the same model on torch —
  3.8x. A 15.7-hour audiobook took **87 minutes on torch, and about 20 on MLX**.
- **The time estimate is measured, not guessed.** Upstream hardcodes 50 chars/sec whenever
  CUDA is missing, so every Mac run mis-reported its ETA by roughly 4x.
- **Better text extraction**, which matters more than speed for a listenable result: page
  numbers, footnote markers, inline tables of contents, and index/copyright sections are no
  longer read aloud. See [What gets read](#what-gets-read).
- **Single-file books are split into real chapters** using the e-book's own table of
  contents, so the `.m4b` has usable chapter marks instead of one 15-hour track.

Everything else — the voices, the GUI, the languages — is upstream's work and is unchanged.
If you are not on Apple Silicon, use [the original project](https://github.com/santinic/audiblez);
it is better suited to you and this fork offers you nothing.

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a text-to-speech model with just 82M
params and very natural sounding output, released under the Apache licence and trained on
< 100 hours of audio. It supports: 🇺🇸 🇬🇧 🇪🇸 🇫🇷 🇮🇳 🇮🇹 🇯🇵 🇧🇷 🇨🇳

## How to install

You need `espeak-ng` and `ffmpeg`, plus Python 3.10–3.12 (spacy has no wheels for 3.13+).

This fork is **not published to PyPI** — install it from a clone:

```bash
brew install ffmpeg espeak-ng                       # on Mac 🍏
git clone https://github.com/simgsr/audiblez_mlx
cd audiblez_mlx
pip install .
```

That installs the MLX backend. To also get the portable torch backend — for comparison, or
on a non-Apple machine — add the extra (this is what pulls in PyTorch):

```bash
pip install ".[torch]"
```

> On a non-Apple machine the `[torch]` extra is **required**, not optional: `mlx-audio` only
> installs on Apple Silicon, so a plain `pip install .` elsewhere leaves you with no speech
> engine at all.

### Environment variables

Both are optional; they are auto-detected from a Homebrew install and only need setting if
detection fails.

| Variable | Purpose |
|---|---|
| `ESPEAK_LIBRARY` | Path to `libespeak-ng.dylib`/`.so`/`.dll` |
| `ESPEAK_DATA_PATH` | Path to the `espeak-ng-data` directory. The MLX path needs this because `misaki`'s `espeakng-loader` ships a hardcoded build-machine path that does not exist on your system |

Then you can convert an .epub directly with:

```
audiblez book.epub -v af_sky
```

It will first create a bunch of `book_chapter_1.wav`, `book_chapter_2.wav`, etc. files in the same directory,
and at the end it will produce a `book.m4b` file with the whole book you can listen with VLC or any
audiobook player.
It will only produce the `.m4b` file if you have `ffmpeg` installed on your machine.

## How to run the GUI

The GUI is a simple graphical interface to use audiblez:

```
brew install ffmpeg espeak-ng
pip install ".[gui]"
```

Then you can run the GUI with:
```
audiblez-ui
```

The GUI shows which backend will actually run, and lets you pick one explicitly. The torch
device selector is greyed out when MLX is in use, since it has no effect there.

## Windows and Linux

This fork targets Apple Silicon and is not tested on Windows or Linux. The torch backend
still works there — `pip install ".[torch]"` and pass `--backend torch` — but if
that is your platform you want [the upstream project](https://github.com/santinic/audiblez)
instead, which supports it properly and has CI for it.

## Speed

By default the audio is generated using a normal speed, but you can make it up to twice slower or faster by specifying a speed argument between 0.5 to 2.0:

```
audiblez book.epub -v af_sky -s 1.5
```

Measured on an M5 Max, synthesizing the same paragraph with `af_sky`:

| Backend | Throughput | Faster than realtime |
|---|---|---|
| MLX (`Kokoro-82M-bf16`) | **906 chars/sec** | 47x |
| Torch (`Kokoro-82M`, CPU) | 238 chars/sec | 12x |

## Supported Voices

Use `-v` option to specify the voice to use. Available voices are listed here.
The first letter is the language code and the second is the gender of the speaker e.g. `im_nicola` is an italian male voice.

[For hearing samples of Kokoro-82M voices, go here](https://claudio.uk/posts/audiblez-v4.html)

| Language                  | Voices                                                                                                                                                                                                                                     |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🇺🇸 American English     | `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| 🇬🇧 British English      | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`                                                                                                                                          |
| 🇪🇸 Spanish              | `ef_dora`, `em_alex`, `em_santa`                                                                                                                                                                                                           |
| 🇫🇷 French               | `ff_siwis`                                                                                                                                                                                                                                 |
| 🇮🇳 Hindi                | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`                                                                                                                                                                                                |
| 🇮🇹 Italian              | `if_sara`, `im_nicola`                                                                                                                                                                                                                     |
| 🇯🇵 Japanese             | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`                                                                                                                                                                         |
| 🇧🇷 Brazilian Portuguese | `pf_dora`, `pm_alex`, `pm_santa`                                                                                                                                                                                                           |
| 🇨🇳 Mandarin Chinese     | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`                                                                                                                                 |

For more detaila about voice quality, check this document: [Kokoro-82M voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)

## Choosing a backend

MLX is used automatically on Apple Silicon. Pick an engine explicitly with `--backend`:

```
audiblez book.epub -v af_sky --backend mlx     # force MLX (Apple Silicon only)
audiblez book.epub -v af_sky --backend torch   # force Torch (needs the [torch] extra)
audiblez book.epub -v af_sky --backend auto    # default: MLX when available, else Torch
```

MLX runs the model in `bf16` where Torch runs `fp32`. Output is equivalent in duration
(within 0.25%) and loudness, but not sample-identical: small timing differences accumulate
across a chapter. Use `--backend torch` if you want upstream's exact output.

Point either backend at a different model repo, for example a quantized one:

```
audiblez book.epub -v af_sky --repo-id mlx-community/Kokoro-82M-8bit
```

If you have an Nvidia GPU, `--cuda` still selects it for the torch backend. It is ignored on
Apple Silicon, where MLX already runs on the GPU.

## What gets read

An e-book contains a lot of text that should not be spoken. This fork drops, before synthesis:

- **Page numbers** embedded in the text, whether marked with `epub:type="pagebreak"` or by a
  page-number CSS class
- **Footnote and endnote markers** — the superscript `12` that otherwise gets read as a number
  in the middle of a sentence — and the note bodies themselves
- **Inline tables of contents**, and any line that is nothing but internal links
- **Index, copyright, colophon and contents sections**, identified from the table of contents

It also fixes text that upstream *misses*: text is extracted by walking the document rather
than matching a fixed tag list, so prose in `<div>`s, table cells, or separated by `<br>` is
no longer dropped. One Chinese novel went from **39 characters extracted to 107,220** — it
had no `<p>` tags at all. Nested `<li><p>` text is also no longer read twice.

Sentence terminators are CJK-aware, so Chinese and Japanese lines no longer get a stray Latin
full stop appended after `。`.

## Chapters

Books that ship as one enormous XHTML file are split into chapters at the anchors their table
of contents points to, so the `.m4b` gets real chapter marks. Chapter titles come from the
book's own table of contents rather than filenames — a book whose files are `0.xhtml`,
`1.xhtml` … used to produce chapters named `0`, `1`, `2`.

Ordinary one-file-per-chapter books are left alone.

## Manually pick chapters to convert

Sometimes you want to manually select which chapters/sections in the e-book to read out loud.
To do so, you can use `--pick` to interactively choose the chapters to convert (without running the GUI).


## Help page

For all the options available, you can check the help page `audiblez --help`:

```
usage: audiblez [-h] [-v VOICE] [-p] [-s SPEED] [-c] [-o FOLDER] epub_file_path

positional arguments:
  epub_file_path        Path to the epub file

options:
  -h, --help            show this help message and exit
  -v VOICE, --voice VOICE
                        Choose narrating voice: a, b, e, f, h, i, j, p, z
  -p, --pick            Interactively select which chapters to read in the
                        audiobook
  -s SPEED, --speed SPEED
                        Set speed from 0.5 to 2.0
  -c, --cuda            Use an Nvidia GPU via Torch. Ignored on Apple Silicon,
                        where the mlx backend already runs on the GPU
  -o FOLDER, --output FOLDER
                        Output folder for the audiobook and temporary files
  -b {auto,torch,mlx}, --backend {auto,torch,mlx}
                        TTS engine: mlx is Apple-Silicon only and faster, auto
                        picks it when available
  --lang CODE           Kokoro language code (a, b, e, f, h, i, j, p, z).
                        Defaults to the first letter of the voice name; set it
                        when using a custom .pt voice
  --repo-id REPO        Hugging Face model repo to use instead of the backend
                        default

example:
  audiblez book.epub -l en-us -v af_sky

to run GUI just run:
  audiblez-ui
```

## Voices beyond the built-in list

Kokoro's `load_voice` accepts more than the names above, and both backends pass it through:

```
audiblez book.epub -v af_heart,af_bella          # blend two voices (averaged)
audiblez book.epub -v /path/to/voice.pt --lang a # a custom voice pack
```

Pass `--lang` with a `.pt` path, since the language can no longer be read off the voice name.

## Author

Upstream audiblez by [Claudio Santini](https://claudio.uk) in 2025, distributed under MIT licence.

Related Article: [Audiblez v4: Generate Audiobooks from E-books](https://claudio.uk/posts/audiblez-v4.html)

This Apple Silicon / MLX fork is maintained separately at
[simgsr/audiblez_mlx](https://github.com/simgsr/audiblez_mlx), under the same MIT licence.
MLX inference via [mlx-audio](https://github.com/Blaizzy/mlx-audio).
