# Changelog

## 0.5.0 — first release of the `audiblez_mlx` fork

Forked from [santinic/audiblez](https://github.com/santinic/audiblez) at v0.4.9 and retargeted
at Apple Silicon. Measurements below are from an M5 Max.

### Added

- **MLX backend for Kokoro**, selected automatically on Apple Silicon. Measured **906
  chars/sec against 238** on the torch path — 3.8x. A 15.7-hour audiobook that took 87
  minutes on torch takes about 20 on MLX.
- `--backend {auto,torch,mlx}` to choose the engine, `--lang` to set the language code
  independently of the voice name, and `--repo-id` to select a different model repo
  (a quantized MLX build, for example).
- Books delivered as one enormous XHTML file are **split into chapters at their
  table-of-contents anchors**, so the `.m4b` gets usable chapter marks. Verified that split
  and unsplit text are byte-identical: the split moves boundaries, it never drops words.
- Chapter marks and the GUI chapter list are **named from the book's table of contents**
  instead of filenames.
- The GUI exposes the backend choice, greys out the torch device selector when MLX is in
  use, and accepts typed voices — comma blends (`af_heart,af_bella`) and paths to `.pt`
  voice packs, both of which the CLI already accepted.

### Changed

- **MLX is the default and only required backend.** torch and kokoro moved to a `[torch]`
  extra, removing a ~2GB dependency that did nothing useful on a Mac. `import audiblez.core`
  no longer imports torch at all.
- **The time estimate is measured rather than guessed.** It was a constant chosen by whether
  CUDA was present, so every Apple run reported the 50 chars/sec CPU figure — announcing
  nearly five hours for a book that took 87 minutes. It now recalibrates from real throughput.
- Text extraction walks the document instead of matching a fixed tag list. Text in `<div>`s,
  table cells and `<br>`-separated prose is no longer dropped: one Chinese novel went from
  **39 characters extracted to 107,220**, having no `<p>` tags at all.
- Sentence terminators are CJK-aware, so Chinese and Japanese lines no longer get a Latin
  full stop appended after `。`.
- CI targets macOS arm64 for the MLX path, with a Linux job covering the torch fallback.
  The upstream workflows installed the `audiblez` package from PyPI, which is not this code.

### Fixed

- **m4b creation was broken on any ffmpeg without libfdk_aac** (which is non-free, and
  excluded from Homebrew's default build). The concat step exited non-zero, the temp file was
  never written, and the run died on `unlink()` with `FileNotFoundError`. The encoder is now
  probed: libfdk_aac → aac_at → aac.
- Page numbers, footnote/endnote markers and their bodies, inline tables of contents, and
  index/copyright/contents sections are no longer read aloud.
- Text inside nested `<li><p>` was read twice.
- Empty paragraphs became a spoken `.` each.
- ffmetadata values are escaped, so chapter titles containing `=`, `;` or `#` no longer
  corrupt the chapter index.
- `audiblez.cli` used a bare `from core import main`, which only resolved when the working
  directory happened to be the package directory.
- The package no longer appends its own directory to `sys.path`, which had made its modules
  importable under bare names like `core` where they could shadow unrelated imports.
- The GUI no longer auto-opens `../epub/lewis.epub` on startup, a developer leftover.

### Removed

- `qwen-tts` and `epub-toc` dependencies — neither was imported anywhere, and `qwen-tts`
  pinned `transformers==4.57.3`, which blocked mlx-audio.
