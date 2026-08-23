# Changelog

## Unreleased

Everything below has landed since the `v0.5.0` tag. Note the breaking change: Qwen3-TTS was
added and then removed again within this range, so it never appeared in a tagged release.

### Added

- **Edge TTS backend** (`--backend edge`, `pip install ".[edge]"`): Microsoft's online neural
  voices, across 16 locales. It needs network at synthesis time and sends the book's text to
  Microsoft's servers, so `auto` never selects it — it has to be asked for by name. Requests
  are retried with a backoff sized to outlast throttling rather than a network round trip.
- **Traditional Chinese support.** Traditional script is converted to simplified before
  phonemization, because misaki's jieba/pypinyin dictionaries are simplified-keyed; the text
  written into the `.m4b` keeps the book's original characters. An Edge `zh-TW` voice reads
  traditional script natively and skips the conversion, and `zh-HK` voices are real Cantonese
  rather than Mandarin.
- **A wake lock for the length of a run** (`caffeinate` on macOS, `systemd-inhibit` on Linux,
  `SetThreadExecutionState` on Windows), held right through the final ffmpeg pass so an idle
  machine cannot suspend halfway through a multi-hour conversion. The display is left alone.
- **A language filter in the GUI voice dropdown**, ticked to English and Chinese by default
  (`en-US`, `en-GB`, `zh-CN`, `zh-TW` on Edge) so the list is not buried under ~50 voices.
- All output now defaults to an **`audiobooks/`** folder, created if missing; `-o` overrides.

### Changed

- The GUI reports synthesis errors in a dialog instead of freezing with a disabled window.
- The time estimate is seeded per language as well as per backend — a CJK character carries
  far more phonemes than a Latin one, and the first ETA for a Chinese book was optimistic by
  4-6x without it.

### Fixed

- Edge no longer silently drops a sentence it returned no audio for. That wrote a truncated
  chapter `.wav`, which the resume path then skipped over, making the loss permanent; the
  chapter is now failed and left unwritten so a re-run redoes it.
- An Edge voice name with a 3-letter subtag (`yue-CN-…`, `fil-PH-…`) is recognised instead of
  being rejected as "not an Edge TTS voice".
- Selecting a Kokoro backend with an Edge voice fails immediately rather than after mlx has
  downloaded a 339 MB model repo.
- A hand-typed voice in the GUI — a `.pt` path, a blend, or an uncurated Edge voice — survives
  ticking a language checkbox instead of being silently replaced.
- jieba can load its dictionary on setuptools 81-83, where `pkg_resources.resource_stream`
  is gone but the module still imports.
- The Start button in the GUI, broken by removing the `sys.path` hack.

### Removed

- **Qwen3-TTS and the model registry it introduced** (breaking, but never released: it was
  added and removed within this range).

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
