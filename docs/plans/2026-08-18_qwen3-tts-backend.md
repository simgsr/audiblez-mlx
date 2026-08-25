# Add Qwen3-TTS as an opt-in model, with a model-aware voice picker in the GUI

Date: 2026-08-18
Branch: `main`
Status: **reverted 2026-08-19.** Implemented, used, and then removed along with the whole
model registry it introduced; the project is Kokoro-only again. Kept as a design record so
the next person to consider Qwen3-TTS can start from the measurements rather than repeat
them.

Why it was dropped, in the order the problems actually mattered:

1. **Too slow.** 62 chars/sec against Kokoro's 666 in English — a 500k-character novel goes
   from ~13 minutes to ~2.2 hours. This alone decided it.
2. **Too few voices.** 9 speakers, of which **2** are English, against Kokoro's 54/28. The
   opt-in model was weakest exactly where most books are.
3. **Tone and pacing drift.** It samples, so identical text came back at different lengths
   and deliveries. Seeding made runs reproducible and `top_p` narrowed the spread (see open
   question 5 below), but neither made the delivery itself pleasant to listen to.

The sampling controls added to tame item 3 (`--temperature`, `--top-p`, `--seed`) went with
it: Kokoro is deterministic and ignores all three, so nothing was left for them to do.

Everything below is the original plan, left as written.

## Goal & scope

The MLX plan deferred Qwen3-TTS explicitly ("different architecture, ~20x the
parameters, needs its own plan and its own benchmark"). That benchmark has now been
run, so this plan settles what to do with the result.

Ship `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (via the `mlx-community` 8-bit port) as an
**opt-in second model on the existing MLX runtime** — never the default, never selected
by `auto`. The headline deliverable is **a model dimension alongside the existing
backend dimension, plus a voice list that follows the chosen model** in both CLI and GUI.

### Measured evidence this plan rests on

Same machine, same harness, warmed up, driven one sentence per call exactly as
`gen_audio_segments` does. English on a 455-char passage, Chinese on a 105-char passage:

| | Kokoro-82M-bf16 | Qwen3-TTS-1.7B-CustomVoice-8bit |
| --- | --- | --- |
| chars/sec, English | **666.5** | **62.3** (~10.7x slower) |
| chars/sec, Chinese | **150** | **27** (~5.6x slower) |
| realtime factor, English | 42.99x | 5.42x |
| model load | 0.3s | 2.6s |
| peak RSS | 0.83 GB | 3.62 GB |
| on disk | 339 MB | 2.9 GB |
| audio for same EN text | 29.4s | 39.6s |

A 500k-char English novel goes from roughly 13 minutes to roughly 2.2 hours. Note the
Kokoro English figure is below the 906 c/s seeded in `backends.py:27` — shorter sample —
so 10.7x is the defensible ratio, not the 14.5x that constant would imply.

**Chinese is the stronger case for this model**, and it is the reason the plan proceeds:
the penalty is roughly half what it is in English, and it competes against Kokoro's
weakest voices rather than its strongest. Five Chinese speakers against two English ones
points the same way.

Chars/sec is **not comparable across languages** — Chinese characters carry far more
phonemes each, which is why Kokoro itself drops from 666 to 150. See step 1.

### Alternatives evaluated and rejected

- **Qwen3-TTS-0.6B-CustomVoice-8bit** — measured, not faster in any way that matters:
  58.5 c/s English (*slower* than the 1.7B) and 32.8 c/s Chinese (~21% faster). The
  bottleneck is audio token count at 12 Hz, not parameter count — the 0.6B emitted 52.3s
  of audio for the same English text the 1.7B did in 39.6s, cancelling the per-token
  saving. Its real wins are 1 GB less disk and 1.1 GB less RAM. Not worth a second model.
- **IndexTTS-1.5** — 16 c/s (~9x slower than Kokoro on Chinese), and structurally
  incompatible: `generate(text, ref_audio, ...)` requires a reference clip and accepts no
  `voice`/`speed`/`lang_code`/`split_pattern`. Also needed `sentencepiece` plus a patched
  `config.json`, because the mlx-community conversion omits the `tokenizer_name` key that
  `ModelArgs` requires (`indextts.py:58`).
- **IndexTTS2** (`index-tts2-mlx`, 8 GB) — v2 architecture; mlx-audio's module is v1.
- **Fun-CosyVoice3-0.5B** — declares `model_type: "cosyvoice3"`; no such module in
  mlx-audio 0.5.0.

Two of the three non-Qwen conversions tested were broken or unloadable. **Pinning
`mlx-audio>=0.5.0` guarantees nothing about any particular mlx-community repo** — treat
each new model as unverified until it loads.

The good news: the existing adapter already drives it. `mlx_audio`'s unified
`generate()` takes `text, voice, speed, lang_code, split_pattern`, which is exactly the
call `MlxKokoroPipeline.__call__` makes at `backends.py:147`. Pointing the *unmodified*
adapter at the Qwen repo produced valid 24000 Hz audio on the first try. The seam built
in the MLX plan generalizes; this plan is mostly registry and UI work, not engine work.

### Out of scope

- **Making Qwen the default, or letting `auto` pick it.** 10.7x slower with no speed
  control is the wrong trade for long-form narration.
- **The `Base` variant and 3-second voice cloning.** A different feature with a different
  UI (reference audio picker, ref-text field) and its own benchmark. The IndexTTS finding
  confirms cloning models cannot share the speaker-based call signature at all, so this is
  a structural boundary rather than a scoping choice. Separate plan if ever wanted.
- **The `VoiceDesign` variant.** Same reasoning.
- **Any second Qwen size or quantization.** The 0.6B was measured and rejected; shipping
  one model keeps the registry and the UI honest.
- **Voice blending.** Kokoro-only; comma-blending has no meaning for Qwen speakers.
- **Streaming synthesis.** `generate()` exposes `stream=`, but nothing downstream of
  `gen_audio_segments` can consume a stream today.
- **Removing or changing anything on the Kokoro path.** `--backend mlx` and
  `--backend torch` must behave exactly as they do today.

## Affected files/modules

| File | Reason |
| --- | --- |
| `audiblez/backends.py` | `MODELS` registry; `model=` parameter on `get_pipeline`; rename the Kokoro-specific adapter; language-code mapping; chars/sec seed. |
| `audiblez/voices.py` | Qwen speaker list + per-model voice lookup, so the GUI and CLI share one source of truth. |
| `audiblez/core.py` | Thread `model` through `main` (`:130`) and `gen_text` (`:290`); the `lang_code in 'ab'` branch at `:257` is Kokoro-specific. |
| `audiblez/ui.py` | Model dropdown, model-aware voice list (`:316-334`), speed field gating, `lang_code` derivation (`:532`), pass `model` into `core.main` (`:584-590`). |
| `audiblez/cli.py` | New `--model` flag; `--speed` warning when unsupported. |
| `pyproject.toml` | Nothing required — `mlx-audio>=0.5.0` already ships `qwen3_tts`. Document the 2.9 GB download. |
| `test/test_backends.py` | Registry dispatch, language mapping, speed-unsupported warning. |
| `README.md` | Document the model choice, the speed caveat, and the honest throughput numbers. |

## Steps

### 1. Add a model dimension to `backends.py`

`backend` currently means *runtime* (`torch` / `mlx`). Qwen is not a runtime — it runs on
the MLX runtime that already exists. Folding it into `BACKENDS` would make `--backend`
mean two different things and would quietly break `resolve_backend`,
`initial_chars_per_sec` and `update_device_row`, all of which key off runtime.

So add a second, small dimension:

```python
MODELS = {
    'kokoro': dict(
        runtimes=('mlx', 'torch'),
        repos={'mlx': 'mlx-community/Kokoro-82M-bf16', 'torch': 'hexgrad/Kokoro-82M'},
        # Keyed by language: CJK characters carry far more phonemes each, so a single
        # per-model number is wrong by ~4x on Chinese. 'default' covers the rest.
        chars_per_sec={'mlx': {'default': 900, 'z': 150},
                       'torch_cuda': {'default': 500},
                       'torch_cpu': {'default': 50}},
        supports_speed=True,
        deterministic=True,
        lang_from_voice=True,      # 'af_sky' -> 'a'
    ),
    'qwen3-tts': dict(
        runtimes=('mlx',),
        repos={'mlx': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit'},
        chars_per_sec={'mlx': {'default': 60, 'z': 27}},   # measured 62.3 / 27
        supports_speed=False,
        deterministic=False,       # see step 3a
        lang_from_voice=False,
    ),
}
```

The per-language table matters more than it looks: `stats.eta` is seeded from this and
only recalibrates from real throughput as the run proceeds (`core.py:278-279`). A 4x-wrong
seed means the first ETA a user sees on a Chinese book is wildly wrong, which is worst
precisely when the run is longest.

`get_pipeline(voice, lang_code=None, backend='auto', repo_id=None, model='kokoro')`.
Default `model='kokoro'` means every existing call site keeps working untouched.

Selecting `model='qwen3-tts'` with a non-MLX runtime must raise a clear error naming the
constraint, in the style of the existing messages at `backends.py:163-170`.

This is the smallest design that works, in the spirit of the MLX plan's rejection of a
general plugin interface: the hot loop, the progress accounting and the adapter all stay
as they are. What varies between Kokoro and Qwen — voice list, language codes, speed
support, throughput seed — varies by *model*, not by runtime, which is precisely what
the registry captures.

Rename `MlxKokoroPipeline` to something honest (`MlxPipeline`); the class was never
Kokoro-specific, only its name and docstring were.

### 2. Fix the language-code mapping (real bug, silent today)

`lang_code` is derived as `voice[0]` in three places (`core.py:128`, `core.py:255`,
`ui.py:532`). For a Qwen speaker that means `'ryan'` -> `'r'`.

`'r'` is not a key in the model's `codec_language_id`, and the lookup is exact
membership, not prefix matching (`qwen3_tts.py:393-395`) — so it does **not** silently
select Russian. It falls through to auto-detect. Benign in effect, but it means the
language argument is quietly doing nothing, which will bite the moment someone
synthesizes a non-English book.

Map Kokoro's single-letter codes onto Qwen's language names:

```
a, b -> english      z -> chinese     j -> japanese
e -> spanish         f -> french      i -> italian     p -> portuguese
h -> (unsupported: Hindi is not in Qwen's list — reject with a clear message)
```

Qwen additionally supports German, Korean and Russian, which Kokoro has no code for;
expose those by name through `--lang`. Gate on `lang_from_voice` in the registry: for
Qwen, the language must come from `--lang` or default to `auto`, never from `voice[0]`.

### 3a. Sampling control — Qwen output is not reproducible

Measured: `vivian` produced **25.9s and then 21.4s of audio for byte-identical input**, a
21% swing between two runs. `generate()` defaults to `temperature=0.9, top_k=50,
top_p=1.0` (`qwen3_tts.py:1147-1160`), so every call samples differently. Kokoro is
deterministic; this is a behaviour change, not a tuning detail.

For a book-length job it means:

- **Chapters are not reproducible.** Re-running one chapter after a crash or a text edit
  yields audibly different pacing from its neighbours — in the same audiobook.
- **Duration cannot be predicted**, so ETA is noisier and any duration-based sanity check
  on a chapter (the kind step 0 of the MLX plan used to catch truncation) gets weaker.
- **The same text can drift in length by ~20%**, which interacts badly with the
  long-output tendency noted in step 3.

The adapter exposes none of this: `MlxKokoroPipeline.__call__` (`backends.py:145-149`)
passes only text/voice/speed/lang/split. Plumb `temperature` (and ideally a seed) through
the adapter and default it **low** for long-form narration, where consistency beats
expressiveness. Expose it via the registry so Kokoro is unaffected.

Worth verifying during implementation: whether a low temperature also suppresses the
over-long outputs seen from `vivian` (25.9s vs a ~14s expectation) and from the 0.6B
model (52.3s vs 39.6s for identical English text). Both look like the dragging/looping
failure mode autoregressive TTS is prone to, and if sampling is the cause, this step
fixes step 3's symptom too.

### 3. Handle `speed` honestly

`generate()` documents `speed` as "not directly supported yet" (`qwen3_tts.py:1175`) and
the custom_voice routing drops it before it reaches the model. Passing `--speed 1.5`
today would be **silently ignored** — the worst failure mode, because the user gets a
full-length audiobook at the wrong pace with no indication anything went wrong.

Compounding it, Qwen already speaks ~35% slower than Kokoro (39.6s vs 29.4s for the same
text), so the natural instinct will be to reach for `--speed`.

Required behaviour: when `supports_speed` is false and the requested speed is not 1.0,
print a clear warning naming the model. Do not fail — the run is still useful — but never
let it pass unremarked. The GUI equivalent is in step 5.

### 4. Voice registry

`voices.py` currently maps Kokoro language codes to voice names and is imported directly
by both `cli.py` and `ui.py`. Add the Qwen speakers beside it, keyed by model, with the
same shape so both consumers can stay dumb:

```python
QWEN_SPEAKERS = {
    'english':  ['ryan', 'aiden'],
    'chinese':  ['serena', 'vivian', 'uncle_fu', 'eric', 'dylan'],
    'japanese': ['ono_anna'],
    'korean':   ['sohee'],
}
```

Nine speakers total, against Kokoro's 54 — and only **two** English ones, against
Kokoro's 28. This is the second real cost of the model, after speed, and the README must
say so plainly. The Chinese side is far better served (five speakers), which is consistent
with Chinese being the case that justifies the model at all.

`beijing_dialect` and `sichuan_dialect` also exist in `codec_language_id` but are filtered
out of `get_supported_languages()` by the `"dialect" not in lang_id` test
(`qwen3_tts.py:200-203`). They appear reachable by passing the id directly. Kokoro has no
equivalent. Untested — worth a look, but not a dependency of this plan.

**Loudness varies ~2x across speakers** (measured RMS 0.046–0.107; `aiden` is ~1.7x
`ryan` and ~2.75x Kokoro). Harmless within one narrator, but it means levels are not
comparable when auditioning voices, and anyone A/B-ing will mistake "louder" for "better".
If Qwen ships, normalize per-chapter audio to a target RMS before the m4b mux, or at
minimum document the variance.

The speakers are the model's own lowercase ids as reported by
`get_supported_speakers()`; the model accepts them case-insensitively
(`qwen3_tts.py:2118`). Store lowercase, display title-cased.

Expose `voices_for(model)` so `cli.py`'s `available_voices_str` and `ui.py`'s dropdown
both derive from one place rather than each growing their own conditional.

### 5. GUI changes (`ui.py`)

The params panel already has the right precedent: `update_device_row()` disables the
torch device radio when MLX will actually run. Model-dependent widgets follow that
pattern exactly.

**5a. Model dropdown.** New row above Backend, since model now determines what the other
rows may contain:

```
Model:    [ kokoro ▾ ]        Kokoro — 54 voices, fast (~666 chars/sec)
Backend:  [ auto ▾ ]          MLX available — "auto" will use it (faster)
```

Only offer `qwen3-tts` when `mlx_available()`, mirroring how `backend_choices` is built
at `ui.py:276-277`. The note line beneath should carry the honest trade — for Qwen,
something like *"Expressive, 9 voices, ~10x slower (~2h for a novel), 2.9 GB download"*.
A user should not discover the cost by watching a progress bar all evening.

**5b. Model-aware voice list — the change requested.** The dropdown is built once at
`ui.py:316-320` from the Kokoro `voices` dict, prefixing `flags[code]`. Extract that into
`update_voice_choices()` and call it from a new `on_select_model` handler, following
`on_select_backend` -> `update_device_row`:

- Repopulate via `voice_dropdown.SetItems(...)` from `voices_for(model)`.
- Reset `self.selected_voice` to the new list's first entry. **This matters**: if the
  user has `af_sky` selected and switches to Qwen, a stale Kokoro voice would reach
  `generate_custom_voice`, which rejects unknown speakers outright
  (`qwen3_tts.py:2118-2121`). Resetting is what stops that becoming a crash at synthesis
  time, long after the mistake.
- Qwen speakers get a flag from their language (`ryan` -> 🇺🇸, `ono_anna` -> 🇯🇵) so
  `get_selected_voice()`'s emoji-stripping at `ui.py:520-526` keeps working unchanged.
  Its `first in flags.values()` test already tolerates any known flag.
- The note at `ui.py:332` — *"Blend voices with commas, or type a path to a .pt voice"* —
  is Kokoro-only advice. Under Qwen it should read that speakers are fixed and neither
  blending nor `.pt` packs apply.
- Consider making the ComboBox read-only under Qwen. It is deliberately editable
  (`ui.py:325-328`) to allow blends and `.pt` paths, neither of which Qwen accepts, so
  free text there can only produce an error.

**5c. Speed field.** Disable the speed `TextCtrl` (`ui.py:337-342`) when the selected
model has `supports_speed=False`, with a note explaining the model ignores it — same
treatment `update_device_row` gives the torch radio. Silently accepting a value the
engine discards is the GUI version of the step-3 bug.

**5d. `lang_code` derivation.** `ui.py:532` does `self.get_selected_voice()[0]`, which is
the step-2 bug in the preview path. Route it through the shared mapping.

**5e. Pass the model through.** `on_start` builds `self.params` at `ui.py:584-590` with
`backend=`; add `model=`. Note `repo_id` is *not* currently passed from the GUI at all —
the registry default covers it, but that gap is why the GUI cannot select a quantization
today.

**5f. Preview.** `on_preview_chapter` (`ui.py:531-566`) constructs its own pipeline and
must pass `model` too, or previews will use Kokoro while the run uses Qwen. Worth a
deliberate look: preview builds a fresh pipeline per click, and at 2.6s load plus ~5x
realtime, a 300-char Qwen preview will take appreciably longer than the near-instant
Kokoro one. Not a blocker, but the button should probably keep its "⏳" state honestly
rather than appearing hung.

**5g. Optional — `instruct`.** Qwen's per-passage emotion/style control
(`generate_custom_voice(instruct=...)`) is the capability that actually justifies the
model, and it has no Kokoro equivalent. A single-line "Style instruction" text field,
shown only for Qwen, would surface it. Deliberately deferred: it needs its own thought
about whether one instruction applies to a whole book or varies per chapter, and this
plan is already large. Flagged here so it is a decision, not an oversight.

### 6. CLI changes

- `--model {kokoro,qwen3-tts}`, default `kokoro`.
- `--speed` warning per step 3.
- `available_voices_str` in the `epilog` (`cli.py:14-16`) is computed at import from the
  Kokoro dict. It should either list both sets or list the chosen model's — note the
  epilog is built before `parse_args`, so showing only the selected model's voices means
  restructuring that. Simplest honest option: list both, grouped by model.

### 7. Documentation

README must state the throughput cost (~10.7x), the two-English-voices limit, the
ignored `--speed`, and the 2.9 GB download. The MLX plan's framing — measured numbers,
not adjectives — is the standard to match.

## Test strategy

Extend `test/test_backends.py`, still with no model downloads:

- **Registry dispatch** — `model='qwen3-tts'` with `backend='torch'` raises a clear
  error; with `backend='mlx'` and `mlx_audio` absent, raises the existing actionable
  message; `model='kokoro'` behaves exactly as today.
- **Default preserved** — `get_pipeline` with no `model=` resolves to Kokoro with the
  same repo as today. This is the regression guard for every existing call site.
- **Language mapping** — `'a'`/`'b'` -> `english`, `'z'` -> `chinese`; `'h'` rejected with
  a message naming Hindi; a Qwen voice never derives its language from `voice[0]`.
- **Speed warning** — `supports_speed=False` plus `speed=1.5` emits a warning and does
  not raise.
- **Sampling control** — the adapter forwards `temperature` for a non-deterministic model
  and does not forward it for Kokoro; the long-form default is low, not the library's 0.9.
- **Throughput seed** — `chars_per_sec` for lang `'z'` differs from `'default'`, for both
  models. Guards the 4x-wrong-ETA bug.
- **Voice registry** — `voices_for('qwen3-tts')` returns the nine speakers; every one is
  accepted by the flag-stripping logic in `get_selected_voice`.
- **Adapter contract unchanged** — the existing fake-`mlx_audio` test must still pass
  against the renamed class, third tuple element still `numpy.ndarray`.

GUI logic worth testing headlessly, separated from wx widgets: switching model resets a
stale voice (the 5b crash path) and the model->voice-list mapping is total.

Integration, `skipUnless(darwin and mlx_available())` and marked slow so it does not run
by default: synthesize one short phrase with `ryan`, assert non-silent audio at 24000 Hz
of plausible duration.

Regression: all existing tests stay green; Kokoro output must be unchanged.

## Risks & trade-offs

- **10.7x slower is the whole risk.** Anyone selecting this for a full book without
  understanding the cost will have a bad evening. Mitigated by the note text in 5a, the
  README, and the `chars_per_sec` seed of 60 so the existing ETA estimate is honest from
  the first sentence rather than after recalibration.
- **Two English voices.** For an English-language audiobook tool this is a severe
  narrowing. It is the strongest argument that the `Base` + cloning variant, not
  CustomVoice, is what this project actually wants.
- **Silent `speed`.** Addressed in steps 3 and 5c, but worth restating: this is the one
  failure mode that produces a plausible-looking wrong result.
- **Nondeterminism** (step 3a). Arguably a bigger correctness issue than silent `speed`,
  because it makes partial re-runs audibly inconsistent with the rest of the book — and
  partial re-runs are exactly what a multi-hour job invites.
- **Over-long output.** Two independent observations of ~30% more audio than expected for
  identical text (`vivian`, and the 0.6B model in English). If this is looping rather than
  pacing, it corrupts chapters silently, deep into a long run, where nobody is listening.
  A duration-vs-char-count outlier check per chapter would catch it — the MLX plan already
  proposed exactly that check for a different reason.
- **Loudness spread** across speakers (~2x RMS). Cosmetic within a book, misleading when
  comparing voices.
- **A second dimension in the UI.** Model x backend is a 2D space where one cell
  (qwen + torch) is invalid. Kept manageable by only offering `qwen3-tts` when MLX is
  available, but it is genuinely more surface than the single Backend dropdown.
- **3.62 GB peak RSS** vs Kokoro's 0.83 GB. Fine on this machine, less so on 8 GB Macs,
  and it runs alongside whatever else the user has open.
- **`mlx-audio` API drift.** The unified `generate()` routing by `tts_model_type` is
  convenient but undocumented as a stable contract; the 8-bit port was converted with
  0.3.0 while the project pins `>=0.5.0`. A pin bump could change routing behaviour. The
  adapter is thin enough to absorb it, and the integration test would catch it.
- **Model quality is unverified.** Numeric checks confirmed non-silent, non-clipping
  audio at plausible duration and RMS. Nobody has listened yet. A/B samples were
  generated for exactly this purpose and remain unjudged — see Open questions.

## Decisions taken

- **Scope is `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`, as an opt-in alternative.** Decided
  2026-08-18. The 0.6B variant, IndexTTS, IndexTTS2 and CosyVoice3 were evaluated and
  rejected; see "Alternatives evaluated and rejected" above. Their weights have been
  removed from the local HF cache — only Kokoro (bf16 + torch) and the 1.7B CustomVoice
  remain.
- **Voice cloning is not part of this work.** Confirmed by the IndexTTS finding: cloning
  models cannot share the speaker-based call signature, so they are a genuinely separate
  feature, not a scoping preference.

## Open questions

1. **Is the quality actually better?** The gate this plan still rests on. Samples were
   generated and played for English (`af_sky` vs `ryan` vs `aiden`) and Chinese
   (`zf_xiaoxiao` vs all five Qwen speakers), but the judgement has not been recorded
   here. The MLX plan set the precedent that a numeric check cannot detect artifacts and
   a human must listen. Record the verdict before implementing — especially for Chinese,
   which is the case carrying this plan.
2. **`--model` naming.** `qwen3-tts` names a family whose members (Base, CustomVoice,
   VoiceDesign) behave quite differently. If Base ever lands, `qwen3-tts-custom` and
   `qwen3-tts-clone` would be clearer, and renaming after release is worse than choosing
   now.
3. **Quantization.** All figures are the 8-bit port; bf16, 6-bit and 4-bit also exist. Not
   measured. Same question the MLX plan left open for Kokoro (its question 4) and never
   closed — and Kokoro's own 8-bit/4-bit variants are still untested, which is the cheaper
   experiment of the two.
4. **Does the `instruct` field belong to the book or the chapter?** Decides whether 5g is
   one text box or a per-chapter column, and therefore whether it is small or large.
5. ~~**Does low temperature fix the over-long outputs?**~~ **Answered 2026-08-19: no, and
   the premise was wrong.** Low temperature does not cause the repetition loop this
   question feared, and does not cure the over-long outputs either. On one 110-character
   English sentence, seeded, every temperature from 0.1 to 0.9 produced 6.3-8.6s of audio
   — all sane, no trend. The cliff is at exactly **0**, which is not the bottom of the
   scale but a different algorithm: mlx-audio branches to greedy argmax at `temperature
   <= 0` and returns before `top_k`/`top_p` are applied. Greedy never emitted a stop token
   and ran to the 4096-token cap — **327.68s of audio for that one sentence**,
   reproducibly. `MlxPipeline` now warns on `temperature=0`.

   Since temperature is not the lever, the drift was attacked from the other two
   directions instead: `top_p` (mlx-audio defaults it to 1.0, i.e. no nucleus filtering at
   all — now 0.8) and a seed. Five unseeded runs of one passage spread 8.48-15.12s at
   `top_p=1.0` versus 8.96-13.84s at 0.8, so the cutoff helps but does not close it.
   Seeding does: MLX's global PRNG is what varies between runs, so `mx.random.seed()`
   before each `generate()` makes output byte-identical. That closes the reproducibility
   half of step 3a. **The duration-outlier check is therefore still required** — the
   over-long outputs remain unexplained, and this question does not close that risk.
