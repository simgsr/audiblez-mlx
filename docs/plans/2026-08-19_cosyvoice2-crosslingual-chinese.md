# Cross-lingual Chinese narration: clone Kokoro's Grade-A English voices to read Mandarin

Date: 2026-08-19
Branch: `feat/traditional-chinese`
Status: **aborted 2026-08-19.** Superseded by the edge-tts backend
([2026-08-19_edge-tts-backend.md](2026-08-19_edge-tts-backend.md)): real Cantonese, native
traditional-script reading, and no model downloads. Kept as a design record — the
measurements and the listening-test gate below are still the right way to revisit this if
edge-tts ever proves insufficient. Everything below rests on measurements taken on
2026-08-19 (an M-series Mac); no production code was written.

## Goal & scope

Kokoro's Chinese output is capped by its training data, not by the language: **all eight of
its Chinese voices are Grade D**, the lowest tier it ships, while `af_heart` is **Grade A**
and `af_bella` **A-**. CosyVoice2's cross-lingual mode clones a reference speaker into a
*different* language, so a Grade-A English timbre can narrate Mandarin.

The deliverable is a GUI and CLI path where the user picks a Kokoro English voice and asks
for Chinese narration, and audiblez does the cloning invisibly: it synthesizes a short
English reference with Kokoro, feeds it to CosyVoice2 as `ref_audio`, and narrates the
book's Chinese text in that voice.

### Measured evidence this plan rests on

Same 84-character Chinese passage throughout, so the numbers are comparable.

| Path | Chars/sec | Notes |
|---|---|---|
| Kokoro `zf_xiaoxiao` (native Chinese) | **270** | Grade D — today's ceiling |
| CosyVoice2 cloning `zf_xiaoxiao` | 30–33 | 8.2x slower than Kokoro |
| CosyVoice2 cloning `af_bella` (cross-lingual) | 19.7 | Grade A- source |
| CosyVoice2 cloning `af_heart` (cross-lingual) | 22.7 | Grade A source |
| Qwen3-TTS, removed 2026-08-19 for being slow | 27 | for scale |

Two facts decide feasibility, and both were verified rather than assumed:

- **One environment can run both engines.** `mlx-audio-plus` ships `kokoro` *and*
  `cosyvoice2`. On Python 3.12 both import cleanly, and Kokoro measured **913–994
  chars/sec** English and **256** Chinese in the fork against **803–1023 / 270** in the
  project venv — no regression, and byte-identical audio durations. So this does not need
  subprocess isolation or two environments.
- **CosyVoice2 has no voices at all.** The repo ships only `config.json`,
  `model.safetensors` and tokenizer files: no speaker table. `generate()` raises
  `ValueError: ref_audio is required for CosyVoice2`. Cloning is not one mode among
  several; it is the only mode.

### Out of scope

- Any model other than Kokoro and CosyVoice2. IndexTTS-2 was tried and **cannot run**:
  mlx-audio's `indextts` module implements IndexTTS-1.x (no `s2mel`, no emotion encoder)
  and mlx-audio 0.5.0 is the latest release, so no upgrade fixes it.
- Reference clips from outside Kokoro. Letting users supply their own recording is the
  obvious follow-up, and it raises a consent question (cloning a real person's voice) that
  deserves its own decision. Ship the Kokoro-to-Kokoro path first.
- CosyVoice2's `instruct` mode (emotion/dialect control) and voice conversion.
- Cantonese. CosyVoice2 supports dialects via instruction text; that is a separate feature.
- Restoring `--temperature` / `--top-p` / `--seed`. They were removed with Qwen for good
  reasons and CosyVoice2 ignores `temperature` anyway.

## Affected files/modules

| File | Why |
|---|---|
| `pyproject.toml` | Swap `mlx-audio` for `mlx-audio-plus`; resolve the version collision below |
| `audiblez/backends.py` | Reinstate a model dimension; add a CosyVoice2 adapter beside `MlxPipeline` |
| `audiblez/voices.py` | Map Kokoro voices to their usability as cross-lingual references |
| `audiblez/core.py` | Thread `model` through `main()` and `gen_text()` again |
| `audiblez/cli.py` | `--model`, and a flag for the reference voice |
| `audiblez/ui.py` | Model row, and the "read in Chinese with this English voice" control |
| `test/test_backends.py` | Adapter tests, with a fake CosyVoice2 model |
| `README.md` | Document the feature and its cost |

Most of this is re-adding the layer deleted in `691940f`. That commit is the reference for
what the code looked like; `git show 691940f` is the cheapest way to see it.

## Steps

### 1. Settle the dependency, on its own

`pip install mlx-audio-plus` pulled in **`mlx-audio 0.2.10` alongside `mlx-audio-plus
0.1.8`**. Both ship a top-level `mlx_audio` package, so the working install measured above
is a *blend* of the two with whichever wrote last winning per-file. It worked, but it is not
a state to depend on.

Before any feature work: determine whether `mlx-audio-plus` declares `mlx-audio` as a
dependency or whether something else (misaki?) dragged it in, and pin a clean resolution.
If both must coexist, this plan is in trouble and step 6's fallback applies.

Also note the fork carries **16 models against mlx-audio 0.5.0's 42**, and 0.2.10 is far
behind 0.5.0. Switching narrows and ages the base.

**This step is independently valuable and independently reversible.** Do it, confirm
Kokoro still hits ~900 chars/sec through the project's own `get_pipeline`, and stop. If that
fails, abandon before touching anything else.

### 2. Reinstate the model dimension

Restore from `691940f`: `MODELS`, `model_spec`, `DEFAULT_MODEL`, and the `model` parameter
through `core.main()` / `gen_text()`. Keep it thinner than the Qwen version — no
`lang_from_voice`, no `chars_per_sec` per model until measured.

Kokoro stays the default and the only model chosen automatically.

### 3. Add a CosyVoice2 adapter

It cannot reuse `MlxPipeline`. Its `generate()` **documents that it ignores `voice`,
`speed`, `lang_code` and `temperature`**, and requires `ref_audio`. Passing Kokoro's
signature would silently discard four arguments — the exact trap the Qwen work had to warn
about for `speed`.

The adapter needs to own reference preparation:

```
CosyVoicePipeline(reference_voice, lang_code)
  __init__: synthesize ~12s of English with Kokoro at `reference_voice`,
            keep it as ref_audio (24kHz mono, model caps at 30s)
  __call__: model.generate(text=..., ref_audio=self.ref)   # cross-lingual mode
```

Cross-lingual mode takes `ref_audio` **alone**. `ref_text` is for same-language zero-shot;
supplying an English transcript for Chinese output is the wrong alignment.

Building the reference means loading Kokoro *and* CosyVoice2 in one run. Both fit in memory
(Kokoro is 339MB, CosyVoice2 1.5GB), but the reference should be built once per run and
cached, not per chapter.

### 4. Voice presentation

The user picks a Kokoro voice; the model dropdown decides whether it narrates directly
(Kokoro) or becomes a cross-lingual reference (CosyVoice2). Only Grade A/B voices are worth
offering as references — `af_heart`, `af_bella`, `af_nicole`, `bf_emma` and similar. Grading
lives in Kokoro's `VOICES.md`, not in this repo, so the shortlist has to be transcribed by
hand with a comment saying where it came from.

### 5. CLI and GUI

CLI: `--model cosyvoice2 --voice af_heart --lang z`.

GUI: the Model row returns, and when CosyVoice2 is selected the Voice dropdown narrows to
the reference shortlist and a note explains that the voice is being cloned into the target
language. Follow how the old `update_speed_row` / `update_model_dependent_rows` greyed out
controls per model — that pattern worked and is in `691940f`.

The GUI must warn about the speed. At ~20 chars/sec a 500k-character novel is **~7 hours**
against ~31 minutes on Kokoro. That is a number the user should see *before* pressing Start,
not discover afterwards.

### 6. Fallback if step 1 fails

If `mlx-audio` and `mlx-audio-plus` cannot be resolved cleanly, the remaining option is a
second venv and a subprocess boundary: audiblez shells out to run CosyVoice2. Two Python
environments to install, keep in sync and document. **Recommend abandoning instead** — the
complexity outweighs a nice-to-have narration mode.

## Test strategy

Unit tests, no model downloads, following the existing fake-model pattern in
`test_backends.py` (`FakeMlxModel`, `fake_mlx_audio`):

- The adapter passes `ref_audio` and does **not** pass `voice`/`speed`/`lang_code`, so a
  future signature change surfaces as a failure rather than silent drift.
- The reference is built once per pipeline, not once per `__call__`.
- A missing/failed reference raises rather than falling through to a broken generate.
- Selecting CosyVoice2 with a non-shortlisted voice is rejected at construction, not at
  synthesis time — the Qwen work established that late failures waste hours.
- Kokoro's path is unchanged: existing tests must pass untouched.

Edge cases worth explicit tests: empty chapter text; a chapter that is entirely punctuation;
a reference voice name that does not exist; and the 30s `ref_audio` cap.

Manual, since no automated check covers it: synthesize one real chapter and listen for
accent drift across chapter boundaries. Each chapter re-clones from the same reference, so
they should match — but that is an assumption until heard.

## Risks & trade-offs

- **~8x slower**, and only marginally faster than the Qwen3-TTS just removed for that exact
  reason. If a 7-hour novel is unacceptable, stop here.
- **Depending on a community fork of the core dependency.** `mlx-audio-plus` is at 0.1.8
  with fewer models than upstream. If it stops being maintained, Kokoro goes with it. This
  is the single biggest structural risk and it affects the working feature, not just the new
  one.
- **Undeclared dependencies.** Getting CosyVoice2 to import needed `mlx-lm`, `einops`,
  `scipy`/`librosa`, `loguru`, and torch (only because `cosyvoice2/__init__.py` imports a
  weight converter at module level). Missing ones surface as the misleading
  `ValueError: Model type cosyvoice2 not supported`, which is a swallowed `ImportError`.
  Budget time for this; it is not a five-minute install.
- **Re-adding complexity deliberately deleted.** `691940f` removed 829 lines to get back to
  one model. This puts a chunk of that back for a mode that may see little use.
- **Non-determinism returns.** CosyVoice2 samples, so pacing varies run to run — the
  complaint that soured Qwen. Its `temperature` argument is ignored, so the `--seed` trick
  that fixed Qwen may not apply. Untested.
- **Accent is unmeasured and may be fatal.** An American-English timbre reading Mandarin
  may carry an accent or blur tone contours. Mandarin is tonal; wrong tones are not a
  stylistic quibble, they change meaning.

## Open questions

1. **Does it actually sound good?** Blocking. Listen to
   `samples/sample_zh_ab_cosyvoice2_af_heart.mp4` and `..._af_bella.mp4` against
   `sample_zh_ab_kokoro_zf_xiaoxiao.mp4` — same passage, so it is a direct comparison. If
   the accent is distracting, close this plan and clean up.
2. **Is a 12s reference the right length?** Untested. The model caps at 30s; longer may
   clone better, or may not.
3. **Does the clone stay stable across a whole book?** Only two short passages were
   generated. Chapter-to-chapter drift is the failure mode that would matter most.
4. **Would traditional-Chinese conversion still apply?** `chinese.normalize` keys on
   lang_code `'z'`; a CosyVoice2 run would need to hit the same path. Probably fine, needs
   checking, and `test_chinese.py` covers the logic.

## Abort and clean up

Everything below was created while investigating and **nothing in it is referenced by the
project**. Removing all of it returns the machine to its pre-investigation state.

### Downloaded models (~6.1 GB)

```bash
rm -rf ~/.cache/huggingface/hub/models--mlx-community--IndexTTS-2-fp16       # 4.6G, unusable
rm -rf ~/.cache/huggingface/hub/models--mlx-community--CosyVoice2-0.5B-fp16  # 1.5G
```

**Do not remove** `models--mlx-community--Kokoro-82M-bf16` or `models--hexgrad--Kokoro-82M`
(the working model), nor `models--mlx-community--Qwen3-*` / `models--BAAI--*` /
`models--FunAudioLLM--CosyVoice2-0.5B` / `models--sentence-transformers--*`, which belong to
other projects on this machine.

### Scratch virtualenvs (~3.7 GB)

Both live under the session scratchpad and are already outside the repo:

```bash
SCRATCH=/private/tmp/claude-501/-Users-randallsim-Documents-interesting-git-project-audiblez/95db1e4b-ce06-4977-a495-93fbc20f6372/scratchpad
rm -rf $SCRATCH/cosyvenv    # 1.3G, python3.14 + mlx-audio-plus + torch
rm -rf $SCRATCH/forkvenv    # 2.4G, python3.12 + mlx-audio-plus + misaki/spacy
rm -f  $SCRATCH/bench_cosyvoice2.py $SCRATCH/bench_indextts.py $SCRATCH/clone_crosslingual.py
```

### Project venv

One package was installed into `.venv` for the IndexTTS-2 attempt and is not a project
dependency:

```bash
.venv/bin/pip uninstall -y sentencepiece
.venv/bin/pip check          # expect: No broken requirements found.
```

`pyproject.toml` was never modified — no dependency changes to revert.

### Samples

Git-ignored (`*.mp4`), so they cost nothing but disk and can stay as a record of the
comparison. To remove:

```bash
rm -f samples/sample_zh_ab_cosyvoice2.mp4 \
      samples/sample_zh_ab_cosyvoice2_af_bella.mp4 \
      samples/sample_zh_ab_cosyvoice2_af_heart.mp4 \
      samples/sample_zh_ab_kokoro_zf_xiaoxiao.mp4
```

Keep `samples/sample_af_bella.mp4` — that was a separate request, unrelated to this plan.

### Code

**None to remove.** No production code was written for this. `audiblez/` is untouched by the
investigation, and the last commits on this branch (`691940f`, `6ae949b`) are the Qwen
removal and a `.gitignore` entry, neither of which relates to CosyVoice2.

If the idea is dropped, the only repo change is this file — keep it as a design record the
way `2026-08-18_qwen3-tts-backend.md` was kept, so the measurements are not repeated.
