# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

AUTOMATIC1111 Stable Diffusion web UI: a Gradio + FastAPI app for Stable Diffusion (SD1.x/2.x/SDXL/SD3) inference, training (embeddings/hypernetworks), and postprocessing. Target Python is 3.10 (3.10.6 in CI; ruff targets py39 syntax).

## Branching

Development happens on `dev`; `master` is the release branch. A CI workflow fails any PR that targets `master`, so open PRs against `dev`.

## Commands

```bash
# Run (creates venv, installs deps, clones repos into repositories/, then launches)
./webui.sh                      # Linux/macOS; user flags go in webui-user.sh (COMMANDLINE_ARGS)
python launch.py [flags]        # inside an already-prepared env
python launch.py --skip-prepare-environment ...   # skip dependency install/clone
python launch.py --nowebui      # API-only server (default port 7861)
python launch.py --api          # UI + REST API at /sdapi/v1/* (docs at /docs)

# Lint (both run in CI)
ruff .                          # CI pins ruff==0.3.3; config in pyproject.toml
npm i && npm run lint           # eslint for javascript/ and extension JS; `npm run fix` to autofix
```

### Tests

Most tests are HTTP tests against a running server (pytest `base_url` defaults to `http://127.0.0.1:7860`). Start a CPU test server first, then run pytest:

```bash
pip install -r requirements-test.txt
python launch.py --skip-prepare-environment --skip-torch-cuda-test --test-server \
  --do-not-download-clip --no-half --disable-opt-split-attention --use-cpu all --api-server-stop &

python -m pytest -vv test                                  # all
python -m pytest -vv test/test_txt2img.py::test_txt2img_simple_performed   # single test
curl -XPOST http://127.0.0.1:7860/sdapi/v1/server-stop     # stop server
```

`--test-server` (`launch_utils.configure_for_tests`) adds `--api`, `--disable-nan-check`, and loads `test/test_files/empty.pt` as the checkpoint, so tests check plumbing, not image quality. Some tests (e.g. `test_torch_utils.py`, `test_face_restorers.py` via the `initialize` fixture) import `modules` in-process instead of hitting the server. `conftest.py` sets `IGNORE_CMD_ARGS_ERRORS=1` so webui's argparse tolerates pytest args.

## Architecture

**Startup flow**: `launch.py` → `modules/launch_utils.py` (`prepare_environment()` pip-installs requirements, git-clones pinned upstream repos — Stability SD, generative-models (SDXL), k-diffusion, BLIP, assets — into `repositories/`, runs each extension's `install.py`) → `start()` imports `webui.py` → `modules/initialize.py` (imports, model loading, script discovery) → `ui.create_ui()` builds Gradio Blocks → API, progress, and extra-network routes are mounted on the FastAPI app. The `webui()` loop supports in-process UI reload/restart.

**Global state** lives in `modules/shared.py`: `cmd_opts` (CLI flags, defined in `modules/cmd_args.py`), `opts` (user settings, persisted to `config.json`, declared in `modules/shared_options.py` via `options_section`/`OptionInfo`), `state` (job progress/interrupt), `sd_model`, `device`. Most modules read config through `shared.opts.<name>`; to add a setting, add an `OptionInfo` in `shared_options.py` (or via `on_ui_settings` in an extension).

**Generation pipeline**: `modules/processing.py` is the core. `StableDiffusionProcessingTxt2Img` / `StableDiffusionProcessingImg2Img` hold all params (`p`); `process_images(p)` applies option overrides, then `process_images_inner` handles prompt parsing (`prompt_parser.py` — attention weights, `[from:to:step]` editing, `AND` composition), conditioning, sampling, VAE decode, face restore, saving, and infotext (`create_infotext`, parsed back by `infotext_utils.py`). The UI handlers (`txt2img.py`, `img2img.py`) and API (`modules/api/api.py`) both build a `p` object and go through the same `ScriptRunner` path. GPU work is serialized via `call_queue.queue_lock`.

**Model layer**: `sd_models.py` loads/switches checkpoints (config detection in `sd_models_config.py`, SDXL specifics in `sd_models_xl.py`, SD3 under `modules/models/`). `sd_hijack*.py` monkey-patches the upstream ldm/sgm code at runtime — custom CLIP text encoding with unlimited tokens/emphasis (`sd_hijack_clip.py`), cross-attention optimizations (`sd_hijack_optimizations.py`, selectable via `list_optimizers`), textual inversion embedding injection. Upstream repos in `repositories/` are not edited; behavior changes go through hijacks. Samplers: `sd_samplers.py` registry, k-diffusion wrappers in `sd_samplers_kdiffusion.py`, CFG logic in `sd_samplers_cfg_denoiser.py`, schedulers in `sd_schedulers.py`. Device/dtype handling lives in `devices.py` (plus `mac_specific.py`, `xpu_specific.py`, `npu_specific.py`); low-VRAM offloading in `lowvram.py`.

**Scripts & extensions** — the main extension mechanism:
- `modules/scripts.py`: subclasses of `scripts.Script` are auto-discovered from `scripts/`, `extensions-builtin/*/scripts/`, and `extensions/*/scripts/`. A script provides `title()`, `show()` (return `scripts.AlwaysVisible` for always-on accordions), `ui()`, and hooks into the pipeline (`process`, `before_process_batch`, `process_batch`, `postprocess`, `postprocess_image`, etc.). Selectable scripts use `run()` to replace normal generation (e.g. `scripts/xyz_grid.py`).
- `modules/processing_scripts/` holds built-in always-visible scripts (seed, sampler, refiner) that are part of the core UI.
- `modules/script_callbacks.py`: global `on_*` hook registration (`on_app_started`, `on_ui_tabs`, `on_ui_settings`, `on_model_loaded`, `on_cfg_denoiser`, `on_image_saved`, …).
- `modules/extra_networks.py`: `<name:args>` prompt syntax dispatched to registered `ExtraNetwork` handlers (hypernetworks in core; Lora lives in `extensions-builtin/Lora`). Card browsing UI in `ui_extra_networks*.py`.
- `extensions-builtin/` ships with the repo (Lora, LDSR, SwinIR, ScuNET, hypertile, soft-inpainting, canvas-zoom…). User-installed extensions go in `extensions/` (git-ignored, excluded from ruff). Extensions may have `install.py`, `preload.py` (adds CLI args), `scripts/`, `javascript/`, `style.css`.
- Postprocessing (Extras tab) uses a parallel system: `scripts_postprocessing.py` with `scripts/postprocessing_*.py`; upscalers subclass `modules/upscaler.py`.

**Frontend**: `script.js` + every file in `javascript/` (and extensions' `javascript/`) is injected into the Gradio page (`ui_gradio_extensions.py`). JS finds elements by Gradio `elem_id`s, so renaming an `elem_id` in Python can silently break JS. UI defaults/limits are persisted in `ui-config.json` (`ui_loadsave.py`).

**API**: `modules/api/api.py` (routes under `/sdapi/v1/`), pydantic request/response models in `modules/api/models.py` are partly generated from the processing class signatures, so adding a parameter to `StableDiffusionProcessing*` exposes it in the API.
