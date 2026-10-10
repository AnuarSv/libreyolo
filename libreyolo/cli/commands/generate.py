"""Text-to-image generation with the LibreGen tier."""

import typer

from ..command_utils import (
    exit_with_error,
    help_json_callback,
    model_call_error_code,
    parse_imgsz_str,
)
from ..output import OutputHandler


def generate_cmd(
    prompt: str | None = typer.Option(
        None, help="Text to render; always literal text, never a path"
    ),
    prompt_file: str | None = typer.Option(
        None, help="Text file with one prompt per line (instead of prompt)"
    ),
    model: str = typer.Option(
        "flux2-klein-4b", help="LibreGen alias or local model directory"
    ),
    imgsz: str | None = typer.Option(
        None, help="Output size: 1024 (square) or HEIGHTxWIDTH such as 768x1024"
    ),
    seed: int | None = typer.Option(None, help="Seed; the same seed repeats the image"),
    n: int = typer.Option(1, help="Images per prompt"),
    steps: int | None = typer.Option(None, help="Sampling steps (family knob)"),
    guidance: float | None = typer.Option(None, help="Prompt guidance (family knob)"),
    device: str = typer.Option("auto", help="auto, cpu, mps, or a CUDA device"),
    dtype: str = typer.Option("auto", help="auto, bf16, fp16, or fp32"),
    offload: bool = typer.Option(False, help="Keep idle components off the GPU"),
    save: bool = typer.Option(True, help="Save images under runs/generate/predict"),
    output_path: str | None = typer.Option(None, help="Single-image output filename"),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    help_json: bool = typer.Option(
        False,
        "--help-json",
        is_eager=True,
        callback=help_json_callback,
        help="Dump command schema as JSON",
    ),
) -> None:
    """Generate images from a text prompt."""
    import os
    import sys
    from contextlib import ExitStack, redirect_stderr, redirect_stdout
    from pathlib import Path

    out = OutputHandler(json_mode=json_output, quiet=quiet)
    if prompt is None and prompt_file is None:
        exit_with_error(
            out, "config_required_key", "Pass prompt=... or prompt_file=... ."
        )
    if prompt is not None and prompt_file is not None:
        exit_with_error(
            out, "config_conflict", "Pass either prompt or prompt_file, not both."
        )
    try:
        size = parse_imgsz_str(imgsz)
    except (ValueError, TypeError) as exc:
        exit_with_error(out, "invalid_imgsz", str(exc))
    try:
        prompts: str | list[str]
        if prompt_file is not None:
            if not Path(prompt_file).is_file():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            lines = Path(prompt_file).read_text(encoding="utf-8").splitlines()
            prompts = [line.strip() for line in lines if line.strip()]
            if not prompts:
                raise ValueError(f"Prompt file has no prompts: {prompt_file}")
            if len(prompts) == 1:
                prompts = prompts[0]
        else:
            prompts = prompt
        knobs = {
            key: value
            for key, value in (("steps", steps), ("guidance", guidance))
            if value is not None
        }
        with ExitStack() as stack:
            destination = (
                stack.enter_context(open(os.devnull, "w")) if quiet else sys.stderr
            )
            stack.enter_context(redirect_stdout(destination))
            if quiet:
                stack.enter_context(redirect_stderr(destination))
            from libreyolo import LibreGen

            model_obj = LibreGen(model, device=device, dtype=dtype, offload=offload)
            results = model_obj(
                prompts,
                imgsz=size,
                seed=seed,
                n=n,
                save=save,
                output_path=output_path,
                **knobs,
            )
        results = results if isinstance(results, list) else [results]
        out.result(
            {
                "task": "generate",
                "model": model,
                "results": [
                    {**r.summary()[0], "path": getattr(r, "saved_path", None)}
                    for r in results
                ],
            }
        )
    except (ValueError, TypeError, NotImplementedError) as exc:
        exit_with_error(out, model_call_error_code(exc), str(exc))
    except ImportError as exc:
        exit_with_error(out, "model_load_failed", str(exc))
    except OSError as exc:
        exit_with_error(out, "io_error", str(exc))
    except Exception as exc:
        # Anything else comes from the upstream runtime; --json still gets a
        # structured error instead of a traceback.
        code = model_call_error_code(exc)
        if code == "io_error":
            code = "model_load_failed"
        exit_with_error(out, code, f"{type(exc).__name__}: {exc}")
