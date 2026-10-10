"""Text-to-image generation with the LibreGen tier."""

import typer

from ..command_utils import exit_with_error, help_json_callback, model_call_error_code
from ..output import OutputHandler


def generate_cmd(
    prompt: str = typer.Option(
        ..., help="Text to render, or a .txt file with one prompt per line"
    ),
    model: str = typer.Option(
        "flux2-klein-4b", help="LibreGen alias or local model directory"
    ),
    imgsz: int | None = typer.Option(None, help="Square output size in pixels"),
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
    try:
        prompts: str | list[str] = prompt
        if prompt.lower().endswith(".txt"):
            if not Path(prompt).is_file():
                raise FileNotFoundError(f"Prompt file not found: {prompt}")
            lines = Path(prompt).read_text(encoding="utf-8").splitlines()
            prompts = [line.strip() for line in lines if line.strip()]
            if len(prompts) == 1:
                prompts = prompts[0]
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
                imgsz=imgsz,
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
