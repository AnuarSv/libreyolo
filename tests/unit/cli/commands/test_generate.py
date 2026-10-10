"""Both generate CLI grammars, machine output and error codes."""

import json

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from libreyolo.cli.commands.generate import generate_cmd
from libreyolo.cli.parsing import KeyValueCommand

pytestmark = pytest.mark.unit


def app():
    application = typer.Typer()
    application.command("generate", cls=KeyValueCommand)(generate_cmd)
    return application


@pytest.fixture
def calls(monkeypatch):
    import libreyolo.models.gen as gen
    from libreyolo.utils.results import GeneratedImage, Results

    recorded = []

    class Model:
        def __init__(self, name, **kwargs):
            recorded.append(("load", name, kwargs))

        def __call__(self, prompt, **kwargs):
            recorded.append(("call", prompt, kwargs))
            print("loading progress")
            if kwargs.get("guidance"):
                raise TypeError("Model does not accept 'guidance'.")
            if prompt == "boom":
                raise AssertionError("Torch not compiled with CUDA enabled")
            if prompt == "oom":
                raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
            prompts = prompt if isinstance(prompt, list) else [prompt]
            return [
                Results(
                    None,
                    (8, 8),
                    generated=GeneratedImage(
                        np.zeros((8, 8, 3), dtype=np.uint8), prompt=text, seed=3
                    ),
                )
                for text in prompts
            ]

    monkeypatch.setattr(gen, "LibreGen", Model)
    return recorded


@pytest.mark.parametrize("key_value", [False, True])
def test_grammars(calls, key_value):
    pairs = {"prompt": "a red fox", "model": "sana-sprint", "seed": "3", "steps": "2"}
    args = (
        [f"{key}={val}" for key, val in pairs.items()]
        if key_value
        else [token for key, val in pairs.items() for token in (f"--{key}", val)]
    )
    args += (
        ["json=true", "save=false", "quiet=true"]
        if key_value
        else ["--json", "--no-save", "--quiet"]
    )
    result = CliRunner().invoke(app(), args)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["task"] == "generate"
    assert payload["results"][0]["prompt"] == "a red fox"
    assert payload["results"][0]["seed"] == 3
    assert calls[0] == (
        "load",
        "sana-sprint",
        {"device": "auto", "dtype": "auto", "offload": False},
    )
    _, prompt, kwargs = calls[1]
    assert prompt == "a red fox"
    assert kwargs == {
        "imgsz": None,
        "seed": 3,
        "n": 1,
        "save": False,
        "output_path": None,
        "steps": 2,
    }
    assert result.stderr == ""


def test_defaults_save_and_read_a_prompt_file(calls, tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("a red fox\n\n  a blue jay  \n")
    result = CliRunner().invoke(app(), [f"prompt={prompts}", "--json"])
    assert result.exit_code == 0, result.output
    assert calls[0][1] == "flux2-klein-4b"
    assert calls[1][1] == ["a red fox", "a blue jay"]
    assert calls[1][2]["save"] is True
    assert len(json.loads(result.stdout)["results"]) == 2


def test_help_and_errors(calls):
    result = CliRunner().invoke(app(), ["--help-json"])
    assert result.exit_code == 0
    assert "guidance" in str(json.loads(result.stdout))
    result = CliRunner().invoke(app(), ["--json"])
    assert result.exit_code == 2
    result = CliRunner().invoke(app(), ["prompt=a fox", "guidance=2.0", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["error"] == "config_type_error"
    # No family has a negative prompt yet, so the CLI does not offer one.
    result = CliRunner().invoke(app(), ["prompt=a fox", "negative=blurry", "--json"])
    assert json.loads(result.stdout)["error"] == "config_unknown_key"


def test_runtime_failures_stay_machine_readable(calls):
    result = CliRunner().invoke(app(), ["prompt=boom", "--json"])
    assert result.exit_code == 4
    payload = json.loads(result.stdout)
    assert payload["error"] == "model_load_failed"
    assert "AssertionError" in payload["message"]
    result = CliRunner().invoke(app(), ["prompt=oom", "--json"])
    assert json.loads(result.stdout)["error"] == "cuda_oom"


def test_prompt_file_edge_cases(calls, tmp_path):
    result = CliRunner().invoke(app(), [f"prompt={tmp_path / 'missing.txt'}", "--json"])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["error"] == "io_error"
    assert calls == []
    one = tmp_path / "one.txt"
    one.write_text("a red fox\n")
    result = CliRunner().invoke(app(), [f"prompt={one}", "--json"])
    assert result.exit_code == 0, result.output
    # A one-line file is one prompt, so output_path stays usable.
    assert calls[1][1] == "a red fox"
