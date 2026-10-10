# LibreGen

`LibreGen` runs generative image models: text in, an image out, behind the
same call as every other LibreYOLO task. The contract lives in
`docs/adr/0030-libregen-contract.md`.

## Install

```bash
pip install "libreyolo[gen]"
```

The extra pulls the Apache-2.0 `diffusers` package. Importing `LibreGen`
never imports it; it loads on first use.

## Generate

```python
from libreyolo import LibreGen

model = LibreGen("flux2-klein-4b")                 # autodownloads
result = model("a red fox sleeping in fresh snow")
result.save("fox.png")
```

Arguments every model understands:

```python
results = model(
    ["a red fox", "a blue jay"],   # one prompt or a list
    imgsz=1024,                    # int, or (height, width)
    seed=0,                        # same seed, same image
    n=2,                           # images per prompt
    save=True,                     # writes runs/generate/predict/*.png
)
```

One prompt with `n=1` returns a `Results`; a list of prompts or `n > 1`
returns a flat list. `stream=True` returns a generator.

- A string is always a prompt, never a path.
- Image `i` of a prompt uses `seed + i`. Without a seed a fresh one is drawn
  and recorded, so any image can be made again.
- `imgsz` sides are rounded up to the model's size multiple with a warning.

## The result

```python
result.generated.array      # (H, W, 3) uint8 RGB
result.generated.image      # PIL.Image
result.generated.prompt     # the text
result.generated.seed       # the seed that reproduces it
result.generated.params     # model, imgsz and the knobs in effect
result.save("fox.png")      # a PNG also stores prompt, seed and params
result.summary()            # one JSON-ready row
```

## Family knobs

Settings that only some models have are per-family. Each family has
defaults, so they are optional. A knob the family does not have raises and
lists the supported ones.

```python
model("a red fox", steps=8, guidance=1.0)
```

| Alias | Parameters | Knobs (defaults) | Download | License |
| --- | --- | --- | --- | --- |
| `flux2-klein-4b` (default) | 4B | `steps` (4), `guidance` (1.0, fixed: the distilled checkpoint runs without guidance) | about 16 GB | Apache-2.0 |
| `sana-sprint-0.6b` | 0.6B | `steps` (2), `guidance` (4.5) | about 7.7 GB | Apache-2.0; Gemma 2 text encoder under the Gemma Terms of Use |

`flux2-klein` and `sana-sprint` are short aliases.

## Devices and memory

```python
model = LibreGen("sana-sprint-0.6b", device="mps")
model = LibreGen("flux2-klein-4b", offload=True)     # CUDA: idle components off the GPU
model = LibreGen("flux2-klein-4b", dtype="fp16")
```

`device="auto"` picks CUDA, then MPS, then CPU. `dtype="auto"` is bf16 on
CUDA and MPS and fp32 on CPU.

## Local directories

```python
model = LibreGen("path/to/pipeline_dir")
```

A directory in a supported family's upstream format loads directly: an
offline copy, or a fine-tune made with upstream tooling.

## CLI

```bash
libreyolo generate prompt="a red fox" model=sana-sprint-0.6b seed=0 n=4
libreyolo generate prompt=prompts.txt steps=8 --json
```

`prompt` is the text, or a `.txt` file with one prompt per line; a `.txt`
path that does not exist is an error. `imgsz` is a single integer (square).
Images are saved by default; pass `--no-save` to skip.

## Not supported yet

`train()`, `val()`, `export()` and `source=` (image-conditioned generation)
raise `NotImplementedError`. Their names are reserved by ADR 0030.

## Licensing

LibreYOLO ships only its own adapter and payload code. Models load through
the Apache-2.0 `diffusers` API. Weights are downloaded from the upstream Hub
repo at a pinned revision; LibreYOLO does not redistribute them. LibreYOLO
adds no content filter; upstream usage policies apply.
