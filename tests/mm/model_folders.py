"""The folders of models that a backend of Multimem claims, with the few files its `claims` reads and an empty model.safetensors (never real weights),
for the tests of `resolve_models`: `model_folder` and its tables, moved unchanged from tests/conftest.py of Multimem at fe0ece5."""

import json
from pathlib import Path


D1_KEYS = {"d1_attention": "set", "d1_set_size": 16, "d1_vision_max_pixels": 1638400, "d1_numbers": True, "d1_codes": 2}


D1_3B_CONFIG = {"model_type": "lfm2_vl", "architectures": ["Lfm2VlForConditionalGeneration"],
                "auto_map": {"AutoModel": "modeling_d1.D1Model"},
                "text_config": {"model_type": "lfm2", "hidden_size": 2048, "num_hidden_layers": 30}}


D3_FORMAT = {"format_version": 1, "format_id": "d3-code-readout-v1", "prompt": "d3"}


MODEL_FOLDERS = ("d1-Omni-2B", "d1-2B", "d1-3B", "d3-edge", "d3-mini", "d3-flash", "h2o-lightning-4b",
                 "pplx-decider-flash-v1.2", "unclaimed")


def model_folder(parent: Path, name: str) -> Path:
    """`parent/name`, holding the few files a backend's `claims` reads, with the values the real downloads have, and an
    empty model.safetensors (never real weights). `name` is one of MODEL_FOLDERS; "unclaimed" has a config.json no backend
    reads and no weights."""
    folder = Path(parent) / name
    folder.mkdir(parents=True, exist_ok=True)

    def write(file: str, data) -> None:
        path = folder / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")

    if name in ("d1-Omni-2B", "d1-2B"):
        write("config.json", {"model_type": "lfm3_omni" if name == "d1-Omni-2B" else "lfm3_vl", **D1_KEYS})
    elif name == "d1-3B":
        write("config.json", D1_3B_CONFIG)
    elif name in ("d3-edge", "d3-mini", "d3-flash"):
        write("config.json", {"model_type": "qwen3_5"})
        write("decision_config.json", {**D3_FORMAT, "base_model": "Qwen/Qwen3.5-9B" if name == "d3-flash" else "Qwen/Qwen3.5-4B"})
        write("MODEL_MANIFEST.json", {"schema": "d3-package-manifest/1", "model_name": name, "format_id": D3_FORMAT["format_id"]})
    elif name == "h2o-lightning-4b":
        write("config.json", {"model_type": "qwen3_5"})
        write("serve_config.json", {"model": "h2oai/h2o-lightning-4b", "base_model": "Qwen/Qwen3.5-4B"})
        write("h2o_lightning_shim.py", "")
    elif name == "pplx-decider-flash-v1.2":
        write("config.json", {"model_type": "qwen3_5"})
        write("joint_config.json", {"version": 3, "template": "joint-question-inline-readout-v1", "marker": "<|fim_suffix|>",
                                    "readout_offset": 0, "max_questions": None, "max_length": None})
        write("decision_config.json", {"format_version": 2, "base_model": "Qwen/Qwen3.5-4B", "max_pixels": 1638400})
        write("source/src/pplx_decider/__init__.py", "")
    elif name == "unclaimed":
        write("config.json", {"model_type": "gpt2"})
    else:
        raise ValueError(f"model_folder knows {', '.join(MODEL_FOLDERS)}, not {name!r}")
    if name != "unclaimed":
        write("model.safetensors", "")
    return folder
