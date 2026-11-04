"""Weights on Drive: the frozen-base manifest hash, the pin in config/model.yaml, a zero-initialized
LoRA adapter so that an untrained role is exactly the base, and the serving command line.

The checkpoint hash is the sha256 over the sorted lines "path:sha256" of every file at the pinned
revision, taken from the Hub's file metadata (the LFS sha256 of each weight file), so pinning does not
re-read 54 GB from Drive. `verify_local` can re-hash the files on Drive to confirm the download.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path

from .config import write_yaml

ATTENTION_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJ = ("gate_proj", "up_proj", "down_proj")


def manifest_hash(entries: dict[str, str]) -> str:
    """entries: relative path -> sha256 hex. Order-independent."""
    lines = sorted(f"{p}:{h}" for p, h in entries.items())
    return "sha256:" + hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hub_manifest(repo_id: str, revision: str | None = None, token: str | None = None) -> dict:
    """File manifest from the Hugging Face Hub: {path: sha256} plus the resolved commit sha and byte total."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    entries, total = {}, 0
    for s in info.siblings:
        total += int(s.size or 0)
        if s.lfs is not None and s.lfs.sha256:
            entries[s.rfilename] = s.lfs.sha256
        else:
            # small text files (config, tokenizer) are not LFS; hash their content from the Hub
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(repo_id, s.rfilename, revision=info.sha, token=token)
            entries[s.rfilename] = sha256_file(p)
    return {"repo": repo_id, "revision": info.sha, "entries": entries, "files": len(entries), "bytes": total,
            "checkpoint_hash": manifest_hash(entries)}


def local_manifest(directory: str | Path, patterns: tuple[str, ...] = ("*.safetensors", "*.json", "*.txt", "*.model")) -> dict:
    """Re-hash the files on disk (slow for 54 GB; optional verification)."""
    d = Path(directory)
    entries = {}
    for pat in patterns:
        for p in d.rglob(pat):
            if p.is_file():
                entries[str(p.relative_to(d)).replace(os.sep, "/")] = sha256_file(p)
    return {"entries": entries, "checkpoint_hash": manifest_hash(entries), "files": len(entries)}


def pin_frozen_base(model_yaml: dict, manifest: dict, drive_dir: str, force: bool = False) -> dict:
    """Write the frozen-base pin into a model.yaml dict; refuse to change an existing different pin."""
    fb = model_yaml.setdefault("frozen_base", {})
    existing = fb.get("checkpoint_hash")
    new = manifest["checkpoint_hash"]
    if existing and existing != new and not force:
        raise RuntimeError(f"config/model.yaml pins {existing} (revision {fb.get('revision')}) but the download "
                           f"resolved to {new} (revision {manifest.get('revision')}). The base is frozen: keep the pin "
                           f"and download that revision, or set force=True to re-pin and log it.")
    fb.update({"hf_repo": manifest.get("repo", fb.get("hf_repo")), "revision": manifest.get("revision"),
               "checkpoint_hash": new, "files": manifest.get("files"), "bytes": manifest.get("bytes"),
               "drive_dir": drive_dir, "dtype": fb.get("dtype", "bfloat16")})
    if not existing or existing == new:
        fb.setdefault("pinned_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    else:
        fb["pinned_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fb["repinned_from"] = existing
    return model_yaml


def record_quantized(model_yaml: dict, *, method: str, drive_dir: str, checkpoint_hash: str | None) -> dict:
    q = model_yaml.setdefault("quantized", {})
    q.update({"method": method, "drive_dir": drive_dir, "checkpoint_hash": checkpoint_hash,
              "built_from": model_yaml.get("frozen_base", {}).get("checkpoint_hash"),
              "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return model_yaml


def save_model_yaml(path: str | Path, model_yaml: dict) -> None:
    write_yaml(path, model_yaml)


def lora_target_shapes(model_dir: str | Path) -> dict[str, tuple[int, int]]:
    """{module name: (out_features, in_features)} for every attention and MLP projection in the checkpoint,
    read from the safetensors headers, so hybrid architectures only get adapters where the modules exist."""
    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    files = sorted(d.glob("*.safetensors"))
    if index.exists():
        wm = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = sorted({d / f for f in wm.values()})
    shapes: dict[str, tuple[int, int]] = {}
    for f in files:
        with open(f, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(n))
        for name, meta in header.items():
            if name == "__metadata__" or not name.endswith(".weight"):
                continue
            mod = name[: -len(".weight")]
            leaf = mod.rsplit(".", 1)[-1]
            if leaf in ATTENTION_PROJ + MLP_PROJ and len(meta["shape"]) == 2:
                shapes[mod] = (int(meta["shape"][0]), int(meta["shape"][1]))
    return shapes


def make_zero_lora(model_dir: str | Path, out_dir: str | Path, rank: int = 64, alpha: int = 128,
                   base_model_name: str = "Qwen/Qwen3.6-27B") -> dict:
    """Write a rank-r LoRA adapter with B = 0 (and a deterministic A) in PEFT format: the adapter changes
    nothing, so a role served through it is exactly the base. Every adapter starts at zero."""
    import numpy as np
    from safetensors.numpy import save_file

    shapes = lora_target_shapes(model_dir)
    if not shapes:
        raise RuntimeError(f"no attention/MLP projections found in {model_dir}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tensors = {}
    for mod, (o, i) in sorted(shapes.items()):
        tensors[f"base_model.model.{mod}.lora_A.weight"] = (rng.standard_normal((rank, i)) / np.sqrt(i)).astype(np.float16)
        tensors[f"base_model.model.{mod}.lora_B.weight"] = np.zeros((o, rank), dtype=np.float16)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    targets = sorted({m.rsplit(".", 1)[-1] for m in shapes})
    cfg = {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "lora_dropout": 0.0, "bias": "none",
           "target_modules": targets, "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model_name,
           "inference_mode": True, "init_lora_weights": True, "fan_in_fan_out": False}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return {"modules": len(shapes), "targets": targets, "rank": rank, "dir": str(out)}


def precision_for_gpu(gpu: str, serving_cfg: dict) -> str:
    table = serving_cfg.get("by_gpu", {})
    for key, prec in table.items():
        if key.lower() in gpu.lower():
            return prec
    return "int4"


def gpu_tier(device_name: str) -> str:
    """Map an nvidia-smi / torch device name to the ledger's tier label."""
    n = device_name.upper()
    for tier in ("H100", "A100", "L4", "T4", "H200"):
        if tier in n:
            return tier
    if "RTX PRO 6000" in n or "B200" in n or "GB" in n and "BLACKWELL" in n:
        return "G4"
    return device_name.replace(" ", "_")


def server_command(*, server: str, model_path: str, served_name: str, precision: str, quant_method: str | None,
                   port: int = 8000, api_key: str | None = None, max_model_len: int = 32768, lora_rank: int = 64,
                   adapters: dict[str, str] | None = None, gpu_memory_utilization: float = 0.92,
                   enforce_eager: bool = False) -> list[str]:
    """The OpenAI-compatible server command line for vLLM (`vllm serve`) or SGLang.

    LoRA serving is enabled only when adapters are given, so a build that refuses LoRA for the model can
    still serve the base. `enforce_eager` skips torch.compile and CUDA graphs (slower, but it sidesteps
    compiler and graph-capture failures on new GPUs).
    """
    adapters = adapters or {}
    if server == "vllm":
        cmd = ["vllm", "serve", model_path, "--served-model-name", served_name, "--port", str(port),
               "--max-model-len", str(max_model_len), "--gpu-memory-utilization", str(gpu_memory_utilization),
               "--trust-remote-code"]
        if adapters:
            cmd += ["--enable-lora", "--max-lora-rank", str(lora_rank)]
        if enforce_eager:
            cmd += ["--enforce-eager"]
        if precision == "bf16":
            cmd += ["--dtype", "bfloat16"]
        elif quant_method and quant_method.startswith("awq-w4a16"):
            # llm-compressor output: compressed-tensors format with AWQ scales
            cmd += ["--quantization", "compressed-tensors"]
        elif quant_method and quant_method.startswith("autoawq"):
            cmd += ["--quantization", "awq_marlin", "--dtype", "float16"]
        elif quant_method and quant_method.startswith("gptq"):
            cmd += ["--quantization", "gptq_marlin", "--dtype", "float16"]
        elif quant_method and quant_method.startswith("bitsandbytes"):
            cmd += ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes", "--dtype", "bfloat16"]
        elif quant_method and quant_method.startswith("compressed"):
            cmd += ["--quantization", "compressed-tensors"]
        if adapters:
            cmd += ["--lora-modules"] + [f"{name}={path}" for name, path in adapters.items()]
        if api_key:
            cmd += ["--api-key", api_key]
        return cmd
    if server == "sglang":
        cmd = ["python", "-m", "sglang.launch_server", "--model-path", model_path, "--served-model-name", served_name,
               "--port", str(port), "--context-length", str(max_model_len), "--trust-remote-code",
               "--mem-fraction-static", str(gpu_memory_utilization)]
        if enforce_eager:
            cmd += ["--disable-cuda-graph"]
        if precision == "bf16":
            cmd += ["--dtype", "bfloat16"]
        elif quant_method and quant_method.startswith("awq"):
            cmd += ["--quantization", "awq"]
        elif quant_method and quant_method.startswith("gptq"):
            cmd += ["--quantization", "gptq"]
        elif quant_method and quant_method.startswith("bitsandbytes"):
            cmd += ["--quantization", "bitsandbytes"]
        if adapters:
            cmd += ["--enable-lora", "--max-lora-rank", str(lora_rank), "--lora-paths"] + \
                   [f"{name}={path}" for name, path in adapters.items()]
        if api_key:
            cmd += ["--api-key", api_key]
        return cmd
    raise ValueError(server)


def shell(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)
