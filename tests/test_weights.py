import json

import numpy as np
import pytest

from lab.weights import (gpu_tier, local_manifest, lora_target_shapes, make_zero_lora, manifest_hash, pin_frozen_base,
                         precision_for_gpu, record_quantized, server_command, sha256_file)


def test_manifest_hash_is_order_independent_and_sensitive():
    a = manifest_hash({"a.safetensors": "1" * 64, "config.json": "2" * 64})
    b = manifest_hash({"config.json": "2" * 64, "a.safetensors": "1" * 64})
    c = manifest_hash({"config.json": "2" * 64, "a.safetensors": "3" * 64})
    assert a == b != c and a.startswith("sha256:")


def test_pin_refuses_a_different_hash_unless_forced():
    y = {"frozen_base": {"hf_repo": "Qwen/Qwen3.6-27B", "checkpoint_hash": None}}
    m1 = {"repo": "Qwen/Qwen3.6-27B", "revision": "aaa", "checkpoint_hash": "sha256:1", "files": 2, "bytes": 10}
    pin_frozen_base(y, m1, "models/x")
    assert y["frozen_base"]["checkpoint_hash"] == "sha256:1" and y["frozen_base"]["pinned_at"]
    pin_frozen_base(y, m1, "models/x")  # same pin again is fine
    m2 = dict(m1, revision="bbb", checkpoint_hash="sha256:2")
    with pytest.raises(RuntimeError):
        pin_frozen_base(y, m2, "models/x")
    pin_frozen_base(y, m2, "models/x", force=True)
    assert y["frozen_base"]["checkpoint_hash"] == "sha256:2" and y["frozen_base"]["repinned_from"] == "sha256:1"
    record_quantized(y, method="awq-w4a16", drive_dir="models/q", checkpoint_hash="sha256:q")
    assert y["quantized"]["built_from"] == "sha256:2"


def test_local_manifest_and_sha256(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "a.safetensors").write_bytes(b"\x00" * 10)
    m = local_manifest(tmp_path)
    assert m["files"] == 2 and m["entries"]["config.json"] == sha256_file(tmp_path / "config.json")


def _fake_checkpoint(d):
    from safetensors.numpy import save_file

    t = {"model.layers.0.self_attn.q_proj.weight": np.zeros((8, 4), np.float16),
         "model.layers.0.self_attn.k_proj.weight": np.zeros((2, 4), np.float16),
         "model.layers.0.mlp.down_proj.weight": np.zeros((4, 6), np.float16),
         "model.layers.0.linear_attn.in_proj.weight": np.zeros((4, 4), np.float16),  # hybrid layer: not a target
         "model.norm.weight": np.zeros((4,), np.float16)}
    save_file(t, str(d / "model-00001-of-00001.safetensors"))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in t}}))


def test_zero_lora_adapter(tmp_path):
    ck = tmp_path / "ck"
    ck.mkdir()
    _fake_checkpoint(ck)
    shapes = lora_target_shapes(ck)
    assert set(shapes) == {"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.mlp.down_proj"}
    info = make_zero_lora(ck, tmp_path / "adapter", rank=4)
    assert info["modules"] == 3 and info["targets"] == ["down_proj", "k_proj", "q_proj"]
    from safetensors.numpy import load_file

    w = load_file(str(tmp_path / "adapter" / "adapter_model.safetensors"))
    assert w["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"].shape == (8, 4)
    assert w["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"].shape == (4, 4)
    assert not w["base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"].any()
    cfg = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert cfg["r"] == 4 and cfg["peft_type"] == "LORA"


def test_gpu_tier_and_precision():
    serving = {"by_gpu": {"H100": "bf16", "G4": "bf16", "A100": "int4", "L4": "int4"}}
    assert gpu_tier("NVIDIA A100-SXM4-40GB") == "A100" and precision_for_gpu("A100", serving) == "int4"
    assert gpu_tier("NVIDIA H100 80GB HBM3") == "H100" and precision_for_gpu("H100", serving) == "bf16"
    assert gpu_tier("NVIDIA L4") == "L4" and gpu_tier("NVIDIA RTX PRO 6000 Blackwell") == "G4"
    assert precision_for_gpu("G4", serving) == "bf16"


def test_server_commands():
    v = server_command(server="vllm", model_path="/m", served_name="qwen3.6-27b", precision="bf16", quant_method=None,
                       adapters={"builder": "/a/builder"}, api_key="k")
    assert v[:3] == ["vllm", "serve", "/m"]
    assert "--enable-lora" in v and "--max-lora-rank" in v and "builder=/a/builder" in v and "--dtype" in v
    assert v[v.index("--max-num-seqs") + 1] == "64"
    fewer = server_command(server="vllm", model_path="/m", served_name="x", precision="bf16", quant_method=None, max_num_seqs=16)
    assert fewer[fewer.index("--max-num-seqs") + 1] == "16"
    base_only = server_command(server="vllm", model_path="/m", served_name="x", precision="bf16", quant_method=None)
    assert "--enable-lora" not in base_only and "--enforce-eager" not in base_only
    eager = server_command(server="vllm", model_path="/m", served_name="x", precision="bf16", quant_method=None,
                           enforce_eager=True, gpu_memory_utilization=0.8)
    assert "--enforce-eager" in eager and eager[eager.index("--gpu-memory-utilization") + 1] == "0.8"
    se = server_command(server="sglang", model_path="/m", served_name="x", precision="bf16", quant_method=None, enforce_eager=True)
    assert "--disable-cuda-graph" in se and "--mem-fraction-static" in se and "--max-running-requests" in se
    q = server_command(server="vllm", model_path="/q", served_name="qwen3.6-27b", precision="int4", quant_method="awq-w4a16")
    assert "compressed-tensors" in q
    a = server_command(server="vllm", model_path="/q", served_name="qwen3.6-27b", precision="int4", quant_method="autoawq")
    assert "awq_marlin" in a
    b = server_command(server="vllm", model_path="/m", served_name="x", precision="int4", quant_method="bitsandbytes-nf4")
    assert "bitsandbytes" in b
    s = server_command(server="sglang", model_path="/m", served_name="x", precision="bf16", quant_method=None, adapters={"r": "/r"})
    assert "sglang.launch_server" in s[2] and "--lora-paths" in s
    with pytest.raises(ValueError):
        server_command(server="tgi", model_path="/m", served_name="x", precision="bf16", quant_method=None)
