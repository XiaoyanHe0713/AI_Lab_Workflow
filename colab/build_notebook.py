"""Generate colab/serve.ipynb (python colab/build_notebook.py). The notebook source lives here so that
changes are reviewable as code; the .ipynb is committed next to it."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# Discovery at 27B: serve the frozen base from Colab

Week 1 of the build plan. This notebook

1. mounts Google Drive and pulls the repository (read-only token from Colab secrets if the repository is private),
2. downloads the bf16 weights of `Qwen/Qwen3.6-27B` once to Drive and **pins the checkpoint hash** as the frozen base in `config/model.yaml`,
3. builds (or reuses) the 4-bit development copy on an A100, or serves bf16 on an H100 / G4,
4. starts an OpenAI-compatible server (vLLM by default, SGLang as the alternative) with LoRA serving enabled, so that a role can later be an adapter name,
5. runs the serving smoke test at the session's precision and the week-2 demo (16 attempts on `quad-d7`), both written to the ledger on Drive.

Rules that never bend: every score comes from a program; quantized runs are marked in the ledger's precision column and never enter a reported cell; weights and the ledger stay on Drive, never in the repository; API keys live in Colab secrets.

Secrets used if present (Runtime > Secrets): `HF_TOKEN` (gated download), `GITHUB_TOKEN` (private repository, read-only).
""")

code("""
#@title 1. Configuration
import time
SESSION_T0 = time.time()

LAB_ROOT = "/content/drive/MyDrive/ai_lab"      #@param {type:"string"}
REPO_URL = "https://github.com/XiaoyanHe0713/AI_Lab_Workflow.git"  #@param {type:"string"}
REPO_BRANCH = ""                                #@param {type:"string"}
SERVER = "vllm"                                 #@param ["vllm", "sglang"]
QUANT_METHOD = "awq"                            #@param ["awq", "bnb"]
UNITS_PER_HOUR = 0                              #@param {type:"number"}
RUN_SMOKE = True                                #@param {type:"boolean"}
RUN_DEMO = True                                 #@param {type:"boolean"}
DEMO_ATTEMPTS = 16                              #@param {type:"integer"}
MAKE_ZERO_ADAPTER = True                        #@param {type:"boolean"}
VERIFY_LOCAL_HASH = False                       #@param {type:"boolean"}
OPEN_TUNNEL = False                             #@param {type:"boolean"}

# REPO_BRANCH: empty clones the repository's default branch; set it to track another branch.
# QUANT_METHOD: "awq" builds a W4A16 copy once with llm-compressor (A100, about an hour) and keeps it on Drive;
#               "bnb" quantizes in flight with bitsandbytes NF4 at load time (nothing to build, slower serving).
# UNITS_PER_HOUR: the rate shown in Colab's Resources panel for this GPU; 0 means use config/prices.yaml.
UNITS_PER_HOUR = UNITS_PER_HOUR or None
""")

code("""
#@title 2. Drive, repository, package
import os, subprocess, secrets, sys
from google.colab import drive, userdata
drive.mount("/content/drive")
for d in ("", "models", "adapters", "logs", "config", "cache"):
    os.makedirs(os.path.join(LAB_ROOT, d), exist_ok=True)

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

def git(*args, token=None):
    \"\"\"Run git with the token supplied through GIT_ASKPASS, never on the command line, never in an error.\"\"\"
    env = dict(os.environ)
    if token:
        askpass = "/content/.git-askpass.sh"
        with open(askpass, "w") as f:
            f.write('#!/bin/sh\\ncase "$1" in\\n  Username*) echo "x-access-token";;\\n  *) echo "$GIT_TOKEN";;\\nesac\\n')
        os.chmod(askpass, 0o700)
        env.update({"GIT_ASKPASS": askpass, "GIT_TOKEN": token, "GIT_TERMINAL_PROMPT": "0"})
    r = subprocess.run(["git", *args], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed (exit {r.returncode}):\\n{r.stderr[-2000:]}")
    return r.stdout

token = secret("GITHUB_TOKEN")  # a fine-grained token with read access to Contents on this repository
REPO = "/content/AI_Lab_Workflow"
if not os.path.exists(REPO):
    git("clone", *(["--branch", REPO_BRANCH] if REPO_BRANCH else []), REPO_URL, REPO, token=token)
else:
    git("-C", REPO, "pull", "--ff-only", token=token)
print("checked out:", git("-C", REPO, "rev-parse", "--abbrev-ref", "HEAD").strip(), git("-C", REPO, "rev-parse", "--short", "HEAD").strip())
os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "huggingface_hub", "hf_transfer", "safetensors", "requests"], check=True)
os.environ.update({"LAB_ROOT": LAB_ROOT, "LAB_REPO": REPO, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
os.environ.setdefault("LAB_ENDPOINT_KEY", secrets.token_urlsafe(24))
print("repository at", REPO, "| lab root", LAB_ROOT)
""")

code("""
#@title 3. GPU tier and precision
import torch
from lab.config import model_config
from lab.cost import CostMeter
from lab.weights import gpu_tier, precision_for_gpu

DEVICE = torch.cuda.get_device_name(0)
GPU = gpu_tier(DEVICE)
mcfg = model_config()
PRECISION = precision_for_gpu(GPU, mcfg["serving"])
meter = CostMeter()
if UNITS_PER_HOUR:
    meter.set_units_per_hour(GPU, UNITS_PER_HOUR)
RATE, RATE_SOURCE = meter.rate(GPU)
os.environ.update({"LAB_GPU": GPU, "LAB_PRECISION": PRECISION})
print(f"{DEVICE} -> tier {GPU}, precision {PRECISION}; {RATE} units/h ({RATE_SOURCE}) = ${meter.gpu_dollars_per_hour(GPU):.3f}/h")
if RATE_SOURCE != "entered from the Colab Resources panel":
    print("Check the rate in the Resources panel and set UNITS_PER_HOUR in cell 1 if it differs; the ledger records the source.")
""")

code("""
#@title 4. The frozen base: bf16 weights on Drive, pinned by checkpoint hash
from huggingface_hub import snapshot_download
from lab.weights import hub_manifest, local_manifest, pin_frozen_base, save_model_yaml

hf_token = secret("HF_TOKEN")
fb = mcfg["frozen_base"]
BF16_DIR = os.path.join(LAB_ROOT, fb["drive_dir"])
manifest = hub_manifest(fb["hf_repo"], revision=fb.get("revision"), token=hf_token)
print(f"{fb['hf_repo']} @ {manifest['revision']}: {manifest['files']} files, {manifest['bytes'] / 1e9:.1f} GB, {manifest['checkpoint_hash']}")
mcfg = pin_frozen_base(mcfg, manifest, fb["drive_dir"])  # refuses to overwrite a different pin
PIN_PATH = os.path.join(LAB_ROOT, "config", "model.yaml")
save_model_yaml(PIN_PATH, mcfg)
snapshot_download(fb["hf_repo"], revision=manifest["revision"], local_dir=BF16_DIR, token=hf_token)
if VERIFY_LOCAL_HASH:
    lm = local_manifest(BF16_DIR)
    assert lm["checkpoint_hash"] == manifest["checkpoint_hash"], "the Drive copy differs from the Hub manifest"
    print("Drive copy verified against the manifest")
print("frozen base pinned in", PIN_PATH, "- commit it to the repository as config/model.yaml")
""")

code("""
#@title 5. The 4-bit development build (A100 and L4 tiers only)
from lab.weights import record_quantized

Q_DIR = os.path.join(LAB_ROOT, mcfg["quantized"]["drive_dir"])
MODEL_PATH, QUANT = BF16_DIR, None
if PRECISION == "int4":
    if QUANT_METHOD == "awq":
        if os.path.exists(os.path.join(Q_DIR, "config.json")):
            QUANT = "awq-w4a16"
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llmcompressor", "datasets"], check=True)
            try:
                from datasets import load_dataset
                from llmcompressor import oneshot
                from llmcompressor.modifiers.awq import AWQModifier
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(BF16_DIR)
                model = AutoModelForCausalLM.from_pretrained(BF16_DIR, torch_dtype="auto")
                ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:256]")
                ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})
                recipe = [AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"])]
                oneshot(model=model, tokenizer=tok, dataset=ds, recipe=recipe, max_seq_length=1024, num_calibration_samples=256)
                model.save_pretrained(Q_DIR, save_compressed=True)
                tok.save_pretrained(Q_DIR)
                QUANT = "awq-w4a16"
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print("AWQ build failed:", repr(e)[:500])
                print("falling back to in-flight bitsandbytes NF4 (recorded as such in the ledger)")
                QUANT = "bitsandbytes-nf4"
    else:
        QUANT = "bitsandbytes-nf4"
    if QUANT == "awq-w4a16":
        MODEL_PATH = Q_DIR
        qhash = local_manifest(Q_DIR)["checkpoint_hash"]
        qdir = mcfg["quantized"]["drive_dir"]
    else:
        MODEL_PATH = BF16_DIR  # quantized from the frozen base at load time
        qhash = mcfg["frozen_base"]["checkpoint_hash"]
        qdir = fb["drive_dir"]
    mcfg = record_quantized(mcfg, method=QUANT, drive_dir=qdir, checkpoint_hash=qhash)
    save_model_yaml(PIN_PATH, mcfg)
print(f"serving {PRECISION} ({QUANT or 'bf16'}) from {MODEL_PATH}")
""")

code("""
#@title 6. A zero-initialised rank-64 LoRA adapter named "builder" (an untrained role is exactly the base)
from lab.weights import make_zero_lora

ADAPTERS = {}
if MAKE_ZERO_ADAPTER:
    adir = os.path.join(LAB_ROOT, "adapters", "builder")
    if not os.path.exists(os.path.join(adir, "adapter_config.json")):
        print(make_zero_lora(BF16_DIR, adir, rank=64, base_model_name=fb["hf_repo"]))
    ADAPTERS["builder"] = adir
print("adapters:", ADAPTERS or "none (base only)")
""")

code("""
#@title 7. Start the OpenAI-compatible server with LoRA serving enabled
import requests
from lab.weights import server_command, shell

pkg = ["vllm"] if SERVER == "vllm" else ["sglang[all]"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkg], check=True)
cmd = server_command(server=SERVER, model_path=MODEL_PATH, served_name=mcfg["serving"]["served_model_name"],
                     precision=PRECISION, quant_method=QUANT, port=8000, api_key=os.environ["LAB_ENDPOINT_KEY"],
                     max_model_len=int(mcfg["serving"]["max_model_len"]), lora_rank=64, adapters=ADAPTERS)
print(shell(cmd))
LOG = os.path.join(LAB_ROOT, "logs", f"{SERVER}.log")
server = subprocess.Popen(cmd, stdout=open(LOG, "a"), stderr=subprocess.STDOUT)
for i in range(360):
    if server.poll() is not None:
        raise RuntimeError(f"the server exited with code {server.returncode}; see {LOG}")
    try:
        if requests.get("http://127.0.0.1:8000/health", timeout=5).status_code == 200:
            break
    except Exception:
        pass
    time.sleep(5)
else:
    raise RuntimeError(f"the server did not become healthy in 30 minutes; see {LOG}")
os.environ["LAB_ENDPOINT"] = "http://127.0.0.1:8000/v1"
print(f"endpoint ready after {5 * i} s: {os.environ['LAB_ENDPOINT']} (model {mcfg['serving']['served_model_name']}, adapters {list(ADAPTERS)})")
""")

code("""
#@title 8. Smoke test: one completion at this precision, tokens per second and units per hour into the ledger
import json
from lab.smoke import smoke

if RUN_SMOKE:
    rows = [smoke(gpu=GPU, precision=PRECISION, server=SERVER, units_per_hour=UNITS_PER_HOUR)]
    if ADAPTERS:
        rows.append(smoke(gpu=GPU, precision=PRECISION, server=SERVER, adapter="builder", units_per_hour=UNITS_PER_HOUR))
    for r in rows:
        print(json.dumps({k: r[k] for k in ("gpu", "precision", "adapter", "completion_tokens", "latency_s",
                                             "tokens_per_second", "units_per_hour", "dollars_per_hour", "ok")}, default=str))
        if not r["ok"]:
            print("  error:", r["meta"]["error"])
""")

code("""
#@title 9. The week-2 demo: 16 attempts on quad-d7, judged in the sandbox, priced, written to the ledger on Drive
from lab.demo import main as demo_main

if RUN_DEMO:
    argv = ["--target", "quad-d7", "--attempts", str(DEMO_ATTEMPTS), "--gpu", GPU, "--precision", PRECISION]
    if UNITS_PER_HOUR:
        argv += ["--units-per-hour", str(UNITS_PER_HOUR)]
    demo_main(argv)
""")

code("""
#@title 10. Optional: reach the endpoint from your own machine (cloudflared quick tunnel)
# The in-notebook path above is the primary one. This exposes the server through a temporary public URL
# protected only by LAB_ENDPOINT_KEY; close the tunnel when done. The binary is Cloudflare's official release.
if OPEN_TUNNEL:
    import re
    subprocess.run("curl -sL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared", shell=True, check=True)
    tunnel = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in tunnel.stdout:
        m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
        if m:
            print("on your machine:")
            print(f"  export LAB_ENDPOINT={m.group(0)}/v1")
            print(f"  export LAB_ENDPOINT_KEY={os.environ['LAB_ENDPOINT_KEY']}")
            print(f"  export LAB_GPU={GPU} LAB_PRECISION={PRECISION}")
            break
""")

code("""
#@title 11. Session cost so far
hours = (time.time() - SESSION_T0) / 3600
units = hours * RATE
print(f"{hours:.2f} h on {GPU} at {RATE} units/h ({RATE_SOURCE}): about {units:.2f} units = ${meter.units_to_dollars(units):.2f}")
print("ledger:", os.path.join(LAB_ROOT, "ledger.sqlite"), "| mirror:", os.path.join(LAB_ROOT, "ledger.attempts.jsonl"))
""")


def build(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.update({"colab": {"name": "serve.ipynb", "provenance": [], "gpuType": "A100"},
                        "accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                        "language_info": {"name": "python"}})
    cells = []
    for kind, src in CELLS:
        cells.append(nbf.v4.new_markdown_cell(src) if kind == "markdown" else nbf.v4.new_code_cell(src))
    nb.cells = cells
    nbf.validate(nb)
    path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells)")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "serve.ipynb")
