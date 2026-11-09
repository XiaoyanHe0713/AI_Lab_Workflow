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
<a href="https://colab.research.google.com/github/XiaoyanHe0713/AI_Lab_Workflow/blob/build%2Fweek-6-diagnosis-second-try/colab/serve.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
""")

md("""
# Discovery at 27B: serve the frozen base from Colab

Week 1 of the build plan. This notebook

1. mounts Google Drive and pulls the repository (read-only token from Colab secrets if the repository is private),
2. downloads the bf16 weights of `Qwen/Qwen3.6-27B` once to Drive and **pins the checkpoint hash** as the frozen base in `config/model.yaml`,
3. builds (or reuses) the 4-bit development copy on an A100, or serves bf16 on an H100 / G4,
4. starts an OpenAI-compatible server (vLLM by default, SGLang as the alternative) with LoRA serving enabled, so that a role can later be an adapter name,
5. runs the serving smoke test at the session's precision and the week-2 demo (16 attempts on `quad-d7`, as parallel requests), both written to the ledger on Drive,
6. optionally (week 3) runs the training gate on the training-family candidates and the ablation harness with GPT for the lessons,
7. optionally (week 4) runs the loop end to end on the quadrature curriculum (theorist, builder, breaker, judge, archive, lesson bank, gate memos), the frozen baseline for every certifier family, the 256-attempt search-only baseline on `quad-square-d5`, and the first transfer-matrix cells.

Rules that never bend: every score comes from a program; quantized runs are marked in the ledger's precision column and never enter a reported cell; weights and the ledger stay on Drive, never in the repository; API keys live in Colab secrets.

Secrets used if present (Runtime > Secrets): `HF_TOKEN` (gated download), `GITHUB_TOKEN` (private repository, read-only).
""")

code("""
#@title 1. Configuration
import time
SESSION_T0 = time.time()

LAB_ROOT = "/content/drive/MyDrive/ai_lab"      #@param {type:"string"}
REPO_URL = "https://github.com/XiaoyanHe0713/AI_Lab_Workflow.git"  #@param {type:"string"}
REPO_BRANCH = "build/week-6-diagnosis-second-try"  #@param {type:"string"}
SERVER = "vllm"                                 #@param ["vllm", "sglang"]
QUANT_METHOD = "awq"                            #@param ["awq", "bnb"]
UNITS_PER_HOUR = 0                              #@param {type:"number"}
RUN_SMOKE = True                                #@param {type:"boolean"}
RUN_DEMO = True                                 #@param {type:"boolean"}
DEMO_ATTEMPTS = 16                              #@param {type:"integer"}
MAKE_ZERO_ADAPTER = True                        #@param {type:"boolean"}
VERIFY_LOCAL_HASH = False                       #@param {type:"boolean"}
OPEN_TUNNEL = False                             #@param {type:"boolean"}
CONCURRENCY = 16                                #@param {type:"integer"}
PROBE_CONCURRENCIES = "16 32 48"                #@param {type:"string"}
RUN_GATE = False                                #@param {type:"boolean"}
GATE_TARGETS = "quad-square-d5 quad-triangle-d4"  #@param {type:"string"}
GATE_ATTEMPTS = 64                              #@param {type:"integer"}
GATE_THINKING = "0 512"                         #@param {type:"string"}
GATE_MAX_TOKENS = 8192                          #@param {type:"integer"}
RUN_ABLATION = False                            #@param {type:"boolean"}
ABLATION_TARGET = "quad-d7"                     #@param {type:"string"}
ABLATION_ATTEMPTS = 16                          #@param {type:"integer"}
RECORD_FIXTURES = False                         #@param {type:"boolean"}
RUN_LOOP = False                                #@param {type:"boolean"}
LOOP_TARGETS = "quad-d7 quad-square-d5 quad-triangle-d2 quad-triangle-d3 quad-triangle-d4"  #@param {type:"string"}
LOOP_ATTEMPTS = 64                              #@param {type:"integer"}
LOOP_THEORIST_ATTEMPTS = 16                     #@param {type:"integer"}
LOOP_PROPOSAL_ATTEMPTS = 16                     #@param {type:"integer"}
LOOP_BREAKER_ATTEMPTS = 8                       #@param {type:"integer"}
LOOP_BREAKER_CANDIDATES = 4                     #@param {type:"integer"}
LOOP_TOOL_CALLS = 0                             #@param {type:"integer"}
LOOP_CARRIER = "lessons"                        #@param ["lessons", "none"]
LOOP_REVISION_MAX_TOKENS = 4096                 #@param {type:"integer"}
RUN_BASELINES = False                           #@param {type:"boolean"}
BASELINE_TARGETS = "pep-gd-strongly-convex-distance order-ode-p4 opcount-dft-n16 ssp-s3-p3 stab-real-s4-p4"  #@param {type:"string"}
BASELINE_ATTEMPTS = 64                          #@param {type:"integer"}
RUN_SEARCH_BASELINE = False                     #@param {type:"boolean"}
SEARCH_BASELINE_TARGET = "quad-square-d5"       #@param {type:"string"}
SEARCH_BASELINE_ATTEMPTS = 256                  #@param {type:"integer"}
SEARCH_BASELINE_SEED = 64                       #@param {type:"integer"}
RUN_MATRIX = False                              #@param {type:"boolean"}
MATRIX_SOURCES = "quad-d7"                      #@param {type:"string"}
MATRIX_TARGETS = "quad-square-d5"               #@param {type:"string"}
MATRIX_CARRIERS = "none unrelated second_try diagnosis_only diagnosis_second_try lessons process_only"  #@param {type:"string"}
MATRIX_SEEDS = 5                                #@param {type:"integer"}
MATRIX_ATTEMPTS = 32                            #@param {type:"integer"}
MATRIX_TOOL_CALLS = 0                           #@param {type:"integer"}
RUN_MINE = False                                #@param {type:"boolean"}
MINE_TARGETS = "quad-d7 quad-square-d5 quad-triangle-d2 quad-triangle-d3 quad-triangle-d4"  #@param {type:"string"}
MINE_ALL_GROUPS = False                         #@param {type:"boolean"}

# RUN_LOOP (week 4): one generation of the loop on LOOP_TARGETS in that order (the curriculum rungs): per rung the theorist
#   proposes (lemmas checked by SymPy in the sandbox), the builder runs its gate group of LOOP_ATTEMPTS from the plain prompt
#   (the gate memo), a lesson group when the bank holds lessons, and LOOP_PROPOSAL_ATTEMPTS builds derived from the proposals;
#   the breaker attacks LOOP_BREAKER_CANDIDATES survivors with LOOP_BREAKER_ATTEMPTS each; survivors enter the archive; GPT
#   (secret OPENAI_API_KEY) extracts lessons and writes dossiers; without the secret the loop runs search-only (--no-harness).
#   LOOP_TOOL_CALLS = 1 gives the builder one certifier dry run and a revision per attempt (the gate memo then says so).
#   LOOP_REVISION_MAX_TOKENS caps a theorist revision round on its own (the 2026-11-06 run let two revisions run to 16,384
#   tokens under the role's cap of 8,192 a round); the lemma file's checks share a time budget below the sandbox cap, so one
#   slow check is cut off (a timeout, a failed check) and the rest still run.
# RUN_BASELINES (week 4): the gate on one target per remaining certifier family (BASELINE_TARGETS, 64 attempts each), then
#   the frozen baseline record over every family in the ledger (python -m lab.harness.checker --baseline --reportable-only).
# RUN_SEARCH_BASELINE (week 4): SEARCH_BASELINE_ATTEMPTS attempts of the frozen base on SEARCH_BASELINE_TARGET as one gate
#   group from seed SEARCH_BASELINE_SEED: does the 7-point rule ever appear without training? The loop's gate group uses seeds
#   0 to LOOP_ATTEMPTS - 1, so a baseline meant to be independent of it starts past them (the 2026-11-06 baseline at seed 0
#   shared its first 64 samples with the gate group).
# RUN_MATRIX (week 4): transfer-matrix cells from the archive of MATRIX_SOURCES to MATRIX_TARGETS with MATRIX_CARRIERS at
#   MATRIX_SEEDS seeds x MATRIX_ATTEMPTS attempts (lessons are never measured on the family they came from; the program refuses).
#   Week 5: the carriers second_try (one certifier dry run, no lesson) and diagnosis_only (the program's diagnosis of the
#   target's own none cell, no remedy) are the two target-side control rows; MATRIX_TOOL_CALLS gives every content carrier that
#   many dry runs; every cell reports its symptom rates (how often each kept detector fires) and the change against the baseline.
#   The defaults since 2026-11-08 are the matrix narrowed for the next session after the 2026-11-07 run: the square alone (the
#   degree-4 triangle's pass rate measured nothing at 0.02), the five rows that were not flat twice at 3 x 16 (the baseline,
#   the unrelated row, which with no other family in the archive is the framing control and the reference every gain should
#   be read against, the second try, the lessons, the redacted lessons; diagnosis_only and skills dropped), and 5 x 32 = 160
#   attempts per cell, where the standard error of a paired difference of pass rates is about 0.05 and the second try's Radon
#   count rests on 160 attempts. The 2026-11-07 run's full set: MATRIX_TARGETS "quad-square-d5 quad-triangle-d4",
#   MATRIX_CARRIERS "none second_try diagnosis_only lessons process_only unrelated skills" at 3 x 16.
#   Week 6: diagnosis_second_try is the third target-side control, the diagnosis-only prompt with one certifier dry run and a
#   revision and no lesson, so that the baseline, diagnosis_only, second_try and diagnosis_second_try are a 2 x 2 (no help,
#   the diagnosis alone, one dry run alone, both). The 2026-11-07 run read the diagnosis alone as doing nothing and the second
#   try as the only row that moved the optimal count; the open question is whether naming the failure helps once the model
#   has seen the certifier's verdict, and the 2 x 2 answers it in one session. The default set since 2026-11-09 is therefore
#   the five rows of 2026-11-08 plus diagnosis_only and diagnosis_second_try, seven rows. A row at 5 x 32 is 160 attempts,
#   about a fifth of the 2026-11-08 estimate for the five-row run (about 7.5 h and $8 at the 2026-11-07 session's rate), so
#   about 1.5 h and $1.60 per row, and a row with a dry run makes two requests per attempt, so it runs longer than a plain
#   one: the seven rows are about 10.5 h and $11.
# RUN_MINE (week 5; cell 16, ahead of the matrix; any runtime, a GPU session included): mine detectors on the largest
#   plain-prompt group (dry-run groups excluded) of each MINE_TARGETS target (the loop's base groups): GPT proposes them when the
#   secret is set, the built-in detectors alone otherwise; every candidate is verified on the held-out half; the kept detectors'
#   firing tables, the librarian queries they name and the diagnosis-only texts are printed and written under detectors/ on
#   Drive, and the matrix's diagnosis row then finds them in the ledger. MINE_ALL_GROUPS mines every base group, not only the largest.
# REPO_BRANCH: the branch this notebook was built from; cell 2 clones or checks it out, so the code matches the cells
#   (empty = the repository's default branch, which may be older than this notebook).
# QUANT_METHOD: "awq" builds a W4A16 copy once with llm-compressor (A100, about an hour) and keeps it on Drive;
#               "bnb" quantizes in flight with bitsandbytes NF4 at load time (nothing to build, slower serving).
# UNITS_PER_HOUR: the rate shown in Colab's Resources panel for this GPU; 0 means use config/prices.yaml.
# CONCURRENCY: attempts of a group sent as parallel requests (the server runs up to --max-num-seqs sequences, 64 by default).
# PROBE_CONCURRENCIES: cell 8 sends a wave of identical requests at each of these and prints the wave's aggregate tokens per
#   second (the 2026-11-07 session moved about 170 tokens per second at 16 in flight on a 96 GB card). Set CONCURRENCY to the
#   largest value whose wave still gained, rerun cell 1, then the long cells: they read CONCURRENCY from this cell.
# GATE_MAX_TOKENS: answer tokens per attempt in the gate (the first G4 run hit the 4096 cap on most triangle attempts).
# GATE_THINKING: thinking budgets; a positive budget caps the reasoning and then asks for the answer (two requests).
# RUN_GATE / RUN_ABLATION (week 3): the training gate on GATE_TARGETS and the ablation harness; the ablation needs
#   the Colab secret OPENAI_API_KEY for the lesson harness (spend capped by config/budget.yaml).
# RECORD_FIXTURES (week 3): record the harness tests' GPT fixtures here and export them to Drive for a commit.
UNITS_PER_HOUR = UNITS_PER_HOUR or None
""")

code("""
#@title 2. Drive, repository, package
import os, subprocess, secrets, sys
from google.colab import drive, userdata
drive.mount("/content/drive")
for d in ("", "models", "adapters", "logs", "config", "cache"):
    os.makedirs(os.path.join(LAB_ROOT, d), exist_ok=True)

_SECRETS = {}
def secret(name, tries=3):
    # A Colab secret (notebook access on), read once per runtime and kept in memory: a read can fail once the browser
    # tab has detached after a long cell, so a failure prints its kind (never the value), retries, and later cells
    # reuse the copy this cell read.
    if name in _SECRETS:
        return _SECRETS[name]
    for i in range(tries):
        try:
            value = userdata.get(name)
            break
        except getattr(userdata, "SecretNotFoundError", ()):   # no such secret: nothing to retry
            return None
        except Exception as err:
            print(f"secret {name}: {type(err).__name__} on read {i + 1} of {tries}")
            time.sleep(3)
    else:
        return os.environ.get(name)
    _SECRETS[name] = value
    return value

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

OPENAI_KEY_PRESENT = bool(secret("OPENAI_API_KEY"))   # read now, while the browser is attached; cells 11, 13 and 17 reuse the copy
print("OPENAI_API_KEY secret:", "present" if OPENAI_KEY_PRESENT else "absent (the frontier harnesses run without GPT)")
token = secret("GITHUB_TOKEN")  # a fine-grained token with read access to Contents on this repository
REPO = "/content/AI_Lab_Workflow"
if not os.path.exists(REPO):
    git("clone", *(["--branch", REPO_BRANCH] if REPO_BRANCH else []), REPO_URL, REPO, token=token)
else:
    git("-C", REPO, "fetch", "origin", token=token)
    if REPO_BRANCH:
        git("-C", REPO, "checkout", REPO_BRANCH)
    git("-C", REPO, "pull", "--ff-only", token=token)
BRANCH, COMMIT = git("-C", REPO, "rev-parse", "--abbrev-ref", "HEAD").strip(), git("-C", REPO, "rev-parse", "--short", "HEAD").strip()
print("checked out:", BRANCH, COMMIT)
if REPO_BRANCH and BRANCH != REPO_BRANCH:
    raise RuntimeError(f"the runtime has {BRANCH} checked out, not {REPO_BRANCH}: the code would not match this notebook")
os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "huggingface_hub", "hf_transfer", "safetensors", "requests"], check=True)
# a running kernel keeps the modules it imported before the pull; drop them so later cells import the new code
for _name in [n for n in sys.modules if n == "lab" or n.startswith("lab.")]:
    del sys.modules[_name]
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
import importlib
import requests
import lab.weights
importlib.reload(lab.weights)  # pick up the repository's current code even if an older copy was imported earlier
from lab.weights import server_command, shell

pkg = ["vllm"] if SERVER == "vllm" else ["sglang[all]"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkg], check=True)
LOG = os.path.join(LAB_ROOT, "logs", f"{SERVER}.log")
KEY = os.environ["LAB_ENDPOINT_KEY"]


def start_server(adapters, max_len, eager=False, mem=0.92, seqs=64):
    \"\"\"Launch the server and wait for /health; returns (process, seconds to ready or None).\"\"\"
    cmd = server_command(server=SERVER, model_path=MODEL_PATH, served_name=mcfg["serving"]["served_model_name"],
                         precision=PRECISION, quant_method=QUANT, port=8000, api_key=KEY,
                         max_model_len=max_len, lora_rank=64, adapters=adapters, enforce_eager=eager,
                         gpu_memory_utilization=mem, max_num_seqs=seqs)
    shown = shell(cmd).replace(KEY, "<LAB_ENDPOINT_KEY>")
    print(shown)
    with open(LOG, "a") as f:
        f.write(f"\\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {shown}\\n")
    proc = subprocess.Popen(cmd, stdout=open(LOG, "a"), stderr=subprocess.STDOUT)
    for i in range(360):
        if proc.poll() is not None:
            return proc, None
        try:
            if requests.get("http://127.0.0.1:8000/health", timeout=5).status_code == 200:
                return proc, 5 * i
        except Exception:
            pass
        time.sleep(5)
    proc.kill()
    return proc, None


def log_excerpt(max_lines=70):
    \"\"\"The failed start's own evidence: the engine core's ERROR lines (they carry the real exception), then the end of the log.
    The API server's long traceback that follows them only says 'engine core initialization failed'.\"\"\"
    with open(LOG, "r", errors="replace") as f:
        lines = f.readlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("===== ")]
    seg = lines[starts[-1]:] if starts else lines
    core = [l for l in seg if "ERROR" in l and "Engine core initialization failed" not in l]
    keep = core[-max_lines:]
    return ("".join(keep) if keep else "(no ERROR lines from the engine core)\\n") + "... end of log:\\n" + "".join(seg[-10:])


def free_gpu(proc=None):
    \"\"\"Kill a failed server and its engine process, then wait until the GPU memory is back.\"\"\"
    if proc is not None and proc.poll() is None:
        proc.kill()
    subprocess.run("pkill -9 -f 'vllm serve'; pkill -9 -f 'EngineCore'; pkill -9 -f 'sglang.launch_server'", shell=True)
    used = None
    for _ in range(45):
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        try:
            used = int(r.stdout.strip().splitlines()[0])
        except Exception:
            return None
        if used < 2000:
            break
        time.sleep(2)
    print(f"GPU memory in use after cleanup: {used} MiB")
    return used


# A short ladder, one change at a time, chosen from the log of the failed start: without LoRA if the build
# refuses LoRA for the model; eager mode if torch.compile or CUDA-graph capture failed; a shorter context
# and less reserved memory if the KV cache did not fit or allocation ran out of memory. The last rung tries
# all three. What actually served is recorded in the ledger through the smoke rows.
import re
START = dict(adapters=dict(ADAPTERS), max_len=int(mcfg["serving"]["max_model_len"]), eager=False, mem=0.92,
             seqs=int(mcfg["serving"].get("max_num_seqs", 64)))
queue = [(START, "as configured")]
tried = set()
server, READY_S, SERVED = None, None, None
while queue and server is None:
    cfg_try, why = queue.pop(0)
    key = (tuple(sorted(cfg_try["adapters"])), cfg_try["max_len"], cfg_try["eager"], cfg_try["mem"], cfg_try["seqs"])
    if key in tried:
        continue
    tried.add(key)
    print(f"starting the server ({why}; adapters {list(cfg_try['adapters'])}; max_model_len {cfg_try['max_len']}; "
          f"max_num_seqs {cfg_try['seqs']}; eager {cfg_try['eager']}; gpu memory {cfg_try['mem']})")
    free_gpu()
    proc, READY_S = start_server(**cfg_try)
    if READY_S is not None:
        server, SERVED = proc, (cfg_try, why)
        break
    tail = log_excerpt()
    print(f"the server exited with code {proc.returncode}; the engine core's errors from {LOG}:\\n{tail[-6000:]}")
    free_gpu(proc)
    low = tail.lower()
    lora_refused = cfg_try["adapters"] and "lora" in low and any(s in low for s in ("not support", "unsupported", "does not implement"))
    seqs_failed = "max_num_seqs" in low and "exceeds" in low
    compile_failed = not seqs_failed and any(s in low for s in ("cuda graph", "cudagraph", "torch.compile", "inductor", "triton", "dynamo"))
    memory_failed = not seqs_failed and any(s in low for s in ("out of memory", "outofmemoryerror", "kv cache", "max seq len", "max_model_len"))
    if seqs_failed and cfg_try["seqs"] > 16:
        m = re.search(r"at most (\\d+)", tail)
        new_seqs = max(16, min(cfg_try["seqs"] // 2, int(m.group(1)) if m else cfg_try["seqs"] // 2))
        queue.append((dict(cfg_try, seqs=new_seqs), "retry with fewer concurrent sequences: one state block each for the hybrid layers"))
    if lora_refused:
        queue.append((dict(cfg_try, adapters={}), "retry without LoRA: this build refuses LoRA for the model"))
    if compile_failed and not cfg_try["eager"]:
        queue.append((dict(cfg_try, eager=True), "retry in eager mode: torch.compile or CUDA-graph capture failed"))
    if memory_failed and cfg_try["max_len"] > 8192:
        queue.append((dict(cfg_try, max_len=cfg_try["max_len"] // 2, mem=0.85), "retry with a shorter context and less reserved memory"))
    if not queue and len(tried) == 1:
        queue.append((dict(cfg_try, adapters={}, eager=True, max_len=min(cfg_try["max_len"], 16384), mem=0.85, seqs=16),
                      "last rung: no LoRA, eager mode, 16k context, 16 sequences"))
if server is None:
    raise RuntimeError(f"the server did not start; the last lines of {LOG} are printed above")
cfg_used, SERVER_NOTE = SERVED
ADAPTERS, MAX_MODEL_LEN = cfg_used["adapters"], cfg_used["max_len"]
SERVER_LABEL = SERVER if SERVER_NOTE == "as configured" else f"{SERVER} ({SERVER_NOTE})"
os.environ["LAB_ENDPOINT"] = "http://127.0.0.1:8000/v1"
print(f"endpoint ready after {READY_S} s: {os.environ['LAB_ENDPOINT']} (model {mcfg['serving']['served_model_name']}, "
      f"adapters {list(ADAPTERS)}, max_model_len {MAX_MODEL_LEN}, max_num_seqs {cfg_used['seqs']}, eager {cfg_used['eager']}, {SERVER_NOTE})")
""")

code("""
#@title 8. Smoke test: one completion at this precision, tokens per second and units per hour into the ledger; then the throughput probe at PROBE_CONCURRENCIES
import json
from lab.smoke import smoke, throughput

if RUN_SMOKE:
    rows = [smoke(gpu=GPU, precision=PRECISION, server=SERVER_LABEL, units_per_hour=UNITS_PER_HOUR)]
    if ADAPTERS:
        rows.append(smoke(gpu=GPU, precision=PRECISION, server=SERVER_LABEL, adapter="builder", units_per_hour=UNITS_PER_HOUR))
    for r in rows:
        print(json.dumps({k: r[k] for k in ("gpu", "precision", "adapter", "completion_tokens", "latency_s",
                                             "tokens_per_second", "units_per_hour", "dollars_per_hour", "ok")}, default=str))
        if not r["ok"]:
            print("  error:", r["meta"]["error"])
    # the throughput probe: a wave of identical requests at each concurrency; tokens_per_second is the wave's aggregate rate,
    # the rate a group of attempts gets at that CONCURRENCY (the single-stream rate above is not)
    for n in [int(v) for v in PROBE_CONCURRENCIES.split()]:
        w = throughput(gpu=GPU, precision=PRECISION, server=SERVER_LABEL, concurrency=n, units_per_hour=UNITS_PER_HOUR)
        print(json.dumps({k: w[k] for k in ("concurrency", "requests_ok", "completion_tokens", "latency_s", "tokens_per_second",
                                             "dollars", "ok")}, default=str))
        if w["meta"]["errors"]:
            print("  errors:", w["meta"]["errors"])
    print("set CONCURRENCY in cell 1 to the largest value whose wave still gained, rerun cell 1, then the long cells")
""")

code("""
#@title 9. The week-2 demo: 16 attempts on quad-d7, judged in the sandbox, priced, written to the ledger on Drive
from lab.demo import main as demo_main

if RUN_DEMO:
    argv = ["--target", "quad-d7", "--attempts", str(DEMO_ATTEMPTS), "--gpu", GPU, "--precision", PRECISION,
            "--concurrency", str(CONCURRENCY)]
    if UNITS_PER_HOUR:
        argv += ["--units-per-hour", str(UNITS_PER_HOUR)]
    demo_main(argv)
else:
    print("RUN_DEMO is False in cell 1: the demo did not run.")
""")

code("""
#@title 10. Week 3: the training gate (64 attempts per target as parallel requests, a memo per target)
from lab.gate import main as gate_main

if RUN_GATE:
    argv = ["--targets", *GATE_TARGETS.split(), "--attempts", str(GATE_ATTEMPTS), "--thinking", *GATE_THINKING.split(),
            "--gpu", GPU, "--precision", PRECISION, "--concurrency", str(CONCURRENCY), "--max-tokens", str(GATE_MAX_TOKENS)]
    if UNITS_PER_HOUR:
        argv += ["--units-per-hour", str(UNITS_PER_HOUR)]
    gate_main(argv)
else:
    print("RUN_GATE is False in cell 1: the gate did not run. Set RUN_GATE = True, rerun cell 1, then this cell.")
""")

code("""
#@title 11. Week 3: the ablation harness online (no notebook, notebook, notebook with gate) with GPT for the lessons
# The lesson extraction and the gate are language parts: GPT through the OpenAI API (key from the Colab secret
# OPENAI_API_KEY; capped by config/budget.yaml). Every attempt of the three arms is in the ledger on Drive.
from lab.harness.ablation import main as ablation_main

if RUN_ABLATION:
    key = secret("OPENAI_API_KEY")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        argv = ["--target", ABLATION_TARGET, "--attempts", str(ABLATION_ATTEMPTS), "--gpu", GPU, "--precision", PRECISION,
                "--concurrency", str(CONCURRENCY), "--harness-provider", "openai", "--harness-mode", "online"]
        if UNITS_PER_HOUR:
            argv += ["--units-per-hour", str(UNITS_PER_HOUR)]
        ablation_main(argv)
    else:
        print("no OPENAI_API_KEY secret: the ablation needs a frontier provider for its lessons; add the secret and rerun this cell")
else:
    print("RUN_ABLATION is False in cell 1: the ablation did not run. Set RUN_ABLATION = True, rerun cell 1, then this cell.")
""")

code("""
#@title 12. Week 3: record the harness fixtures with GPT and export them to Drive (no GPU needed)
# The harness tests replay recorded GPT responses. This cell records them here: the key comes from the Colab secret
# OPENAI_API_KEY, every call passes the budget guard and lands in the ledger on Drive. It then rebuilds the notebook
# arms' local sequences, writes the librarian's reviews of the provisional scoreboard entries, replays the recorded
# tests offline once to prove the fixtures, and copies the files to LAB_ROOT/exports/week3-fixtures/ for a commit
# from your own machine (Colab pulls the repository read-only). Cost: well under a dollar at gpt-5-mini prices.
import shutil
if RECORD_FIXTURES:
    key = secret("OPENAI_API_KEY")
    assert key, "add the Colab secret OPENAI_API_KEY (notebook access on) and rerun this cell"
    rec_env = dict(os.environ, OPENAI_API_KEY=key, LAB_FRONTIER_MODE="record")
    off_env = dict(os.environ, LAB_FRONTIER_MODE="offline")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    tests = ["tests/test_librarian.py", "tests/test_lessons.py", "tests/test_checker.py", "tests/test_ablation.py", "tests/test_loop.py"]
    r = subprocess.run([sys.executable, "-m", "pytest", *tests, "-k", "openai", "-p", "no:cacheprovider", "-rs"],
                       cwd=REPO, env=rec_env, text=True, capture_output=True)
    print("record run:", r.stdout[-2500:], r.stderr[-1000:])
    subprocess.run([sys.executable, "tests/make_fixtures.py"], cwd=REPO, check=True)
    for board, mode in (("quadrature", "offline"), ("time_integrators", "record"), ("fast_transforms", "record")):
        subprocess.run([sys.executable, "-m", "lab.harness.librarian", "--review", board, "--provider", "openai", "--mode", mode],
                       cwd=REPO, env=rec_env if mode == "record" else off_env, check=True)
    r2 = subprocess.run([sys.executable, "-m", "pytest", *tests, "-k", "openai", "-p", "no:cacheprovider"],
                        cwd=REPO, env=off_env, text=True, capture_output=True)
    print("offline replay:", r2.stdout[-1200:], r2.stderr[-600:])
    out = os.path.join(LAB_ROOT, "exports", "recorded-fixtures")
    for rel in ("tests/fixtures/providers/openai", "tests/fixtures/providers/local", "scoreboard/reviews"):
        src, dst = os.path.join(REPO, rel), os.path.join(out, rel)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
    print("exported to", out, ": copy its three folders into the repository (same relative paths) and commit them")
else:
    print("RECORD_FIXTURES is False in cell 1: nothing was recorded. Set RECORD_FIXTURES = True, rerun cell 1, then this cell.")
""")

code("""
#@title 13. Week 4: the loop end to end on the quadrature curriculum (theorist, builder, breaker, judge, archive, lessons)
# One generation on LOOP_TARGETS in order. Every attempt of every role is a ledger row on Drive with its cost; a gate memo
# per target under gates/, the generation record (archive snapshot, ledger pointer, delayed credit) under generations/.
# GPT extracts the lessons and writes the dossiers (secret OPENAI_API_KEY, capped by config/budget.yaml); without the
# secret the loop runs search-only with the program's template dossiers.
from lab.loop import main as loop_main

if RUN_LOOP:
    argv = ["--targets", *LOOP_TARGETS.split(), "--attempts", str(LOOP_ATTEMPTS), "--theorist-attempts", str(LOOP_THEORIST_ATTEMPTS),
            "--proposal-attempts", str(LOOP_PROPOSAL_ATTEMPTS), "--breaker-attempts", str(LOOP_BREAKER_ATTEMPTS),
            "--breaker-candidates", str(LOOP_BREAKER_CANDIDATES), "--tool-calls", str(LOOP_TOOL_CALLS), "--carrier", LOOP_CARRIER,
            "--max-tokens", str(GATE_MAX_TOKENS), "--revision-max-tokens", str(LOOP_REVISION_MAX_TOKENS), "--thinking", "0",
            "--concurrency", str(CONCURRENCY), "--gpu", GPU, "--precision", PRECISION]
    if UNITS_PER_HOUR:
        argv += ["--units-per-hour", str(UNITS_PER_HOUR)]
    key = secret("OPENAI_API_KEY")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        argv += ["--harness-provider", "openai", "--harness-mode", "online"]
    else:
        print("no OPENAI_API_KEY secret: the loop runs search-only (no lesson extraction, template dossiers)")
        argv += ["--no-harness"]
    loop_main(argv)
else:
    print("RUN_LOOP is False in cell 1: the loop did not run. Set RUN_LOOP = True, rerun cell 1, then this cell.")
""")

code("""
#@title 14. Week 4: the frozen baseline for every certifier family (the gate on one target per family, then the baseline record)
from lab.gate import main as gate_main
from lab.harness.checker import main as checker_main

if RUN_BASELINES:
    argv = ["--targets", *BASELINE_TARGETS.split(), "--attempts", str(BASELINE_ATTEMPTS), "--thinking", "0", "--gpu", GPU,
            "--precision", PRECISION, "--concurrency", str(CONCURRENCY), "--max-tokens", str(GATE_MAX_TOKENS)]
    if UNITS_PER_HOUR:
        argv += ["--units-per-hour", str(UNITS_PER_HOUR)]
    gate_main(argv)
    checker_main(["--baseline", "--reportable-only"])
else:
    print("RUN_BASELINES is False in cell 1: the family baselines did not run.")
""")

code("""
#@title 15. Week 4: the search-only baseline, 256 attempts of the frozen base on quad-square-d5 (does the 7-point rule ever appear?)
from lab.gate import main as gate_main

if RUN_SEARCH_BASELINE:
    argv = ["--targets", SEARCH_BASELINE_TARGET, "--attempts", str(SEARCH_BASELINE_ATTEMPTS), "--thinking", "0", "--gpu", GPU,
            "--precision", PRECISION, "--concurrency", str(CONCURRENCY), "--max-tokens", str(GATE_MAX_TOKENS),
            "--seed", str(SEARCH_BASELINE_SEED)]  # past the loop's gate group (seeds 0 to LOOP_ATTEMPTS - 1): an independent sample
    if UNITS_PER_HOUR:
        argv += ["--units-per-hour", str(UNITS_PER_HOUR)]
    gate_main(argv)
    print("verify-first in the memo above says whether the best known value (7 nodes) appeared; the memo is under gates/ on Drive")
else:
    print("RUN_SEARCH_BASELINE is False in cell 1: the search-only baseline did not run.")
""")

code("""
#@title 16. Week 5: mine detectors on the base groups already in the ledger and print their firing tables (any runtime; before the matrix)
# The loop's plain-prompt groups are on Drive; this names their failures after the fact: a program splits each group, GPT
# (secret OPENAI_API_KEY, read once in cell 2) proposes detectors or the built-in ones stand alone, the held-out half keeps or
# rejects each, and the report (firing table, the librarian query the diagnosis names, the diagnosis-only text) lands under
# detectors/ on Drive. Nothing here needs the GPU, so a CPU runtime serves as well; in a GPU session the cell comes before the
# matrix so that the detectors GPT proposed and the program kept are in the ledger when cell 17 writes the diagnosis row
# (minutes of GPU time, not hours).
from lab.harness.checker import main as checker_main

if RUN_MINE:
    argv = ["--mine", "--targets", *MINE_TARGETS.split()]
    if MINE_ALL_GROUPS:
        argv.append("--all-groups")
    key = secret("OPENAI_API_KEY")
    if key:
        os.environ["OPENAI_API_KEY"] = key
        argv += ["--provider", "openai", "--mode", "online"]
    else:
        print("no OPENAI_API_KEY secret: the built-in detectors alone are mined")
    checker_main(argv)
else:
    print("RUN_MINE is False in cell 1: no detectors were mined.")
""")

code("""
#@title 17. Week 4: transfer-matrix cells from the archive (lessons never measured on the family they came from)
# Rows: the experience set of MATRIX_SOURCES (its archived methods and the lessons the loop extracted) plus the two controls
# (process-only lessons with content redacted; unrelated-domain experience) and the target-side control rows (week 5:
# second_try, one dry run and no lesson; diagnosis_only, the program's diagnosis of the target's own baseline and no remedy;
# week 6: diagnosis_second_try, the diagnosis and one dry run, which completes the 2 x 2 of the two controls; the diagnosis is
# mined once per target for both diagnosis rows); columns: MATRIX_TARGETS in their settings. Every cell reports its symptom
# rates and the change against the baseline.
# Needs the loop (cell 13) to have run with GPT so the bank holds lessons; the report lands under matrices/ on Drive.
# Cell 16 (RUN_MINE) runs first so that the detectors GPT proposed and the program kept enter the diagnosis row; this cell
# mines the built-in ones itself. The defaults since 2026-11-09 are the square and seven rows at 5 x 32: the five rows of 2026-11-08
# plus the two diagnosis rows, so that one session measures the 2 x 2 (cell 1 says why and what a row costs); the unrelated row
# with no other family in the archive is the framing control, the reference every gain is read against.
from lab.harness.matrix import main as matrix_main

if RUN_MATRIX:
    argv = ["--sources", *MATRIX_SOURCES.split(), "--targets", *MATRIX_TARGETS.split(), "--carriers", *MATRIX_CARRIERS.split(),
            "--seeds", str(MATRIX_SEEDS), "--attempts", str(MATRIX_ATTEMPTS), "--max-tokens", str(GATE_MAX_TOKENS), "--thinking", "0",
            "--concurrency", str(CONCURRENCY), "--gpu", GPU, "--precision", PRECISION, "--no-harness", "--tool-calls", str(MATRIX_TOOL_CALLS)]
    if UNITS_PER_HOUR:
        argv += ["--units-per-hour", str(UNITS_PER_HOUR)]
    matrix_main(argv)
else:
    print("RUN_MATRIX is False in cell 1: no matrix cells ran.")
""")

code("""
#@title 18. Optional: reach the endpoint from your own machine (cloudflared quick tunnel)
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
#@title 19. Session cost so far
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
