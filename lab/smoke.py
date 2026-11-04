"""Serving smoke test: one chat completion through the endpoint, tokens per second, units per hour,
dollars per hour, all written to the ledger (a `serving` row and an attempt row).

    python -m lab.smoke --gpu A100 --precision int4 [--adapter builder] [--units-per-hour 11.77]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import lab_config, model_config
from .cost import CostMeter
from .ledger import Attempt, Ledger, prompt_hash
from .roles import OpenAIChatProvider, make_provider


def smoke(*, gpu: str, precision: str, endpoint: str | None = None, model: str | None = None,
          adapter: str | None = None, units_per_hour: float | None = None, ledger_path: str | None = None,
          max_tokens: int = 128, prompt: str = "In one sentence: what is a Gauss quadrature rule?",
          server: str | None = None, cfg: dict | None = None, provider=None) -> dict:
    cfg = cfg or lab_config()
    mcfg = model_config()
    if provider is None:
        if endpoint or model:
            provider = OpenAIChatProvider(kind="local", base_url=endpoint or cfg["endpoint"]["base_url"],
                                          api_key=None, model=model or cfg["endpoint"]["model"], name="local")
        else:
            provider = make_provider("local", cfg, mode="online")  # a smoke test is always live
    meter = CostMeter()
    if units_per_hour is not None:
        meter.set_units_per_hour(gpu, units_per_hour)
    rate, source = meter.rate(gpu)
    served = adapter or provider.model
    messages = [{"role": "user", "content": prompt}]
    ok, err, comp = True, None, None
    try:
        comp = provider.complete(messages, temperature=0.0, max_tokens=max_tokens, seed=0, model=served,
                                 extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    except Exception as e:
        ok, err = False, f"{type(e).__name__}: {e}"
    base = mcfg.get("frozen_base", {})
    q = mcfg.get("quantized", {})
    model_version = (f"{q.get('method') or 'int4'}:{q.get('checkpoint_hash') or 'unpinned'}" if precision == "int4"
                     else f"bf16:{base.get('checkpoint_hash') or 'unpinned'}")
    latency = comp.latency_s if comp else 0.0
    tps = comp.tokens_per_second if comp else None
    cost = meter.gpu_cost(gpu, latency)
    row = dict(gpu=gpu, precision=precision, model=str(base.get("hf_repo") or provider.model), model_version=model_version,
               server=server or mcfg.get("serving", {}).get("server"), adapter=adapter,
               prompt_tokens=comp.prompt_tokens if comp else None, completion_tokens=comp.completion_tokens if comp else None,
               latency_s=latency, tokens_per_second=tps, units_per_hour=rate, units_source=source,
               dollars_per_hour=meter.gpu_dollars_per_hour(gpu), dollars=cost.dollars, ok=int(ok),
               meta={"endpoint": getattr(provider, "base_url", None), "served_model": served, "error": err,
                     "provider": getattr(provider, "name", "local"), "mode": getattr(provider, "mode", "online"),
                     "text": comp.text[:500] if comp else None, "finish_reason": comp.finish_reason if comp else None})
    ledger = Ledger(ledger_path or cfg["storage"]["ledger"])
    sid = ledger.add_serving(**row)
    att = Attempt(role="smoke", model=row["model"], model_version=model_version, precision=precision, target="smoke-test",
                  prompt_hash=prompt_hash(messages), context={"serving_id": sid, "adapter": adapter, "gpu": gpu},
                  output=comp.text if comp else None, tool_log=[{"tool": "chat.completions", "latency_s": latency}],
                  certificate={"certifier": "none", "version": "", "passed": ok, "score": tps,
                               "lower_is_better": False, "details": {"tokens_per_second": tps}},
                  diagnosis=err, gpu_seconds=latency, cpu_seconds=0.0, dollars=cost.dollars, seed=0,
                  provider=getattr(provider, "name", "local"), mode=getattr(provider, "mode", "online"), api_dollars=0.0,
                  meta={"serving_row": row})
    ledger.add_attempt(att)
    ledger.close()
    row["serving_id"] = sid
    row["attempt_id"] = att.id
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.smoke")
    ap.add_argument("--gpu", required=True, help="H100 | G4 | A100 | L4 | T4")
    ap.add_argument("--precision", required=True, help="bf16 | int4")
    ap.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default=None, help="served model name")
    ap.add_argument("--adapter", default=None, help="LoRA adapter (role) name to address instead of the base")
    ap.add_argument("--units-per-hour", type=float, default=None, help="rate shown in the Colab Resources panel")
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--server", default=None, help="vllm | sglang (recorded)")
    args = ap.parse_args(argv)
    row = smoke(gpu=args.gpu, precision=args.precision, endpoint=args.endpoint, model=args.model, adapter=args.adapter,
                units_per_hour=args.units_per_hour, ledger_path=args.ledger, max_tokens=args.max_tokens, server=args.server)
    print(json.dumps(row, indent=1, default=str))
    return 0 if row["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
