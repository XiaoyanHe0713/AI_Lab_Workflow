"""The cost meter and the frontier budget.

Cost meter: Colab compute units and API tokens to dollars, from config/prices.yaml. Matched-dollar
comparisons are read from here and nowhere else. GPU seconds times the unit rate of the GPU tier times
the dollar price of a unit, plus API tokens times the per-million price, plus CPU seconds at the CPU box
rate. The meter never guesses a rate it was not given: an unknown GPU raises. Offline replays cost
nothing: no GPU seconds and no tokens are charged in offline mode.

Budget: config/budget.yaml caps the monthly frontier spend (GPT, later Claude) at $200 and warns at
80 percent. The guard reads this month's spend from the ledger (API dollars of rows with provider openai
or anthropic in online or record mode, by calendar month) plus what this session has spent since, and is
consulted before every online or record call: above the warning fraction it warns once per session; at or
above the cap it raises and no call is made. To raise the cap, edit config/budget.yaml by hand; every
attempt row records the cap in force, so the change stays visible in the ledger.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import load_yaml, prices_config, repo_root

log = logging.getLogger("lab.budget")


@dataclass
class Cost:
    gpu_seconds: float = 0.0
    cpu_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    dollars: float = 0.0
    breakdown: dict = field(default_factory=dict)

    @property
    def api_dollars(self) -> float:
        return float(self.breakdown.get("api", 0.0))

    def __add__(self, other: "Cost") -> "Cost":
        b = dict(self.breakdown)
        for k, v in other.breakdown.items():
            b[k] = b.get(k, 0.0) + v
        return Cost(
            self.gpu_seconds + other.gpu_seconds,
            self.cpu_seconds + other.cpu_seconds,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.dollars + other.dollars,
            b,
        )


class CostMeter:
    def __init__(self, prices: dict | None = None, path: str | Path | None = None):
        self.prices = prices if prices is not None else prices_config(path)
        colab = self.prices.get("colab", {})
        self.dollars_per_unit = float(colab.get("dollars_per_unit", 0.0))
        self.units_per_hour = {str(k): float(v) for k, v in (colab.get("units_per_hour") or {}).items()}
        self.cpu_dollars_per_hour = float(self.prices.get("cpu", {}).get("dollars_per_hour", 0.0))
        self.api = {str(k): v for k, v in (self.prices.get("api") or {}).items()}
        self.overrides: dict[str, float] = {}

    # --- Colab ----------------------------------------------------------------------------
    def set_units_per_hour(self, gpu: str, rate: float) -> None:
        """Override the table with the rate read from the Colab Resources panel."""
        self.overrides[gpu] = float(rate)

    def rate(self, gpu: str) -> tuple[float, str]:
        """(units per hour, source) for a GPU tier."""
        if gpu in self.overrides:
            return self.overrides[gpu], "entered from the Colab Resources panel"
        if gpu in self.units_per_hour:
            return self.units_per_hour[gpu], "config/prices.yaml"
        raise KeyError(f"no compute-unit rate for GPU {gpu!r}; add it to config/prices.yaml")

    def gpu_dollars_per_hour(self, gpu: str) -> float:
        return self.rate(gpu)[0] * self.dollars_per_unit

    def units_to_dollars(self, units: float) -> float:
        return units * self.dollars_per_unit

    def gpu_cost(self, gpu: str, seconds: float) -> Cost:
        rate, _ = self.rate(gpu)
        units = rate * seconds / 3600.0
        dollars = units * self.dollars_per_unit
        return Cost(gpu_seconds=seconds, dollars=dollars, breakdown={"gpu": dollars, "units": units})

    # --- API tokens -------------------------------------------------------------------------
    def api_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> Cost:
        if model not in self.api:
            raise KeyError(f"no API price for model {model!r}; add it to config/prices.yaml")
        p = self.api[model]
        dollars = prompt_tokens * float(p["input"]) / 1e6 + completion_tokens * float(p["output"]) / 1e6
        return Cost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, dollars=dollars,
                    breakdown={"api": dollars})

    # --- CPU --------------------------------------------------------------------------------
    def cpu_cost(self, seconds: float) -> Cost:
        dollars = seconds / 3600.0 * self.cpu_dollars_per_hour
        return Cost(cpu_seconds=seconds, dollars=dollars, breakdown={"cpu": dollars})

    # --- one attempt ------------------------------------------------------------------------
    def attempt_cost(self, *, provider_kind: str, mode: str = "online", gpu: str | None = None,
                     model: str | None = None, gpu_seconds: float = 0.0, cpu_seconds: float = 0.0,
                     prompt_tokens: int = 0, completion_tokens: int = 0) -> Cost:
        """Cost of one attempt: a local endpoint is priced by GPU seconds, a frontier API by tokens, an
        offline replay by nothing; the judge's CPU seconds are added in every mode."""
        total = Cost()
        if mode == "offline":
            total.prompt_tokens += prompt_tokens
            total.completion_tokens += completion_tokens
        elif provider_kind in ("local", "openai_compatible"):
            if gpu is None:
                raise ValueError("a local endpoint attempt needs the GPU tier to be priced")
            total = total + self.gpu_cost(gpu, gpu_seconds)
            total.prompt_tokens += prompt_tokens
            total.completion_tokens += completion_tokens
        elif provider_kind in ("openai", "anthropic", "frontier"):
            total = total + self.api_cost(model or "", prompt_tokens, completion_tokens)
        else:
            raise ValueError(f"unknown provider kind {provider_kind!r}")
        if cpu_seconds:
            total = total + self.cpu_cost(cpu_seconds)
        return total


# --- the frontier budget ------------------------------------------------------------------------
@dataclass
class Budget:
    monthly_cap_usd: float = 200.0
    warning_fraction: float = 0.8
    providers: tuple[str, ...] = ("openai", "anthropic")
    modes: tuple[str, ...] = ("online", "record")
    path: str | None = None

    @property
    def warning_usd(self) -> float:
        return self.monthly_cap_usd * self.warning_fraction


def load_budget(path: str | Path | None = None) -> Budget:
    p = Path(path) if path else repo_root() / "config" / "budget.yaml"
    d = load_yaml(p) if p.exists() else {}
    f = d.get("frontier", d)
    return Budget(monthly_cap_usd=float(f.get("monthly_cap_usd", 200.0)),
                  warning_fraction=float(f.get("warning_fraction", 0.8)),
                  providers=tuple(f.get("providers", ("openai", "anthropic"))),
                  modes=tuple(f.get("modes", ("online", "record"))), path=str(p))


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    """Checked before every online or record call to a frontier provider."""

    def __init__(self, budget: Budget | None = None, ledger=None, clock=None):
        self.budget = budget or load_budget()
        self.ledger = ledger
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.session_spend = 0.0
        self.warned = False

    def ledger_spend(self) -> float:
        if self.ledger is None:
            return 0.0
        now = self.clock()
        return float(self.ledger.api_spend(now.year, now.month, kinds=self.budget.providers, modes=self.budget.modes))

    def month_spend(self) -> float:
        return self.ledger_spend() + self.session_spend

    def status(self) -> dict:
        spent = self.month_spend()
        cap = self.budget.monthly_cap_usd
        return {"cap_usd": cap, "spent_usd": spent, "fraction": spent / cap if cap else float("inf"),
                "warning_usd": self.budget.warning_usd, "month": self.clock().strftime("%Y-%m"),
                "budget_file": self.budget.path}

    def check(self, provider_kind: str, mode: str) -> dict:
        """Raise BudgetExceeded at or above the cap; warn once per session above the warning fraction."""
        if mode not in self.budget.modes or provider_kind not in self.budget.providers:
            return {"counted": False, "cap_usd": self.budget.monthly_cap_usd}
        st = self.status()
        if st["spent_usd"] >= st["cap_usd"]:
            raise BudgetExceeded(
                f"frontier budget exhausted: ${st['spent_usd']:.2f} of ${st['cap_usd']:.2f} spent in {st['month']} "
                f"(providers {list(self.budget.providers)}, modes {list(self.budget.modes)}). No call was made. "
                f"Raise the cap by hand in {st['budget_file']}; the ledger keeps the old cap on every row.")
        if st["spent_usd"] >= st["warning_usd"] and not self.warned:
            self.warned = True
            log.warning("frontier budget at %.0f%%: $%.2f of $%.2f spent in %s (cap in %s)",
                        100 * st["fraction"], st["spent_usd"], st["cap_usd"], st["month"], st["budget_file"])
        return {"counted": True, **st}

    def charge(self, dollars: float) -> None:
        self.session_spend += float(dollars)
