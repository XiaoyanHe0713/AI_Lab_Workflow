"""The source-target matrix (proposal, "Transfer settings, carriers and controls"; E1): experience sets drawn from
the archive per source family, the two controls, a cell runner that applies a carrier from a source set to a
target setting at matched dollars with N seeds, and the analysis that reads certified gain per cell with a
bootstrap interval from the ledger. The ablation harness (lab/harness/ablation.py) is the single-cell version: three
carriers on one target from the target's own experience; this module runs any carrier from any set on any target.

    python -m lab.harness.matrix --sources quad-d7 --targets quad-square-d5 quad-triangle-d4 \\
        --carriers none lessons process_only unrelated --seeds 5 --attempts 16 --harness-mode offline
    python -m lab.harness.matrix --analyze mtx_...                                   # the matrix from the ledger

Rules the program enforces. Lessons are never measured on the family they came from: a content carrier (lessons,
lessons_gate, skills, process_only) is refused when the target is one of the set's instances, and refused when the
target's family key (certificate family and structure class, lab/targets.py) is one of the set's, unless
`allow_same_family` is passed explicitly, in which case the cell is marked and excluded from the reported matrix.
The `none` carrier is the baseline of every row and is run once per target and seed. Gain in a cell is the certified
pass rate (and optimal rate, and best score) with the carrier minus without, paired by seed, with a percentile
bootstrap interval over seeds (over attempts when there is one seed). Open targets carry their flag in every cell.

Week 5 adds two target-side control rows that carry no experience and are never refused: `second_try` (one certifier
dry run and a revision, no lesson: the floor every lesson gain is read against) and `diagnosis_only` (the program's
diagnosis of the target's own `none` cell, the kept detectors that fired with their counts and no remedy, which
separates knowing the fix from seeing the problem). The built-in detectors are mined on the `none` cell when the
diagnosis-only row runs. Every cell also reports its symptom rates, how often each kept detector of the target fires,
and every gain the symptom change against the baseline: the evidence that a carrier removed the failure it addresses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import lab_config
from ..demo import run_group
from ..ledger import Attempt, Ledger, new_id
from ..roles import KINDS, MODES
from ..targets import get_target, transfer_setting
from .archive import Archive
from .base import harness_provider
from .carriers import CARRIERS, CONTENT_CARRIERS, CONTROL_CARRIERS, apply_carrier
from .checker import base_groups, firing_table, kept_detectors, largest_base_group, mine_builtin, symptom_rates
from .lessons import LessonBank, LessonHarness


class TransferLeak(ValueError):
    """A content carrier was about to be measured on the family its experience came from."""


@dataclass
class ExperienceSet:
    """What the lab recorded on a set of source targets: the certified methods in the archive, the lessons extracted
    from those targets' groups, and the attempts behind them."""
    id: str
    name: str
    kind: str                                   # archive | unrelated | process_only
    source_targets: list[str]
    family_keys: list[str]
    certificate_families: list[str]
    lesson_ids: list[str] = field(default_factory=list)
    artifact_ids: list[str] = field(default_factory=list)
    attempt_ids: list[str] = field(default_factory=list)
    content_terms: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "source_targets": self.source_targets, "family_keys": self.family_keys,
                "certificate_families": self.certificate_families, "lessons": len(self.lesson_ids), "artifacts": len(self.artifact_ids),
                "attempts": len(self.attempt_ids), "note": self.note}


def _set_id(kind: str, sources: list[str], lesson_ids: list[str], artifact_ids: list[str]) -> str:
    blob = json.dumps({"k": kind, "s": sorted(sources), "l": sorted(lesson_ids), "a": sorted(artifact_ids)}, sort_keys=True)
    return "exp_" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def lessons_of_targets(bank: LessonBank, targets: list[str]) -> list:
    """Active lessons whose targets or source group lie in the given targets (process lessons included only when they
    came from these targets, so a set never borrows another family's craft by accident)."""
    want = set(targets)
    return [l for l in bank.all() if not l.retired and (set(l.targets) & want)]


def experience_from_archive(ledger: Ledger, archive: Archive, bank: LessonBank, source_targets: list[str], *, name: str | None = None) -> ExperienceSet:
    """The experience set of a source family: its archived methods, the lessons extracted from its groups, its attempts."""
    blocks = [get_target(t).transfer for t in source_targets]
    if any(b is None for b in blocks):
        raise ValueError("every source target needs a transfer block")
    artifact_ids = []
    for t in source_targets:
        artifact_ids += [m["artifact_id"] for m in archive.methods(t)]
    lessons = lessons_of_targets(bank, source_targets)
    attempt_ids = [r.id for t in source_targets for r in ledger.attempts(target=t, role="builder")]
    terms = sorted({w for b in blocks for w in b.content_terms})
    lesson_ids = sorted(l.id for l in lessons)
    return ExperienceSet(id=_set_id("archive", source_targets, lesson_ids, artifact_ids), name=name or "+".join(source_targets), kind="archive",
                         source_targets=list(source_targets), family_keys=sorted({b.family_key for b in blocks}),
                         certificate_families=sorted({b.family for b in blocks}), lesson_ids=lesson_ids, artifact_ids=sorted(set(artifact_ids)),
                         attempt_ids=attempt_ids, content_terms=terms, note="experience from the archive of the source targets")


def process_only_control(base: ExperienceSet) -> ExperienceSet:
    """The content-redaction control: the same set, applied through the process_only carrier (the redaction happens
    in the carrier; the set records the terms it redacts)."""
    return ExperienceSet(id=_set_id("process_only", base.source_targets, base.lesson_ids, []), name=f"process-only({base.name})",
                         kind="process_only", source_targets=list(base.source_targets), family_keys=list(base.family_keys),
                         certificate_families=list(base.certificate_families), lesson_ids=list(base.lesson_ids), artifact_ids=[],
                         attempt_ids=list(base.attempt_ids), content_terms=list(base.content_terms),
                         note="process-level lessons of the source targets with digits and content terms redacted")


def unrelated_control(ledger: Ledger, archive: Archive, bank: LessonBank, target_id: str, *, name: str | None = None) -> ExperienceSet:
    """The unrelated-domain control: every archived method and lesson from targets under a different certificate than
    the target's. Empty when the ledger holds no other family yet (the cell then measures an empty notebook)."""
    t = get_target(target_id).transfer
    sources = []
    for m in ledger.rows("provenance", order="created"):
        try:
            b = get_target(m["target"]).transfer
        except KeyError:
            continue
        if b.family != t.family and m["target"] not in sources:
            sources.append(m["target"])
    for l in bank.all():
        if l.retired:
            continue
        for lt in l.targets:
            try:
                b = get_target(lt).transfer
            except KeyError:
                continue
            if b.family != t.family and lt not in sources:
                sources.append(lt)
    if not sources:
        return ExperienceSet(id=_set_id("unrelated", [], [], []), name=name or "unrelated(empty)", kind="unrelated", source_targets=[],
                             family_keys=[], certificate_families=[], note="no other family in the ledger yet: the plain prompt plus the empty-notebook sentence (a framing control)")
    base = experience_from_archive(ledger, archive, bank, sources, name=name or f"unrelated({'+'.join(sources)})")
    base.id = _set_id("unrelated", sources, base.lesson_ids, base.artifact_ids)
    base.kind = "unrelated"
    base.note = "experience from targets under a different certificate than the target's"
    return base


def check_transfer(experience: ExperienceSet, target_id: str, carrier: str, *, allow_same_family: bool = False) -> dict:
    """The setting of a cell, and the rule that lessons are never measured on the family they came from. The two
    target-side controls carry no experience, so the rule has nothing to refuse: they are reported as controls with a flag."""
    if carrier in CONTROL_CARRIERS:
        t = get_target(target_id).transfer
        flag = "diagnosis of the target's own baseline, no remedy" if carrier == "diagnosis_only" else "one dry run and a revision, no lesson"
        return {"setting": "control", "same_family": False, "open": t.open, "target": t.to_dict(), "allowed": True, "flag": flag}
    if carrier == "none" or experience is None or not experience.source_targets:
        # the unrelated control with no other family in the archive runs as the plain prompt plus the empty-notebook
        # sentence: a framing control, reported as such rather than as unrelated experience
        framing = carrier == "unrelated" and experience is not None and not experience.source_targets
        setting = {"setting": "baseline" if carrier == "none" else ("framing" if framing else "no_sources"), "same_family": False,
                   "open": get_target(target_id).transfer.open, "target": get_target(target_id).transfer.to_dict()}
        flag = "no other family in the archive: the plain prompt plus the empty-notebook sentence (a framing control)" if framing else None
        return {**setting, "allowed": True, "flag": flag}
    st = transfer_setting(experience.source_targets, target_id)
    if carrier in CONTENT_CARRIERS and experience.kind != "unrelated":
        if st["setting"] == "same_instances":
            raise TransferLeak(f"{carrier} from {experience.name} on {target_id}: the target is one of the set's own instances; "
                               f"lessons are never measured on the family they came from")
        if st["same_family"] and not allow_same_family:
            raise TransferLeak(f"{carrier} from {experience.name} on {target_id}: same family {st['target_family_key']} "
                               f"({st['setting']}); pass allow_same_family to run it as an unreported cell")
    flag = "same family (unreported)" if (st["same_family"] and carrier in CONTENT_CARRIERS) else None
    return {**st, "allowed": True, "flag": flag}


def run_cell(*, experience: ExperienceSet | None, carrier: str, target_id: str, seeds: int, attempts: int, cfg: dict, ledger: Ledger,
             bank: LessonBank, archive: Archive, harness: LessonHarness | None = None, matrix_id: str, budget_usd: float | None = None,
             baseline_dollars: dict | None = None, allow_same_family: bool = False, token_budget: int | None = None, run_kw: dict | None = None,
             quiet: bool = True, seed0: int = 0, seed_stride: int = 1000, tool_calls: int = 0, diagnosis_table: dict | None = None) -> dict:
    """One cell: `seeds` groups of `attempts` with the carrier applied from the experience set, each group a seed block
    (seed0 + k * seed_stride), stopped at the matched budget when one is given. Every attempt row carries the matrix id,
    the cell id, the experience, the carrier, the setting and the seed index. `tool_calls` gives the builder that many
    certifier dry runs (the second-try control fixes it at one; a content carrier with dry runs is a cell of its own);
    `diagnosis_table` is the firing table the diagnosis-only control is written from."""
    target = get_target(target_id)
    setting = check_transfer(experience, target_id, carrier, allow_same_family=allow_same_family)
    token_budget = int(token_budget or cfg.get("harness", {}).get("lesson_token_budget", 2000))
    app = apply_carrier(carrier, target, experience, bank=bank, archive=archive, harness=harness, token_budget=token_budget,
                        diagnosis_table=diagnosis_table)
    tool_calls = max(int(tool_calls or 0), int(app.get("tool_calls", 0)))
    key = f"{matrix_id}|{experience.id if experience else 'none'}|{carrier}|{target_id}" + (f"|t{tool_calls}" if tool_calls else "")
    cell_id = "cell_" + hashlib.sha256(key.encode()).hexdigest()[:12]
    kw = dict(run_kw or {})
    groups = []
    for k in range(seeds):
        seed = seed0 + k * seed_stride
        stop = None
        if budget_usd is not None:
            stop = budget_usd
        elif baseline_dollars and k in baseline_dollars:
            stop = baseline_dollars[k]
        ctx = {**app["context"], "matrix_id": matrix_id, "cell_id": cell_id, "experience_kind": experience.kind if experience else None,
               "source_targets": experience.source_targets if experience else [], "setting": setting["setting"], "same_family": setting.get("same_family", False),
               "seed_index": k, "open_target": setting.get("open", False), "transfer_flag": setting.get("flag")}
        s = run_group(target_id=target_id, attempts=attempts, seed=seed, provider_name=kw.pop("provider_name", "local"), role_name="builder",
                      ledger=ledger, ledger_path=ledger.path, cfg=cfg, quiet=quiet, stop_at_dollars=stop, prompt=app["prompt"] if app["prompt_override"] else None,
                      context_extra=ctx, post_check=app["post_check"], tool_calls=tool_calls, **kw)
        groups.append({"seed_index": k, "seed": seed, "group_id": s["group_id"], "attempts": s["attempts"], "passed": s["passed"], "optimal": s["optimal"],
                       "pass_rate": s["pass_rate"], "optimal_rate": s["optimal_rate"], "best_score": s["best_score"], "dollars": s["dollars"],
                       "stopped_at": s.get("stopped_at"), "gate_checks": s.get("gate_checks"),
                       "revised": s.get("revised", 0), "revised_passed": s.get("revised_passed", 0)})
    return {"cell_id": cell_id, "matrix_id": matrix_id, "experience": experience.to_dict() if experience else None, "carrier": carrier,
            "target": target_id, "setting": setting, "lesson_ids": app["context"].get("lesson_ids", []), "skill_ids": app["context"].get("skill_ids", []),
            "redacted": app["context"].get("redacted", False), "tool_calls": tool_calls, "detectors_named": app["context"].get("detectors_named"),
            "groups": groups, "dollars": sum(g["dollars"] for g in groups), "revised": sum(g["revised"] for g in groups),
            "revised_passed": sum(g["revised_passed"] for g in groups),
            "gate": ({"lesson_id": app["gate"]["lesson"].id if app["gate"].get("lesson") else None, "check": app["gate"].get("check")} if app.get("gate") else None)}


def diagnosis_for(ledger: Ledger, target_id: str, none_cell: dict | None, *, seed: int = 0) -> tuple[dict, dict]:
    """The firing table the diagnosis-only row is written from: the built-in detectors are mined on the target's `none`
    cell of this matrix (or, without one, on the largest base group the ledger holds), then every detector the program
    has kept for the target, the loop's included, is run over those attempts. Returns the table and a record of it."""
    target = get_target(target_id)
    if none_cell is not None:
        rows = [a for g in none_cell["groups"] for a in ledger.attempts(group_id=g["group_id"])]
        source = {"from": "none cell", "cell_id": none_cell["cell_id"]}
    else:
        group = largest_base_group(base_groups(ledger, target_id))
        rows = group[1] if group else []
        source = {"from": "ledger base group", "group_id": group[0] if group else None}
    if rows:
        mine_builtin(ledger, target, rows, seed=seed)
    kept = kept_detectors(ledger, target_id)
    table = firing_table(kept, rows)
    rec = {**source, "attempts": table["n"], "failed": table["failed"], "kept": [d["name"] for d in kept],
           "firing": [{k: d[k] for k in ("name", "origin", "fires", "failures_fired", "rate")} for d in table["detectors"]]}
    return table, rec


def run_matrix(*, sources: list[ExperienceSet], carriers: list[str], targets: list[str], seeds: int, attempts: int, cfg: dict, ledger: Ledger,
               bank: LessonBank, archive: Archive, harness: LessonHarness | None = None, match: str = "attempts", budget_usd: float | None = None,
               allow_same_family: bool = False, token_budget: int | None = None, run_kw: dict | None = None, quiet: bool = True,
               out_dir: str | Path | None = None, matrix_id: str | None = None, controls: bool = True, seed0: int = 0, tool_calls: int = 0) -> dict:
    """Rows: the experience sets (and, with controls, the process-only and unrelated controls, and the two target-side
    controls of week 5); columns: the targets in their settings; the `none` carrier once per target. `tool_calls` gives
    every content carrier that many certifier dry runs (a lessons-plus-second-try cell). Cells that would leak are
    skipped and listed."""
    matrix_id = matrix_id or new_id("mtx")
    for c in carriers:
        if c not in CARRIERS:
            raise ValueError(f"unknown carrier {c!r}")
    rows: list[tuple[ExperienceSet | None, str]] = []
    for e in sources:
        for c in carriers:
            if c in ("none", "process_only", "unrelated") or c in CONTROL_CARRIERS:
                continue
            rows.append((e, c))
        if controls and "process_only" in carriers:
            rows.append((process_only_control(e), "process_only"))
    record: dict = {"id": matrix_id, "sources": [e.to_dict() for e in sources], "carriers": carriers, "targets": targets, "seeds": seeds,
                    "attempts": attempts, "match": match, "budget_usd": budget_usd, "tool_calls": tool_calls, "cells": [], "skipped": [],
                    "baselines": {}, "diagnoses": {}, "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for tid in targets:
        base_dollars = None
        none_cell = None
        if "none" in carriers:
            none_cell = run_cell(experience=None, carrier="none", target_id=tid, seeds=seeds, attempts=attempts, cfg=cfg, ledger=ledger, bank=bank,
                                 archive=archive, harness=harness, matrix_id=matrix_id, budget_usd=budget_usd, token_budget=token_budget, run_kw=run_kw,
                                 quiet=quiet, seed0=seed0)
            record["cells"].append(none_cell)
            record["baselines"][tid] = none_cell["cell_id"]
            if match == "dollars" and budget_usd is None:
                base_dollars = {g["seed_index"]: g["dollars"] for g in none_cell["groups"]}
        # the target-side controls (week 5): no experience, never refused
        if "second_try" in carriers:
            record["cells"].append(run_cell(experience=None, carrier="second_try", target_id=tid, seeds=seeds, attempts=attempts, cfg=cfg, ledger=ledger,
                                            bank=bank, archive=archive, harness=harness, matrix_id=matrix_id, budget_usd=budget_usd,
                                            baseline_dollars=base_dollars, token_budget=token_budget, run_kw=run_kw, quiet=quiet, seed0=seed0))
        if "diagnosis_only" in carriers:
            table, diag = diagnosis_for(ledger, tid, none_cell, seed=seed0)
            record["diagnoses"][tid] = diag
            record["cells"].append(run_cell(experience=None, carrier="diagnosis_only", target_id=tid, seeds=seeds, attempts=attempts, cfg=cfg,
                                            ledger=ledger, bank=bank, archive=archive, harness=harness, matrix_id=matrix_id, budget_usd=budget_usd,
                                            baseline_dollars=base_dollars, token_budget=token_budget, run_kw=run_kw, quiet=quiet, seed0=seed0,
                                            diagnosis_table=table))
        trows = list(rows)
        if controls and "unrelated" in carriers:
            trows.append((unrelated_control(ledger, archive, bank, tid), "unrelated"))
        for e, c in trows:
            try:
                cell = run_cell(experience=e, carrier=c, target_id=tid, seeds=seeds, attempts=attempts, cfg=cfg, ledger=ledger, bank=bank, archive=archive,
                                harness=harness, matrix_id=matrix_id, budget_usd=budget_usd, baseline_dollars=base_dollars, allow_same_family=allow_same_family,
                                token_budget=token_budget, run_kw=run_kw, quiet=quiet, seed0=seed0, tool_calls=tool_calls)
            except TransferLeak as err:
                record["skipped"].append({"experience": e.name, "carrier": c, "target": tid, "reason": str(err)})
                continue
            record["cells"].append(cell)
    record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["analysis"] = analyze(ledger, matrix_id)
    out_dir = Path(out_dir) if out_dir else Path(ledger.path).resolve().parent / "matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{matrix_id}.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    (out_dir / f"{matrix_id}.md").write_text(matrix_markdown(record["analysis"], record), encoding="utf-8")
    record["files"] = {"json": str(out_dir / f"{matrix_id}.json"), "markdown": str(out_dir / f"{matrix_id}.md")}
    if not quiet:
        print(matrix_markdown(record["analysis"], record))
    return record


# --- analysis: the ledger query and the bootstrap -------------------------------------------------------------------------
CELL_SQL = """
    SELECT id, target, seed, group_id, dollars, certificate,
           json_extract(context, '$.cell_id') AS cell_id, json_extract(context, '$.carrier') AS carrier,
           json_extract(context, '$.experience_id') AS experience_id, json_extract(context, '$.experience_kind') AS experience_kind,
           json_extract(context, '$.source_targets') AS source_targets, json_extract(context, '$.setting') AS setting,
           json_extract(context, '$.same_family') AS same_family, json_extract(context, '$.seed_index') AS seed_index,
           json_extract(context, '$.open_target') AS open_target, json_extract(context, '$.transfer_flag') AS transfer_flag,
           json_extract(context, '$.tool_calls') AS tool_calls
    FROM attempts WHERE json_extract(context, '$.matrix_id') = ? AND role = 'builder' ORDER BY timestamp, id
"""


def _stats(rows: list[dict]) -> dict:
    n = len(rows)
    passed = [r for r in rows if r["_passed"]]
    optimal = [r for r in passed if r["_optimal"]]
    scores = [r["_score"] for r in passed if r["_score"] is not None]
    lower = all(r["_lower"] for r in passed) if passed else True
    best = (min(scores) if lower else max(scores)) if scores else None
    return {"attempts": n, "passed": len(passed), "optimal": len(optimal), "pass_rate": len(passed) / n if n else float("nan"),
            "optimal_rate": len(optimal) / n if n else float("nan"), "best_score": best, "lower_is_better": lower,
            "dollars": sum(r["dollars"] or 0.0 for r in rows)}


def bootstrap_interval(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float] | None:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n_boot))
    lo, hi = means[int(alpha / 2 * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def bootstrap_gain_over_attempts(with_rows: list[dict], without_rows: list[dict], key: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """One seed: resample attempts within each group and read the gain in the rate."""
    if not with_rows or not without_rows:
        return None
    rng = random.Random(seed)
    a = [1.0 if r[key] else 0.0 for r in with_rows]
    b = [1.0 if r[key] else 0.0 for r in without_rows]
    gains = sorted((sum(rng.choice(a) for _ in a) / len(a)) - (sum(rng.choice(b) for _ in b) / len(b)) for _ in range(n_boot))
    return (gains[int(0.025 * n_boot)], gains[min(n_boot - 1, int(0.975 * n_boot))])


def cell_symptoms(ledger: Ledger, target: str, group_ids: list[str], kept: list[dict] | None = None) -> dict:
    """The symptom rates of a cell: how often each detector the program kept for the target fires over the cell's attempts."""
    kept = kept_detectors(ledger, target) if kept is None else kept
    if not kept:
        return {}
    rows = [a for g in group_ids for a in ledger.attempts(group_id=g)]
    return symptom_rates(kept, rows)


def analyze(ledger: Ledger, matrix_id: str, n_boot: int = 2000) -> dict:
    """Certified gain per cell from the ledger: the carrier's cell minus the target's `none` cell, paired by seed index,
    mean over seeds with a percentile bootstrap interval (over attempts when there is a single seed); and, per cell, the
    symptom rates of the target's kept detectors, with the change against the baseline inside every gain."""
    rows = ledger.query(CELL_SQL, (matrix_id,))
    for r in rows:
        cert = json.loads(r["certificate"]) if isinstance(r["certificate"], str) and r["certificate"] else {}
        r["_passed"], r["_optimal"], r["_score"], r["_lower"] = bool(cert.get("passed")), cert.get("optimal") is True, cert.get("score"), cert.get("lower_is_better", True)
        r["seed_index"] = int(r["seed_index"]) if r["seed_index"] is not None else 0
    cells: dict[str, list[dict]] = {}
    for r in rows:
        cells.setdefault(r["cell_id"], []).append(r)
    baselines: dict[str, str] = {}
    for cid, rs in cells.items():
        if rs[0]["carrier"] == "none":
            baselines[rs[0]["target"]] = cid
    kept_by_target = {t: kept_detectors(ledger, t) for t in {r["target"] for r in rows}}
    symptoms_by_cell = {cid: cell_symptoms(ledger, rs[0]["target"], sorted({r["group_id"] for r in rs}), kept_by_target[rs[0]["target"]])
                        for cid, rs in cells.items()}
    out = {"matrix_id": matrix_id, "cells": [], "n_rows": len(rows), "detectors": {t: [d["name"] for d in k] for t, k in kept_by_target.items()}}
    for cid, rs in cells.items():
        first = rs[0]
        by_seed = {}
        for r in rs:
            by_seed.setdefault(r["seed_index"], []).append(r)
        seeds = sorted(by_seed)
        per_seed = {k: _stats(v) for k, v in by_seed.items()}
        entry = {"cell_id": cid, "target": first["target"], "carrier": first["carrier"], "experience_id": first["experience_id"],
                 "experience_kind": first["experience_kind"], "source_targets": json.loads(first["source_targets"]) if first["source_targets"] else [],
                 "setting": first["setting"], "same_family": bool(first["same_family"]), "open": bool(first["open_target"]), "flag": first["transfer_flag"],
                 "tool_calls": int(first["tool_calls"] or 0), "seeds": len(seeds), "attempts": len(rs), "dollars": sum(r["dollars"] or 0.0 for r in rs),
                 "pass_rate": sum(s["pass_rate"] for s in per_seed.values()) / len(per_seed), "optimal_rate": sum(s["optimal_rate"] for s in per_seed.values()) / len(per_seed),
                 "best_score": None, "per_seed": per_seed, "symptoms": symptoms_by_cell[cid], "gain": None}
        scores = [s["best_score"] for s in per_seed.values() if s["best_score"] is not None]
        lower = per_seed[seeds[0]]["lower_is_better"]
        entry["best_score"] = (min(scores) if lower else max(scores)) if scores else None
        base_id = baselines.get(first["target"])
        if first["carrier"] != "none" and base_id and base_id in cells:
            base_by_seed = {}
            for r in cells[base_id]:
                base_by_seed.setdefault(r["seed_index"], []).append(r)
            common = [k for k in seeds if k in base_by_seed]
            if common:
                gains_p = [per_seed[k]["pass_rate"] - _stats(base_by_seed[k])["pass_rate"] for k in common]
                gains_o = [per_seed[k]["optimal_rate"] - _stats(base_by_seed[k])["optimal_rate"] for k in common]
                if len(common) >= 2:
                    ci_p, ci_o = bootstrap_interval(gains_p, n_boot), bootstrap_interval(gains_o, n_boot)
                    method = f"percentile bootstrap over {len(common)} seeds"
                else:
                    k = common[0]
                    ci_p = bootstrap_gain_over_attempts(by_seed[k], base_by_seed[k], "_passed", n_boot)
                    ci_o = bootstrap_gain_over_attempts(by_seed[k], base_by_seed[k], "_optimal", n_boot)
                    method = "percentile bootstrap over attempts (one seed)"
                bb = [s for s in (_stats(base_by_seed[k])["best_score"] for k in common) if s is not None]
                base_best = (min(bb) if lower else max(bb)) if bb else None
                base_sym = symptoms_by_cell.get(base_id, {})
                names = sorted(set(entry["symptoms"]) | set(base_sym))
                entry["gain"] = {"pass_rate": sum(gains_p) / len(gains_p), "pass_rate_ci": ci_p, "optimal_rate": sum(gains_o) / len(gains_o),
                                 "optimal_rate_ci": ci_o, "best_score_gain": ((base_best - entry["best_score"]) if lower else (entry["best_score"] - base_best))
                                 if (base_best is not None and entry["best_score"] is not None) else None,
                                 "seeds": len(common), "method": method, "baseline_cell": base_id,
                                 "matched_dollars_spread": abs(entry["dollars"] - sum(r["dollars"] or 0.0 for r in cells[base_id])),
                                 "symptom_change": {n: ((entry["symptoms"].get(n) or {}).get("rate") or 0.0) - ((base_sym.get(n) or {}).get("rate") or 0.0)
                                                    for n in names}}
        out["cells"].append(entry)
    return out


def symptom_text(changes: dict | None, k: int = 3) -> str:
    """The largest symptom changes of a cell, as a short cell of the markdown table (negative is a symptom removed)."""
    if not changes:
        return "-"
    top = sorted(changes.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:k]
    return "; ".join(f"{n} {v:+.2f}" for n, v in top)


def matrix_markdown(analysis: dict, record: dict | None = None) -> str:
    cells = [c for c in analysis["cells"] if c["carrier"] != "none"]
    lines = [f"# Transfer matrix {analysis['matrix_id']}", ""]
    if record:
        lines += [f"Sources: {', '.join(s['name'] for s in record['sources'])}; carriers {', '.join(record['carriers'])}; targets {', '.join(record['targets'])}; "
                  f"{record['seeds']} seed(s) x {record['attempts']} attempts per cell; matched by {record['match']}"
                  + (f" at ${record['budget_usd']}" if record.get('budget_usd') else "") + ".", ""]
    lines += ["| experience (row) | carrier | target | setting | open | gain in pass rate [95% CI] | gain in optimal rate [95% CI] | best score gain | symptom change | seeds | $ | flag |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (c["experience_kind"] or "", ",".join(c["source_targets"]), c["carrier"], c["target"])):
        g = c.get("gain")
        if g:
            ci = g["pass_rate_ci"]
            cio = g["optimal_rate_ci"]
            gp = f"{g['pass_rate']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else f"{g['pass_rate']:+.3f}"
            go = f"{g['optimal_rate']:+.3f} [{cio[0]:+.3f}, {cio[1]:+.3f}]" if cio else f"{g['optimal_rate']:+.3f}"
            bs = f"{g['best_score_gain']:+g}" if g["best_score_gain"] is not None else "-"
            sy = symptom_text(g.get("symptom_change"))
        else:
            gp = go = bs = "no baseline"
            sy = "-"
        if c["setting"] == "framing":
            row = "framing: (the empty-notebook sentence)"
        else:
            row = f"{c['experience_kind'] or 'control'}: {'+'.join(c['source_targets']) or ('(target-side)' if c['carrier'] in CONTROL_CARRIERS else '(empty)')}"
        carrier = c["carrier"] + (" (+dry run)" if c.get("tool_calls") and c["carrier"] != "second_try" else "")
        lines.append(f"| {row} | {carrier} | {c['target']} | {c['setting']} | {'yes' if c['open'] else 'no'} | {gp} | {go} | {bs} | {sy} | {c['seeds']} | {c['dollars']:.4f} | {c['flag'] or ''} |")
    base = [c for c in analysis["cells"] if c["carrier"] == "none"]
    if base:
        lines += ["", "Baselines (carrier none): " + "; ".join(f"{c['target']}: pass rate {c['pass_rate']:.3f}, optimal rate {c['optimal_rate']:.3f}, best {c['best_score']}" for c in base)]
        sym = [f"{c['target']}: " + ", ".join(f"{n} {s['rate']:.2f}" for n, s in sorted(c["symptoms"].items())) for c in base if c.get("symptoms")]
        if sym:
            lines += ["", "Symptom rates of the baselines (how often each kept detector fires; a detector is a program verified on held-out attempts): "
                          + "; ".join(sym)]
    if record and record.get("skipped"):
        lines += ["", "Cells refused by the program (lessons are never measured on the family they came from):"]
        lines += [f"- {s['experience']} / {s['carrier']} on {s['target']}: {s['reason']}" for s in record["skipped"]]
    lines += ["", "A cell is the certified gain with the experience minus without, paired by seed; the interval is a percentile bootstrap. "
                  "The symptom change is the firing rate of a kept detector with the carrier minus without (negative: the failure it names went away). "
                  "The two target-side controls carry no experience: diagnosis_only names the failure and no remedy, second_try is one dry run and no lesson. A framing row is the unrelated control with no other family in the archive: the plain prompt plus the sentence that the notebook holds no lessons yet, which measures the framing alone. "
                  "Every number is a certificate or a detector's firing; nothing a carrier adds to a prompt is scored."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lab.harness.matrix", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sources", nargs="*", default=[], help="source target ids of one experience set (repeat --sources-2 for a second)")
    ap.add_argument("--sources-2", nargs="*", default=None, help="a second experience set")
    ap.add_argument("--targets", nargs="+", default=[])
    ap.add_argument("--carriers", nargs="+", default=["none", "lessons", "process_only", "unrelated"], choices=list(CARRIERS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--attempts", type=int, default=16)
    ap.add_argument("--match", default="attempts", choices=["attempts", "dollars"])
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--allow-same-family", action="store_true", help="run same-family cells anyway (they are marked and unreported)")
    ap.add_argument("--tool-calls", type=int, default=0, help="certifier dry runs for every content carrier (a lessons-plus-second-try cell); the second_try control always has one")
    ap.add_argument("--provider", default="local", choices=list(KINDS))
    ap.add_argument("--mode", default=None, choices=list(MODES))
    ap.add_argument("--harness-provider", default=None)
    ap.add_argument("--harness-mode", default=None, choices=list(MODES))
    ap.add_argument("--no-harness", action="store_true")
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--units-per-hour", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--thinking", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--analyze", default=None, help="matrix id: recompute the analysis from the ledger and print it")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    cfg = lab_config()
    ledger_path = args.ledger or cfg["storage"]["ledger"]
    ledger = Ledger(ledger_path)
    if args.analyze:
        an = analyze(ledger, args.analyze)
        print(json.dumps(an, indent=1, default=str) if args.json else matrix_markdown(an))
        ledger.close()
        return 0
    root = Path(ledger_path).resolve().parent
    archive, bank = Archive(ledger, root), LessonBank(ledger)
    harness = None
    if not args.no_harness:
        hp = harness_provider(cfg, name=args.harness_provider, mode=args.harness_mode, ledger=ledger, fixtures_root=args.fixtures)
        harness = LessonHarness(hp, ledger, bank, cfg=cfg)
    sources = []
    if args.sources:
        sources.append(experience_from_archive(ledger, archive, bank, args.sources))
    if args.sources_2:
        sources.append(experience_from_archive(ledger, archive, bank, args.sources_2))
    run_kw = dict(provider_name=args.provider, mode=args.mode, gpu=args.gpu, precision=args.precision, units_per_hour=args.units_per_hour,
                  temperature=None, max_tokens=args.max_tokens, timeout_s=args.timeout, fixtures_root=args.fixtures, cache_dir=None,
                  concurrency=args.concurrency, thinking_budget=args.thinking)
    rec = run_matrix(sources=sources, carriers=args.carriers, targets=args.targets, seeds=args.seeds, attempts=args.attempts, cfg=cfg, ledger=ledger,
                     bank=bank, archive=archive, harness=harness, match=args.match, budget_usd=args.budget_usd, allow_same_family=args.allow_same_family,
                     token_budget=args.token_budget, run_kw=run_kw, quiet=args.json, out_dir=args.out, tool_calls=args.tool_calls)
    ledger.close()
    if args.json:
        print(json.dumps(rec, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
