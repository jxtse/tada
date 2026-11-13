"""Opt-in probes and bounded, work-preserving repairs; no implicit paid calls."""
from __future__ import annotations

import copy
from pathlib import Path
from .clients import ModelClient

DEFECT_CLASSES = {"missing_information", "contradiction", "ambiguous_scope", "answer_leakage",
                  "unsupported_label", "shortcut", "invalid_deliverable", "other"}


def spec_probe(instruction: str, workspace: dict[str, str], client: ModelClient) -> dict:
    response = client.complete("spec_probe", {
        "instruction": instruction, "workspace": workspace,
        "request": "Report missing information, contradictions and unresolvable decisions. Classify blocker/non-blocker and defect_class.",
    })
    findings = response.get("findings")
    if not isinstance(findings, list):
        raise ValueError("Probe must return findings")
    for item in findings:
        if item.get("severity") not in {"blocker", "non-blocker"}:
            raise ValueError("Invalid probe severity")
        if item.get("defect_class", "other") not in DEFECT_CLASSES:
            raise ValueError("Invalid defect class")
    return {"findings": findings, "pass": not any(x["severity"] == "blocker" for x in findings),
            "model": client.model_id, "fake": getattr(client, "fake", False)}


def difficulty_decision(score: float, keyword: float) -> str:
    if not 0 <= score <= 1 or not 0 <= keyword <= 1:
        raise ValueError("Scores must be in [0, 1]")
    if score >= 0.90:
        return "deepen"
    if score <= keyword + 0.1:
        return "repair"
    return "continue"


def difficulty_probe(task: Path, jobs: Path, keyword: float, runner) -> dict:
    """Runner is explicitly injected. Its implementation may invoke run_trial.sh."""
    result = runner(task, jobs, "difficulty-probe", "deepseek-flash")
    if result.get("valid") is not True:
        return {"action": "blocked", "reason": "No valid difficulty result", "trial": result}
    return {"action": difficulty_decision(result["score"], keyword), "trial": result}


def trial_command(task: Path, jobs: Path, name: str, model: str) -> list[str]:
    repo = Path(__file__).resolve().parents[2]
    return ["sh", str(repo / "solver/run_trial_remote.sh"), str(task), str(jobs), name, model]


def structure(spec: dict) -> dict:
    """Only content/parameters may change; all work/graph/output structure stays."""
    frozen = copy.deepcopy(spec)
    frozen.pop("parameters", None)
    frozen.pop("instruction", None)
    frozen.pop("instruction_text", None)
    frozen.pop("design", None)
    for doc in frozen.get("documents", []):
        doc.pop("text", None)
        # Values may change, but no fact may disappear or change identity.
        if isinstance(doc.get("facts"), dict):
            doc["facts"] = sorted(doc["facts"])
    for node in frozen.get("dependency_graph", {}).get("nodes", []):
        node.pop("value", None)
    return frozen


def preserve_work(before: dict, after: dict) -> None:
    if structure(before) != structure(after):
        raise ValueError("Repair changed structure, work items, scored cells or semantic steps")


def repair_loop(spec: dict, inspect, repair, max_rounds: int = 3) -> dict:
    if not 0 <= max_rounds <= 3:
        raise ValueError("At most three repair rounds")
    current = copy.deepcopy(spec)
    history = []
    for round_index in range(max_rounds + 1):
        report = inspect(current)
        blockers = [f for f in report.get("findings", []) if f.get("severity") == "blocker"]
        history.append({"round": round_index, "findings": report.get("findings", [])})
        if not blockers:
            return {"status": "passed", "spec": current, "rounds": round_index, "history": history}
        if round_index == max_rounds:
            return {"status": "dropped", "spec": current, "rounds": round_index, "history": history,
                    "defect_classes": sorted({x.get("defect_class", "other") for x in blockers})}
        candidate = repair(copy.deepcopy(current), blockers)
        preserve_work(current, candidate)
        current = candidate
