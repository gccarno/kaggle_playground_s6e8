#!/usr/bin/env python3
"""Run s6e8-model.ipynb LOCALLY with a config override, archive its artifacts,
optionally submit the resulting file, and append a row to experiments/runs.csv.

The local loop is ~5 minutes against ~15 for a kernel push/queue/run/collect cycle,
and a locally-produced submission.csv can be submitted directly -- so probes are
screened here and only champions get pushed to Kaggle for the reproducible record.

Usage:
    python scripts/run_local.py --cfg '{"run_tag":"B1","fe_interaction":true}' \
        --description "B1: interaction ratios" --notes "HYPOTHESIS... GATE..." --submit

    # strict-twin check: refuse to run unless exactly one CFG field differs
    python scripts/run_local.py --cfg '{...}' --diff-vs '{"run_tag":"champion"}' ...

Rows land in the same experiments/runs.csv as kernel runs, with kernel_ref="local".
"""
import argparse, importlib.util, json, os, shutil, subprocess, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "s6e8-model.ipynb"

# Reuse collect_run.py's log plumbing rather than reimplementing the schema logic.
_spec = importlib.util.spec_from_file_location("collect_run", Path(__file__).with_name("collect_run.py"))
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

DEFAULTS = {
    "run_tag": "champion", "learner": "lgb", "model_seed": 42,
    "fe_interaction": False, "fe_composition": False, "fe_normalization": False,
    "drop_flat_cats": False, "lgb_params": {}, "n_estimators": 8000, "early_stop": 100,
}


def git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_ROOT)
    return out.stdout.strip() if out.returncode == 0 else ""


def check_strict_twin(cfg, baseline):
    """KAGGLE_PLAYBOOK.md section 3: a probe changes exactly ONE thing. A probe that
    moved two knobs cannot attribute its delta to either, so it is not evidence --
    refuse to run it rather than produce an uninterpretable row."""
    a, b = {**DEFAULTS, **baseline}, {**DEFAULTS, **cfg}
    diff = [k for k in a if a[k] != b[k] and k != "run_tag"]
    if len(diff) != 1:
        raise SystemExit(
            f"NOT A STRICT TWIN: {len(diff)} fields differ from the baseline: {diff}\n"
            f"  baseline: { {k: a[k] for k in diff} }\n"
            f"  probe   : { {k: b[k] for k in diff} }\n"
            "Change exactly one thing, or drop --diff-vs if this is deliberate.")
    print(f"strict-twin OK: only {diff[0]!r} differs "
          f"({a[diff[0]]!r} -> {b[diff[0]]!r})")


def execute_notebook(cfg, out_dir):
    """Execute the notebook in-process via nbclient, streaming stdout as it goes."""
    import nbformat
    from nbclient import NotebookClient

    env = dict(os.environ, S6E8_CFG=json.dumps(cfg), S6E8_OUT=str(out_dir))
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(nb, timeout=7200, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO_ROOT)}})
    # nbclient inherits the parent env, so set it here for the duration of the run.
    old = {k: os.environ.get(k) for k in ("S6E8_CFG", "S6E8_OUT")}
    os.environ.update(S6E8_CFG=env["S6E8_CFG"], S6E8_OUT=env["S6E8_OUT"])
    try:
        client.execute()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    text = "".join(
        "".join(o.get("text", "") for o in c.get("outputs", []) if o.get("output_type") == "stream")
        for c in nb.cells if c.cell_type == "code")
    print(text)
    for c in nb.cells:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                raise RuntimeError(f"notebook cell failed: {o['ename']}: {o['evalue']}\n"
                                   + "\n".join(o.get("traceback", [])[-8:]))
    marker = "RUN_METRICS_JSON:"
    line = next((l for l in text.splitlines() if l.startswith(marker)), None)
    if line is None:
        raise RuntimeError("RUN_METRICS_JSON not found in notebook output")
    return json.loads(line[len(marker):])


def submit(sub_file, message, competition="playground-series-s6e8"):
    out = subprocess.run(["kaggle", "competitions", "submit", competition,
                          "-f", str(sub_file), "-m", message],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    print(out.stdout, out.stderr, file=sys.stderr if out.returncode else sys.stdout)
    if out.returncode != 0:
        raise RuntimeError(f"submit failed: {out.stderr}")


def poll_score(competition="playground-series-s6e8", timeout_min=20, poll_interval=30):
    import csv, time
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        out = subprocess.run(["kaggle", "competitions", "submissions", competition, "--csv"],
                             capture_output=True, text=True, cwd=REPO_ROOT)
        rows = list(csv.DictReader(out.stdout.splitlines()))
        if rows:
            s = rows[0].get("publicScore") or rows[0].get("public_score")
            if s not in (None, "", "None", "pending"):
                return s
        time.sleep(poll_interval)
    print("  WARNING: timed out waiting for a public score")
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", default="{}", help="JSON CFG override for the notebook")
    ap.add_argument("--description", required=True)
    ap.add_argument("--notes", default="", help="hypothesis / gate / mechanism -- write the gate BEFORE the result")
    ap.add_argument("--diff-vs", default=None, help="JSON baseline CFG; enforces a strict-twin (exactly one field differs)")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="run without appending to runs.csv")
    args = ap.parse_args()

    cfg = {**DEFAULTS, **json.loads(args.cfg)}
    if args.diff_vs is not None:
        check_strict_twin(cfg, json.loads(args.diff_vs))

    run_id = uuid.uuid4().hex[:8]
    out_dir = REPO_ROOT / "experiments" / "preds" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id}  cfg={json.dumps(cfg)}\n")

    metrics = execute_notebook(cfg, out_dir)
    print(f"\n{'='*70}\nOOF AUC = {metrics['final_oof_auc']}   ({metrics['n_features']} features)")

    score = ""
    if args.submit:
        submit(out_dir / "submission.csv", args.description)
        score = poll_score()
        print(f"Public LB = {score}")

    if not args.no_log:
        row = cr.build_row(
            metrics,
            run_id=run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=git("rev-parse", "HEAD"),
            git_dirty=bool(git("status", "--porcelain")),
            kernel_ref="local",
            kernel_version="",
            description=args.description,
            public_lb_score=score,
            private_lb_score="",
            preds_dir=out_dir.relative_to(REPO_ROOT).as_posix(),
            notes=args.notes,
        )
        row["run_tag"] = metrics.get("run_tag", "")
        row["engineered"] = ",".join(metrics.get("engineered", []))
        cr.append_run_row(row)
        print(f"Appended run {run_id} to experiments/runs.csv")


if __name__ == "__main__":
    main()
