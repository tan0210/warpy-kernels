import argparse
import importlib
import json
import pathlib
import re
import sys
import time
import traceback

import torch

import tests
from tests.fixtures import Ctx
from tests.registry import BENCHES, CASES, Skip


def _matcher(expr):
    if not expr:
        return lambda _id: True
    code = re.sub(r"[^\s()]+", lambda m: m.group(0) if m.group(0) in ("and", "or", "not") else f"({m.group(0)!r} in _id)", expr)
    return lambda _id: bool(eval(code, {"__builtins__": {}}, {"_id": _id}))


def _fmt(name, value, tol):
    text = f"{value:.3g}" if tol is None else f"{value:.2e}"
    return f"{name} {text}"


def _run(entry):
    ctx = Ctx()
    start = time.perf_counter()
    status, detail = "PASS", ""
    try:
        entry.fn(ctx, **entry.params)
        torch.cuda.synchronize()
        if ctx.failures:
            status, detail = "FAIL", "\n".join(ctx.failures)
    except Skip as e:
        status, detail = "SKIP", str(e)
    except Exception:  # noqa: BLE001  # any test error is reported, not raised
        status, detail = "ERROR", traceback.format_exc()
    return ctx, status, detail, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(prog="python -m tests")
    parser.add_argument("-k", default="", help="select tests by substring; combine with and/or/not")
    parser.add_argument("-x", action="store_true", help="stop at the first failure")
    parser.add_argument("--list", action="store_true", help="list the selected tests and exit")
    parser.add_argument("--bench", action="store_true", help="run the benchmarks instead of the tests")
    parser.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    args = parser.parse_args()

    for root in map(pathlib.Path, tests.__path__):
        for path in sorted(root.rglob("test_*.py")):  # includes subfolders
            importlib.import_module(".".join(("tests", *path.relative_to(root).with_suffix("").parts)))
    match = _matcher(args.k)
    selected = [e for e in (BENCHES if args.bench else CASES) if match(e.id)]
    if args.list:
        print("\n".join(e.id for e in selected))
        return 0

    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    print(f"collected {len(selected)} {'benchmarks' if args.bench else 'tests'}\n")
    width = max((len(e.id) for e in selected), default=0)
    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIP": 0}
    problems = []
    records = []
    t0 = time.perf_counter()
    for entry in selected:
        ctx, status, detail, secs = _run(entry)
        counts[status] += 1
        records.append({"id": entry.id, "name": entry.name, "params": entry.params, "status": status, "seconds": secs,
                            "metrics": {n: v for n, v, _ in ctx.metrics}, "data": ctx.data})
        metrics = "  ".join(_fmt(*m) for m in ctx.metrics)
        first = detail.splitlines()[-1] if status == "ERROR" else detail.splitlines()[0] if detail else ""
        print(f"{status:<5} {entry.id:<{width}}  {metrics if status in ('PASS', 'FAIL') else first}  ({secs:.1f}s)", flush=True)
        for note in ctx.notes:
            print(f"      {note}")
        if status in ("FAIL", "ERROR"):
            problems.append((entry.id, status, detail))
            if args.x:
                break
    for ident, status, detail in problems:
        print(f"\n{'=' * 20} {status}: {ident} {'=' * 20}\n{detail}")
    if args.json:
        meta = {"device": torch.cuda.get_device_name(), "torch": torch.__version__}
        with open(args.json, "w") as f:
            json.dump(dict(meta, results=records), f, indent=1, default=str)
        print(f"\nwrote {args.json}")
    summary = ", ".join(f"{n} {k.lower()}" for k, n in counts.items() if n)
    print(f"\n{'=' * 10} {summary} in {time.perf_counter() - t0:.1f}s {'=' * 10}")
    return 1 if problems else 0


sys.exit(main())
