import argparse
import importlib
import json
import os
import pathlib
import re
import sys
import threading
import time
import traceback

import torch

import tests
from tests.fixtures import Ctx
from tests.registry import BENCHES, CASES, Skip
from warpy_kernels import device_arch


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
    except NotImplementedError as e:
        # SM120 supports everything, so a gap there is an error
        if device_arch() == 120:
            status, detail = "ERROR", traceback.format_exc()
        else:
            status, detail = "SKIP", str(e)
    except Exception:  # noqa: BLE001  # any test error is reported, not raised
        status, detail = "ERROR", traceback.format_exc()
    return ctx, status, detail, time.perf_counter() - start


def _regressions(ctx, base, slack):
    return [
        f"{name} = {value:.3e} regressed from baseline {base[name]:.3e}"
        for name, value, _ in ctx.metrics
        if name in ctx.gated and name in base and not value <= base[name] * (1 + slack) + 1e-9
    ]


def _write_json(path, records):
    meta = {"device": torch.cuda.get_device_name(), "torch": torch.__version__}
    with open(path, "w") as f:
        json.dump(dict(meta, results=records), f, indent=1, default=str)


def main():
    parser = argparse.ArgumentParser(prog="python -m tests")
    parser.add_argument("-k", default="", help="select tests by substring; combine with and/or/not")
    parser.add_argument("-x", action="store_true", help="stop at the first failure")
    parser.add_argument("--list", action="store_true", help="list the selected tests and exit")
    parser.add_argument("--bench", action="store_true", help="run the benchmarks instead of the tests")
    parser.add_argument("--json", metavar="PATH", help="also write the results as JSON")
    parser.add_argument("--baseline", metavar="PATH", help="fail gated metrics that are worse than in this --json file")
    parser.add_argument("--slack", type=float, default=0.1, help="relative slack for --baseline (default 0.1)")
    parser.add_argument("--timeout", type=float, default=0, help="seconds per test before the run aborts (0 = off)")
    args = parser.parse_args()
    baseline = {}
    if args.baseline:
        with open(args.baseline) as f:
            baseline = {r["id"]: r["metrics"] for r in json.load(f)["results"]}

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
        watchdog = None
        if args.timeout > 0:

            def _abort(ident=entry.id):
                # a hung kernel can't be interrupted, so name the test and end the process
                print(f"TIMEOUT {ident} after {args.timeout:.0f}s", flush=True)
                if args.json:
                    _write_json(args.json, records)
                os._exit(3)

            watchdog = threading.Timer(args.timeout, _abort)
            watchdog.daemon = True
            watchdog.start()
        ctx, status, detail, secs = _run(entry)
        if watchdog is not None:
            watchdog.cancel()
        regressed = _regressions(ctx, baseline.get(entry.id, {}), args.slack)
        if regressed and status in ("PASS", "FAIL"):
            status = "FAIL"
            detail = "\n".join(filter(None, [detail, *regressed]))
        counts[status] += 1
        records.append({"id": entry.id, "name": entry.name, "params": entry.params, "status": status, "seconds": secs,
                            "metrics": {n: v for n, v, _ in ctx.metrics}, "gated": sorted(ctx.gated), "data": ctx.data})
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
        _write_json(args.json, records)
        print(f"\nwrote {args.json}")
    summary = ", ".join(f"{n} {k.lower()}" for k, n in counts.items() if n)
    print(f"\n{'=' * 10} {summary} in {time.perf_counter() - t0:.1f}s {'=' * 10}")
    return 1 if problems else 0


sys.exit(main())
