import argparse
import itertools
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASELINE_COLOR = "#8c8c8c"
COLORS = ["#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#f0e442"]
LABELS = {"n": "Sequence length", "nk": "Context length"}
X_PARAMS = ("n", "nk")
STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.dpi": 300,
}


def _key(value):
    return json.dumps(value, sort_keys=True)


def _unique(values):
    seen = {}
    for v in values:
        seen.setdefault(_key(v), v)
    return list(seen.values())


def _tick(value):
    if isinstance(value, int) and value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def _title(combo):
    parts = []
    for name, value in combo:
        if isinstance(value, bool):
            parts.append(name if value else f"not {name}")
        else:
            parts.append(f"{name}={value}")
    return ", ".join(parts)


def _panel(ax, rows, xs, x, series, colors, baseline, peaks, title):
    width = 0.8 / len(series)
    for i, label in enumerate(series):
        for pos, xv in enumerate(xs):
            hit = [r for r in rows if _key(r["params"][x]) == _key(xv)]
            if not hit or f"tf:{label}" not in hit[0]["metrics"]:
                continue
            value = hit[0]["metrics"][f"tf:{label}"]
            left = pos - 0.4 + width * (i + 0.5)
            ax.bar(left, value, width, color=colors[label], label=label if pos == 0 else None)
            ref = hit[0]["metrics"].get(f"tf:{baseline}")
            if label != baseline and ref:
                ax.text(left, value, f"{value / ref:.2f}x", ha="center", va="bottom", fontsize=7)
    for label, tf in peaks.items():
        ax.axhline(tf, color="black", linestyle=":", linewidth=1, label=f"{label} ({tf:g} TF)")
    ax.set_xticks(range(len(xs)), [_tick(v) for v in xs])
    ax.set_title(title)
    ax.set_ylabel("TFLOP/s")
    ax.set_xlabel(LABELS.get(x, x))


def draw(data, name, path, x=None, baseline=None, peaks=None):
    """Write a grouped bar figure of the ``tf:<series>`` metrics of the ``name`` results to ``path``.

    Every series found in the results becomes a bar, every other varying parameter becomes a panel, and each
    non-baseline bar is annotated with its ratio to the baseline series. A ``peaks`` mapping in a result's data
    draws a dotted line per entry.

    Args:
        data: Parsed JSON written by ``python -m tests --bench --json``.
        name: Benchmark name to plot.
        path: Output file; its extension (``.png``, ``.pdf``, ...) sets the format.
        x: Parameter on the x axis; the first of ``n`` and ``nk`` that the results have if ``None``.
        baseline: Series the ratios are computed against; the first series found if ``None``.
        peaks: Extra ``{label: TFLOP/s}`` lines to draw; they replace entries with the same label in the results.

    Returns:
        ``True`` if the figure was written, ``False`` if there are no results to plot for ``name``.
    """
    rows = [r for r in data["results"] if r["name"] == name and r["status"] == "PASS"]
    series = _unique(k[3:] for r in rows for k in r["metrics"] if k.startswith("tf:"))
    if not rows or not series:
        return False
    baseline = baseline or series[0]
    x = x or next(k for k in X_PARAMS if k in rows[0]["params"])
    colors, palette = {}, iter(COLORS)
    for label in series:
        colors[label] = BASELINE_COLOR if label == baseline else next(palette)
    xs = _unique(r["params"][x] for r in rows)
    if all(isinstance(v, (int, float)) for v in xs):
        xs.sort()
    varying = [k for k in rows[0]["params"] if k != x and len(_unique(r["params"][k] for r in rows)) > 1]
    values = [_unique(r["params"][k] for r in rows) for k in varying]
    combos = list(itertools.product(*values))
    if len(varying) == 2:
        nrows, ncols = len(values[0]), len(values[1])
    else:
        ncols = min(2, len(combos))
        nrows = math.ceil(len(combos) / ncols)
    lines = {}
    for r in rows:
        lines.update(r["data"].get("peaks", {}))
    lines.update(peaks or {})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(max(3.6 * ncols, 5.0), 2.9 * nrows + 0.5), squeeze=False, sharey=True, layout="constrained")
        flat = [ax for line in axes for ax in line]
        for ax, combo in zip(flat, combos):
            sel = [r for r in rows if all(_key(r["params"][k]) == _key(v) for k, v in zip(varying, combo))]
            _panel(ax, sel, xs, x, series, colors, baseline, lines, _title(zip(varying, combo)))
        for ax in flat[len(combos):]:
            ax.set_visible(False)
        handles, labels = flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 2 if ncols == 1 else 3))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(prog="python -m tests.plot")
    parser.add_argument("results", help="JSON from python -m tests --bench --json")
    parser.add_argument("--out", default="assets/figures/{name}.png",
                        help="output file; {name} is replaced by the benchmark name")
    parser.add_argument("--peaks", nargs="?", const="-", metavar="PATH",
                        help="JSON with a \"peaks\" mapping from python -m tests.peak; stdin if no path is given")
    parser.add_argument("--only", nargs="*", help="benchmark names to plot (default: all with tf: metrics)")
    parser.add_argument("--x", help="parameter on the x axis (default: n, or nk)")
    parser.add_argument("--baseline", help="series the ratios are computed against (default: the first)")
    args = parser.parse_args()
    with open(args.results) as f:
        data = json.load(f)
    extra = {}
    if args.peaks:
        extra = json.load(sys.stdin if args.peaks == "-" else open(args.peaks))["peaks"]
    names = args.only or _unique(r["name"] for r in data["results"] if any(k.startswith("tf:") for k in r["metrics"]))
    if len(names) > 1 and "{name}" not in args.out:
        parser.error("--out needs {name} when plotting more than one benchmark")
    for name in names:
        path = args.out.format(name=name)
        print(f"wrote {path}" if draw(data, name, path, args.x, args.baseline, extra) else f"no results for {name}")


if __name__ == "__main__":
    main()
