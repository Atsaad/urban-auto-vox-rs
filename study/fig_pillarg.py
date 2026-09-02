"""Render the human roof-type legibility figure from the frozen snapshot.

Reads only the frozen copy taken on 2026-08-16, so the figure cannot drift
as further responses arrive. Colour convention matches the rest of the
evaluation figures: blue = real reference, green = generated.

Usage:
    model/.venv/bin/python study/fig_pillarg.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FROZEN = Path("/tmp/claude-4220471/-home-ge27lof-Documents-github-urban-auto-vox-rs-master/"
              "78eccde0-f00e-4591-bb6b-d163c033fc21/scratchpad/pillarG_frozen_2026-08-16")
OUT = Path("/home/ge27lof/Documents/github/urban-auto-vox-rs-master/local comments/"
           "Overleaf Projects (2 items)/tum-thesis-latex-master/figures/pillarG_results.png")

C_REAL, C_GEN = "#1f77b4", "#2ca02c"
CHANCE = 100 / 14          # 13 roof options plus "none of the above"


def load():
    rows = []
    for f in sorted(glob.glob(str(FROZEN / "page_*.json"))):
        rows += json.load(open(f))
    key = json.load(open(FROZEN / "answer_key.json"))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["session"]].append(r)
    seen = {}
    for s, v in by.items():
        v.sort(key=lambda r: r["client_time"])
        for r in v:
            img, c = (r["img"] or "").strip(), (r["choice"] or "").strip()
            if not img or c == "(skipped)" or img not in key or (s, img) in seen:
                continue
            seen[(s, img)] = c
    return seen, key


def main() -> None:
    seen, key = load()
    pp = collections.defaultdict(lambda: {"real": [0, 0], "v7": [0, 0]})
    for (s, img), c in seen.items():
        m = key[img]
        k = "real" if m["kind"] == "real" else "v7"
        pp[s][k][0] += (c == m["roof_type"])
        pp[s][k][1] += 1
    full = {s: d for s, d in pp.items() if d["real"][1] >= 10 and d["v7"][1] >= 10}
    pairs = sorted(((100 * d["real"][0] / d["real"][1], 100 * d["v7"][0] / d["v7"][1])
                    for d in full.values()), key=lambda t: t[0] - t[1])

    byrt = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for (s, img), c in seen.items():
        m = key[img]
        k = "real" if m["kind"] == "real" else "v7"
        byrt[m["roof_type"]][k][0] += (c == m["roof_type"])
        byrt[m["roof_type"]][k][1] += 1

    conf = collections.Counter(c for (s, img), c in seen.items()
                               if key[img]["kind"] != "real" and key[img]["roof_type"] == "gabled")
    tot = sum(conf.values())

    fig = plt.figure(figsize=(11.0, 3.6), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.05], wspace=0.34,
                          left=0.055, right=0.985, top=0.80, bottom=0.17)

    # (a) per-participant paired comparison -------------------------------
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(pairs))
    for i, (r, g) in enumerate(pairs):
        ax.plot([g, r], [i, i], color="#bbbbbb", lw=1.5, zorder=1)
    ax.scatter([p[1] for p in pairs], y, s=42, color=C_GEN, zorder=3, label="selected model")
    ax.scatter([p[0] for p in pairs], y, s=42, color=C_REAL, zorder=3, label="real")
    ax.axvline(CHANCE, color="#999999", ls=":", lw=1.2)
    ax.annotate("chance", (CHANCE + 1.5, len(pairs) - 0.6), fontsize=8, color="#666666")
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xlabel("roof type identified correctly (%)", fontsize=9.5)
    # "qualifying session", not "participant": the snapshot holds 34
    # sessions and these 12 are the subset clearing >=10 trials per arm.
    # Calling them participants reads as "the study had 12 people".
    ax.set_ylabel(f"qualifying session  (n = {len(pairs)})", fontsize=9.5)
    ax.set_title("(a)  every qualifying session, both arms", fontsize=10)
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)

    # (b) by roof type ----------------------------------------------------
    ax = fig.add_subplot(gs[1])
    rts = ["gabled", "flat", "monopitch"]
    x, w = np.arange(3), 0.36
    rv = [100 * byrt[t]["real"][0] / byrt[t]["real"][1] for t in rts]
    gv = [100 * byrt[t]["v7"][0] / byrt[t]["v7"][1] for t in rts]
    ax.bar(x - w/2, rv, w, color=C_REAL, label="real")
    ax.bar(x + w/2, gv, w, color=C_GEN, label="selected model")
    for i in range(3):
        ax.text(x[i] - w/2, rv[i] + 2, f"{rv[i]:.0f}", ha="center", fontsize=8.5, color=C_REAL)
        ax.text(x[i] + w/2, gv[i] + 2, f"{gv[i]:.0f}", ha="center", fontsize=8.5, color=C_GEN)
    ax.axhline(CHANCE, color="#999999", ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(["gabled", "flat", "monopitch\n(see caption)"], fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_ylabel("identified correctly (%)", fontsize=9.5)
    ax.set_title("(b)  by requested roof type", fontsize=10)
    ax.grid(axis="y", alpha=0.22, lw=0.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

    # (c) what a generated gable is called --------------------------------
    ax = fig.add_subplot(gs[2])
    items = conf.most_common(6)
    lab = [k for k, _ in items][::-1]
    val = [100 * v / tot for _, v in items][::-1]
    cols = [C_GEN if l == "gabled" else "#c8c8c8" for l in lab]
    yy = np.arange(len(lab))
    ax.barh(yy, val, 0.62, color=cols)
    for i, v in enumerate(val):
        ax.text(v + 0.8, yy[i], f"{v:.1f}%", va="center", fontsize=8.4, color="#444444")
    ax.set_yticks(yy)
    ax.set_yticklabels([l.replace("_", " ") for l in lab], fontsize=9)
    ax.set_xlim(0, 36)
    ax.set_xlabel("share of responses (%)", fontsize=9.5)
    ax.set_title("(c)  a generated gable is called…", fontsize=10)
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)

    # One legend for the whole figure, above the panels: inside (a) it
    # overlapped the dumbbells, and (b)/(c) have no room either.
    h = [plt.Line2D([], [], marker="o", ls="", color=C_REAL, ms=8),
         plt.Line2D([], [], marker="o", ls="", color=C_GEN, ms=8)]
    fig.legend(h, ["real buildings", "selected model"], loc="upper center",
               ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"  participants {len(pairs)}   trials {len(seen)}")
    print(f"  correct-only bar in (c): gabled {100*conf['gabled']/tot:.1f}%")


if __name__ == "__main__":
    main()
