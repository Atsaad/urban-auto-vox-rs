"""Score Pillar G roof-type responses against the answer key.

The task asks a participant to name the roof type of each building. The
comparison that matters is **generated against real**, both scored the
same way:

  - accuracy on REAL buildings is the ceiling -- how legibly a roof type
    can be read from a voxel render at all, independent of any model;
  - accuracy on GENERATED buildings is how well the model expressed the
    roof type it was asked for.

A model whose roofs are as readable as real ones scores equally on both.
This is a human validation of the automatic D2 roof-control metric, and
unlike a "which is real" task it has a ground truth per image and needs
no comparison partner.

The page offers every roof type in the model's vocabulary (12), while
only a few actually occur in the stimuli. The remainder are distractors:
choosing one is a false positive and is reported, since a model producing
ambiguous roofs will attract them. Which types are distractors is derived
from the answer key rather than hardcoded, so it stays correct if the
stimulus set changes.

Partial responses are expected and fully usable -- every trial is scored
independently, so a participant who answered twelve buildings
contributes twelve trials.

Usage
-----
    python study/score.py --sheet study/responses/responses.csv
    python study/score.py study/responses/*.json
    python study/score.py --codes study/responses/codes.txt

Sheet rows are de-duplicated on (session, image): a retry that lands
twice would otherwise weight one participant above another. Skips are
recorded but excluded from accuracy.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

KEY = Path("study/answer_key.json")   # outside docs/, never deployed


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct for small n, unlike normal approx."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _row(label: str, k: int, n: int) -> str:
    if n == 0:
        return f"  {label:34s} {'--':>7s} {'':>16s} {0:6d}"
    lo, hi = wilson(k, n)
    return (f"  {label:34s} {100*k/n:6.1f}% "
            f"[{100*lo:5.1f}, {100*hi:5.1f}] {n:6d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="participant JSON response files")
    ap.add_argument("--codes", help="text file of base64 result codes, one per line")
    ap.add_argument("--sheet", help="CSV exported from the Google Sheet "
                                    "(File -> Download -> CSV)")
    a = ap.parse_args()

    if not KEY.exists():
        raise SystemExit(f"answer key not found at {KEY} -- run make_stimuli.py first")
    key = json.loads(KEY.read_text())

    payloads = [json.loads(Path(f).read_text()) for f in a.files]

    if a.sheet:
        # Rows are session, server_time, client_time, img, choice --
        # appended one per trial by the Apps Script receiver.
        #
        # De-duplicate on (session, img): a retry that succeeds twice
        # appends the same trial again, and counting it twice would
        # silently weight one participant's answer above another's. The
        # FIRST answer is kept, since a later duplicate is a resend of
        # the same choice rather than a change of mind.
        # Read by HEADER NAME, not column position: a Google Sheet export
        # and a Supabase export carry the same fields in different orders
        # (Supabase adds id/created_at), and positional parsing would
        # silently mis-assign them.
        import csv as _csv
        by_session: dict[str, dict[str, str]] = defaultdict(dict)
        dupes = skipped = 0
        with open(a.sheet, newline="") as fh:
            rdr = _csv.DictReader(fh)
            need = {"session", "img", "choice"}
            missing = need - set(rdr.fieldnames or [])
            if missing:
                raise SystemExit(
                    f"{a.sheet}: missing column(s) {sorted(missing)}; "
                    f"found {rdr.fieldnames}")
            for rec in rdr:
                sess = (rec.get("session") or "").strip()
                img = (rec.get("img") or "").strip()
                choice = (rec.get("choice") or "").strip()
                if not sess or not img:
                    continue
                if choice == "(skipped)":
                    skipped += 1
                    continue                       # not an answer
                if img in by_session[sess]:
                    dupes += 1
                    continue
                by_session[sess][img] = choice
        for sess, ans in by_session.items():
            payloads.append({"v": 1, "task": "rooftype", "session": sess,
                             "n": len(ans), "answers": ans})
        print(f"sheet: {len(by_session)} sessions, "
              f"{sum(len(v) for v in by_session.values())} unique trials "
              f"({dupes} duplicate rows dropped, {skipped} skips ignored)")
    if a.codes:
        for line in Path(a.codes).read_text().split():
            if line.strip():
                payloads.append(json.loads(base64.b64decode(line.strip())))
    if not payloads:
        raise SystemExit("no responses given")

    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: dict[tuple[str, str], Counter] = defaultdict(Counter)
    n_trials = 0

    for p in payloads:
        for img, chose in (p.get("answers") or {}).items():
            meta = key.get(img)
            if meta is None:
                continue                       # image not in this key
            truth = meta["roof_type"]
            kind = meta["kind"]
            arm = meta["model"] if kind == "generated" else "real"
            correct = int(chose == truth)
            n_trials += 1
            for b in ("ALL", f"kind:{kind}", f"arm:{arm}",
                      f"arm:{arm}|roof:{truth}", f"roof:{truth}"):
                buckets[b][0] += correct
                buckets[b][1] += 1
            confusion[(arm, truth)][chose] += 1

    print(f"participants: {len(payloads)}   scored trials: {n_trials}\n")
    print(f"  {'bucket':34s} {'acc':>7s} {'95% CI':>16s} {'n':>6s}")
    print("  " + "-" * 66)
    order = (["ALL"]
             + sorted(b for b in buckets if b.startswith("kind:"))
             + sorted(b for b in buckets if b.startswith("arm:") and "|" not in b)
             + sorted(b for b in buckets if b.startswith("roof:"))
             + sorted(b for b in buckets if "|" in b))
    for b in order:
        if b in buckets:
            print(_row(b, *buckets[b]))

    print("\n  The comparison that matters is `kind:real` against")
    print("  `kind:generated`. Real accuracy is the ceiling set by the")
    print("  rendering itself; a model whose roofs are as readable as real")
    print("  ones scores the same on both.")

    print("\n=== what was chosen, per true roof type ===")
    for (arm, truth) in sorted(confusion):
        c = confusion[(arm, truth)]
        tot = sum(c.values())
        picks = ", ".join(f"{k} {100*v/tot:.0f}%"
                          for k, v in c.most_common())
        print(f"  {arm:8s} true={truth:10s} (n={tot:4d}) -> {picks}")

    # Any roof type absent from the stimuli is a distractor, so picking
    # it is a false positive. Derived from the key so that changing the
    # stimulus set cannot silently invalidate this count.
    present = {m["roof_type"] for m in key.values()}
    fp = sum(v for (_arm, _t), c in confusion.items()
             for k, v in c.items() if k not in present and k != "none")
    if n_trials:
        absent = sorted({k for (_a, _t), c in confusion.items()
                         for k in c} - present - {"none"})
        print(f"\n  distractor picks ({', '.join(absent) if absent else 'none chosen'}): "
              f"{fp} / {n_trials} = {100*fp/n_trials:.1f}%")
        print(f"  roof types actually present in the stimuli: "
              f"{', '.join(sorted(present))}")


if __name__ == "__main__":
    main()
