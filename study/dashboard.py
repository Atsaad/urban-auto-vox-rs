"""Build a local results dashboard for the Pillar G study.

Deliberately NOT deployed. A results view needs two things that must
never be public: read access to the responses, and the answer key.
Publishing either would let a participant look up the answers and the
study would be void. That is true of GitHub Pages, Vercel, Netlify or
anything else serving a static page — the constraint is the secret, not
the host. So this runs on your machine and writes a self-contained HTML
file you open locally.

Usage
-----
    export SUPABASE_URL="https://xxxx.supabase.co"
    export SUPABASE_SECRET="sb_secret_..."      # bypasses RLS, read-only use
    python study/dashboard.py

    # or from an exported CSV, with no key at all:
    python study/dashboard.py --csv study/responses/responses.csv

Writes study/dashboard.html (gitignored).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

KEY_PATH = Path("study/answer_key.json")
OUT = Path("study/dashboard.html")

# Categorical slots 1-4, validated in both modes (adjacent pairlist):
# light worst CVD dE 9.1, normal-vision 22.9; dark 8.4 / 19.8.
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]


# --------------------------------------------------------------- stats
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval — correct at the small n a voluntary study yields."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fetch_supabase(url: str, key: str) -> list[dict]:
    rows, page, size = [], 0, 1000
    while True:
        req = urllib.request.Request(
            f"{url}/rest/v1/responses?select=*&order=id.asc"
            f"&offset={page * size}&limit={size}",
            headers={"apikey": key, "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req) as r:
            batch = json.loads(r.read())
        rows.extend(batch)
        if len(batch) < size:
            return rows
        page += 1


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def analyse(raw: list[dict], key: dict) -> dict:
    """De-duplicate on (session, img) and score every trial."""
    seen: dict[tuple[str, str], str] = {}
    skips = dupes = unknown = 0
    for r in raw:
        sess = (r.get("session") or "").strip()
        img = (r.get("img") or "").strip()
        choice = (r.get("choice") or "").strip()
        if not sess or not img:
            continue
        if choice == "(skipped)":
            skips += 1
            continue
        if img not in key:
            unknown += 1
            continue
        if (sess, img) in seen:       # a retry that landed twice
            dupes += 1
            continue
        seen[(sess, img)] = choice

    present = {m["roof_type"] for m in key.values()}
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_img: dict[str, list] = defaultdict(lambda: [0, 0, Counter()])
    confusion: dict[str, Counter] = defaultdict(Counter)
    distractors = 0

    for (sess, img), choice in seen.items():
        m = key[img]
        truth, kind = m["roof_type"], m["kind"]
        arm = m["model"] if kind == "generated" else "real"
        ok = int(choice == truth)
        for b in ("ALL", f"kind:{kind}", f"arm:{arm}", f"roof:{truth}",
                  f"arm:{arm}|roof:{truth}"):
            buckets[b][0] += ok
            buckets[b][1] += 1
        per_img[img][0] += ok
        per_img[img][1] += 1
        per_img[img][2][choice] += 1
        confusion[truth][choice] += 1
        if choice not in present and choice != "none":
            distractors += 1

    sessions = Counter(s for s, _ in seen)
    return {
        "trials": len(seen), "sessions": sessions, "skips": skips,
        "dupes": dupes, "unknown": unknown, "distractors": distractors,
        "buckets": dict(buckets), "per_img": dict(per_img),
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "present": sorted(present),
        "covered": len(per_img), "total_stimuli": len(key),
    }


# ---------------------------------------------------------------- html
def bar_row(label: str, k: int, n: int, colour_idx: int, note: str = "") -> str:
    if n == 0:
        return (f'<tr><td class="lbl">{html.escape(label)}</td>'
                f'<td colspan="3" class="muted">no data</td></tr>')
    acc = k / n
    lo, hi = wilson(k, n)
    return f'''<tr>
  <td class="lbl">{html.escape(label)}{note}</td>
  <td class="barcell">
    <div class="track">
      <div class="fill s{colour_idx}" style="width:{acc*100:.1f}%"></div>
      <div class="ci" style="left:{lo*100:.1f}%;width:{max(hi-lo,0)*100:.1f}%"></div>
    </div>
  </td>
  <td class="num">{acc*100:.1f}%</td>
  <td class="ci-txt">[{lo*100:.0f}–{hi*100:.0f}]</td>
  <td class="num muted">{n}</td>
</tr>'''


def build(a: dict, key: dict, source: str) -> str:
    b = a["buckets"]

    def get(name):
        return b.get(name, [0, 0])

    real_k, real_n = get("kind:real")
    gen_k, gen_n = get("kind:generated")
    all_k, all_n = get("ALL")
    gap = ((real_k / real_n if real_n else 0)
           - (gen_k / gen_n if gen_n else 0)) * 100

    arms = sorted(k for k in b if k.startswith("arm:") and "|" not in k)
    roofs = sorted(k for k in b if k.startswith("roof:"))

    # per-image, worst first: this is the "what is wrong" view
    rows = []
    for img, (k, n, picks) in a["per_img"].items():
        m = key[img]
        wrong = [(c, v) for c, v in picks.items() if c != m["roof_type"]]
        wrong.sort(key=lambda t: -t[1])
        rows.append((k / n if n else 0, n, img, m, wrong[:2]))
    rows.sort(key=lambda r: (r[0], -r[1]))

    img_rows = "".join(
        f'''<tr>
      <td><img src="docs/img/{html.escape(img)}" loading="lazy" alt=""></td>
      <td class="lbl">{html.escape(m["roof_type"])}</td>
      <td><span class="chip {'gen' if m['kind']=='generated' else 'real'}">
          {html.escape(m['model'] if m['kind']=='generated' else 'real')}</span></td>
      <td class="num {'bad' if acc < .5 else ''}">{acc*100:.0f}%</td>
      <td class="num muted">{n}</td>
      <td class="muted small">{', '.join(f"{html.escape(c)} ×{v}" for c, v in wrong) or '—'}</td>
    </tr>''' for acc, n, img, m, wrong in rows[:40])

    # confusion matrix, sequential blue by row-share
    picked = sorted({c for v in a["confusion"].values() for c in v})
    head = "".join(f"<th>{html.escape(c)}</th>" for c in picked)
    cm = ""
    for truth in sorted(a["confusion"]):
        tot = sum(a["confusion"][truth].values()) or 1
        cells = ""
        for c in picked:
            v = a["confusion"][truth].get(c, 0)
            share = v / tot
            hit = " hit" if c == truth else ""
            cells += (f'<td class="cell{hit}" style="--v:{share:.3f}">'
                      f'{v or ""}</td>')
        cm += f'<tr><th class="lbl">{html.escape(truth)}</th>{cells}</tr>'

    n_sess = len(a["sessions"])
    med = (sorted(a["sessions"].values())[n_sess // 2] if n_sess else 0)
    cov = a["covered"] / a["total_stimuli"] * 100 if a["total_stimuli"] else 0

    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pillar G — results</title>
<style>
:root{{color-scheme:light;
  --surface-1:#fcfcfb; --surface-2:#fff; --line:#e4e4e0;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#86857f;
  --s1:{LIGHT[0]}; --s2:{LIGHT[1]}; --s3:{LIGHT[2]}; --s4:{LIGHT[3]};
  --seq:#2a78d6; --bad:#d03b3b;}}
@media (prefers-color-scheme:dark){{:root:where(:not([data-theme="light"])){{
  color-scheme:dark;
  --surface-1:#1a1a19; --surface-2:#232322; --line:#33332f;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
  --s1:{DARK[0]}; --s2:{DARK[1]}; --s3:{DARK[2]}; --s4:{DARK[3]};
  --seq:#3987e5;}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--surface-1);color:var(--text-primary);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 padding:32px 20px 80px}}
.wrap{{max-width:1080px;margin:0 auto}}
h1{{font-size:1.5rem;margin:0 0 .2em;letter-spacing:-.02em}}
h2{{font-size:1rem;margin:2.4em 0 .8em;letter-spacing:-.01em}}
.sub{{color:var(--text-secondary);margin:0 0 2em}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.tile{{background:var(--surface-2);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.tile b{{display:block;font-size:1.7rem;line-height:1.15;letter-spacing:-.02em}}
.tile span{{font-size:.78rem;color:var(--text-secondary)}}
.tile.hero b{{color:var(--s1)}}
table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:middle}}
th{{font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);font-weight:600}}
.lbl{{font-weight:600;white-space:nowrap}}
.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.muted{{color:var(--text-muted)}} .small{{font-size:.82rem}}
.bad{{color:var(--bad);font-weight:700}}
.barcell{{width:52%}}
.track{{position:relative;height:16px;background:var(--line);border-radius:4px;overflow:hidden}}
.fill{{height:100%;border-radius:4px}}
.fill.s0{{background:var(--s1)}} .fill.s1{{background:var(--s2)}}
.fill.s2{{background:var(--s3)}} .fill.s3{{background:var(--s4)}}
.ci{{position:absolute;top:4px;height:8px;border-left:2px solid var(--text-primary);
 border-right:2px solid var(--text-primary);opacity:.45}}
.ci-txt{{font-size:.78rem;color:var(--text-muted);font-variant-numeric:tabular-nums;white-space:nowrap}}
img{{width:56px;height:56px;object-fit:contain;border-radius:6px;background:var(--surface-2);display:block}}
.chip{{display:inline-block;padding:.1em .5em;border-radius:5px;font-size:.75rem;
 border:1px solid var(--line);white-space:nowrap}}
.chip.real{{color:var(--s1)}} .chip.gen{{color:var(--s2)}}
.cell{{text-align:center;font-variant-numeric:tabular-nums;
 background:color-mix(in oklab,var(--seq) calc(var(--v)*100%),transparent)}}
.cell.hit{{outline:2px solid var(--s3);outline-offset:-2px}}
.callout{{background:var(--surface-2);border:1px solid var(--line);border-left:3px solid var(--s1);
 border-radius:8px;padding:12px 16px;margin:1em 0;color:var(--text-secondary);font-size:.9rem}}
.scroll{{overflow-x:auto}}
</style></head><body><div class="wrap">

<h1>Pillar G — perceptual study results</h1>
<p class="sub">{a['trials']:,} scored trials · {n_sess} participant(s) ·
generated from {html.escape(source)} on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>

<div class="tiles">
  <div class="tile hero"><b>{(real_k/real_n*100 if real_n else 0):.1f}%</b><span>real buildings (the ceiling)</span></div>
  <div class="tile hero"><b>{(gen_k/gen_n*100 if gen_n else 0):.1f}%</b><span>generated buildings</span></div>
  <div class="tile"><b>{gap:+.1f}</b><span>gap, percentage points</span></div>
  <div class="tile"><b>{a['trials']:,}</b><span>scored trials</span></div>
  <div class="tile"><b>{n_sess}</b><span>participants (median {med} each)</span></div>
  <div class="tile"><b>{cov:.0f}%</b><span>of {a['total_stimuli']} stimuli seen</span></div>
</div>

<div class="callout"><strong>How to read this.</strong> Accuracy on real
buildings is the benchmark — it shows how legibly a roof type can be read from
a voxel render at all, independent of any model. The generated figure is only
meaningful beside it. A small gap means the model expresses roof type about as
clearly as reality does; chance is {100/13:.1f}% across 13 options, but that is a
weak floor and not the comparison that matters.</div>

<h2>Real against generated</h2>
<table>
<tr><th>Group</th><th>Accuracy</th><th></th><th>95% CI</th><th>n</th></tr>
{bar_row("Real (benchmark)", real_k, real_n, 0)}
{bar_row("Generated", gen_k, gen_n, 1)}
{bar_row("All trials", all_k, all_n, 2)}
</table>

<h2>By model version</h2>
<table>
<tr><th>Arm</th><th>Accuracy</th><th></th><th>95% CI</th><th>n</th></tr>
{"".join(bar_row(k.split(":",1)[1], *b[k], i % 4) for i, k in enumerate(arms))}
</table>

<h2>By roof type</h2>
<table>
<tr><th>True roof</th><th>Accuracy</th><th></th><th>95% CI</th><th>n</th></tr>
{"".join(bar_row(k.split(":",1)[1], *b[k], 2) for k in roofs)}
</table>

<h2>What people chose, per true roof type</h2>
<div class="scroll"><table>
<tr><th>True ↓ / chosen →</th>{head}</tr>
{cm}
</table></div>
<p class="muted small">Cell shade is the share of that row. The outlined cell on
each row is the correct answer. Off-diagonal weight shows which roof types the
model — or the rendering — makes ambiguous.</p>

<h2>Hardest stimuli — worst first</h2>
<p class="muted small">The “what is wrong” view: individual buildings people
misread most often, with the answers they gave instead. A generated building
here means the model failed to express that roof; a <em>real</em> building here
means the rendering itself is hard to read, which is a caveat on the whole task
rather than a fault of the model.</p>
<div class="scroll"><table>
<tr><th></th><th>True roof</th><th>Source</th><th>Correct</th><th>n</th><th>Most common wrong answers</th></tr>
{img_rows}
</table></div>

<h2>Data quality</h2>
<table>
<tr><td class="lbl">Duplicate rows dropped</td><td class="num">{a['dupes']}</td>
    <td class="muted small">retries that landed twice; first answer kept</td></tr>
<tr><td class="lbl">Skipped trials</td><td class="num">{a['skips']}</td>
    <td class="muted small">recorded but excluded from accuracy</td></tr>
<tr><td class="lbl">Distractor picks</td><td class="num">{a['distractors']}</td>
    <td class="muted small">roof types not present in any stimulus</td></tr>
<tr><td class="lbl">Unrecognised images</td><td class="num">{a['unknown']}</td>
    <td class="muted small">responses whose image is not in the answer key</td></tr>
</table>

</div></body></html>'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="scored from an exported CSV instead of the API")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if not KEY_PATH.exists():
        raise SystemExit(f"answer key not found at {KEY_PATH}")
    key = json.loads(KEY_PATH.read_text())

    if a.csv:
        raw, source = read_csv(Path(a.csv)), a.csv
    else:
        url = os.environ.get("SUPABASE_URL")
        sec = os.environ.get("SUPABASE_SECRET")
        if not (url and sec):
            raise SystemExit(
                "set SUPABASE_URL and SUPABASE_SECRET, or pass --csv.\n"
                "The secret key is read from the environment on purpose: it "
                "bypasses row-level security and must never be written into a "
                "file or committed.")
        raw, source = fetch_supabase(url.rstrip("/"), sec), "Supabase"

    stats = analyse(raw, key)
    Path(a.out).write_text(build(stats, key, source))
    print(f"{len(raw)} row(s) in -> {stats['trials']} scored trials "
          f"from {len(stats['sessions'])} participant(s)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
