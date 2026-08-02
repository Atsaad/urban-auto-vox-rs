# Pillar G — Perceptual Study

A roof-type identification task: a participant sees one building at a
time and names its roof type.

The comparison that matters is **generated against real**, both scored
identically. Accuracy on real buildings is the ceiling — how legibly a
roof type can be read from a voxel render at all, independent of any
model. Accuracy on generated buildings is how well the model expressed
the roof type it was asked for. A model whose roofs are as readable as
real ones scores the same on both.

This is a human validation of the automatic D2 roof-control metric
(§`sec:eval-d2`), and it is preferred to a "which one is real" task for
three reasons: it has a ground truth per image, it needs no comparison
partner, and it measures something the thesis can act on. A realism
judgement would in any case have been confounded — the defect v5
improved is *interior* closure, which an exterior render cannot show.

`hipped` and `mansard` are offered but never occur in the stimuli. They
are deliberate distractors: picking one is a false positive, and a model
producing ambiguous gables will attract them.

---

## Why this design

**Blind by construction.** Image files are named by SHA-256 hash, and the
manifest that ships to the browser (`docs/stimuli.json`) lists only which
two files form a pair — it never says which is real. The page itself
does not know the answer. Scoring happens offline against `study/answer_key.json`, which lives
**outside** the deployed `docs/` directory so it cannot be served even by
accident, and is gitignored as well.

**Anonymous.** The page records only which image was clicked, per trial.
No name, email, account, cookie, free text, or analytics. There is
nothing collected that could identify a participant.

**Order-randomised per participant.** Trial order is shuffled in the
browser, so any fatigue effect is spread across images rather than always
falling on the same ones.

**Partial responses are fully usable.** Every trial is scored
independently, so a participant who answers twelve buildings contributes
twelve trials. Progress is saved to `localStorage` after every answer, so
closing the tab loses nothing and the participant is offered *Resume* on
return. There is a *Skip* button for buildings they cannot judge, and a
*Stop & submit* button available at all times.

---

## 1. Generate stimuli

```bash
PYTHONPATH=. ./model/.venv/bin/python study/make_stimuli.py \
    --gen model/checkpoints/phase_b_v4/eval_w0/eval_samples.npz \
    --label v4_w0 \
    --pairs 32
```

| Argument | Meaning |
|---|---|
| `--gen` | an `eval_samples.npz` produced by `evaluate.py build-samples` (contains both `gen` and `real`) |
| `--label` | name for this model arm, e.g. `v4_w0`, `v5`, `phase_c` |
| `--pairs` | approximate number of pairs, split evenly across conditions |
| `--salt` | changes the filename hashes; keep it fixed across arms |

**Outputs**

- `docs/img/*.png` — the rendered buildings, opaque filenames
- `docs/stimuli.json` — trial list, ships to the browser, **contains no answers**
- `study/answer_key.json` — the key, **outside `docs/`, gitignored, keep locally**

### Adding a model arm later

Re-run with a different `--gen` and `--label`. Both files are **appended**
to, not overwritten, so v4, v5 and Phase C can coexist in one study and be
scored separately:

```bash
PYTHONPATH=. ./model/.venv/bin/python study/make_stimuli.py \
    --gen model/checkpoints/phase_b_v5/eval/eval_samples.npz \
    --label v5 --pairs 32
```

Back up `study/answer_key.json` before re-running — losing it makes every
collected response unscoreable.

---

## 2. Deploy

The page is a single self-contained `index.html` with no dependencies, no
build step, and no external requests.

1. Push `study/docs/` to your repository (the answer key is gitignored).
2. Repository → **Settings** → **Pages** → Source: *Deploy from a branch*,
   branch `main`, folder `/docs`.
3. If `study/docs` is not the repository root's `/docs`, either move it
   there or point Pages at the correct branch/folder.

Verify locally first:

```bash
cd study/docs && python3 -m http.server 8000
# open http://localhost:8000
```

---

## 3. Collecting responses automatically

Answers are sent **as they happen**, one POST per answer. A participant
who answers twelve buildings and closes the tab still contributes twelve
trials — nothing depends on them reaching a submit button or emailing a
code back. Failed sends stay queued in `localStorage` and retry on the
next answer and the next visit, and `sendBeacon` makes a final attempt if
the tab is closed mid-session.

GitHub Pages is static and cannot receive a POST, so the endpoint is a
free Google Apps Script web app writing into a Google Sheet.

### Set it up (about five minutes)

1. Create a new Google Sheet. Name the first tab `responses`.
2. **Extensions → Apps Script**, delete the placeholder, paste:

```javascript
function doPost(e) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet()
                            .getSheetByName('responses');
  var body = JSON.parse(e.postData.contents);
  var now  = new Date();
  // One row per trial: session, server time, client time, image, choice.
  (body.rows || []).forEach(function (r) {
    sheet.appendRow([body.session, now, new Date(r.ts), r.img, r.choice]);
  });
  return ContentService
    .createTextOutput(JSON.stringify({ok: true, n: (body.rows || []).length}))
    .setMimeType(ContentService.MimeType.JSON);
}
```

3. **Deploy → New deployment → Web app**.
   - *Execute as*: **Me**
   - *Who has access*: **Anyone**  ← required; participants are not
     logged in
4. Copy the `/exec` URL it gives you.
5. Paste it into `docs/index.html`:

```javascript
const RESULTS_ENDPOINT = "https://script.google.com/macros/s/AKfy.../exec";
```

6. Redeploy the page. Do one run yourself and confirm rows appear.

### Notes

- Rows are appended, so duplicates are possible if a retry succeeds twice.
  `score.py` de-duplicates on `(session, img)`, keeping the first answer.
- The page posts `Content-Type: text/plain` deliberately: it avoids a
  CORS preflight, which Apps Script does not answer. The script parses
  the body as JSON regardless.
- **`session` is a random 16-hex-digit value generated in the browser.**
  It is not derived from anything about the participant and is never
  linked to any other record; it exists only so trials from one sitting
  can be grouped when scoring. The study still collects no name, email,
  account, or free text.
- Apps Script's free quota is far above anything this study will reach.

### If you would rather not use Google

Leave `RESULTS_ENDPOINT = null` and the page falls back to showing a
result code the participant sends back manually. It works, but expect to
lose responses — which is the reason automatic submission exists.

Any endpoint accepting a POST will do (Formspree, a Cloudflare Worker, a
small Flask app). Note that any third-party endpoint may log the request
IP at the network level, which is outside this page's control.

---

## 4. Score

```bash
# from downloaded JSON files
./model/.venv/bin/python study/score.py study/responses/*.json

# or from pasted base64 codes, one per line
./model/.venv/bin/python study/score.py --codes study/responses/codes.txt
```

Reports roof-identification accuracy overall, split real vs generated,
per model arm, per roof type, and per arm×roof-type, each with a Wilson
95 % confidence interval (correct for small n, unlike the normal
approximation). Also prints what was chosen for each true roof type, and
the rate at which the never-present distractors were picked.

---

## Reporting in the thesis

Report **n participants**, **n trials**, and accuracy with CI for real
and generated separately — the real figure is the ceiling and must be
given, or the generated number cannot be interpreted. Chance is 1/6
≈ 16.7 % against six options, but that is a weak floor; the meaningful
reference is the real-building accuracy.

The headline is the CI, not the point estimate. Because participants may
stop early, report the trial count per bucket rather than assuming
n_participants × 128, and do not exclude partial responses — they are
unbiased with respect to the arms, since trial order is randomised.
