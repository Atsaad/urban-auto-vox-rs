"""Build a deployable, passphrase-protected results page.

The results view needs two secrets: the answer key (which roof each
stimulus really has) and a Supabase key that can read the responses.
Publishing either in plaintext would void the study, because a
participant could look up the answers.

So they are encrypted, not hidden. The page ships an AES-256-GCM
ciphertext; the passphrase you choose derives the key via PBKDF2-SHA256
at 600,000 iterations. Without the passphrase the page is inert — not
"hidden behind a prompt you can skip by viewing source", but genuinely
undecryptable. This is why a client-side PIN check would have been
useless and this is not: there is nothing to bypass.

Choose a real passphrase. Six digits is a million guesses, which at
600k iterations is slow but not impossible for someone who downloads the
page. Three or four unrelated words is beyond reach.

Usage
-----
    export SUPABASE_URL="https://xxxx.supabase.co"
    export SUPABASE_SECRET="sb_secret_..."
    python study/build_results_page.py

It prompts for the passphrase, writes docs/results.html, and that file is
safe to commit and deploy.
"""
from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import secrets
from pathlib import Path

KEY_PATH = Path("study/answer_key.json")
OUT = Path("study/docs/results.html")
ITERATIONS = 600_000


def encrypt(payload: dict, passphrase: str) -> dict:
    """AES-256-GCM with a PBKDF2-SHA256 derived key.

    Mirrors exactly what the browser's Web Crypto does on the other side,
    so the page can decrypt without any library.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = secrets.token_bytes(16)
    iv = secrets.token_bytes(12)
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, ITERATIONS, 32)
    ct = AESGCM(key).encrypt(iv, json.dumps(payload).encode(), None)
    b64 = lambda b: base64.b64encode(b).decode()
    return {"salt": b64(salt), "iv": b64(iv), "ct": b64(ct), "iter": ITERATIONS}


def main() -> None:
    url = os.environ.get("SUPABASE_URL")
    sec = os.environ.get("SUPABASE_SECRET")
    if not (url and sec):
        raise SystemExit("set SUPABASE_URL and SUPABASE_SECRET first")
    if not KEY_PATH.exists():
        raise SystemExit(f"answer key not found at {KEY_PATH}")

    p1 = os.environ.get("RESULTS_PASSPHRASE")
    if p1:
        print("using RESULTS_PASSPHRASE from the environment")
    else:
        p1 = getpass.getpass("Passphrase for the results page: ")
        if p1 != getpass.getpass("Repeat: "):
            raise SystemExit("passphrases do not match")
    if len(p1) < 8:
        raise SystemExit("too short — at least 8 characters, ideally "
                         "three or four unrelated words")

    key = json.loads(KEY_PATH.read_text())
    blob = encrypt({"key": key, "url": url.rstrip("/"), "sb": sec}, p1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PAGE.replace("__BLOB__", json.dumps(blob)))
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KiB)")
    print(f"  {len(key)} stimuli in the encrypted answer key")
    print("  safe to commit: without the passphrase this is undecryptable")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pillar G — results</title>
<style>
:root{color-scheme:light;
 --surface-1:#fcfcfb;--surface-2:#fff;--line:#e4e4e0;
 --text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#86857f;
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--seq:#2a78d6;--bad:#d03b3b;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
 color-scheme:dark;
 --surface-1:#1a1a19;--surface-2:#232322;--line:#33332f;
 --text-primary:#fff;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--seq:#3987e5;}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-1);color:var(--text-primary);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
 padding:32px 20px 80px}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 .2em;letter-spacing:-.02em}
h2{font-size:1rem;margin:2.4em 0 .8em;letter-spacing:-.01em}
.sub{color:var(--text-secondary);margin:0 0 2em}
.gate{max-width:420px;margin:12vh auto;background:var(--surface-2);
 border:1px solid var(--line);border-radius:14px;padding:28px}
input{width:100%;padding:.7em .9em;font:inherit;border:1px solid var(--line);
 border-radius:9px;background:var(--surface-1);color:var(--text-primary);margin:.8em 0}
button{font:inherit;font-weight:600;background:var(--s1);color:#fff;border:0;
 border-radius:9px;padding:.7em 1.4em;cursor:pointer}
.err{color:var(--bad);font-size:.88rem;min-height:1.3em}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--surface-2);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile b{display:block;font-size:1.7rem;line-height:1.15;letter-spacing:-.02em}
.tile span{font-size:.78rem;color:var(--text-secondary)}
.tile.hero b{color:var(--s1)}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:middle}
th{font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);font-weight:600}
.lbl{font-weight:600;white-space:nowrap}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.muted{color:var(--text-muted)}.small{font-size:.82rem}
.bad{color:var(--bad);font-weight:700}
.track{position:relative;height:16px;background:var(--line);border-radius:4px;overflow:hidden}
.fill{height:100%;border-radius:4px}
.ci{position:absolute;top:4px;height:8px;border-left:2px solid var(--text-primary);
 border-right:2px solid var(--text-primary);opacity:.45}
.ci-txt{font-size:.78rem;color:var(--text-muted);font-variant-numeric:tabular-nums}
img{width:56px;height:56px;object-fit:contain;border-radius:6px;background:var(--surface-2);display:block}
.chip{display:inline-block;padding:.1em .5em;border-radius:5px;font-size:.75rem;border:1px solid var(--line)}
.cell{text-align:center;font-variant-numeric:tabular-nums;
 background:color-mix(in oklab,var(--seq) calc(var(--v)*100%),transparent)}
.cell.hit{outline:2px solid var(--s3);outline-offset:-2px}
.callout{background:var(--surface-2);border:1px solid var(--line);border-left:3px solid var(--s1);
 border-radius:8px;padding:12px 16px;margin:1em 0;color:var(--text-secondary);font-size:.9rem}
.scroll{overflow-x:auto}
.bar{display:flex;gap:10px;align-items:center;margin-bottom:1.6em}
</style></head><body>

<div id="gate" class="gate">
  <h1>Results</h1>
  <p class="muted small">This page holds the study's answer key, encrypted.
    Enter the passphrase to decrypt it and load the responses.</p>
  <input id="pw" type="password" placeholder="Passphrase" autofocus
         onkeydown="if(event.key==='Enter')unlock()">
  <button onclick="unlock()">Unlock</button>
  <p class="err" id="err"></p>
</div>

<div class="wrap" id="app" hidden></div>

<script>
const BLOB = __BLOB__;
const dec = new TextDecoder(), enc = new TextEncoder();
const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));

async function unlock(){
  const pw = document.getElementById('pw').value;
  const err = document.getElementById('err');
  err.textContent = 'Deriving key…';
  try{
    const base = await crypto.subtle.importKey('raw', enc.encode(pw), 'PBKDF2', false, ['deriveKey']);
    const key = await crypto.subtle.deriveKey(
      {name:'PBKDF2', salt:b64(BLOB.salt), iterations:BLOB.iter, hash:'SHA-256'},
      base, {name:'AES-GCM', length:256}, false, ['decrypt']);
    const pt = await crypto.subtle.decrypt({name:'AES-GCM', iv:b64(BLOB.iv)}, key, b64(BLOB.ct));
    const payload = JSON.parse(dec.decode(pt));
    document.getElementById('gate').hidden = true;
    document.getElementById('app').hidden = false;
    render(payload);
  }catch(e){
    // AES-GCM authenticates, so a wrong passphrase fails to decrypt
    // rather than yielding garbage. There is nothing to brute-force in
    // the page itself.
    err.textContent = 'Wrong passphrase.';
  }
}

const wilson = (k,n) => {
  if(!n) return [0,0];
  const z=1.96, p=k/n, d=1+z*z/n;
  const c=(p+z*z/(2*n))/d, h=z*Math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;
  return [Math.max(0,c-h), Math.min(1,c+h)];
};
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function bar(label,k,n,slot){
  if(!n) return `<tr><td class="lbl">${esc(label)}</td><td colspan="4" class="muted">no data</td></tr>`;
  const acc=k/n, [lo,hi]=wilson(k,n);
  return `<tr><td class="lbl">${esc(label)}</td>
    <td style="width:52%"><div class="track">
      <div class="fill" style="width:${(acc*100).toFixed(1)}%;background:var(--s${slot})"></div>
      <div class="ci" style="left:${(lo*100).toFixed(1)}%;width:${((hi-lo)*100).toFixed(1)}%"></div>
    </div></td>
    <td class="num">${(acc*100).toFixed(1)}%</td>
    <td class="ci-txt">[${(lo*100).toFixed(0)}–${(hi*100).toFixed(0)}]</td>
    <td class="num muted">${n}</td></tr>`;
}

async function render(p){
  const app = document.getElementById('app');
  app.innerHTML = '<p class="muted">Loading responses…</p>';
  let raw = [], from = 0;
  while(true){
    const r = await fetch(`${p.url}/rest/v1/responses?select=*&order=id.asc&offset=${from}&limit=1000`,
      {headers:{apikey:p.sb, Authorization:'Bearer '+p.sb}});
    const b = await r.json();
    raw = raw.concat(b);
    if(b.length < 1000) break;
    from += 1000;
  }

  // de-duplicate on (session,img): a retry that lands twice must not
  // weight one participant above another
  const seen = new Map(); let skips=0, dupes=0, unknown=0;
  for(const r of raw){
    const s=(r.session||'').trim(), img=(r.img||'').trim(), c=(r.choice||'').trim();
    if(!s||!img) continue;
    if(c==='(skipped)'){skips++;continue;}
    if(!p.key[img]){unknown++;continue;}
    const id=s+'|'+img;
    if(seen.has(id)){dupes++;continue;}
    seen.set(id,c);
  }

  const present=new Set(Object.values(p.key).map(m=>m.roof_type));
  const B={}, perImg={}, conf={}, sessions=new Set();
  const add=(b,ok)=>{(B[b]=B[b]||[0,0])[0]+=ok; B[b][1]++;};
  let distract=0;
  for(const [id,c] of seen){
    const [sess,img]=id.split('|'); sessions.add(sess);
    const m=p.key[img], truth=m.roof_type, kind=m.kind;
    const arm = kind==='generated' ? m.model : 'real';
    const ok = c===truth ? 1 : 0;
    add('ALL',ok); add('kind:'+kind,ok); add('arm:'+arm,ok); add('roof:'+truth,ok);
    perImg[img]=perImg[img]||[0,0,{}];
    perImg[img][0]+=ok; perImg[img][1]++;
    perImg[img][2][c]=(perImg[img][2][c]||0)+1;
    conf[truth]=conf[truth]||{}; conf[truth][c]=(conf[truth][c]||0)+1;
    if(!present.has(c)&&c!=='none') distract++;
  }
  const g=n=>B[n]||[0,0];
  const [rk,rn]=g('kind:real'), [gk,gn]=g('kind:generated'), [ak,an]=g('ALL');
  const gap=((rn?rk/rn:0)-(gn?gk/gn:0))*100;

  const arms=Object.keys(B).filter(k=>k.startsWith('arm:')).sort();
  const roofs=Object.keys(B).filter(k=>k.startsWith('roof:')).sort();

  const rows=Object.entries(perImg).map(([img,[k,n,picks]])=>{
    const m=p.key[img];
    const wrong=Object.entries(picks).filter(([c])=>c!==m.roof_type).sort((a,b)=>b[1]-a[1]).slice(0,2);
    return {acc:n?k/n:0,n,img,m,wrong};
  }).sort((a,b)=>a.acc-b.acc||b.n-a.n);

  const picked=[...new Set(Object.values(conf).flatMap(o=>Object.keys(o)))].sort();
  let cm='';
  for(const truth of Object.keys(conf).sort()){
    const tot=Object.values(conf[truth]).reduce((a,b)=>a+b,0)||1;
    cm+=`<tr><th class="lbl">${esc(truth)}</th>`+picked.map(c=>{
      const v=conf[truth][c]||0;
      return `<td class="cell${c===truth?' hit':''}" style="--v:${(v/tot).toFixed(3)}">${v||''}</td>`;
    }).join('')+'</tr>';
  }

  app.innerHTML = `
  <div class="bar"><h1 style="margin:0">Pillar G — results</h1></div>
  <p class="sub">${seen.size.toLocaleString()} scored trials · ${sessions.size} participant(s)
    · ${raw.length.toLocaleString()} raw rows · loaded ${new Date().toLocaleString()}</p>

  <div class="tiles">
    <div class="tile hero"><b>${rn?(rk/rn*100).toFixed(1):'0.0'}%</b><span>real buildings (the ceiling)</span></div>
    <div class="tile hero"><b>${gn?(gk/gn*100).toFixed(1):'0.0'}%</b><span>generated buildings</span></div>
    <div class="tile"><b>${gap>=0?'+':''}${gap.toFixed(1)}</b><span>gap, percentage points</span></div>
    <div class="tile"><b>${seen.size}</b><span>scored trials</span></div>
    <div class="tile"><b>${sessions.size}</b><span>participants</span></div>
    <div class="tile"><b>${(Object.keys(perImg).length/Object.keys(p.key).length*100).toFixed(0)}%</b>
      <span>of ${Object.keys(p.key).length} stimuli seen</span></div>
  </div>

  <div class="callout"><strong>How to read this.</strong> Accuracy on real
  buildings is the benchmark — how legibly a roof type can be read from a voxel
  render at all, independent of any model. The generated figure only means
  something beside it. Chance is ${(100/13).toFixed(1)}% across 13 options, but that is a
  weak floor and not the comparison that matters.</div>

  <h2>Real against generated</h2>
  <table><tr><th>Group</th><th>Accuracy</th><th></th><th>95% CI</th><th>n</th></tr>
  ${bar('Real (benchmark)',rk,rn,1)}${bar('Generated',gk,gn,2)}${bar('All trials',ak,an,3)}</table>

  <h2>By model version</h2>
  <table><tr><th>Arm</th><th>Accuracy</th><th></th><th>95% CI</th><th>n</th></tr>
  ${arms.map((k,i)=>bar(k.slice(4),...B[k],(i%4)+1)).join('')}</table>

  <h2>By roof type</h2>
  <table><tr><th>True roof</th><th>Accuracy</th><th></th><th>95% CI</th><th>n</th></tr>
  ${roofs.map(k=>bar(k.slice(5),...B[k],3)).join('')}</table>

  <h2>What people chose, per true roof type</h2>
  <div class="scroll"><table><tr><th>True ↓ / chosen →</th>${picked.map(c=>`<th>${esc(c)}</th>`).join('')}</tr>
  ${cm}</table></div>
  <p class="muted small">Shade is the share of that row; the outlined cell is the
  correct answer. Off-diagonal weight shows which roof types are ambiguous.</p>

  <h2>Hardest stimuli — worst first</h2>
  <p class="muted small">A <em>generated</em> building here means the model failed
  to express that roof. A <em>real</em> one means the rendering itself is hard to
  read — a caveat on the task, not a fault of the model.</p>
  <div class="scroll"><table>
  <tr><th></th><th>True roof</th><th>Source</th><th>Correct</th><th>n</th><th>Most common wrong answers</th></tr>
  ${rows.slice(0,40).map(r=>`<tr>
    <td><img src="img/${esc(r.img)}" loading="lazy" alt=""></td>
    <td class="lbl">${esc(r.m.roof_type)}</td>
    <td><span class="chip">${esc(r.m.kind==='generated'?r.m.model:'real')}</span></td>
    <td class="num ${r.acc<0.5?'bad':''}">${(r.acc*100).toFixed(0)}%</td>
    <td class="num muted">${r.n}</td>
    <td class="muted small">${r.wrong.map(([c,v])=>esc(c)+' ×'+v).join(', ')||'—'}</td></tr>`).join('')}
  </table></div>

  <h2>Data quality</h2>
  <table>
  <tr><td class="lbl">Duplicate rows dropped</td><td class="num">${dupes}</td>
      <td class="muted small">retries that landed twice; first answer kept</td></tr>
  <tr><td class="lbl">Skipped trials</td><td class="num">${skips}</td>
      <td class="muted small">recorded, excluded from accuracy</td></tr>
  <tr><td class="lbl">Distractor picks</td><td class="num">${distract}</td>
      <td class="muted small">roof types not present in any stimulus</td></tr>
  <tr><td class="lbl">Unrecognised images</td><td class="num">${unknown}</td>
      <td class="muted small">image not in the answer key</td></tr>
  </table>`;
}
</script></body></html>
"""


if __name__ == "__main__":
    main()
