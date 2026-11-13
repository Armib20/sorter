"""Live dashboard for sorter runs: what the robot is doing now, every item's images,
scores, decision, diagnostics and timing waterfall. Standard library only, no internet.

    python dashboard.py                      # http://localhost:8765  (follows the newest run)
    python dashboard.py --host 0.0.0.0       # view the NUC's dashboard from your laptop
    ssh -L 8765:localhost:8765 nuc           # ... or tunnel it
    python dashboard.py --snapshot logs/<run>   # -> logs/<run>/report.html, open in any browser

It only reads the run folders written by sorter.py (diagnostics.RunRecorder), so it can
be started, stopped or crash at any time without affecting the robot.
Keys: <- / -> previous/next item, f follow latest, l cycle image layer.
"""

import argparse
import json
import mimetypes
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import settings


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def _read_jsonl(path, since=0):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    out = []
    for i, line in enumerate(lines[since:]):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if since + i == len(lines) - 1:  # half-written last line: pick it up next poll
                break
            raise
    return out


def run_dir(root: Path, run_id: str):
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        return None
    d = (root / run_id).resolve()
    return d if d.parent == root.resolve() and (d / "run.json").exists() else None


def list_runs(root: Path):
    if not root.exists():
        return []
    runs = []
    for d in sorted((p for p in root.iterdir() if (p / "run.json").exists()),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        meta, status = _read_json(d / "run.json"), _read_json(d / "status.json")
        try:
            n = sum(1 for line in (d / "items.jsonl").open() if line.strip())
        except OSError:
            n = 0
        det = meta.get("detector", {})
        runs.append({"id": d.name, "verb": meta.get("verb"), "started": meta.get("started"), "items": n,
                     "state": status.get("state"), "model": Path(det.get("model_path", "")).parent.name,
                     "device": det.get("device")})
    return runs


def load_run(d: Path, since=0, events=300):
    ev = _read_jsonl(d / "events.jsonl")
    return {"id": d.name, "meta": _read_json(d / "run.json"), "status": _read_json(d / "status.json"),
            "items": _read_jsonl(d / "items.jsonl", since), "since": since, "events": ev[-events:],
            "now": time.time()}


class Handler(BaseHTTPRequestHandler):
    root: Path = settings.LOG_DIR

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype="application/json", cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode())

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path in ("/", "/index.html"):
                return self._send(200, page(None).encode(), "text/html; charset=utf-8")
            if url.path == "/api/runs":
                return self._json(list_runs(self.root))
            if url.path == "/api/run":
                d = run_dir(self.root, q.get("id", [""])[0])
                if d is None:
                    return self._json({"error": "unknown run"}, 404)
                return self._json(load_run(d, int(q.get("since", ["0"])[0] or 0)))
            if url.path.startswith("/files/"):
                parts = unquote(url.path[len("/files/"):]).split("/", 1)
                d = run_dir(self.root, parts[0]) if len(parts) == 2 else None
                f = (d / parts[1]).resolve() if d else None
                if f is None or d not in f.parents or not f.is_file():
                    return self._json({"error": "not found"}, 404)
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype, cache=True)
            return self._json({"error": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def page(snapshot):
    data = "null" if snapshot is None else json.dumps(snapshot, default=str).replace("</", "<\\/")
    return PAGE.replace("/*SNAPSHOT*/null", data)


def snapshot(run: Path):
    run = Path(run).resolve()
    if not (run / "run.json").exists():
        raise SystemExit(f"{run} is not a run folder")
    out = run / "report.html"
    out.write_text(page(load_run(run, 0, events=1000)))
    return out


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def serve(host, port, root):
    Handler.root = Path(root).resolve()
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other machines on the network")
    ap.add_argument("--port", type=int, default=settings.DASHBOARD_PORT)
    ap.add_argument("--logs", default=str(settings.LOG_DIR))
    ap.add_argument("--snapshot", default=None, metavar="RUN_DIR", help="write RUN_DIR/report.html and exit")
    args = ap.parse_args()
    if args.snapshot:
        print(f"wrote {snapshot(args.snapshot)}")
        return
    Path(args.logs).mkdir(parents=True, exist_ok=True)
    srv = serve(args.host, args.port, args.logs)
    where = lan_ip() if args.host == "0.0.0.0" else "localhost"
    print(f"dashboard: http://{where}:{args.port}   (logs: {Path(args.logs).resolve()})  Ctrl-C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sorter dashboard</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e8ee;--dim:#8a93a6;--good:#3ecf8e;--bad:#ff5a5f;
--warn:#f5a524;--acc:#7c8cff;--thr:#ffd84d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;gap:14px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5;flex-wrap:wrap}
header h1{font-size:16px;margin:0}select,button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font:inherit;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc)}.meta{color:var(--dim);font-size:12px}
.pill{padding:3px 10px;border-radius:12px;background:var(--card);border:1px solid var(--line);font-weight:600;white-space:nowrap}
.pill.running{border-color:var(--good);color:var(--good)}.pill.error{border-color:var(--bad);color:var(--bad)}.pill.stopped,.pill.stale{border-color:var(--warn);color:var(--warn)}
.pill.running:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--good);margin-right:6px;animation:p 1s infinite}
@keyframes p{50%{opacity:.25}}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;padding:10px 16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px}.kpi b{font-size:20px;display:block}.kpi span{color:var(--dim);font-size:11px}
main{display:grid;grid-template-columns:minmax(440px,1.15fr) 1fr 1fr;gap:10px;padding:0 16px 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}
.card h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}
.item{grid-row:span 4}.wide{grid-column:2/4}canvas{width:100%;display:block}
table{border-collapse:collapse;width:100%}td,th{padding:3px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--dim);font-weight:500}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.viewer{position:relative;background:#000;border-radius:6px;overflow:hidden;min-height:200px;display:flex;align-items:center;justify-content:center}
.viewer img{max-width:100%;max-height:520px;display:block}.ph{color:var(--dim);padding:40px;text-align:center}
.tabs{display:flex;gap:4px;margin:6px 0;flex-wrap:wrap}.tabs button{padding:2px 8px;font-size:12px}
.act{display:inline-block;padding:2px 10px;border-radius:10px;font-weight:700;color:#000}
.gauge{position:relative;height:22px;border-radius:4px;background:linear-gradient(90deg,#1d3b2c,#3b1d22);margin:8px 0}
.gauge .band{position:absolute;top:0;bottom:0;background:rgba(255,216,77,.15)}.gauge .thr{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--thr)}
.gauge .mk{position:absolute;top:2px;width:4px;height:18px;margin-left:-2px;background:#fff;border-radius:2px}.gauge .vk{position:absolute;top:6px;width:2px;height:10px;background:#aaa}
.wf{font-size:11px}.wf .r{display:grid;grid-template-columns:170px 1fr;align-items:center;gap:6px;height:18px}
.wf .lbl{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.wf .trk{position:relative;height:12px}
.wf .bar{position:absolute;height:12px;border-radius:2px;min-width:2px}.wf .bar span{position:absolute;left:100%;padding-left:4px;white-space:nowrap;color:var(--dim)}
.wf .bar span.lft{left:auto;right:100%;padding:0 4px 0 0}
.wf .bar span.in{left:auto;right:4px;padding:0;color:#000;font-weight:600;line-height:12px}
.warn{color:var(--warn)}.err{color:var(--bad)}.ok{color:var(--good)}.dim{color:var(--dim)}
.gal{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:6px;max-height:440px;overflow:auto}
.th{position:relative;border:2px solid transparent;border-radius:6px;overflow:hidden;cursor:pointer;background:#000}
.th img{width:100%;display:block;aspect-ratio:1;object-fit:cover}.th .cap{position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);font-size:11px;padding:1px 4px}
.th.wrong{border-color:var(--bad)}.th.sel{border-color:#fff}.th .w{position:absolute;top:2px;right:4px}
.ev{max-height:260px;overflow:auto;font:11px/1.5 ui-monospace,Menlo,monospace}.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;color:var(--dim);margin-top:4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
details summary{cursor:pointer;color:var(--dim)}pre{white-space:pre-wrap;font-size:11px;margin:4px 0}
@media(max-width:1100px){main{grid-template-columns:1fr}.item,.wide{grid-column:auto;grid-row:auto}}
</style></head><body>
<header><h1>Sorter</h1><span id="runsel"></span><span id="pill" class="pill">…</span><span id="meta" class="meta"></span></header>
<section id="kpis" class="kpis"></section>
<main>
 <div class="card item" id="item"><div class="ph">waiting for items…</div></div>
 <div class="card"><h3>Score per item</h3><canvas id="c_scores"></canvas><div class="legend" id="lg_scores"></div></div>
 <div class="card"><h3>Score distribution</h3><canvas id="c_hist"></canvas><div class="legend" id="lg_hist"></div></div>
 <div class="card"><h3>Latency per item (ms)</h3><canvas id="c_lat"></canvas><div class="legend" id="lg_lat"></div></div>
 <div class="card"><h3>Stage latency</h3><div id="stages"></div></div>
 <div class="card"><h3>Accuracy</h3><div id="acc"></div></div>
 <div class="card"><h3>Run</h3><div id="runinfo"></div></div>
 <div class="card wide"><h3>Items</h3><div class="tabs" id="filters"></div><div class="gal" id="gal"></div></div>
 <div class="card wide"><h3>Events</h3><div class="ev" id="events"></div></div>
</main>
<script>
const SNAP = /*SNAPSHOT*/null;
const S = {runs: [], run: null, meta: {}, status: {}, items: [], events: [], sel: null, follow: true, view: 0,
           layer: 'overlay', filter: 'all', offset: 0, inflight: false, sig: ''};
const STAGES = ['pick', 'capture', 'detect', 'decide', 'recheck', 'place'];
const SCOL = {pick: '#7c8cff', capture: '#4fb3ff', detect: '#ff8a3d', decide: '#d16dff', recheck: '#f5a524', place: '#3ecf8e'};
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const fmt = (v, d = 1) => (v == null || Number.isNaN(Number(v))) ? '–' : Number(v).toFixed(d);
const file = p => SNAP ? p : `/files/${encodeURIComponent(S.run)}/${p}`;
const cfg = (k, d) => { const s = S.meta.settings || {}; return (k in s) ? s[k] : d; };
const THR = () => cfg('THRESHOLD', 0.5), MARGIN = () => cfg('UNCERTAIN_MARGIN', 0.1), DZ = () => cfg('DRIFT_Z', 4);
const AF = () => cfg('ACTION_FOR', {good: 'box', defect: 'reject'});
const ok = it => it.status !== 'error' && it.score != null;
const correct = it => (it.label && ok(it)) ? it.action === AF()[it.label] : null;
const uncertain = it => ok(it) && it.confidence < 0.75;
const dtype = it => { if (it.source) { const p = it.source.split(/[\\/]/); return p[p.length - 2]; } return it.label || '?'; };
function pct(arr, p) { if (!arr.length) return NaN; const a = [...arr].sort((x, y) => x - y); const i = (a.length - 1) * p;
  const lo = Math.floor(i), hi = Math.ceil(i); return a[lo] + (a[hi] - a[lo]) * (i - lo); }
function median(a) { return pct(a, 0.5); }
function acolor(a) { const af = AF(); if (a === af.defect) return '#ff5a5f'; if (a === af.good) return '#3ecf8e';
  const pal = ['#f5a524', '#7c8cff', '#d16dff', '#4fb3ff']; const acts = cfg('ACTIONS', []); return pal[Math.max(0, acts.indexOf(a)) % pal.length]; }

// ------------------------------------------------------------------ data
async function getJSON(u) { const r = await fetch(u, {cache: 'no-store'}); if (!r.ok) throw new Error(u + ' ' + r.status); return r.json(); }
async function loadRuns() {
  if (SNAP) return;
  try { S.runs = await getJSON('/api/runs'); } catch (e) { return; }
  const sel = S.runs.map(r => `<option value="${esc(r.id)}" ${r.id === S.run ? 'selected' : ''}>${esc(r.id)} · ${r.items} items · ${esc(r.state || '')}</option>`).join('');
  $('runsel').innerHTML = `<select id="rs">${sel}</select> <button id="newest" title="always jump to the newest run">newest</button>`;
  $('rs').onchange = e => selectRun(e.target.value, false);
  $('newest').onclick = () => { S.pinned = false; if (S.runs.length) selectRun(S.runs[0].id, false); };
  if (S.runs.length && (!S.run || (!S.pinned && S.runs[0].id !== S.run && S.runs[0].state === 'running'))) selectRun(S.runs[0].id, false);
}
function selectRun(id, pinned = true) {
  if (id === S.run) return; S.run = id; S.pinned = pinned || S.pinned; S.items = []; S.events = []; S.sel = null; S.follow = true; S.sig = ''; poll();
}
async function poll() {
  if (SNAP || !S.run || S.inflight) return; S.inflight = true;
  const id = S.run;
  try { const d = await getJSON(`/api/run?id=${encodeURIComponent(id)}&since=${S.items.length}`); if (id === S.run && d.since === S.items.length) apply(d); }
  catch (e) { /* server restarting: keep last view */ } finally { S.inflight = false; }
}
var apply = function (d) {
  S.meta = d.meta || {}; S.status = d.status || {}; S.events = d.events || []; S.items.push(...(d.items || []));
  S.offset = (d.now || Date.now() / 1000) - Date.now() / 1000;
  if (S.follow && S.items.length) S.sel = S.items.length - 1;
  render();
}

// ------------------------------------------------------------------ render
function render() {
  const sig = [S.run, S.items.length, S.sel, S.view, S.layer, S.filter, S.events.length, S.status.state].join('|');
  renderPill(); if (sig === S.sig) return; S.sig = sig;
  renderMeta(); renderKPIs(); renderItem(); drawScores(); drawHist(); drawLatency(); renderStages(); renderAcc(); renderRunInfo(); renderGallery(); renderEvents();
}
function renderPill() {
  const st = S.status || {}, now = Date.now() / 1000 + S.offset; let cls = st.state || '', txt = st.state || '…';
  if (SNAP) { $('pill').className = 'pill'; $('pill').textContent = 'report · ' + (st.state || ''); return; }
  if (st.state === 'running') {
    if (st.updated && now - st.updated > 15 && st.stage !== 'waiting for operator' && st.stage !== 'idle') { cls = 'stale'; txt = `no update for ${fmt(now - st.updated, 0)}s – process gone?`; }
    else { const a = st.stage_attrs || {}; const tgt = a.target ? ' → ' + a.target : '';
      txt = `item ${st.item ?? '–'} · ${st.stage || 'starting'}${tgt}` + (st.stage_started ? ` · ${fmt(now - st.stage_started, 1)}s` : ''); }
  }
  $('pill').className = 'pill ' + cls; $('pill').textContent = txt;
}
function renderMeta() {
  const m = S.meta, d = m.detector || {}, r = m.robot || {}, md = d.metadata || {};
  const model = (d.model_path || '').split(/[\\/]/).slice(-2, -1)[0] || '–';
  $('meta').innerHTML = [`<b>${esc(m.verb || '')}</b>`, esc(m.started || ''), `model <b>${esc(model)}</b> (${esc(md.model || '')} ${esc((md.image_size || []).join('×'))})`,
    `device <b>${esc(d.device || '–')}</b> ${esc(d.precision || '')}`, `threshold <b>${THR()}</b>`, `decider ${esc((m.decider || {}).type || '')}`,
    r.connected ? `robot ${esc(r.address)}${r.dry_run ? ' <span class=warn>(dry run)</span>' : ''}` : '<span class=dim>robot offline</span>',
    m.git && m.git.commit ? `git ${esc(m.git.commit)}${m.git.dirty ? '*' : ''}` : ''].filter(Boolean).join(' · ');
  if (SNAP) $('runsel').innerHTML = `<b>${esc(SNAP.id)}</b>`;
}
function kpi(v, l, cls = '') { return `<div class="kpi"><b class="${cls}">${v}</b><span>${l}</span></div>`; }
function renderKPIs() {
  const it = S.items, good = it.filter(ok), lab = good.filter(x => x.label), errs = it.length - good.length;
  const tot = good.map(x => x.ms.total), inf = good.flatMap(x => x.ms.infer_per_view || []);
  let h = kpi(it.length, `items${errs ? ` · <span class=err>${errs} errors</span>` : ''}`);
  if (lab.length) {
    const c = lab.filter(correct).length, d = lab.filter(x => x.label === 'defect'), g = lab.filter(x => x.label === 'good');
    const acc = c / lab.length;
    h += kpi(`${fmt(acc * 100)}%`, `accuracy (${c}/${lab.length})`, acc >= 0.99 ? 'ok' : acc >= 0.9 ? 'warn' : 'err');
    h += kpi(d.length ? `${fmt(d.filter(correct).length / d.length * 100)}%` : '–', 'defect recall');
    h += kpi(g.length ? `${fmt(g.filter(correct).length / g.length * 100)}%` : '–', 'good recall');
  } else {
    const counts = {}; good.forEach(x => counts[x.action] = (counts[x.action] || 0) + 1);
    h += kpi(Object.entries(counts).map(([a, n]) => `<span style="color:${acolor(a)}">${n}</span>`).join(' / ') || '–', Object.keys(counts).join(' / ') || 'actions');
  }
  h += kpi(fmt(median(tot), 0), `total p50 ms (p95 ${fmt(pct(tot, .95), 0)})`);
  h += kpi(fmt(median(inf), 1), `inference p50 ms (p95 ${fmt(pct(inf, .95), 1)})`);
  h += kpi(tot.length ? fmt(60000 / median(tot), 1) : '–', 'items / min');
  h += kpi(good.filter(uncertain).length, 'uncertain', good.filter(uncertain).length ? 'warn' : '');
  const w = it.filter(x => (x.warnings || []).length).length; h += kpi(w, 'items with frame warnings', w ? 'warn' : '');
  $('kpis').innerHTML = h;
}
function gauge(it, scores) {
  const t = THR(), m = MARGIN(), s = Math.min(1, Math.max(0, it.score));
  return `<div class="gauge" title="score ${fmt(it.score, 3)} · threshold ${t} · uncertain band ±${m}">
    <div class="band" style="left:${(t - m) * 100}%;width:${2 * m * 100}%"></div><div class="thr" style="left:${t * 100}%"></div>
    ${(scores.length > 1 ? scores : []).map(v => `<div class="vk" style="left:${Math.min(1, v) * 100}%"></div>`).join('')}
    <div class="mk" style="left:${s * 100}%"></div></div>`;
}
function renderItem() {
  const it = S.items[S.sel];
  if (!it) { $('item').innerHTML = '<div class="ph">waiting for items…</div>'; return; }
  const arts = it.artifacts || [], v = Math.min(S.view, Math.max(0, (it.views || 1) - 1)), art = arts[v] || {};
  const c = correct(it), d = (it.diag || [])[v] || {};
  const layers = ['overlay', 'frame', 'map'].filter(l => art[l]);
  const src = art[S.layer] || art.overlay;
  let h = `<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
    <div><button id="prev">◀</button> <b style="font-size:16px">#${it.item}</b> <button id="next">▶</button>
    <button id="follow" class="${S.follow ? 'on' : ''}">follow latest</button></div>
    <div>${ok(it) ? `<span class="act" style="background:${acolor(it.action)}">${esc(it.action)}</span>
      <b style="font-size:16px">${fmt(it.score, 3)}</b> conf ${fmt(it.confidence, 2)}` : '<span class="err"><b>ERROR</b></span>'}
      ${c === true ? '<span class=ok>✔ correct</span>' : c === false ? '<span class=err>✘ wrong</span>' : ''}
      ${it.label ? `<span class=dim>truth: ${esc(it.label)} (${esc(dtype(it))})</span>` : ''}</div></div>`;
  h += `<div class="tabs">${layers.map(l => `<button data-layer="${l}" class="${l === S.layer ? 'on' : ''}">${l}</button>`).join('')}
    ${(it.views || 1) > 1 ? (it.scores || []).map((s, i) => `<button data-view="${i}" class="${i === v ? 'on' : ''}">view ${i}: ${fmt(s, 3)}</button>`).join('') : ''}</div>`;
  h += `<div class="viewer">${src ? `<img src="${file(src)}" onerror="this.outerHTML='<div class=ph>image not available</div>'">` :
    `<div class="ph">${(S.meta.artifacts === false) ? 'images off for this run (--no-artifacts)' : 'no image'}</div>`}</div>`;
  if (ok(it)) h += gauge(it, it.scores || []);
  if (it.status === 'error') h += `<div class="err"><b>${esc(it.error)}</b></div><details><summary>traceback</summary><pre>${esc(it.traceback)}</pre></details>`;
  if ((it.warnings || []).length) h += `<div class="warn" style="margin:6px 0">${it.warnings.map(w => '⚠ ' + esc(w)).join('<br>')}</div>`;
  // facts
  const ms = it.ms || {};
  h += `<table style="margin-top:6px">
    <tr><th>reason</th><td>${esc(it.reason || '')}</td></tr>
    <tr><th>model / device</th><td>${esc(it.model)} · ${esc(it.device)}${it.dry_run ? ' · <span class=warn>dry run</span>' : ''}${it.grabbed === false ? ' · <span class=err>grab failed</span>' : ''}</td></tr>
    ${it.source ? `<tr><th>source</th><td>${esc(it.source.split(/[\\/]/).slice(-2).join('/'))}</td></tr>` : ''}
    <tr><th>view ${v} timing</th><td>preprocess ${fmt((ms.preprocess_per_view || [])[v])} · infer <b>${fmt((ms.infer_per_view || [])[v])}</b> · diagnostics ${fmt((ms.diag_per_view || [])[v])} ms · frame ${esc((d.frame_size || []).join('×'))}</td></tr>
    <tr><th>anomaly region</th><td>${d.map ? `peak ${fmt(d.map.peak, 3)} at (${(d.map.peak_xy || []).map(x => fmt(x, 2)).join(', ')}) · area over threshold <b>${fmt((d.map.area || 0) * 100, 1)}%</b>${d.map.bbox ? ' · box ' + d.map.bbox.map(x => fmt(x, 2)).join(',') : ' · none'}` : '–'}</td></tr></table>`;
  // frame quality vs training
  const ref = ((S.meta.detector || {}).metadata || {}).image_stats;
  if (d.stats) {
    h += `<table style="margin-top:6px"><tr><th>frame quality</th><th class=n>value</th><th class=n>training mean ± std</th><th class=n>z</th></tr>` +
      Object.entries(d.stats).map(([k, val]) => { const z = (d.z || {})[k], r = ref && ref[k];
        const cls = z == null ? '' : Math.abs(z) >= DZ() ? 'err' : Math.abs(z) >= DZ() / 2 ? 'warn' : 'ok';
        return `<tr><td>${k}</td><td class=n>${fmt(val, 3)}</td><td class=n>${r ? fmt(r.mean, 3) + ' ± ' + fmt(r.std, 3) : '<span class=dim>no reference</span>'}</td><td class="n ${cls}">${z == null ? '–' : fmt(z, 1)}</td></tr>`; }).join('') + '</table>';
  }
  // waterfall
  const sp = it.spans || [];
  if (sp.length) {
    const total = Math.max(...sp.map(s => s.start_ms + s.dur_ms), 1); let top = '';
    h += `<h3 style="margin:10px 0 4px">timeline · ${fmt(total, 1)} ms</h3><div class="wf">` + sp.map(s => {
      if (s.depth === 0) top = s.name; const col = SCOL[top] || '#888';
      const a = s.attrs || {}; const note = a.target || a.used || a.file || a.device || a.decider || '';
      return `<div class="r" title="${esc(JSON.stringify(a))}${s.error ? ' ERROR ' + esc(s.error) : ''}"><div class="lbl" style="padding-left:${s.depth * 14}px">${esc(s.name)} <span class=dim>${esc(note)}</span></div>
        <div class="trk"><div class="bar" style="left:${s.start_ms / total * 100}%;width:${Math.max(s.dur_ms / total * 100, 0.2)}%;background:${s.error ? 'var(--bad)' : col};opacity:${s.depth ? .65 : 1}"><span${(s.start_ms + s.dur_ms) / total > 0.8 ? (s.dur_ms / total > 0.1 ? ' class="in"' : ' class="lft"') : ''}>${fmt(s.dur_ms, s.dur_ms < 10 ? 1 : 0)}</span></div></div></div>`;
    }).join('') + '</div>';
  }
  $('item').innerHTML = h;
  $('prev').onclick = () => step(-1); $('next').onclick = () => step(1);
  $('follow').onclick = () => { S.follow = !S.follow; if (S.follow) S.sel = S.items.length - 1; render(); };
  document.querySelectorAll('[data-layer]').forEach(b => b.onclick = () => { S.layer = b.dataset.layer; render(); });
  document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => { S.view = +b.dataset.view; render(); });
}
function step(d) { if (!S.items.length) return; S.follow = false; S.sel = Math.max(0, Math.min(S.items.length - 1, (S.sel ?? 0) + d)); S.view = 0; render(); }
function selectItem(i) { S.follow = false; S.sel = i; S.view = 0; render(); }

// ------------------------------------------------------------------ charts
function cv(id, h) { const c = $(id), w = c.clientWidth || 300, r = window.devicePixelRatio || 1;
  c.width = w * r; c.height = h * r; c.style.height = h + 'px'; const g = c.getContext('2d'); g.setTransform(r, 0, 0, r, 0, 0);
  g.clearRect(0, 0, w, h); g.font = '10px sans-serif'; return [c, g, w, h]; }
function axisY(g, L, R, T, B, ticks, lab) { g.strokeStyle = '#262b36'; g.fillStyle = '#8a93a6'; g.lineWidth = 1;
  ticks.forEach(v => { const y = lab(v); g.beginPath(); g.moveTo(L, y); g.lineTo(R, y); g.stroke(); g.fillText(String(v), 2, y + 3); }); }
function drawScores() {
  const [c, g, w, h] = cv('c_scores', 210), L = 30, R = w - 8, T = 8, B = h - 18, it = S.items;
  const n = Math.max(it.length, 2), X = i => L + (R - L) * (i / (n - 1)), Y = v => B - (B - T) * Math.min(1.05, Math.max(0, v));
  axisY(g, L, R, T, B, [0, 0.25, 0.5, 0.75, 1], Y);
  const t = THR(), m = MARGIN(); g.fillStyle = 'rgba(255,216,77,.10)'; g.fillRect(L, Y(t + m), R - L, Y(t - m) - Y(t + m));
  g.strokeStyle = '#ffd84d'; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(L, Y(t)); g.lineTo(R, Y(t)); g.stroke(); g.setLineDash([]);
  const pts = [];
  it.forEach((x, i) => {
    if (!ok(x)) { g.fillStyle = '#888'; g.fillRect(X(i) - 3, B - 6, 6, 6); return; }
    if ((x.scores || []).length > 1) { g.fillStyle = 'rgba(255,255,255,.25)'; x.scores.forEach(s => g.fillRect(X(i) - 1, Y(s) - 1, 2, 2)); }
    const px = X(i), py = Y(x.score); pts.push([px, py, i]);
    g.fillStyle = acolor(x.action); g.beginPath(); g.arc(px, py, 3.5, 0, 7); g.fill();
    if (correct(x) === false) { g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(px - 5, py - 5); g.lineTo(px + 5, py + 5); g.moveTo(px + 5, py - 5); g.lineTo(px - 5, py + 5); g.stroke(); }
    if (i === S.sel) { g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.arc(px, py, 7, 0, 7); g.stroke(); }
  });
  g.fillStyle = '#8a93a6'; [0, Math.floor((it.length - 1) / 2), it.length - 1].filter(i => i >= 0).forEach(i => g.fillText('#' + (it[i] || {}).item, X(i) - 6, h - 4));
  const near = e => { const r = c.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top; let best = null, bd = 1e9;
    pts.forEach(p => { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; best = p[2]; } }); return bd < 400 ? best : null; };
  c.onclick = e => { const i = near(e); if (i != null) selectItem(i); };
  c.onmousemove = e => { const i = near(e); c.title = i == null ? '' : `#${it[i].item} score ${fmt(it[i].score, 3)} → ${it[i].action}${it[i].label ? ' (truth ' + it[i].label + ')' : ''}`; c.style.cursor = i == null ? '' : 'pointer'; };
  $('lg_scores').innerHTML = cfg('ACTIONS', []).map(a => `<span><i style="background:${acolor(a)}"></i>${esc(a)}</span>`).join('') +
    `<span><i style="background:#ffd84d"></i>threshold ${t}</span><span><i style="background:rgba(255,216,77,.3)"></i>uncertain band</span><span>✕ wrong</span>`;
}
function drawHist() {
  const [c, g, w, h] = cv('c_hist', 210), L = 30, R = w - 8, T = 8, B = h - 18, it = S.items.filter(ok), nb = 25;
  const lab = it.some(x => x.label);
  const groups = lab ? [['good', '#3ecf8e', x => x.label === 'good'], ['defect', '#ff5a5f', x => x.label === 'defect']]
                     : cfg('ACTIONS', []).map(a => [a, acolor(a), x => x.action === a]);
  const counts = groups.map(([, , f]) => { const b = new Array(nb).fill(0); it.filter(f).forEach(x => b[Math.min(nb - 1, Math.floor(Math.max(0, x.score) * nb))]++); return b; });
  const mx = Math.max(1, ...counts.flat()), X = v => L + (R - L) * v, Y = v => B - (B - T) * v / mx;
  axisY(g, L, R, T, B, [0, Math.ceil(mx / 2), mx], Y);
  const bw = (R - L) / nb / groups.length;
  counts.forEach((b, gi) => { g.fillStyle = groups[gi][1]; b.forEach((v, i) => { if (v) g.fillRect(X(i / nb) + gi * bw, Y(v), bw - 1, B - Y(v)); }); });
  g.strokeStyle = '#ffd84d'; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(X(THR()), T); g.lineTo(X(THR()), B); g.stroke(); g.setLineDash([]);
  g.fillStyle = '#8a93a6'; [0, 0.25, 0.5, 0.75, 1].forEach(v => g.fillText(v, X(v) - 6, h - 4));
  $('lg_hist').innerHTML = groups.map(([n, col]) => `<span><i style="background:${col}"></i>${lab ? 'truth: ' : ''}${esc(n)}</span>`).join('') +
    (lab ? (() => { const gs = it.filter(x => x.label === 'good').map(x => x.score), ds = it.filter(x => x.label === 'defect').map(x => x.score);
      return gs.length && ds.length ? `<span>max good ${fmt(Math.max(...gs), 3)} · min defect ${fmt(Math.min(...ds), 3)} · gap ${fmt(Math.min(...ds) - Math.max(...gs), 3)}</span>` : ''; })() : '');
}
function drawLatency() {
  const [c, g, w, h] = cv('c_lat', 210), L = 38, R = w - 8, T = 8, B = h - 18;
  const it = S.items.slice(-80), off = S.items.length - it.length;
  const tot = it.map(x => (x.ms || {}).total || 0), mx = Math.max(1, pct(tot, 0.98) * 1.1, ...tot.map(v => v * 0)), bw = (R - L) / Math.max(it.length, 1);
  const Y = v => B - (B - T) * Math.min(1, v / mx);
  const tick = Math.pow(10, Math.floor(Math.log10(mx))); axisY(g, L, R, T, B, [0, tick * Math.floor(mx / tick / 2), tick * Math.floor(mx / tick)].filter((v, i, a) => a.indexOf(v) === i), Y);
  it.forEach((x, i) => { let acc = 0; STAGES.forEach(s => { const v = (x.ms || {})[s]; if (!v) return; g.fillStyle = SCOL[s];
    g.fillRect(L + i * bw, Y(acc + v), Math.max(1, bw - 1), Y(acc) - Y(acc + v)); acc += v; });
    if (off + i === S.sel) { g.strokeStyle = '#fff'; g.strokeRect(L + i * bw - 0.5, T, bw, B - T); }
    if (x.status === 'error') { g.fillStyle = '#ff5a5f'; g.fillRect(L + i * bw, B + 2, bw - 1, 3); } });
  c.onclick = e => { const r = c.getBoundingClientRect(), i = Math.floor((e.clientX - r.left - L) / bw); if (i >= 0 && i < it.length) selectItem(off + i); };
  $('lg_lat').innerHTML = STAGES.map(s => `<span><i style="background:${SCOL[s]}"></i>${s}</span>`).join('') + `<span>last ${it.length} items</span>`;
}

// ------------------------------------------------------------------ tables
function renderStages() {
  const it = S.items.filter(ok), keys = ['pick', 'capture', 'detect', 'preprocess_per_view', 'infer_per_view', 'diag_per_view', 'decide', 'recheck', 'place', 'total'];
  const rows = keys.map(k => { const v = it.flatMap(x => { const m = (x.ms || {})[k]; return m == null ? [] : Array.isArray(m) ? m : [m]; });
    return v.length ? `<tr><td>${k.replace('_per_view', ' /view')}</td><td class=n>${v.length}</td><td class=n><b>${fmt(median(v))}</b></td><td class=n>${fmt(pct(v, .95))}</td><td class=n>${fmt(Math.max(...v))}</td></tr>` : ''; }).join('');
  $('stages').innerHTML = rows ? `<table><tr><th>stage</th><th class=n>n</th><th class=n>p50</th><th class=n>p95</th><th class=n>max</th></tr>${rows}</table>` : '<span class=dim>no items</span>';
}
function renderAcc() {
  const it = S.items.filter(x => ok(x) && x.label);
  if (!it.length) { $('acc').innerHTML = '<span class=dim>no ground truth in this run (live). Offline runs and <code>detect --image folder</code> carry labels; or use <code>evaluate.py --labels</code>.</span>'; return; }
  const af = AF(), acts = [af.good, af.defect, ...cfg('ACTIONS', []).filter(a => a !== af.good && a !== af.defect)];
  const cell = (l, a) => it.filter(x => x.label === l && x.action === a).length;
  let h = `<table><tr><th>truth \\ action</th>${acts.map(a => `<th class=n style="color:${acolor(a)}">${esc(a)}</th>`).join('')}</tr>` +
    ['good', 'defect'].map(l => `<tr><td>${l}</td>${acts.map(a => { const v = cell(l, a), right = af[l] === a; return `<td class="n ${v ? (right ? 'ok' : 'err') : 'dim'}"><b>${v}</b></td>`; }).join('')}</tr>`).join('') + '</table>';
  const types = {}; it.forEach(x => { const k = dtype(x); (types[k] = types[k] || []).push(x); });
  h += `<table style="margin-top:8px"><tr><th>type</th><th class=n>n</th><th class=n>correct</th><th class=n>median score</th><th class=n>min–max</th></tr>` +
    Object.entries(types).sort().map(([k, xs]) => { const c = xs.filter(correct).length, sc = xs.map(x => x.score);
      return `<tr><td>${esc(k)}</td><td class=n>${xs.length}</td><td class="n ${c === xs.length ? 'ok' : 'err'}">${fmt(c / xs.length * 100, 0)}%</td><td class=n>${fmt(median(sc), 3)}</td><td class=n>${fmt(Math.min(...sc), 2)}–${fmt(Math.max(...sc), 2)}</td></tr>`; }).join('') + '</table>';
  $('acc').innerHTML = h;
}
function renderRunInfo() {
  const m = S.meta, d = m.detector || {}, md = d.metadata || {}, v = m.versions || {}, dev = m.openvino_devices || {};
  const metr = md.metrics || {};
  $('runinfo').innerHTML = `<table>
    <tr><th>model</th><td>${esc(md.model || '')} · coreset ${esc(md.coreset_ratio ?? '–')} · ${esc(md.compression || '')} · trained ${esc(md.created || '')}</td></tr>
    <tr><th>test metrics</th><td>${Object.entries(metr).map(([k, x]) => `${esc(k)} ${fmt(x, 3)}`).join(' · ') || '–'}</td></tr>
    <tr><th>compile</th><td>${fmt(d.compile_ms, 0)} ms on ${esc(d.device)} · input ${esc((d.input_hw || []).join('×'))}</td></tr>
    <tr><th>OpenVINO devices</th><td>${Object.entries(dev).map(([k, x]) => `<b>${esc(k)}</b> ${esc(x)}`).join('<br>') || '–'}</td></tr>
    <tr><th>views / aggregate</th><td>${cfg('NUM_VIEWS', '–')} / ${esc(cfg('VIEW_AGGREGATE', ''))}</td></tr>
    <tr><th>versions</th><td>${Object.entries(v).map(([k, x]) => `${esc(k)} ${esc(x)}`).join(' · ')}</td></tr>
    <tr><th>command</th><td><code>${esc((m.argv || []).join(' '))}</code></td></tr></table>`;
}
function renderGallery() {
  const F = {all: () => true, wrong: x => correct(x) === false, uncertain: uncertain, warnings: x => (x.warnings || []).length, errors: x => x.status === 'error'};
  cfg('ACTIONS', []).forEach(a => F[a] = x => x.action === a);
  const cnt = k => S.items.filter(F[k]).length;
  $('filters').innerHTML = Object.keys(F).map(k => `<button data-f="${k}" class="${k === S.filter ? 'on' : ''}">${k} (${cnt(k)})</button>`).join('');
  document.querySelectorAll('[data-f]').forEach(b => b.onclick = () => { S.filter = b.dataset.f; render(); });
  const list = S.items.map((x, i) => [x, i]).filter(([x]) => (F[S.filter] || F.all)(x)).reverse().slice(0, 300);
  $('gal').innerHTML = list.map(([x, i]) => { const a = (x.artifacts || [])[0] || {};
    return `<div class="th ${correct(x) === false ? 'wrong' : ''} ${i === S.sel ? 'sel' : ''}" data-i="${i}">
      ${a.thumb ? `<img loading="lazy" src="${file(a.thumb)}">` : '<div style="aspect-ratio:1"></div>'}
      ${(x.warnings || []).length ? '<span class="w warn">⚠</span>' : ''}
      <div class="cap">#${x.item} ${ok(x) ? `<b style="color:${acolor(x.action)}">${fmt(x.score, 2)}</b> ${esc(x.action)}` : '<span class=err>error</span>'}</div></div>`; }).join('') || '<span class=dim>none</span>';
  document.querySelectorAll('.th').forEach(el => el.onclick = () => selectItem(+el.dataset.i));
}
function renderEvents() {
  $('events').innerHTML = [...S.events].reverse().slice(0, 200).map(e => `<div class="${e.level === 'error' ? 'err' : e.level === 'warning' ? 'warn' : ''}">${esc(e.ts)} ${esc(e.level)} ${esc(e.msg)}${e.traceback ? `<details><summary>traceback</summary><pre>${esc(e.traceback)}</pre></details>` : ''}</div>`).join('') || '<span class=dim>no events</span>';
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') step(-1); else if (e.key === 'ArrowRight') step(1);
  else if (e.key === 'f') { S.follow = !S.follow; if (S.follow) S.sel = S.items.length - 1; render(); }
  else if (e.key === 'l') { const ls = ['overlay', 'frame', 'map']; S.layer = ls[(ls.indexOf(S.layer) + 1) % 3]; render(); }
});
window.addEventListener('resize', () => { S.sig = ''; render(); });
// deep link: #item=12 opens that item (and stops following); #run=<id> picks a run
const HASH = Object.fromEntries(location.hash.slice(1).split('&').filter(Boolean).map(kv => kv.split('=').map(decodeURIComponent)));
const _apply = apply;
apply = function (d) { _apply(d); if (HASH.item && !HASH.done) { const i = S.items.findIndex(x => String(x.item) === HASH.item);
  if (i >= 0) { HASH.done = true; S.follow = false; S.sel = i; S.sig = ''; render(); } } };
if (HASH.run && !SNAP) { S.run = HASH.run; S.pinned = true; poll(); }
if (SNAP) { S.run = SNAP.id; apply(SNAP); }
else { loadRuns(); setInterval(loadRuns, 4000); setInterval(poll, 1000); setInterval(renderPill, 250); }
</script></body></html>
"""

if __name__ == "__main__":
    main()
