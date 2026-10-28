(() => {
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const MOCK = new URLSearchParams(location.search).has('mock');
let cfg = {}, strategy = null, fg = null /* foreground job id */, jobs = [], seenDone = new Set();

// ---------- mock backend (?mock=1) ----------
if (MOCK) {
  const real = window.fetch, mj = {}; let n = 0;
  const mkBt = () => { const nav = [], dd = [], reg = []; let s = 1, b = 1, pk = 1; const labels = ['risk_on','cautious','defensive','risk_off'];
    for (let i = 0; i < 300; i++) { const d = new Date(2024, 0, 1 + i).toISOString().slice(0, 10);
      s *= 1 + (Math.sin(i / 9) * .004 + .0012 + (Math.random() - .5) * .006); b *= 1 + (Math.random() - .5) * .007 + .0005; pk = Math.max(pk, s);
      nav.push({date: d, strategy: s * 1e5, benchmark: b * 1e5}); dd.push({date: d, value: s / pk - 1}); reg.push({date: d, label: labels[Math.floor(i / 40) % 4]}); }
    const rts = Array.from({length: 25}, (_, i) => { const bp = 100 + i, pn = (Math.random() - .4) * .2; return {stock_id: ['AAPL','NVDA','005930','MSFT'][i % 4], buy_date: `2024-0${i % 4 + 1}-0${i % 9 + 1}`, sell_date: `2024-0${i % 4 + 5}-1${i % 9}`, shares: 10 + i, buy_price: bp, sell_price: bp * (1 + pn), pnl_pct: pn, holding_days: 20 + i}; });
    return {stats: {total_return: .31, cagr: .24, sharpe_ratio: 1.42, sortino_ratio: 1.9, max_drawdown: -.11, annualized_volatility: .15, win_rate: .58, total_trades: 84, beta: .8, alpha: .05},
      bench_stats: {total_return: .18, cagr: .14, sharpe_ratio: .9, sortino_ratio: 1.1, max_drawdown: -.17, annualized_volatility: .16, win_rate: .5, total_trades: 0, beta: 1, alpha: 0},
      nav, drawdown: dd, regimes: reg, warning: 'Mock data: price coverage for 3 tickers starts after the requested start date.',
      round_trips: rts, trades: rts.flatMap(t => [{date: t.buy_date, stock_id: t.stock_id, action: 'BUY', shares: t.shares, price: t.buy_price, exec_price: t.buy_price, gross_value: t.shares * t.buy_price}, {date: t.sell_date, stock_id: t.stock_id, action: 'SELL', shares: t.shares, price: t.sell_price, exec_price: t.sell_price, gross_value: t.shares * t.sell_price}]).sort((x, y) => x.date < y.date ? -1 : 1)}; };
  const res = o => Promise.resolve(new Response(JSON.stringify(o), {headers: {'Content-Type': 'application/json'}}));
  const strat = {id: 's1', name: 'Balanced Momentum', rationale: 'Mock rationale.', top_n: 8, rebalance: 'biweekly', equity_by_regime: {risk_on: 1, cautious: .7, defensive: .4, risk_off: .1}, sell_rules: {stop_loss_pct: 8}, rebalance_triggers: [], risk_notes: 'mock'};
  window.fetch = (u, o = {}) => { const m = (o.method || 'GET').toUpperCase(), p = u.split('?')[0];
    if (!p.startsWith('/api/')) return real(u, o);
    if (p === '/api/config') return res({ai_enabled: true, models: ['claude-sonnet-5-5'], default_model: 'claude-sonnet-5-5', currencies: {USD: 1, KRW: 1380}, benchmarks: ['SPY', 'QQQ', 'KOSPI'], data_end: '2025-10-24', tickers: []});
    if (p === '/api/backtest') return new Promise(r => setTimeout(() => r(res(mkBt())), 400)).then(x => x);
    if (p === '/api/strategies') return res(m === 'GET' ? [strat, {...strat, id: 's2', name: 'Defensive Quality', top_n: 15, rebalance: 'monthly'}] : strat);
    if (p === '/api/analyze') { const id = 'j' + ++n; mj[id] = {id, kind: JSON.parse(o.body).mode, status: 'running', created: Date.now(), progress: 0, title: 'AI ' + JSON.parse(o.body).mode, t: Date.now()}; return res({job_id: id}); }
    if (p === '/api/jobs') return res(Object.values(mj).map(j => upd(j)));
    let mm = p.match(/^\/api\/jobs\/(\w+)(\/cancel)?$/);
    if (mm) { const j = mj[mm[1]]; if (mm[2]) { j.status = 'cancelled'; return res({}); } upd(j);
      if (j.status === 'done') j.result = {strategy: strat, report_markdown: '## Mock report\n\n- **Bold** point <script>alert(1)</script>\n- `code`\n\nRegime looks *cautious*.', backtest: mkBt()}; return res(j); }
    return res({}); };
  function upd(j) { if (j.status === 'running') { const e = (Date.now() - j.t) / 8000; j.progress = Math.min(e, 1); if (e >= 1) j.status = 'done'; } return j; }
}

// ---------- api ----------
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  if (!r.ok) { let t = await r.text(); try { t = JSON.parse(t).detail || t; } catch {} throw new Error(typeof t === 'string' ? t : JSON.stringify(t)); }
  return r.json();
}
const showErr = m => { $('err').hidden = !m; $('err').textContent = m || ''; };

// ---------- formatting ----------
const PCT = /return|cagr|drawdown|volatility|rate|alpha|excess|_pct|turnover/;
const fmt = (k, v) => v == null || isNaN(v) ? '–' : PCT.test(k) ? (v * 100).toFixed(2) + '%' : Number.isInteger(v) ? v : (+v).toFixed(2);
const label = k => k.replace(/_/g, ' ');
const arrow = v => v > 0 ? '▲' : v < 0 ? '▼' : '▬';
const cls = v => v > 0 ? 'up' : v < 0 ? 'down' : '';
const LOWER_BETTER = /drawdown|volatility|kurtosis/;  // drawdown stored negative: higher (closer to 0) is better, so only vol/kurt flip
const flip = k => /volatility|kurtosis/.test(k);

function kpis(s) {
  const keys = ['total_return','cagr','sharpe_ratio','sortino_ratio','max_drawdown','annualized_volatility','win_rate','total_trades'].filter(k => k in s);
  $('kpis').innerHTML = keys.map(k => { const v = s[k], c = /ratio|return|cagr|alpha|rate/.test(k) || /drawdown/.test(k) ? cls(v) : '';
    return `<div class="kpi"><div class="l">${esc(label(k))}</div><div class="v ${c}">${c ? arrow(v) + ' ' : ''}${fmt(k, v)}</div></div>`; }).join('');
}
function compare(s, b) {
  const keys = Object.keys(s).filter(k => typeof s[k] === 'number' && !/^benchmark_/.test(k));
  $('cmp').innerHTML = '<tr><th>Metric</th><th>Strategy</th><th>Benchmark</th><th>Difference</th></tr>' + keys.map(k => {
    const d = (s[k] ?? 0) - (b?.[k] ?? 0), g = flip(k) ? -d : d, has = b && k in b;
    return `<tr><td>${esc(label(k))}</td><td>${fmt(k, s[k])}</td><td>${has ? fmt(k, b[k]) : '–'}</td><td>${has ? `<span class="${cls(+g.toFixed(6))}">${arrow(+g.toFixed(6))} ${fmt(k, Math.abs(d))}</span>` : '–'}</td></tr>`; }).join('');
}
const RC = {risk_on: '#34d399', cautious: '#fbbf24', defensive: '#fb923c', risk_off: '#f87171'};
function chart(r) {
  const t = getComputedStyle(document.documentElement), col = n => t.getPropertyValue(n).trim() || '#888';
  const x = r.nav.map(p => p.date), shapes = []; let st = null;
  (r.regimes || []).forEach((g, i, a) => { if (!st || g.label !== st.label) { if (st) shapes.push(box(st, g.date)); st = g; } if (i === a.length - 1) shapes.push(box(st, g.date)); });
  function box(s, e) { return {type: 'rect', xref: 'x', yref: 'paper', x0: s.date, x1: e, y0: 0, y1: 1, fillcolor: RC[s.label] || '#888', opacity: .09, line: {width: 0}, layer: 'below'}; }
  const ax = {gridcolor: col('--border'), zerolinecolor: col('--border'), color: col('--muted')};
  Plotly.newPlot('chart', [
    {x, y: r.nav.map(p => p.strategy), name: 'Strategy', line: {color: col('--accent'), width: 2}},
    {x, y: r.nav.map(p => p.benchmark), name: 'Benchmark', line: {color: col('--muted'), width: 1.5, dash: 'dot'}},
    {x: r.drawdown.map(p => p.date), y: r.drawdown.map(p => p.value * 100), name: 'Drawdown %', yaxis: 'y2', fill: 'tozeroy', line: {color: col('--down'), width: 1}, fillcolor: 'rgba(248,113,113,.2)'}],
    {paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: {color: col('--text')}, margin: {l: 60, r: 20, t: 10, b: 30}, shapes, showlegend: true, legend: {orientation: 'h', y: 1.08},
      xaxis: {...ax, anchor: 'y2'}, yaxis: {...ax, domain: [.32, 1], title: 'NAV'}, yaxis2: {...ax, domain: [0, .26], title: 'DD %'}, hovermode: 'x unified', hoverlabel: {bgcolor: '#fff', bordercolor: '#cbd5e1', font: {color: '#000'}}},
    {responsive: true, displaylogo: false});
  const seen = [...new Set((r.regimes || []).map(g => g.label))];
  $('legend').innerHTML = seen.map(l => `<span><i style="background:${RC[l] || '#888'}"></i>${esc(l)}</span>`).join('');
}
function trades(rows) {
  if (!rows?.length) { $('trades').innerHTML = '<tr><td class="muted">No trades</td></tr>'; return; }
  const ks = Object.keys(rows[0]);
  $('trades').innerHTML = '<tr>' + ks.map(k => `<th>${esc(label(k))}</th>`).join('') + '</tr>' + rows.slice(0, 500).map(r => '<tr>' + ks.map(k => {
    const v = r[k]; return `<td>${typeof v === 'number' ? (/pct/.test(k) ? `<span class="${cls(v)}">${arrow(v)} ${(v * (Math.abs(v) < 5 ? 100 : 1)).toFixed(2)}%</span>` : +v.toFixed(4)) : esc(v)}</td>`; }).join('') + '</tr>').join('');
}
// ---------- trades: filters + execution / round-trip view ----------
let tData = {exec: [], rt: []}, tRound = false;
const dparts = d => String(d || '').split('-');  // YYYY-MM-DD
function fillSel(id, all, vals) { const el = $(id), cur = el.value; el.innerHTML = `<option value="">${all}</option>` + vals.map(v => `<option>${esc(v)}</option>`).join(''); if (vals.map(String).includes(cur)) el.value = cur; }
function setTrades(ex, rt) {
  tData = {exec: ex || [], rt: rt || []};
  const rows = tData.exec, dates = rows.map(r => r.date);
  fillSel('fStock', 'All stocks', [...new Set(rows.map(r => r.stock_id))].sort());
  fillSel('fYear', 'All years', [...new Set(dates.map(d => dparts(d)[0]))].sort());
  fillSel('fMonth', 'All months', Array.from({length: 12}, (_, i) => String(i + 1).padStart(2, '0')));
  fillSel('fDay', 'All days', Array.from({length: 31}, (_, i) => String(i + 1).padStart(2, '0')));
  renderTrades();
}
function renderTrades() {
  const src = tRound ? tData.rt : tData.exec, f = {s: $('fStock').value, a: $('fAction').value, y: $('fYear').value, m: $('fMonth').value, d: $('fDay').value};
  const dateOk = ds => ds.some(x => { const [y, m, d] = dparts(x); return (!f.y || y === f.y) && (!f.m || m === f.m) && (!f.d || d === f.d); });
  const out = src.filter(r => (!f.s || r.stock_id === f.s) && (tRound || !f.a || r.action === f.a)
    && dateOk(tRound ? [r.buy_date, r.sell_date] : [r.date]));
  $('fAction').disabled = tRound; $('tView').textContent = tRound ? 'Show individual executions' : 'Show buy→sell round trips';
  $('tCount').textContent = `${out.length} / ${src.length} rows` + (out.length > 500 ? ' (first 500 shown)' : '');
  trades(out);
}
['fStock', 'fAction', 'fYear', 'fMonth', 'fDay'].forEach(i => $(i).addEventListener('change', renderTrades));
$('fReset').onclick = () => { ['fStock', 'fAction', 'fYear', 'fMonth', 'fDay'].forEach(i => $(i).value = ''); renderTrades(); };
$('tView').onclick = () => { tRound = !tRound; renderTrades(); };
function showBacktest(r) {
  $('warn').hidden = !r.warning; $('warn').textContent = r.warning ? '⚠ ' + r.warning : '';
  kpis(r.stats); compare(r.stats, r.bench_stats); chart(r); setTrades(r.trades, r.round_trips);
}

// ---------- markdown (escape first, then a few safe transforms) ----------
function md(src) {
  const h = esc(src).split('\n'); let out = '', inList = false;
  const inl = s => s.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/\*([^*]+)\*/g, '<i>$1</i>');
  for (const l of h) {
    const li = l.match(/^\s*[-*] (.*)/), hd = l.match(/^(#{1,3}) (.*)/);
    if (li) { if (!inList) { out += '<ul>'; inList = true; } out += `<li>${inl(li[1])}</li>`; continue; }
    if (inList) { out += '</ul>'; inList = false; }
    out += hd ? `<h${hd[1].length}>${inl(hd[2])}</h${hd[1].length}>` : l.trim() ? `<p>${inl(l)}</p>` : '';
  }
  return out + (inList ? '</ul>' : '');
}
function showResult(res) {
  if (res.strategy) { strategy = res.strategy; applyStrategy(res.strategy, true); }
  $('report').className = 'md'; $('report').innerHTML = (res.strategy ? `<h2>${esc(res.strategy.name)}</h2>` : '') + md(res.report_markdown || '');
  if (res.backtest) showBacktest(res.backtest);
  $('report').scrollIntoView({behavior: 'smooth'});
}

// ---------- inputs ----------
const params = () => ({start: $('start').value, end: $('end').value, rebalance: $('rebalance').value,
  ...($('rebalance').value === 'custom' ? {rebalance_days: +$('rebalance_days').value} : {}),
  top_n: +$('top_n').value, capital: +$('capital').value, currency: $('currency').value, fx: +$('fx').value,
  benchmark: $('benchmark').value, slippage_pct: +$('slippage_pct').value, commission_pct: +$('commission_pct').value});
function applyStrategy(s, quiet) {
  $('top_n').value = s.top_n; $('rebalance').value = s.rebalance || 'monthly';
  if (s.rebalance_days) $('rebalance_days').value = s.rebalance_days; sync();
  strategy = s; if (!quiet) showErr('');
}
const sync = () => { $('daysWrap').hidden = $('rebalance').value !== 'custom'; };
async function run() {
  showErr(''); $('runBtn').disabled = true; $('runBtn').textContent = 'Running…';
  try { showBacktest(await api('/api/backtest', {...params(), ...(strategy ? {strategy} : {})})); }
  catch (e) { showErr('Backtest failed: ' + e.message); }
  $('runBtn').disabled = false; $('runBtn').textContent = 'Run backtest';
}

// ---------- AI jobs ----------
async function startAI(mode) {
  showErr('');
  try {
    const {job_id} = await api('/api/analyze', {model: $('model').value || undefined, mode, params: params()});
    fg = job_id; $('ovTitle').textContent = mode === 'full_ai' ? 'Complete AI Mode running…' : 'AI Recommend running…'; $('overlay').hidden = false; poll();
  } catch (e) { showErr('AI request failed: ' + e.message); }
}
async function poll() {
  try { jobs = await api('/api/jobs'); } catch { return; }
  renderJobs();
  if (fg) { const j = jobs.find(x => x.id === fg);
    if (j) { $('ovProg').textContent = j.progress != null ? Math.round((j.progress <= 1 ? j.progress * 100 : j.progress)) + '%' : j.status;
      if (['done', 'error', 'cancelled'].includes(j.status)) { const id = fg; fg = null; $('overlay').hidden = true; seenDone.add(id); await openJob(id); } } }
}
async function openJob(id) {
  try { const j = await api('/api/jobs/' + id);
    if (j.status === 'error') return showErr('Job failed: ' + (j.error || j.result?.error || 'unknown error'));
    if (j.result) { showResult(j.result); $('tray').hidden = true; } } catch (e) { showErr(e.message); }
}
function renderJobs() {
  const act = jobs.filter(j => ['queued', 'running'].includes(j.status)).length, fin = jobs.filter(j => j.status === 'done' && !seenDone.has(j.id)).length;
  $('jobsBadge').hidden = !(act + fin); $('jobsBadge').textContent = act + fin;
  $('tray').innerHTML = jobs.length ? [...jobs].reverse().map(j => `<div class="job"><span class="t" title="${esc(j.title)}">${esc(j.title || j.kind)}</span><span class="st">${esc(j.status)}</span>
    ${j.status === 'done' ? `<button data-open="${esc(j.id)}">Open</button>` : ['queued', 'running'].includes(j.status) ? `<button data-cancel="${esc(j.id)}">Cancel</button>` : ''}</div>`).join('') : '<div class="muted small" style="padding:8px">No jobs yet.</div>';
}

// ---------- init ----------
async function init() {
  $('rebalance').onchange = sync;
  $('currency').onchange = () => { $('fx').value = cfg.currencies?.[$('currency').value] ?? 1; };
  $('runBtn').onclick = run; $('menuBtn').onclick = () => $('side').classList.toggle('open');
  $('recBtn').onclick = () => startAI('recommend'); $('fullBtn').onclick = () => startAI('full_ai');
  $('bgBtn').onclick = () => { fg = null; $('overlay').hidden = true; $('tray').hidden = false; };
  $('ovCancel').onclick = async () => { if (fg) { await api('/api/jobs/' + fg + '/cancel', {}).catch(() => {}); } };
  $('jobsBtn').onclick = () => { $('tray').hidden = !$('tray').hidden; };
  $('tray').onclick = async e => { const o = e.target.dataset.open, c = e.target.dataset.cancel;
    if (o) { seenDone.add(o); renderJobs(); openJob(o); } if (c) { await api('/api/jobs/' + c + '/cancel', {}).catch(() => {}); poll(); } };
  try { cfg = await api('/api/config'); } catch (e) { showErr('Cannot reach backend (/api/config): ' + e.message + '. Append ?mock=1 for demo mode.'); cfg = {currencies: {USD: 1}, benchmarks: [], models: []}; }
  $('currency').innerHTML = Object.keys(cfg.currencies).map(c => `<option>${esc(c)}</option>`).join('');
  $('fx').value = cfg.currencies[$('currency').value] ?? 1;
  $('benchmark').innerHTML = (cfg.benchmarks || []).map(b => `<option>${esc(b)}</option>`).join('');
  $('model').innerHTML = (cfg.models || []).map(m => `<option ${m === cfg.default_model ? 'selected' : ''}>${esc(m)}</option>`).join('');
  $('model').hidden = !(cfg.models || []).length;
  if (cfg.data_end) { $('end').value = cfg.data_end; const d = new Date(cfg.data_end); d.setFullYear(d.getFullYear() - 2); $('start').value = d.toISOString().slice(0, 10); $('dataEnd').textContent = 'Data through ' + cfg.data_end; }
  if (!cfg.ai_enabled) { $('recBtn').disabled = $('fullBtn').disabled = true; $('aiHint').textContent = 'AI disabled: set ANTHROPIC_API_KEY in the backend environment and restart the server.'; $('aiHint').style.color = 'var(--warn)'; }
  try { const ss = await api('/api/strategies'); $('templates').innerHTML = ss.length ? '' : '<div class="muted small">No saved strategies yet. AI Recommend creates them.</div>';
    ss.forEach(s => { const d = document.createElement('div'); d.className = 'card';
      d.innerHTML = `<b>${esc(s.name)}</b><p>${esc(s.rationale || '')}</p><span class="muted small">Top ${esc(s.top_n)} · ${esc(s.rebalance)}</span> `;
      const b = document.createElement('button'); b.textContent = 'Apply'; b.onclick = () => applyStrategy(s); d.appendChild(b); $('templates').appendChild(d); });
  } catch { $('templates').innerHTML = '<div class="muted small">Strategies unavailable.</div>'; }
  sync(); renderJobs(); setInterval(poll, 2000); poll();
  if (cfg.data_end || MOCK) run();
}
init();
})();
