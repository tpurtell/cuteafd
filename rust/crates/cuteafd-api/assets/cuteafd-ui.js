// CuteAFD UI: the page shell, number formatting and SVG chart primitives shared
// by the engine's built-in pages (live console at /, benchmarks at /bench).
// Served at /assets/cuteafd-ui.js next to /assets/cuteafd-ui.css; compiled into
// the binary, no external resources. Exposes `window.CuteUI`.
(() => {
'use strict';
const root = getComputedStyle(document.documentElement);
const token = (name) => root.getPropertyValue('--' + name).trim();
// Palette tokens by name (`accepted`, `target`, `spark`, ...); unknown names pass through as CSS colors.
const color = (name) => (name && /^[a-z][a-z0-9-]*$/.test(name) && token(name)) || name || token('ink-2');

// ---------------------------------------------------------------- formatting
const nf0 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const fmt = {
  n0: (v) => (Number.isFinite(v) ? nf0.format(v) : '–'),
  n1: (v) => (Number.isFinite(v) ? nf1.format(v) : '–'),
  compact(v) {
    if (!Number.isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(2) + 'B';
    if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
    if (a >= 1e4) return (v / 1e3).toFixed(1) + 'k';
    return nf0.format(v);
  },
  bytes(v) {
    if (!Number.isFinite(v)) return '–';
    const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (i ? v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2) : v) + ' ' + u[i];
  },
  pct: (a, b) => (b > 0 ? (100 * a / b).toFixed(0) + '%' : '–'),
  // Microseconds as HTML with a small unit.
  us(v) {
    if (!Number.isFinite(v)) return '–';
    if (v >= 1e6) return `${(v / 1e6).toFixed(2)} <small>s</small>`;
    if (v >= 1000) return `${(v / 1000).toFixed(v >= 1e4 ? 1 : 2)} <small>ms</small>`;
    return `${v.toFixed(0)} <small>µs</small>`;
  },
  ms: (v) => (!Number.isFinite(v) ? '–' : v >= 1000 ? (v / 1000).toFixed(2) + ' s' : v.toFixed(0) + ' ms'),
  clock(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  },
  esc: (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])),
};

// ---------------------------------------------------------------- page shell
// Fills `el` (a <header>) with the logo (and the favicon when the page has
// none), a slot for page status, the header
// facts and the page navigation. Returns {status, meta, extra} slot elements.
// opts: {page: 'console' | 'bench', subtitle}.
const PAGES = [
  { id: 'console', href: '/', label: 'LIVE CONSOLE' },
  { id: 'usage', href: '/usage', label: 'USAGE' },
  { id: 'agent', href: '/agent', label: 'AGENT (EXPERIMENTAL)' },
  { id: 'bench', href: '/bench', label: 'BENCHMARK', primary: true },
];
function header(el, opts = {}) {
  const page = opts.page || 'console';
  if (!document.querySelector('link[rel="icon"]')) {
    const icon = document.createElement('link');
    Object.assign(icon, { rel: 'icon', type: 'image/svg+xml', href: '/assets/cuteafd-mark.svg' });
    document.head.appendChild(icon);
  }
  el.innerHTML = `<div class="brand"><a href="/" class="logo-link"><img class="logo" src="/assets/cuteafd-logo.svg" alt="cuteafd" width="95" height="26"></a><span>${fmt.esc(opts.subtitle || 'Engine console')}</span></div>
    <span class="slot-status"></span><div class="meta"></div>
    <nav class="nav" aria-label="Pages">${PAGES.map((p) => `<a href="${p.href}"${p.primary ? ' class="primary"' : ''}${p.id === page ? ' aria-current="page"' : ''}>${p.label}</a>`).join('')}</nav>
    <div class="header-extra"><span class="slot-extra"></span></div>`;
  return { status: el.querySelector('.slot-status'), meta: el.querySelector('.meta'), extra: el.querySelector('.slot-extra') };
}
// Header facts: [[key, valueHtml, title?]] -> `key <b>value</b>` spans.
function facts(el, items) {
  el.innerHTML = items.filter(([, v]) => v != null && v !== '').map(([k, v, title]) =>
    `<span${title ? ` title="${fmt.esc(title)}"` : ''}>${fmt.esc(k)} <b>${v}</b></span>`).join('');
}
// The running build from a `{release, commit, dirty}` object, as HTML.
function build(b) {
  if (!b) return 'unknown';
  const commit = b.commit ? fmt.esc(String(b.commit).slice(0, 12)) : 'unknown';
  const dirty = b.dirty ? '<span class="dirty">+dirty</span>' : '';
  return b.release ? `${fmt.esc(b.release)} · ${commit}${dirty}` : `dev · ${commit}${dirty}`;
}

// ---------------------------------------------------------------- SVG charts
let ids = 0;
const NS = 'http://www.w3.org/2000/svg';
function svgIn(el, cls, h) {
  const w = el.clientWidth || el.getBoundingClientRect().width;
  let svg = el.tagName === 'svg' ? el : el.querySelector(`svg.${cls}`);
  if (!svg) { svg = document.createElementNS(NS, 'svg'); svg.setAttribute('class', cls); el.appendChild(svg); }
  const height = h ?? (svg.clientHeight || 30);
  svg.setAttribute('viewBox', `0 0 ${Math.max(1, w)} ${height}`);
  svg.setAttribute('height', height);
  return { svg, w, h: height };
}
// A rate sparkline: newest value at the right edge, `points` slots wide, area
// fill fading to transparent, a dot on the newest value. `el` is an <svg> or
// its container. opts: {color, points = 60, height = 30, max}.
function sparkline(el, data, opts = {}) {
  const { svg, w, h } = svgIn(el, 'spark', opts.height ?? 30);
  if (!w) return;
  const c = color(opts.color);
  const finite = data.filter(Number.isFinite);
  if (finite.length < 2) { svg.innerHTML = ''; return; }
  const n = opts.points ?? 60, step = w / (n - 1);
  const max = (opts.max ?? Math.max(1e-9, ...finite)) * 1.1;
  const x = (i) => w - (data.length - 1 - i) * step, y = (v) => h - 1.5 - (v / max) * (h - 4);
  let line = '', first = null;
  data.forEach((v, i) => { if (!Number.isFinite(v)) return; line += `${first == null ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`; first ??= i; });
  const id = `cg${++ids}`, last = data[data.length - 1];
  svg.innerHTML = `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${c}" stop-opacity=".25"/><stop offset="1" stop-color="${c}" stop-opacity="0"/></linearGradient></defs>
    <path d="${line}L${w},${h}L${x(first).toFixed(1)},${h}Z" fill="url(#${id})"/>
    <path d="${line}" fill="none" stroke="${c}" stroke-width="1.5" stroke-linejoin="round" style="filter:drop-shadow(0 0 3px ${c})"/>
    ${Number.isFinite(last) ? `<circle cx="${w - 2}" cy="${y(last).toFixed(1)}" r="2.5" fill="${c}"/>` : ''}`;
}
// Vertical columns (a per-layer profile): one slot per value, a wide muted bar
// for `opts.median[i]` behind a narrow colored bar for the value; labels under
// slots `opts.label(i)` returns text for. opts: {colors: [i] -> color, median,
// height = 64, note}.
function columns(el, values, opts = {}) {
  const { svg, w, h } = svgIn(el, 'columns', opts.height ?? 64);
  if (!w) return;
  if (!values.length) {
    svg.innerHTML = `<text x="4" y="${h / 2}" fill="${token('muted')}" font-size="11" font-family="${fmt.esc(token('mono'))}">${fmt.esc(opts.empty || 'No data yet')}</text>`;
    return;
  }
  const med = opts.median || [];
  const max = Math.max(1, ...values.filter(Number.isFinite), ...med.filter(Number.isFinite));
  const bw = w / values.length, top = 12, base = h - 13;
  const muted = token('muted'), line2 = token('line-2'), mono = fmt.esc(token('mono'));
  let out = '';
  values.forEach((v, i) => {
    const x0 = i * bw, m = med[i];
    if (Number.isFinite(m)) { const mh = (base - top) * m / max; out += `<rect x="${(x0 + 1).toFixed(1)}" y="${(base - mh).toFixed(1)}" width="${Math.max(0, bw - 2).toFixed(1)}" height="${mh.toFixed(1)}" fill="${line2}"/>`; }
    if (Number.isFinite(v)) { const vh = Math.max(1, (base - top) * v / max); out += `<rect x="${(x0 + bw * .22).toFixed(1)}" y="${(base - vh).toFixed(1)}" width="${(bw * .56).toFixed(1)}" height="${vh.toFixed(1)}" fill="${color(opts.colors ? opts.colors(i) : 'target')}" opacity=".9"/>`; }
    const label = opts.label ? opts.label(i) : null;
    if (label != null) out += `<text x="${(x0 + bw / 2).toFixed(1)}" y="${h - 2}" fill="${muted}" font-size="9.5" text-anchor="middle" font-family="${mono}">${fmt.esc(label)}</text>`;
  });
  if (opts.note) out += `<text x="2" y="9" fill="${muted}" font-size="9.5" font-family="${mono}">${fmt.esc(opts.note(max))}</text>`;
  svg.innerHTML = out;
}
// Horizontal timeline rows: rows = [{label, segments: [{t0, t1, color, title}]}]
// over [opts.t0, opts.t1]. opts: {t0, t1, row = 18, gutter = 120}.
function timeline(el, rows, opts = {}) {
  const rowH = opts.row ?? 18, gutter = opts.gutter ?? 120;
  const { svg, w } = svgIn(el, 'timeline', rows.length * rowH + 4);
  if (!w) return;
  const t0 = opts.t0 ?? Math.min(...rows.flatMap((r) => r.segments.map((s) => s.t0))), t1 = opts.t1 ?? Math.max(...rows.flatMap((r) => r.segments.map((s) => s.t1)));
  const span = Math.max(1e-9, t1 - t0), X = (t) => gutter + (w - gutter - 2) * (t - t0) / span;
  const mono = fmt.esc(token('mono')), ink2 = token('ink-2');
  svg.innerHTML = rows.map((r, i) => {
    const y = 2 + i * rowH;
    return `<text x="0" y="${y + rowH / 2 + 3.5}" fill="${ink2}" font-size="11" font-family="${mono}">${fmt.esc(r.label)}</text>` +
      r.segments.map((s) => `<rect x="${X(s.t0).toFixed(1)}" y="${y + 2}" width="${Math.max(1, X(s.t1) - X(s.t0) - .5).toFixed(1)}" height="${rowH - 4}" rx="1.5" fill="${color(s.color)}">${s.title ? `<title>${fmt.esc(s.title)}</title>` : ''}</rect>`).join('');
  }).join('');
}
// A horizontal stacked meter as HTML: segments = [{value, color}] of `total`.
function meter(segments, total) {
  return `<div class="track">${segments.map((s) => `<span style="background:${color(s.color)};width:${(100 * Math.max(0, s.value) / Math.max(1e-9, total)).toFixed(2)}%"></span>`).join('')}</div>`;
}

// Usage charts share a tooltip and a keyboard/hover/click layer. Callers supply
// numeric data; labels always pass through escaping or textContent.
let chartTip;
function tooltip(text, event, target) {
  chartTip ||= Object.assign(document.body.appendChild(document.createElement('div')), { className: 'tip chart-tip', hidden: true });
  chartTip.textContent = text; chartTip.hidden = false;
  const box = target.getBoundingClientRect();
  chartTip.style.left = Math.max(8, Math.min(innerWidth - chartTip.offsetWidth - 8, event.clientX || box.left)) + 'px';
  chartTip.style.top = Math.max(8, Math.min(innerHeight - chartTip.offsetHeight - 8, (event.clientY || box.top) + 16)) + 'px';
}
function marks(svg, data, describe, click) {
  svg.setAttribute('role', 'group');
  svg.querySelectorAll('[data-i]').forEach((node) => {
    const i = Number(node.dataset.i), item = data[i];
    node.setAttribute('tabindex', '0'); node.setAttribute('role', click ? 'button' : 'img');
    node.setAttribute('aria-label', describe(item, i));
    const show = (e) => tooltip(describe(item, i), e, node);
    node.onpointermove = show; node.onfocus = show;
    node.onpointerleave = node.onblur = () => { if (chartTip) chartTip.hidden = true; };
    node.onclick = () => click?.(item, i);
    node.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); click?.(item, i); } };
  });
}
function chartText(x, y, text, anchor = 'start') {
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" fill="${token('ink-2')}" font-size="10" font-family="${fmt.esc(token('mono'))}">${fmt.esc(text)}</text>`;
}
function emptyChart(svg, h) { svg.innerHTML = chartText(8, h / 2, 'No retained data in this range'); }
// series = [{name, color, values: [number]}], times = epoch-ms positions.
// A brush calls onBrush([from,to]); clicking a series calls onClick(series,index).
function area(el, series, opts = {}) {
  const { svg, w, h } = svgIn(el, 'usage-area', opts.height ?? 160);
  const times = opts.times || series[0]?.values.map((_, i) => i) || [];
  if (!times.length || !series.length) { emptyChart(svg, h); return; }
  const left = 42, right = Math.max(left + 1, w - 8), top = 10, bottom = h - 23;
  const span = Math.max(1, times.at(-1) - times[0]), X = (t) => left + (right - left) * (t - times[0]) / span;
  const sums = times.map((_, i) => series.reduce((n, s) => n + Math.max(0, s.values[i] || 0), 0));
  const max = opts.max ?? Math.max(1, ...(opts.stacked === false ? series.flatMap((s) => s.values.filter(Number.isFinite)) : sums)), Y = (v) => bottom - v / max * (bottom - top);
  let out = [0, .5, 1].map((f) => `<path d="M${left},${Y(f * max)}H${right}" stroke="${token('line')}"/>` + chartText(left - 5, Y(f * max) + 3, fmt.compact(f * max), 'end')).join('');
  const floors = times.map(() => 0);
  series.forEach((s, j) => {
    const lower = floors.slice(), upper = floors.map((n, i) => n + Math.max(0, s.values[i] || 0));
    const path = opts.columns ? upper.map((n, i) => { const bw = (right - left) / times.length, x = left + i * bw, width = Math.max(1, bw - 2); return `M${x},${Y(n)}h${width}V${Y(lower[i])}H${x}Z`; }).join('') : upper.map((n, i) => `${i ? 'L' : 'M'}${X(times[i])},${Y(n)}`).join('') + lower.map((n, i) => [X(times[i]), Y(n)]).reverse().map(([x, y]) => `L${x},${y}`).join('') + 'Z';
    if (opts.stacked === false) {
      const line = s.values.map((v, i) => `${i ? 'L' : 'M'}${X(times[i])},${Y(v || 0)}`).join('');
      out += `<path d="${line}" fill="none" stroke="${fmt.esc(color(s.color))}" stroke-width="2" stroke-dasharray="${s.dash || (j % 2 ? '5 3' : '')}"/><path data-i="${j}" d="${line}" fill="none" stroke="transparent" stroke-width="14"/>`;
    } else {
      out += `<path data-i="${j}" d="${path}" fill="${fmt.esc(color(s.color))}" fill-opacity="${s.opacity ?? .7}" stroke="${token('panel')}" stroke-width="2"/>`;
      upper.forEach((n, i) => { floors[i] = n; });
    }
  });
  const label = opts.label || ((t) => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
  out += chartText(left, h - 4, label(times[0])) + chartText(right, h - 4, label(times.at(-1)), 'end');
  out += `<rect class="chart-brush" x="0" y="${top}" width="0" height="${bottom - top}" fill="${token('ink')}" opacity=".1" pointer-events="none"/><path class="chart-crosshair" d="" stroke="${token('ink-2')}" stroke-dasharray="3 3" pointer-events="none"/>`;
  svg.innerHTML = out;
  marks(svg, series, (s) => s.name, (s, i) => opts.onClick?.(s, i));
  const nearest = (e) => {
    const rect = svg.getBoundingClientRect(), x = Math.max(left, Math.min(right, (e.clientX - rect.left) * w / rect.width));
    if (opts.columns) return Math.min(times.length - 1, Math.floor((x - left) / (right - left) * times.length));
    const t = times[0] + (x - left) / (right - left) * span;
    return times.reduce((best, v, i) => Math.abs(v - t) < Math.abs(times[best] - t) ? i : best, 0);
  };
  let start = null, moved = false;
  svg.onpointermove = (e) => {
    const i = nearest(e), x = X(times[i]);
    svg.querySelector('.chart-crosshair').setAttribute('d', `M${x},${top}V${bottom}`);
    tooltip(`${label(times[i])}\n${series.map((s) => `${fmt.n1(s.values[i] || 0)}  ${s.name}`).join('\n')}`, e, svg);
    if (start != null) {
      moved ||= Math.abs(i - start) > 0;
      const brush = svg.querySelector('.chart-brush');
      brush.setAttribute('x', Math.min(x, X(times[start]))); brush.setAttribute('width', Math.abs(x - X(times[start])));
    }
  };
  svg.onpointerdown = (e) => { if (!opts.onBrush) return; start = nearest(e); moved = false; svg.setPointerCapture(e.pointerId); };
  svg.onpointerup = (e) => {
    if (start != null && moved) { const end = nearest(e); opts.onBrush?.([times[Math.min(start, end)], times[Math.max(start, end)]]); }
    start = null; svg.querySelector('.chart-brush')?.setAttribute('width', 0);
  };
  svg.onpointercancel = () => { start = null; };
  svg.onpointerleave = () => { if (chartTip) chartTip.hidden = true; svg.querySelector('.chart-crosshair').setAttribute('d', ''); };
  svg.addEventListener('click', (e) => { if (moved) { e.stopPropagation(); moved = false; } }, { capture: true });
}
// Backend bins are floor(log2(value)): bin 0 includes zero; bin 23 is open-ended.
function histogram(el, bins, opts = {}) {
  const { svg, w, h } = svgIn(el, 'usage-histogram', opts.height ?? 130);
  if (!bins?.length || !bins.some((n) => n > 0)) { emptyChart(svg, h); return; }
  const last = Math.max(1, ...bins.map((n, i) => n || opts.ghost?.[i] ? i : 0));
  const count = last + 1, left = 32, top = 22, bottom = h - 23, bw = Math.max(1, (w - left - 4) / count);
  const max = Math.max(1, ...bins, ...(opts.ghost || [])), Y = (n) => bottom - n / max * (bottom - top);
  let out = chartText(0, top, fmt.compact(max));
  bins.slice(0, count).forEach((n, i) => {
    const x = left + i * bw, ghost = opts.ghost?.[i];
    if (ghost) out += `<rect x="${x + 1}" y="${Y(ghost)}" width="${Math.max(1, bw - 2)}" height="${bottom - Y(ghost)}" fill="none" stroke="${token('ink-2')}" stroke-dasharray="3 2"/>`;
    out += `<rect data-i="${i}" x="${x + 1}" y="${Y(n)}" width="${Math.max(1, bw - 2)}" height="${Math.max(1, bottom - Y(n))}" rx="3" fill="${fmt.esc(color(opts.color || 'target'))}"/>`;
    if (i % Math.max(1, Math.ceil(count / 5)) === 0) out += chartText(x + bw / 2, h - 5, fmt.compact(2 ** i), 'middle');
  });
  [['p50', opts.p50], ['p95', opts.p95], ['p99', opts.p99]].forEach(([name, v], i) => {
    if (!Number.isFinite(v)) return;
    const x = Math.max(left, Math.min(w - 4, left + (Math.log2(Math.max(1, v)) + .5) * bw));
    out += `<path d="M${x},${top}V${bottom}" stroke="${token('ink-2')}" stroke-dasharray="2 3"/>`;
    out += chartText(left + i * (w - left) / 3, 12, `${name} ${fmt.n1(v)}`);
  });
  svg.innerHTML = out;
  marks(svg, bins, (n, i) => `${fmt.n0(n)} requests\n${i === 0 ? '0' : 2 ** i} - ${i === 23 ? 'infinity' : 2 ** (i + 1)} ${opts.unit || ''}`, (_, i) => opts.onClick?.({ from: i === 0 ? 0 : 2 ** i, to: i === 23 ? Infinity : 2 ** (i + 1), index: i }));
}
// rows = [{client, protocol, model, requests, tokens}]. Node widths and links
// share one normalization so incoming and outgoing ribbon mass is conserved.
function ribbons(el, rows, opts = {}) {
  const h = opts.height ?? 220, { svg, w } = svgIn(el, 'usage-ribbons', h);
  if (!rows?.length) { emptyChart(svg, h); return; }
  const fields = ['client', 'protocol', 'model'], metric = opts.metric || 'requests';
  const nodes = fields.map((field) => [...new Set(rows.map((r) => r[field] || 'unknown'))].sort().map((name) => ({ name, field, value: rows.filter((r) => (r[field] || 'unknown') === name).reduce((n, r) => n + (r[metric] || 0), 0) })));
  const total = Math.max(1, nodes[0].reduce((n, r) => n + r.value, 0)), maxN = Math.max(...nodes.map((n) => n.length));
  const scale = Math.max(1, h - 45 - maxN * 9) / total, xs = [8, w / 2 - 6, w - 20];
  nodes.forEach((list) => { let y = 30; list.forEach((n) => { n.y = y; n.h = n.value * scale; n.offset = 0; y += n.h + 9; }); });
  let out = fields.map((field, i) => chartText(xs[i], 12, field, i === 2 ? 'end' : 'start')).join('');
  for (let col = 0; col < 2; col++) {
    nodes[col].forEach((n) => { n.offset = 0; }); nodes[col + 1].forEach((n) => { n.offset = 0; });
    const links = new Map();
    rows.forEach((r) => { const a = r[fields[col]] || 'unknown', b = r[fields[col + 1]] || 'unknown', k = JSON.stringify([a, b, r.protocol]); const link = links.get(k) || { a, b, protocol: r.protocol, value: 0 }; link.value += r[metric] || 0; links.set(k, link); });
    for (const link of links.values()) {
      const a = nodes[col].find((n) => n.name === link.a), b = nodes[col + 1].find((n) => n.name === link.b), thick = link.value * scale;
      const y0 = a.y + a.offset, y1 = b.y + b.offset, x0 = xs[col] + 12, x1 = xs[col + 1], mid = (x0 + x1) / 2;
      out += `<path d="M${x0},${y0}C${mid},${y0} ${mid},${y1} ${x1},${y1}L${x1},${y1 + thick}C${mid},${y1 + thick} ${mid},${y0 + thick} ${x0},${y0 + thick}Z" fill="${fmt.esc(color(opts.color?.(link.protocol) || 'target'))}" opacity=".3" stroke="${token('panel')}" stroke-width="2"><title>${fmt.esc(`${link.a} -> ${link.b}: ${fmt.n0(link.value)} ${metric}`)}</title></path>`;
      a.offset += thick; b.offset += thick;
    }
  }
  const flat = nodes.flat();
  flat.forEach((n, i) => {
    const col = fields.indexOf(n.field), x = xs[col];
    out += `<rect data-i="${i}" x="${x}" y="${n.y}" width="12" height="${Math.max(4, n.h)}" rx="2" fill="${fmt.esc(color(n.field === 'protocol' ? opts.color?.(n.name) || 'target' : 'ink-2'))}"/>`;
    out += chartText(col === 2 ? x - 5 : x + 17, n.y + Math.max(10, n.h / 2 + 3), n.name.length > 22 ? n.name.slice(0, 21) + '...' : n.name, col === 2 ? 'end' : 'start');
  });
  svg.innerHTML = out; marks(svg, flat, (n) => `${n.name}: ${fmt.n0(n.value)} ${metric}`, (n) => opts.onClick?.(n.field, n.name));
}
// turns = [{tokens_in,tokens_cached,tokens_out,rid}]. Cached input uses exactly
// 38% of the series color; outputs remain target-colored regardless of split.
function strip(el, turns, opts = {}) {
  const { svg, w, h } = svgIn(el, 'usage-strip', opts.height ?? 24);
  if (!turns?.length) { emptyChart(svg, h); return; }
  const total = Math.max(1, turns.reduce((n, t) => n + (t.tokens_in || 0), 0)), gap = 2;
  const usable = Math.max(1, w - gap * turns.length), c = fmt.esc(color(opts.color || 'target'));
  let x = 0, out = '';
  turns.forEach((t, i) => {
    const bw = usable * (t.tokens_in || 0) / total, cached = Math.min(1, (t.tokens_cached || 0) / Math.max(1, t.tokens_in || 0));
    out += `<g data-i="${i}"><rect x="${x}" y="1" width="${bw}" height="${h - 7}" rx="2" fill="${c}"/><rect x="${x}" y="1" width="${bw * cached}" height="${h - 7}" fill="${token('well')}"/><rect x="${x}" y="1" width="${bw * cached}" height="${h - 7}" fill="${c}" opacity=".38"/><rect x="${x}" y="${h - 4}" width="${bw * Math.min(1, (t.tokens_out || 0) / Math.max(1, t.tokens_in || 0))}" height="3" fill="${token('target')}"/></g>`;
    x += bw + gap;
  });
  svg.innerHTML = out; marks(svg, turns, (t, i) => `Turn ${i + 1}\n${fmt.n0(t.tokens_in)} input / ${fmt.n0(t.tokens_cached)} cached / ${fmt.n0(t.tokens_out)} output`, (t, i) => opts.onClick?.(t, i));
}
// cells = [{day: 'YYYY-MM-DD', hour: 0..23, value}]. One sequential hue.
function heat(el, cells, opts = {}) {
  const days = [...new Set((cells || []).map((c) => c.day))].sort();
  const { svg, w, h } = svgIn(el, 'usage-heat', Math.max(50, days.length * 20 + 25));
  if (!days.length) { emptyChart(svg, h); return; }
  const left = Math.min(90, w / 3), cw = Math.max(1, (w - left) / 24), max = Math.max(1, ...cells.map((c) => c.value));
  let out = days.map((d, i) => chartText(0, 36 + i * 20, d)).join('');
  for (let hour = 0; hour < 24; hour += 4) out += chartText(left + hour * cw, 12, String(hour).padStart(2, '0'));
  days.forEach((day, row) => {
    for (let hour = 0; hour < 24; hour++) {
      const index = cells.findIndex((c) => c.day === day && c.hour === hour), value = cells[index]?.value || 0;
      out += `<rect${index >= 0 ? ` data-i="${index}"` : ''} x="${left + hour * cw}" y="${22 + row * 20}" width="${Math.max(1, cw - 2)}" height="18" rx="2" fill="${fmt.esc(color(opts.color || 'target'))}" opacity="${.08 + .92 * value / max}"/>`;
    }
  });
  svg.innerHTML = out; marks(svg, cells, (c) => `${c.day} ${String(c.hour).padStart(2, '0')}:00\n${fmt.n0(c.value)} ${opts.unit || 'requests'}`, (c) => opts.onClick?.(c));
}

window.CuteUI = { token, color, fmt, header, facts, build, area, histogram, ribbons, strip, heat, svg: { sparkline, columns, timeline, meter, area, histogram, ribbons, strip, heat } };
})();
