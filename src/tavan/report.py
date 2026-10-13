"""Self-contained HTML reports with inline SVG charts (no dependencies).

Produces a single ``.html`` file that opens in any browser — no JavaScript,
no plotting library — summarising a scenario set: quantile fan chart, loss
distribution, risk table and (if a reference is given) quality metrics.
"""

from __future__ import annotations

import datetime as _dt
import html
from typing import List, Optional, Sequence, Tuple

import numpy as np

_CSS = """
:root{--fg:#1a2733;--muted:#5b6b7a;--line:#e2e8f0;--accent:#08519c;--band:#9ecae1;--bg:#fff;--card:#f8fafc}
*{box-sizing:border-box}body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);
margin:0;background:var(--bg);line-height:1.5}.wrap{max-width:960px;margin:0 auto;padding:32px 20px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--muted);font-size:13px;margin-bottom:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card .k{font-size:12px;color:var(--muted)}.card .v{font-size:20px;font-weight:600;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
svg{max-width:100%;height:auto;background:var(--bg);border:1px solid var(--line);border-radius:10px}
.foot{color:var(--muted);font-size:12px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}
"""


def _svg_fan(scen, channel: int = 0, w: int = 900, h: int = 300, pad: int = 44) -> str:
    V = scen.values[:, :, channel]
    H = scen.horizon
    q = {p: np.quantile(V, p, axis=0) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}
    ymin, ymax = float(np.min([q[0.05]])), float(np.max([q[0.95]]))
    ymin, ymax = min(ymin, V.min()), max(ymax, V.max())
    rng = ymax - ymin or 1.0
    ymin -= 0.05 * rng
    ymax += 0.05 * rng

    def X(i):
        return pad + i * (w - 2 * pad) / max(H - 1, 1)

    def Y(v):
        return h - pad - (v - ymin) * (h - 2 * pad) / (ymax - ymin)

    def band(lo, hi):
        pts = [f"{X(i):.1f},{Y(lo[i]):.1f}" for i in range(H)] + \
              [f"{X(i):.1f},{Y(hi[i]):.1f}" for i in range(H - 1, -1, -1)]
        return "M" + "L".join(pts) + "Z"

    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    # grid + y labels
    for g in range(5):
        yv = ymin + g * (ymax - ymin) / 4
        y = Y(yv)
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w-pad}" y2="{y:.1f}" stroke="#eef2f6"/>')
        parts.append(f'<text x="{pad-6}" y="{y+4:.1f}" font-size="10" fill="#5b6b7a" text-anchor="end">{yv:.1f}</text>')
    parts.append(f'<path d="{band(q[0.05], q[0.95])}" fill="#c6dbef" opacity="0.7"/>')
    parts.append(f'<path d="{band(q[0.25], q[0.75])}" fill="#6baed6" opacity="0.6"/>')
    med = "M" + "L".join(f"{X(i):.1f},{Y(q[0.5][i]):.1f}" for i in range(H))
    parts.append(f'<path d="{med}" fill="none" stroke="#08306b" stroke-width="2"/>')
    # a few sample paths
    rg = np.random.default_rng(0)
    for i in rg.choice(len(V), min(15, len(V)), replace=False):
        p = "M" + "L".join(f"{X(j):.1f},{Y(V[i][j]):.1f}" for j in range(H))
        parts.append(f'<path d="{p}" fill="none" stroke="#2a6099" stroke-width="0.5" opacity="0.2"/>')
    parts.append(f'<text x="{w/2}" y="{h-8}" font-size="11" fill="#5b6b7a" text-anchor="middle">horizon step</text>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_hist(values, w: int = 900, h: int = 230, pad: int = 44, bins: int = 30,
              vlines: Optional[List[Tuple[float, str, str]]] = None) -> str:
    v = np.asarray(values, dtype=float)
    counts, edges = np.histogram(v, bins=bins)
    cmax = counts.max() or 1
    xmin, xmax = edges[0], edges[-1]
    rng = xmax - xmin or 1.0

    def X(val):
        return pad + (val - xmin) * (w - 2 * pad) / rng

    def Y(c):
        return h - pad - c * (h - 2 * pad) / cmax

    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    bw = (w - 2 * pad) / bins
    for i, c in enumerate(counts):
        x = pad + i * bw
        parts.append(f'<rect x="{x:.1f}" y="{Y(c):.1f}" width="{bw-1:.1f}" height="{h-pad-Y(c):.1f}" fill="#6baed6"/>')
    for val, color, label in (vlines or []):
        x = X(val)
        parts.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{h-pad}" stroke="{color}" stroke-width="1.5" stroke-dasharray="4"/>')
        parts.append(f'<text x="{x+3:.1f}" y="{pad+12}" font-size="10" fill="{color}">{html.escape(label)}</text>')
    parts.append(f'<text x="{w/2}" y="{h-8}" font-size="11" fill="#5b6b7a" text-anchor="middle">scenario loss</text>')
    parts.append("</svg>")
    return "".join(parts)


def _card(k: str, v: str) -> str:
    return f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>'


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def scenario_report(scen, path: str = "tavan_report.html", reference=None, title: str = "Scenario report",
                    loss="sum") -> str:
    """Write a self-contained HTML report for a :class:`ScenarioSet`."""
    losses = scen.losses(loss)
    rs = scen.risk_summary(loss=loss)
    cards = [
        _card("Scenarios", str(scen.n_scenarios)),
        _card("Horizon", str(scen.horizon)),
        _card("Channels", ", ".join(scen.names)),
        _card("Mean loss", f"{rs['mean']:.3g}"),
        _card("CVaR 95%", f"{rs['CVaR_0.95']:.3g}"),
        _card("VaR 95%", f"{rs['VaR_0.95']:.3g}"),
    ]
    risk_rows = [[f"{a:g}", f"{rs[f'VaR_{a:g}']:.4g}", f"{rs[f'CVaR_{a:g}']:.4g}"] for a in (0.9, 0.95, 0.99)]
    sections = [
        f"<h1>{html.escape(title)}</h1>",
        f'<div class="sub">Generated by tavan on {_dt.datetime.now():%Y-%m-%d %H:%M} · '
        f'model: {html.escape(str(scen.meta.get("model", "n/a")))}</div>',
        '<div class="grid">' + "".join(cards) + "</div>",
        "<h2>Scenario fan chart</h2>", _svg_fan(scen),
        "<h2>Loss distribution</h2>",
        _svg_hist(losses, vlines=[(rs["VaR_0.95"], "#d94801", "VaR95"), (rs["CVaR_0.95"], "#b30000", "CVaR95")]),
        "<h2>Risk measures</h2>", _table(["alpha", "VaR", "CVaR"], risk_rows),
    ]
    vr = scen.meta.get("violation_rates")
    if vr:
        sections += ["<h2>Constraint violations (raw samples)</h2>",
                     _table(["constraint", "violation rate"], [[k, f"{v:.1%}"] for k, v in vr.items()])]
    if reference is not None:
        try:
            m = scen.evaluate(reference)
            sections += ["<h2>Quality vs reference</h2>",
                         _table(["metric", "value"], [[k, f"{v:.4g}"] for k, v in m.items()])]
        except Exception:  # pragma: no cover
            pass
    body = "".join(sections)
    doc = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
           f"<body><div class='wrap'>{body}"
           f"<div class='foot'>tavan · lightweight time-series scenarios &amp; forecasting</div>"
           f"</div></body></html>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return path
