import { useState, type ReactNode } from "react";
import { ChartFrame, XAxis, YGrid, extent, linear, rampFill, divergingFill, useCellInk, useTip } from "./core";

// ================================================================ ForestPlot
export interface ForestRow { id: string; label: string; sub?: string; value: number; lo: number | null; hi: number | null; tier: number }

/** Point estimates with 95% CIs, one row per entry; tier bands shaded. */
export function ForestPlot({ rows, format, label }: { rows: ForestRow[]; format: (v: number) => string; label: string }) {
  const { tip, show, hide } = useTip();
  const rowH = 28;
  const m = { l: 190, r: 70, t: 8, b: 36 };
  const height = m.t + m.b + rows.length * rowH;
  return (
    <ChartFrame height={height} tip={tip} label={label}>
      {(width) => {
        const vals = rows.flatMap((r) => [r.value, r.lo ?? r.value, r.hi ?? r.value]);
        const x = linear(extent(vals, 0.08), [m.l, width - m.r]);
        return (
          <>
            <g className="grid">
              {x.ticks(Math.max(3, Math.floor((width - m.l) / 90))).map((t) => (
                <line key={t} x1={x(t)} x2={x(t)} y1={m.t} y2={height - m.b} />
              ))}
            </g>
            {rows.map((r, i) => {
              const cy = m.t + i * rowH + rowH / 2;
              const first = i === 0 || rows[i - 1].tier !== r.tier;
              return (
                <g key={r.id} onMouseEnter={() => show(x(r.value), cy - 6, (
                      <><div className="tt-title">{r.label}</div>
                        <div className="tt-row">Score <b>{format(r.value)}</b></div>
                        {r.lo != null && <div className="tt-row">95% CI <b>{format(r.lo)} – {format(r.hi!)}</b></div>}
                        <div className="tt-row">Tier <b>{r.tier}</b></div></>))}
                   onMouseLeave={hide}>
                  {r.tier % 2 === 1 && <rect x={0} y={cy - rowH / 2} width={width} height={rowH} fill="var(--tier-a)" />}
                  {first && i > 0 && <line x1={0} x2={width} y1={cy - rowH / 2} y2={cy - rowH / 2} stroke="var(--hairline)" />}
                  <text x={8} y={cy + 4} fontSize={12.5} fill="var(--ink)" fontWeight={550}>
                    {r.label.length > 26 ? r.label.slice(0, 25) + "…" : r.label}
                  </text>
                  {r.lo != null && r.hi != null && (
                    <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} stroke="var(--s1)" strokeWidth={2} strokeLinecap="round" />
                  )}
                  <circle cx={x(r.value)} cy={cy} r={4.5} fill="var(--s1)" stroke="var(--surface)" strokeWidth={2} />
                  <text x={width - 8} y={cy + 4} textAnchor="end" fontSize={12} fill="var(--ink-2)" className="tnum">{format(r.value)}</text>
                  <rect x={0} y={cy - rowH / 2} width={width} height={rowH} fill="transparent" />
                </g>
              );
            })}
            <XAxis scale={x} y={height - m.b + 6} format={format} width={width - m.l} />
          </>
        );
      }}
    </ChartFrame>
  );
}

// ================================================================ Heatmap
export function Heatmap({
  rows, cols, values, format, domain, diverging = false, cellH = 30, rowLabelW = 180, colLabelH = 90,
  tipFor, label, onRowClick,
}: {
  rows: string[]; cols: string[]; values: (number | null)[][]; format: (v: number) => string;
  domain: [number, number]; diverging?: boolean; cellH?: number; rowLabelW?: number; colLabelH?: number;
  tipFor?: (r: number, c: number) => ReactNode; label: string; onRowClick?: (r: number) => void;
}) {
  const { tip, show, hide } = useTip();
  const ink = useCellInk();
  const height = colLabelH + rows.length * cellH + 4;
  const [hc, setHc] = useState<[number, number] | null>(null);
  return (
    <ChartFrame height={height} tip={tip} label={label}>
      {(width) => {
        // rotated column labels lean right: reserve room so the last one stays inside the card
        const padR = Math.min(100, Math.max(...cols.map((c) => Math.min(c.length, 22))) * 4.2);
        const cw = Math.max(14, (width - rowLabelW - padR) / cols.length);
        const showText = cw >= 38;
        return (
          <>
            {cols.map((c, j) => (
              <text key={c} transform={`translate(${rowLabelW + j * cw + cw / 2},${colLabelH - 8}) rotate(-50)`}
                    fontSize={11.5} fill={hc?.[1] === j ? "var(--ink)" : "var(--ink-2)"} fontWeight={hc?.[1] === j ? 650 : 450}>
                {c.length > 22 ? c.slice(0, 21) + "…" : c}
              </text>
            ))}
            {rows.map((r, i) => (
              <g key={r}>
                <text x={rowLabelW - 10} y={colLabelH + i * cellH + cellH / 2 + 4} textAnchor="end" fontSize={12.5}
                      fill={hc?.[0] === i ? "var(--ink)" : "var(--ink-2)"} fontWeight={hc?.[0] === i ? 650 : 500}
                      style={{ cursor: onRowClick ? "pointer" : undefined }} onClick={() => onRowClick?.(i)}>
                  {r.length > 26 ? r.slice(0, 25) + "…" : r}
                </text>
                {cols.map((_, j) => {
                  const v = values[i][j];
                  const t = v == null ? 0 : diverging
                    ? v / Math.max(Math.abs(domain[0]), Math.abs(domain[1]) || 1)
                    : (v - domain[0]) / (domain[1] - domain[0] || 1);
                  const cx = rowLabelW + j * cw, cy = colLabelH + i * cellH;
                  return (
                    <g key={j}
                       onMouseEnter={() => { setHc([i, j]); show(cx + cw / 2, cy, tipFor ? tipFor(i, j) : v == null ? "—" : format(v)); }}
                       onMouseLeave={() => { setHc(null); hide(); }}>
                      <rect x={cx + 1} y={cy + 1} width={cw - 2} height={cellH - 2} rx={3}
                            fill={v == null ? "var(--surface-2)" : diverging ? divergingFill(t) : rampFill(t)} />
                      {showText && v != null && (
                        <text x={cx + cw / 2} y={cy + cellH / 2 + 4} textAnchor="middle" fontSize={11}
                              fill={diverging ? "var(--ink)" : ink(t)} className="tnum">{format(v)}</text>
                      )}
                      {hc && hc[0] === i && hc[1] === j && (
                        <rect x={cx + 1} y={cy + 1} width={cw - 2} height={cellH - 2} rx={3} fill="none" stroke="var(--ink)" strokeWidth={1.5} />
                      )}
                    </g>
                  );
                })}
              </g>
            ))}
          </>
        );
      }}
    </ChartFrame>
  );
}

export function RampLegend({ lo, hi, diverging = false }: { lo: string; hi: string; diverging?: boolean }) {
  const bg = diverging
    ? "linear-gradient(90deg, var(--div-neg), var(--div-mid), var(--div-pos))"
    : "linear-gradient(90deg, var(--q100), var(--q700))";
  return <div className="ramp-legend"><span>{lo}</span><div className="bar" style={{ background: bg }} /><span>{hi}</span></div>;
}

// ================================================================ LineChart
export interface LineSeries { name: string; color: string; points: [number, number][] }

export function LineChart({ series, xLabel, yFormat, xFormat = (v) => String(v), height = 240, label, yDomain }: {
  series: LineSeries[]; xLabel?: string; yFormat: (v: number) => string; xFormat?: (v: number) => string;
  height?: number; label: string; yDomain?: [number, number];
}) {
  const { tip, show, hide } = useTip();
  const [cx, setCx] = useState<number | null>(null);
  const m = { l: 48, r: 16, t: 14, b: xLabel ? 42 : 30 };
  const xsAll = series.flatMap((s) => s.points.map((p) => p[0]));
  const xVals = [...new Set(xsAll)].sort((a, b) => a - b);
  return (
    <ChartFrame height={height} tip={tip} label={label}>
      {(width) => {
        const x = linear([Math.min(...xVals), Math.max(...xVals)], [m.l, width - m.r]);
        const y = linear(yDomain ?? extent(series.flatMap((s) => s.points.map((p) => p[1])), 0.08), [height - m.b, m.t]);
        const ticks = xVals.length <= 12 ? xVals : undefined;
        return (
          <>
            <YGrid scale={y} x0={m.l} x1={width - m.r} format={yFormat} />
            <XAxis scale={x} y={height - m.b} format={xFormat} label={xLabel} width={width} ticks={ticks} />
            {series.map((s) => (
              <path key={s.name} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round"
                    d={s.points.map((p, i) => `${i ? "L" : "M"}${x(p[0])},${y(p[1])}`).join(" ")} />
            ))}
            {series.map((s) => {
              const last = s.points[s.points.length - 1];
              return last ? <circle key={s.name + "e"} cx={x(last[0])} cy={y(last[1])} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} /> : null;
            })}
            {cx !== null && (
              <g>
                <line x1={x(cx)} x2={x(cx)} y1={m.t} y2={height - m.b} stroke="var(--axis)" />
                {series.map((s) => {
                  const p = s.points.find((q) => q[0] === cx);
                  return p ? <circle key={s.name} cx={x(p[0])} cy={y(p[1])} r={4.5} fill={s.color} stroke="var(--surface)" strokeWidth={2} /> : null;
                })}
              </g>
            )}
            <rect x={m.l} y={m.t} width={Math.max(0, width - m.l - m.r)} height={height - m.t - m.b} fill="transparent"
                  onMouseMove={(e) => {
                    const r = (e.currentTarget as SVGRectElement).getBoundingClientRect();
                    const px = e.clientX - r.left + m.l;
                    const near = xVals.reduce((a, b) => (Math.abs(x(b) - px) < Math.abs(x(a) - px) ? b : a), xVals[0]);
                    setCx(near);
                    show(x(near), m.t + 10, (
                      <><div className="tt-title">{xLabel ? `${xLabel} ${xFormat(near)}` : xFormat(near)}</div>
                        {series.map((s) => {
                          const p = s.points.find((q) => q[0] === near);
                          return p ? <div className="tt-row" key={s.name}>
                            <span style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
                              <i style={{ width: 10, height: 2, background: s.color, display: "inline-block" }} />{s.name}
                            </span><b>{yFormat(p[1])}</b></div> : null;
                        })}</>));
                  }}
                  onMouseLeave={() => { setCx(null); hide(); }} />
          </>
        );
      }}
    </ChartFrame>
  );
}

// ================================================================ BarList
/** Horizontal bars, one series, value at the tip. */
export function BarList({ items, format, max = 1, label, color = "var(--s1)" }: {
  items: { label: string; value: number | null; sub?: string }[]; format: (v: number) => string;
  max?: number; label: string; color?: string;
}) {
  const { tip, show, hide } = useTip();
  const rowH = 24;
  const labelW = 170;
  const height = items.length * rowH + 4;
  return (
    <ChartFrame height={height} tip={tip} label={label}>
      {(width) => {
        const x = linear([0, max], [labelW, width - 56]);
        return items.map((it, i) => {
          const cy = i * rowH + rowH / 2;
          const w = it.value == null ? 0 : Math.max(2, x(it.value) - labelW);
          return (
            <g key={it.label} onMouseEnter={() => show(labelW + w, cy - 4, <><div className="tt-title">{it.label}</div>
                 <div className="tt-row">Value <b>{it.value == null ? "—" : format(it.value)}</b></div>
                 {it.sub && <div className="tt-row">{it.sub}</div>}</>)} onMouseLeave={hide}>
              <rect x={0} y={cy - rowH / 2} width={width} height={rowH} fill="transparent" />
              <text x={labelW - 10} y={cy + 4} textAnchor="end" fontSize={12} fill="var(--ink-2)">
                {it.label.length > 24 ? it.label.slice(0, 23) + "…" : it.label}
              </text>
              <rect x={labelW} y={cy - 5} width={Math.max(0, x(max) - labelW)} height={10} rx={3} fill="var(--surface-2)" />
              <path d={`M${labelW},${cy - 5} h${Math.max(0, w - 4)} a4,4 0 0 1 4,4 v2 a4,4 0 0 1 -4,4 h${-Math.max(0, w - 4)} z`} fill={color} />
              <text x={labelW + w + 6} y={cy + 4} fontSize={11.5} fill="var(--ink)" className="tnum">{it.value == null ? "n/a" : format(it.value)}</text>
            </g>
          );
        });
      }}
    </ChartFrame>
  );
}

// ================================================================ Reliability
export function Reliability({ bins, ece }: { bins: { lo: number; hi: number; count: number; accuracy: number; confidence: number }[]; ece: number }) {
  const { tip, show, hide } = useTip();
  const height = 280;
  const m = { l: 44, r: 12, t: 12, b: 40 };
  const total = bins.reduce((a, b) => a + b.count, 0);
  return (
    <ChartFrame height={height} tip={tip} label={`Reliability diagram, ECE ${(ece * 100).toFixed(2)}%`}>
      {(width) => {
        const size = Math.min(width - m.l - m.r, height - m.t - m.b);
        const x = linear([0, 1], [m.l, m.l + size]);
        const y = linear([0, 1], [m.t + size, m.t]);
        const ticks = [0, 0.25, 0.5, 0.75, 1];
        const pct = (v: number) => `${Math.round(v * 100)}`;
        return (
          <>
            <YGrid scale={y} x0={m.l} x1={m.l + size} ticks={ticks} format={pct} />
            <XAxis scale={x} y={m.t + size} ticks={ticks} format={pct} label="confidence %" width={size} />
            <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--ink-3)" strokeWidth={1} />
            {bins.map((b) => {
              const bw = x(b.hi) - x(b.lo);
              return (
                <g key={b.lo} onMouseEnter={() => show(x(b.lo) + bw / 2, Math.min(y(b.accuracy), y(b.confidence)), (
                    <><div className="tt-title">Confidence {pct(b.lo)}–{pct(b.hi)}%</div>
                      <div className="tt-row">Accuracy <b>{(b.accuracy * 100).toFixed(1)}%</b></div>
                      <div className="tt-row">Mean confidence <b>{(b.confidence * 100).toFixed(1)}%</b></div>
                      <div className="tt-row">{b.accuracy >= b.confidence ? "Under-confident by" : "Over-confident by"} <b>{(Math.abs(b.accuracy - b.confidence) * 100).toFixed(1)} pt</b></div>
                      <div className="tt-row">Samples <b>{b.count} ({((b.count / total) * 100).toFixed(1)}%)</b></div></>))}
                   onMouseLeave={hide}>
                  {b.accuracy > 0 && (
                    <path fill="var(--s1)" d={`M${x(b.lo) + 1},${y(0)} V${y(b.accuracy) + 3} a3,3 0 0 1 3,-3 H${x(b.hi) - 4} a3,3 0 0 1 3,3 V${y(0)} Z`} />
                  )}
                  <line x1={x(b.lo) + 1} x2={x(b.hi) - 1} y1={y(b.confidence)} y2={y(b.confidence)} stroke="var(--s2)" strokeWidth={2.5} strokeLinecap="round" />
                  <rect x={x(b.lo)} y={m.t} width={bw} height={size} fill="transparent" />
                </g>
              );
            })}
            <g transform={`translate(${m.l + size + 18}, ${m.t + 4})`} className="legend-svg">
              {width - size > 160 && (
                <>
                  <rect width={10} height={10} rx={2} fill="var(--s1)" /><text x={16} y={9} fontSize={12} fill="var(--ink-2)">accuracy per bin</text>
                  <line x1={0} x2={10} y1={25} y2={25} stroke="var(--s2)" strokeWidth={2.5} strokeLinecap="round" /><text x={16} y={29} fontSize={12} fill="var(--ink-2)">mean confidence</text>
                  <line x1={0} x2={10} y1={45} y2={45} stroke="var(--ink-3)" /><text x={16} y={49} fontSize={12} fill="var(--ink-2)">perfect calibration</text>
                  <text y={80} fontSize={12} fill="var(--ink-3)">ECE</text>
                  <text y={104} fontSize={24} fontWeight={700} fill="var(--ink)">{(ece * 100).toFixed(2)}%</text>
                </>
              )}
            </g>
          </>
        );
      }}
    </ChartFrame>
  );
}

// ================================================================ CIBar (inline, in tables)
export function CIBar({ value, lo, hi, domain, width = 110 }: { value: number; lo: number | null; hi: number | null; domain: [number, number]; width?: number }) {
  const x = linear(domain, [4, width - 4]);
  return (
    <svg width={width} height={14} aria-hidden="true" style={{ display: "block" }}>
      <line x1={4} x2={width - 4} y1={7} y2={7} stroke="var(--hairline)" />
      {lo != null && hi != null && <line x1={x(lo)} x2={x(hi)} y1={7} y2={7} stroke="var(--s1)" strokeWidth={3} strokeLinecap="round" opacity={0.45} />}
      <circle cx={x(value)} cy={7} r={3.5} fill="var(--s1)" stroke="var(--surface)" strokeWidth={1.5} />
    </svg>
  );
}
