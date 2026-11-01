import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

// ---------------------------------------------------------------- scales
export interface Scale { (v: number): number; domain: [number, number]; ticks: (n?: number) => number[] }

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const s = ((v: number) => r0 + (v - d0) * k) as Scale;
  s.domain = domain;
  s.ticks = (n = 5) => niceTicks(d0, d1, n);
  return s;
}

export function log(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain.map(Math.log10) as [number, number];
  const inner = linear([d0, d1], range);
  const s = ((v: number) => inner(Math.log10(Math.max(v, 1e-12)))) as Scale;
  s.domain = domain;
  s.ticks = () => {
    const out: number[] = [];
    for (let e = Math.floor(d0); e <= Math.ceil(d1); e++) {
      for (const m of [1, 2, 5]) {
        const v = m * 10 ** e;
        if (v >= domain[0] * 0.999 && v <= domain[1] * 1.001) out.push(v);
      }
    }
    if (out.length < 2) return niceTicks(domain[0], domain[1], 4).filter((v) => v > 0);  // narrow range: linear ticks
    return out.length > 7 ? out.filter((v) => String(v).replace(/0+$/, "").replace(".", "").startsWith("1")) : out;
  };
  return s;
}

export function niceTicks(a: number, b: number, n = 5): number[] {
  if (a === b) return [a];
  const span = b - a;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.5) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export function extent(vals: number[], pad = 0.06): [number, number] {
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const p = (hi - lo || Math.abs(hi) || 1) * pad;
  return [lo - p, hi + p];
}

// ---------------------------------------------------------------- layout
export function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    if (!ref.current) return;
    setW(ref.current.clientWidth);
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export interface TipState { x: number; y: number; content: ReactNode }

/** Chart frame: measures width, hosts the tooltip layer. */
export function ChartFrame({ height, children, tip, label }: {
  height: number; tip: TipState | null; label: string;
  children: (width: number) => ReactNode;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  return (
    <div className="chart" ref={ref} style={{ minHeight: height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={label}>{children(width)}</svg>
      )}
      {tip && (
        <div className="tooltip" style={{ left: Math.min(Math.max(tip.x, 90), Math.max(width - 90, 90)), top: tip.y }}>
          {tip.content}
        </div>
      )}
    </div>
  );
}

export function useTip() {
  const [tip, setTip] = useState<TipState | null>(null);
  return { tip, show: (x: number, y: number, content: ReactNode) => setTip({ x, y, content }), hide: () => setTip(null) };
}

// ---------------------------------------------------------------- axes
export function XAxis({ scale, y, ticks, format, label, width }: {
  scale: Scale; y: number; ticks?: number[]; format: (v: number) => string; label?: string; width: number;
}) {
  const ts = ticks ?? scale.ticks(Math.max(2, Math.floor(width / 90)));
  return (
    <g className="axis">
      <line x1={scale.domain ? scale(scale.domain[0]) : 0} x2={scale(scale.domain[1])} y1={y} y2={y} />
      {ts.map((t) => (
        <text key={t} className="tick" x={scale(t)} y={y + 16} textAnchor="middle">{format(t)}</text>
      ))}
      {label && <text className="axis-label" x={scale(scale.domain[1])} y={y + 34} textAnchor="end">{label}</text>}
    </g>
  );
}

export function YGrid({ scale, x0, x1, ticks, format, label }: {
  scale: Scale; x0: number; x1: number; ticks?: number[]; format: (v: number) => string; label?: string;
}) {
  const ts = ticks ?? scale.ticks(5);
  return (
    <g>
      <g className="grid">
        {ts.map((t) => <line key={t} x1={x0} x2={x1} y1={scale(t)} y2={scale(t)} />)}
      </g>
      {ts.map((t) => (
        <text key={t} className="tick" x={x0 - 8} y={scale(t) + 4} textAnchor="end">{format(t)}</text>
      ))}
      {label && <text className="axis-label" x={x0 - 8} y={scale(scale.domain[1]) - 12} textAnchor="start" transform="translate(0,0)">{label}</text>}
    </g>
  );
}

// ---------------------------------------------------------------- marks
export type Shape = "circle" | "square" | "triangle" | "diamond";

export function Marker({ shape, x, y, r = 5, fill, ring = true, className }: {
  shape: Shape; x: number; y: number; r?: number; fill: string; ring?: boolean; className?: string;
}) {
  const common = { fill, stroke: ring ? "var(--surface)" : "none", strokeWidth: ring ? 2 : 0, className };
  if (shape === "square") return <rect x={x - r * 0.9} y={y - r * 0.9} width={r * 1.8} height={r * 1.8} rx={1.5} {...common} />;
  if (shape === "triangle") {
    const h = r * 1.15;
    return <path d={`M${x},${y - h} L${x + h},${y + h * 0.8} L${x - h},${y + h * 0.8} Z`} strokeLinejoin="round" {...common} />;
  }
  if (shape === "diamond") return <path d={`M${x},${y - r * 1.25} L${x + r * 1.25},${y} L${x},${y + r * 1.25} L${x - r * 1.25},${y} Z`} {...common} />;
  return <circle cx={x} cy={y} r={r} {...common} />;
}

export function LegendSwatch({ shape, color }: { shape: Shape; color: string }) {
  return <svg width="14" height="14" viewBox="0 0 14 14"><Marker shape={shape} x={7} y={7} r={4.5} fill={color} ring={false} /></svg>;
}

/** Sequential ramp fill via CSS color-mix, so light/dark tokens apply automatically. */
export const rampFill = (t: number) =>
  `color-mix(in oklab, var(--q700) ${Math.round(Math.max(0, Math.min(1, t)) * 100)}%, var(--q100))`;
export const divergingFill = (t: number) =>       // t in [-1, 1]
  t >= 0
    ? `color-mix(in oklab, var(--div-pos) ${Math.round(Math.min(1, t) * 100)}%, var(--div-mid))`
    : `color-mix(in oklab, var(--div-neg) ${Math.round(Math.min(1, -t) * 100)}%, var(--div-mid))`;

export function useIsDark() {
  const get = () => {
    const t = document.documentElement.dataset.theme;
    return t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  };
  const [dark, setDark] = useState(get);
  useEffect(() => {
    const mo = new MutationObserver(() => setDark(get()));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => setDark(get());
    mq.addEventListener("change", on);
    return () => { mo.disconnect(); mq.removeEventListener("change", on); };
  }, []);
  return dark;
}

/** Text colour inside a ramp cell: ink on the light end, inverse on the deep end. */
export function useCellInk() {
  const dark = useIsDark();
  return (t: number) => (dark ? (t > 0.6 ? "#0b0b0b" : "#ffffff") : (t > 0.55 ? "#ffffff" : "#0b0b0b"));
}
