import { useMemo, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { ChartFrame, Marker, XAxis, YGrid, extent, linear, log, useTip, type Shape } from "./core";

export interface ScatterPoint {
  id: string; x: number; y: number; label: string; color: string; shape: Shape;
  yLo?: number | null; yHi?: number | null; href?: string; tip?: ReactNode;
}

/** Pareto frontier: best y for the cost on x (minimize x) or the benefit on x (maximize x). */
function frontier(points: ScatterPoint[], xBetter: "low" | "high"): ScatterPoint[] {
  const sorted = [...points].sort((a, b) => (xBetter === "low" ? a.x - b.x : b.x - a.x) || b.y - a.y);
  const out: ScatterPoint[] = [];
  let best = -Infinity;
  for (const p of sorted) if (p.y > best) { out.push(p); best = p.y; }
  return out;
}

export default function Scatter({
  points, xLabel, yLabel, xFormat, yFormat, xLog = false, frontierDir, height = 340, highlight, label, labelIds = [],
}: {
  points: ScatterPoint[]; xLabel: string; yLabel: string; xFormat: (v: number) => string;
  yFormat: (v: number) => string; xLog?: boolean; frontierDir?: "low" | "high"; height?: number;
  highlight?: string | null; label: string; labelIds?: string[];
}) {
  const { tip, show, hide } = useTip();
  const [hover, setHover] = useState<string | null>(null);
  const navigate = useNavigate();
  const front = useMemo(() => (frontierDir ? frontier(points, frontierDir) : []), [points, frontierDir]);
  const frontIds = new Set(front.map((p) => p.id));
  const active = hover ?? highlight ?? null;
  const m = { l: 52, r: 24, t: 24, b: 44 };

  return (
    <ChartFrame height={height} tip={tip} label={label}>
      {(width) => {
        const xs = points.map((p) => p.x);
        const ys = points.flatMap((p) => [p.y, p.yLo ?? p.y, p.yHi ?? p.y]);
        const pad = Math.max(...xs) / Math.min(...xs) < 3 ? 1.12 : 1.6;   // narrow ranges need less log padding
        const xd: [number, number] = xLog
          ? [Math.min(...xs) / pad, Math.max(...xs) * pad]
          : extent(xs, 0.08);
        const x = (xLog ? log : linear)(xd, [m.l, width - m.r]);
        const y = linear(extent(ys, 0.1), [height - m.b, m.t]);
        const fpath = front.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join(" ");
        // greedy label placement: try above-right/left, then below; skip if every slot collides
        const placed: { x0: number; x1: number; y0: number; y1: number }[] = [];
        const labelPos = new Map<string, { x: number; y: number; anchor: "start" | "end"; text: string }>();
        const wanted = [...points].filter((p) => frontIds.has(p.id) || labelIds.includes(p.id)).sort((a, b) => b.y - a.y);
        for (const p of wanted) {
          const text = p.label.length > 24 ? p.label.slice(0, 23) + "…" : p.label;
          const w = text.length * 6.3, lx = x(p.x), ly = y(p.y);
          const prefer: "start" | "end" = lx > width - 170 ? "end" : "start";
          const other: "start" | "end" = prefer === "start" ? "end" : "start";
          const slots: [number, "start" | "end"][] = [[ly - 8, prefer], [ly + 16, prefer], [ly - 8, other], [ly + 16, other]];
          for (const [ty, anchor] of slots) {
            const tx = lx + (anchor === "start" ? 10 : -10);
            const box = { x0: anchor === "start" ? tx : tx - w, x1: anchor === "start" ? tx + w : tx, y0: ty - 11, y1: ty + 3 };
            if (box.x0 < 0 || box.x1 > width) continue;
            if (!placed.some((b) => b.x0 < box.x1 && box.x0 < b.x1 && b.y0 < box.y1 && box.y0 < b.y1)) {
              placed.push(box);
              labelPos.set(p.id, { x: tx, y: ty, anchor, text });
              break;
            }
          }
        }
        return (
          <>
            <YGrid scale={y} x0={m.l} x1={width - m.r} format={yFormat} label={yLabel} />
            <XAxis scale={x} y={height - m.b} format={xFormat} label={xLabel} width={width} />
            {fpath && (
              <path d={fpath} fill="none" stroke="var(--ink-3)" strokeWidth={1.5} strokeLinejoin="round"
                    strokeDasharray="0" opacity={0.55} />
            )}
            {points.map((p) => (p.yLo != null && p.yHi != null ? (
              <line key={p.id + "ci"} x1={x(p.x)} x2={x(p.x)} y1={y(p.yLo)} y2={y(p.yHi)}
                    stroke={p.color} strokeWidth={2} strokeLinecap="round" opacity={active && active !== p.id ? 0.15 : 0.45} />
            ) : null))}
            {points.map((p) => {
              const dim = active && active !== p.id;
              const lx = x(p.x), ly = y(p.y);
              const anchorHover: "start" | "end" = lx > width - 170 ? "end" : "start";
              const lp = active === p.id
                ? { x: lx + (anchorHover === "start" ? 10 : -10), y: ly - 8, anchor: anchorHover, text: p.label }
                : !active ? labelPos.get(p.id) : undefined;
              return (
                <g key={p.id} style={{ cursor: p.href ? "pointer" : "default" }}
                   onMouseEnter={() => { setHover(p.id); show(lx, ly, p.tip ?? p.label); }}
                   onMouseLeave={() => { setHover(null); hide(); }}
                   onClick={() => p.href && navigate(p.href)}>
                  <circle cx={lx} cy={ly} r={14} fill="transparent" />
                  <Marker shape={p.shape} x={lx} y={ly} r={active === p.id ? 6.5 : 5} fill={p.color}
                          className={dim ? "dimmed" : undefined} />
                  {lp && <text className="label" x={lp.x} y={lp.y} textAnchor={lp.anchor}>{lp.text}</text>}
                </g>
              );
            })}
          </>
        );
      }}
    </ChartFrame>
  );
}
