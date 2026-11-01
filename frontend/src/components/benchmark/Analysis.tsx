import { useQueries } from "@tanstack/react-query";
import { Fragment, useMemo, useState } from "react";
import { api, media, useAsset, useLeaderboard } from "../../lib/api";
import { compact, fmtMetric, metricUnit, pretty, shortPretraining } from "../../lib/format";
import type { BenchmarkDetail, LeaderboardEntry, SegmentationAssets } from "../../lib/types";
import { BarList, ForestPlot, Heatmap, LineChart, RampLegend } from "../charts/Charts";
import Scatter, { type ScatterPoint } from "../charts/Scatter";
import { Empty, FAMILIES, Legend, familyStyle } from "../ui/ui";
import MaskCompare from "./MaskCompare";

const pct1 = (v: number) => (v * 100).toFixed(1);

/** Points worth a direct label: the two highest on y, the lowest on y, the highest on x. */
export function notable(pts: ScatterPoint[]): string[] {
  if (!pts.length) return [];
  const byY = [...pts].sort((a, b) => b.y - a.y);
  const byX = [...pts].sort((a, b) => b.x - a.x);
  return [...new Set([byY[0].id, byY[1]?.id, byY[byY.length - 1].id, byX[0].id].filter(Boolean))] as string[];
}

function familyLegend(entries: LeaderboardEntry[]) {
  const fams = [...new Set(entries.map((e) => e.card?.family).filter(Boolean))] as string[];
  return fams.filter((f) => FAMILIES[f]).map((f) => ({ label: f, ...familyStyle(f) }));
}

function tipFor(e: LeaderboardEntry, bm: BenchmarkDetail) {
  const unit = metricUnit(bm.metrics, bm.primary_metric);
  return (
    <>
      <div className="tt-title">{e.model_name}</div>
      <div className="tt-row">{bm.primary_metric_label} <b>{fmtMetric(e.score, unit)}</b></div>
      {e.ci_low != null && <div className="tt-row">95% CI <b>{fmtMetric(e.ci_low, unit)}–{fmtMetric(e.ci_high, unit)}</b></div>}
      <div className="tt-row">Params <b>{compact(e.card?.num_params)}</b></div>
      <div className="tt-row">GFLOPs <b>{e.card?.gflops?.toFixed(2) ?? "—"}</b></div>
      <div className="tt-row">Throughput <b>{compact(e.card?.throughput)} img/s</b></div>
      <div className="tt-row">Pretraining <b>{shortPretraining(e.card?.pretraining ?? "")}</b></div>
    </>
  );
}

// ---------------------------------------------------------------- efficiency
function Efficiency({ bm, entries }: { bm: BenchmarkDetail; entries: LeaderboardEntry[] }) {
  const [axis, setAxis] = useState<"params" | "gflops" | "throughput">("params");
  const unit = metricUnit(bm.metrics, bm.primary_metric);
  const pts: ScatterPoint[] = entries
    .map((e) => {
      const xv = axis === "params" ? e.card?.num_params : axis === "gflops" ? e.card?.gflops : e.card?.throughput;
      if (!xv) return null;
      const st = familyStyle(e.card?.family);
      return { id: e.submission_id, x: xv, y: e.score, yLo: e.ci_low, yHi: e.ci_high, label: e.model_name,
               color: st.color, shape: st.shape, href: `/submissions/${e.submission_id}`, tip: tipFor(e, bm) };
    })
    .filter(Boolean) as ScatterPoint[];
  if (pts.length < 2) return null;
  const xLabel = { params: "parameters (log)", gflops: "GFLOPs per image (log)", throughput: "images / s on T4 (log)" }[axis];
  return (
    <div className="card">
      <div className="card-title">
        <div>
          <h3>Accuracy–efficiency frontier</h3>
          <div className="sub">Grey line: Pareto frontier — no other model is both {axis === "throughput" ? "faster" : "smaller"} and better.</div>
        </div>
        <div className="seg" role="group" aria-label="Efficiency axis">
          {(["params", "gflops", "throughput"] as const).map((a) => (
            <button key={a} aria-pressed={axis === a} onClick={() => setAxis(a)}>{a === "params" ? "Params" : a === "gflops" ? "GFLOPs" : "Throughput"}</button>
          ))}
        </div>
      </div>
      <Scatter points={pts} xLog xLabel={xLabel} yLabel={`${bm.primary_metric_label}${unit === "ratio" ? " %" : ""}`}
               xFormat={(v) => compact(v, 0)} yFormat={(v) => fmtMetric(v, unit, 1)}
               frontierDir={axis === "throughput" ? "high" : "low"} label="Accuracy versus model cost" />
      <Legend items={familyLegend(entries)} />
    </div>
  );
}

// ---------------------------------------------------------------- per-class heatmap
function PerClass({ bm, entries }: { bm: BenchmarkDetail; entries: LeaderboardEntry[] }) {
  const details = useQueries({
    queries: entries.map((e) => ({ queryKey: ["submission", e.submission_id], queryFn: () => api.submission(e.submission_id), staleTime: 60_000 })),
  });
  const ready = details.every((d) => d.data);
  // per-class score shown: IoU (segmentation), AP (multi-label), F1 (single-label)
  const key = bm.task_type === "semantic_segmentation" ? "iou" : bm.task_type === "multilabel_classification" ? "ap" : "f1";
  const classes = bm.classes ?? [];
  const matrix = useMemo(() => (ready ? details.map((d) => classes.map((c) => d.data!.details?.per_class?.[c]?.[key] ?? null)) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ready, entries.length]);
  if (!ready) return <div className="card"><div className="skeleton" style={{ height: 260 }} /></div>;
  const flat = matrix.flat().filter((v): v is number => v != null);
  const lo = Math.floor(Math.min(...flat) * 20) / 20;
  const mean = classes.map((c, j) => {
    const vals = matrix.map((r) => r[j]).filter((v): v is number => v != null);
    return { label: c, value: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN };
  });
  const hardest = mean.filter((m) => !Number.isNaN(m.value)).sort((a, b) => a.value - b.value).slice(0, 8);
  const metricName = { iou: "IoU", ap: "AP", f1: "F1" }[key];
  const many = classes.length > 15;
  return (
    <div className={many ? "stack" : "grid g2"} style={many ? undefined : { gridTemplateColumns: "minmax(0,2fr) minmax(0,1fr)" }}>
      <div className="card">
        <div className="card-title">
          <div><h3>Per-class {metricName} by model</h3><div className="sub">Rows in leaderboard order. Hover a cell for details.</div></div>
          <RampLegend lo={(lo * 100).toFixed(0)} hi="100" />
        </div>
        <Heatmap rows={entries.map((e) => e.model_name)} cols={classes.map(pretty)} values={matrix}
                 domain={[lo, 1]} format={(v) => (v * 100).toFixed(0)} cellH={many ? 22 : 30} colLabelH={many ? 120 : 96}
                 label={`Per-class ${metricName} heatmap`}
                 tipFor={(i, j) => (<><div className="tt-title">{entries[i].model_name}</div>
                   <div className="tt-row">{pretty(classes[j])} <b>{matrix[i][j] == null ? "—" : `${(matrix[i][j]! * 100).toFixed(2)}%`}</b></div>
                   <div className="tt-row">Class mean <b>{(mean[j].value * 100).toFixed(2)}%</b></div></>)} />
      </div>
      <div className="card">
        <div className="card-title"><div><h3>Hardest classes</h3>
          <div className="sub">{key !== "f1" ? `Mean ${metricName} across all entries` : "Mean error (100 − F1) across all entries — zero-based so small gaps stay visible"}</div></div></div>
        {key !== "f1"
          ? <BarList items={hardest.map((h) => ({ label: pretty(h.label), value: h.value }))} format={(v) => (v * 100).toFixed(1)}
                     max={1} label={`Hardest classes by mean ${metricName}`} />
          : <BarList items={hardest.map((h) => ({ label: pretty(h.label), value: 1 - h.value, sub: `mean F1 ${(h.value * 100).toFixed(2)}%` }))}
                     format={(v) => (v * 100).toFixed(2)} max={Math.max(...hardest.map((h) => 1 - h.value))} color="var(--s8)"
                     label="Hardest classes by mean error" />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- robustness
function Robustness({ bm, entries }: { bm: BenchmarkDetail; entries: LeaderboardEntry[] }) {
  const details = useQueries({
    queries: entries.map((e) => ({ queryKey: ["submission", e.submission_id], queryFn: () => api.submission(e.submission_id), staleTime: 60_000 })),
  });
  const clean = useLeaderboard("eurosat");
  const [pick, setPick] = useState(0);
  const ready = details.every((d) => d.data);
  if (!ready) return <div className="card"><div className="skeleton" style={{ height: 300 }} /></div>;
  const corr: string[] = bm.config?.corruptions ?? [];
  const per = details.map((d) => d.data!.details?.per_corruption ?? {});
  const matrix = per.map((p) => corr.map((c) => p[c]?.mean ?? null));
  const flat = matrix.flat().filter((v): v is number => v != null);
  const lo = Math.floor(Math.min(...flat) * 20) / 20;
  const sel = per[pick] ?? {};
  const colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)", "var(--s7)"];
  const cleanAcc = entries[pick]?.metrics.clean_accuracy ?? 1;

  // accuracy vs robustness: x = clean EuroSAT accuracy (the clean board), y = mCA
  const cleanByName = Object.fromEntries((clean.data ?? []).map((e) => [e.model_name, e]));
  const pts: ScatterPoint[] = entries.flatMap((e) => {
    const c = cleanByName[e.model_name];
    const st = familyStyle(e.card?.family);
    const x = c ? c.score : e.metrics.clean_accuracy;
    return x == null ? [] : [{ id: e.submission_id, x, y: e.score, yLo: e.ci_low, yHi: e.ci_high, label: e.model_name,
      color: st.color, shape: st.shape, href: `/submissions/${e.submission_id}`,
      tip: <><div className="tt-title">{e.model_name}</div>
        <div className="tt-row">Clean accuracy <b>{pct1(x)}%</b></div>
        <div className="tt-row">mCA <b>{pct1(e.score)}%</b></div>
        <div className="tt-row">Retained <b>{pct1(e.metrics.relative_robustness)}%</b></div></> }];
  });

  return (
    <div className="stack" style={{ ["--gap" as any]: "16px" }}>
      <div className="grid g2">
        <div className="card">
          <div className="card-title"><div><h3>Clean accuracy is not robustness</h3>
            <div className="sub">EuroSAT clean accuracy (x) vs mean corruption accuracy (y)</div></div></div>
          <Scatter points={pts} xLabel="clean accuracy %" yLabel="mCA %" xFormat={pct1} yFormat={pct1} height={320}
                   label="Clean accuracy versus corruption accuracy" labelIds={notable(pts)} />
          <Legend items={familyLegend(entries)} />
        </div>
        <div className="card">
          <div className="card-title">
            <div><h3>Accuracy vs severity</h3><div className="sub">Severity 0 = clean images</div></div>
            <select value={pick} onChange={(e) => setPick(+e.target.value)} style={{ width: "auto", maxWidth: 230 }} aria-label="Model">
              {entries.map((e, i) => <option key={e.submission_id} value={i}>{e.model_name}</option>)}
            </select>
          </div>
          <LineChart label="Accuracy by corruption severity" xLabel="severity" xFormat={(v) => String(v)} yFormat={pct1} height={286}
                     series={corr.map((c, i) => ({ name: pretty(c), color: colors[i % colors.length],
                       points: [[0, cleanAcc], [1, sel[c]?.s1], [2, sel[c]?.s2], [3, sel[c]?.s3]].filter((p) => p[1] != null) as [number, number][] }))} />
          <Legend items={corr.map((c, i) => ({ label: pretty(c), color: colors[i % colors.length] }))} />
        </div>
      </div>
      <div className="card">
        <div className="card-title">
          <div><h3>Accuracy under each corruption</h3><div className="sub">Mean over three severities; rows in leaderboard (mCA) order</div></div>
          <RampLegend lo={(lo * 100).toFixed(0)} hi="100" />
        </div>
        <Heatmap rows={entries.map((e) => e.model_name)} cols={corr.map(pretty)} values={matrix} domain={[lo, 1]}
                 format={pct1} colLabelH={110} label="Corruption accuracy heatmap"
                 tipFor={(i, j) => {
                   const p = per[i][corr[j]] ?? {};
                   return <><div className="tt-title">{entries[i].model_name}</div><div className="tt-row">{pretty(corr[j])}</div>
                     {[1, 2, 3].map((s) => <div className="tt-row" key={s}>Severity {s} <b>{p[`s${s}`] != null ? pct1(p[`s${s}`]) + "%" : "—"}</b></div>)}</>;
                 }} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- segmentation samples
function SegSamples({ bm }: { bm: BenchmarkDetail }) {
  const { data } = useAsset<SegmentationAssets>(bm.slug);
  const modelIds = data ? Object.keys(data.samples[0]?.predictions ?? {}) : [];
  const [model, setModel] = useState<string | null>(null);
  const [idx, setIdx] = useState(0);
  if (!data || !data.samples.length) return null;
  const m = model ?? modelIds[0];
  const s = data.samples[idx];
  const nameOf = (id: string) => data.model_names?.[id] ?? id;
  return (
    <div className="card">
      <div className="card-title">
        <div><h3>Qualitative comparison</h3><div className="sub">Public test tiles · drag the handle: left = prediction, right = ground truth</div></div>
        <div className="seg" role="group" aria-label="Model">
          {modelIds.map((id) => <button key={id} aria-pressed={m === id} onClick={() => setModel(id)}>{nameOf(id)}</button>)}
        </div>
      </div>
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.3fr) minmax(0, 1fr)", alignItems: "start" }}>
        <MaskCompare image={media(bm.slug, s.image)} left={media(bm.slug, s.predictions[m].mask)} right={media(bm.slug, s.gt)} />
        <div className="stack" style={{ ["--gap" as any]: "12px" }}>
          <div className="thumbs" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
            {data.samples.map((t, i) => (
              <button key={t.sample_id} onClick={() => setIdx(i)} style={{ padding: 0, border: i === idx ? "2px solid var(--accent)" : "2px solid transparent", borderRadius: 10, background: "none", cursor: "pointer" }}>
                <img src={media(bm.slug, t.image)} alt={`Tile ${t.sample_id}`} style={{ borderRadius: 8, aspectRatio: "1", objectFit: "cover" }} />
              </button>
            ))}
          </div>
          <dl className="kv small">
            <dt>Tile</dt><dd className="mono">{s.sample_id}</dd>
            {modelIds.map((id) => (
              <Fragment key={id}><dt>{nameOf(id)}</dt><dd className="tnum">tile mIoU {(s.predictions[id].miou * 100).toFixed(1)}</dd></Fragment>
            ))}
          </dl>
          <Legend items={(data.classes ?? []).map((c, i) => ({ label: pretty(c), color: `rgb(${data.palette[i].join(",")})`, shape: "square" as const }))} />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- main
export default function Analysis({ bm, entries }: { bm: BenchmarkDetail; entries: LeaderboardEntry[] }) {
  if (entries.length < 2) return <Empty title="Not enough entries to analyse yet" />;
  const unit = metricUnit(bm.metrics, bm.primary_metric);
  return (
    <div className="stack" style={{ ["--gap" as any]: "16px" }}>
      {bm.task_type === "scene_classification_robustness" && <Robustness bm={bm} entries={entries} />}
      {bm.task_type === "semantic_segmentation" && <SegSamples bm={bm} />}
      <div className="grid g2">
        <Efficiency bm={bm} entries={entries} />
        <div className="card">
          <div className="card-title"><div><h3>Confidence intervals</h3>
            <div className="sub">95% bootstrap CI on the hidden split; shaded rows share a tier</div></div></div>
          <ForestPlot rows={entries.map((e) => ({ id: e.submission_id, label: e.model_name, value: e.score, lo: e.ci_low, hi: e.ci_high, tier: e.tier }))}
                      format={(v) => fmtMetric(v, unit)} label="Forest plot of scores with confidence intervals" />
        </div>
      </div>
      {bm.task_type !== "scene_classification_robustness" && <PerClass bm={bm} entries={entries} />}
    </div>
  );
}
