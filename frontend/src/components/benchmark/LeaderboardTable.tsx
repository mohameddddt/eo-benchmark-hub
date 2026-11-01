import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { BOARD_METRICS, compact, fmtMetric, metricUnit, shortPretraining } from "../../lib/format";
import type { BenchmarkDetail, LeaderboardEntry } from "../../lib/types";
import { CIBar } from "../charts/Charts";
import { Empty, FamilyChip, Icon, Verified } from "../ui/ui";

type SortKey = "rank" | "params" | "gflops" | "throughput" | string;

export default function LeaderboardTable({ bm, entries }: { bm: BenchmarkDetail; entries: LeaderboardEntry[] }) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "rank", desc: false });
  const [family, setFamily] = useState<string | null>(null);
  const [pre, setPre] = useState<string | null>(null);
  const extra = BOARD_METRICS[bm.task_type] ?? [];
  const metricInfo = Object.fromEntries(bm.metrics.map((m) => [m.name, m]));
  const families = [...new Set(entries.map((e) => e.card?.family).filter(Boolean))] as string[];
  const pretrainings = [...new Set(entries.map((e) => shortPretraining(e.card?.pretraining ?? "")))];

  const domain = useMemo<[number, number]>(() => {
    const v = entries.flatMap((e) => [e.ci_low ?? e.score, e.ci_high ?? e.score]);
    const lo = Math.min(...v), hi = Math.max(...v);
    const pad = (hi - lo || 0.01) * 0.05;
    return [lo - pad, hi + pad];
  }, [entries]);

  const rows = useMemo(() => {
    let r = entries.filter((e) => (!family || e.card?.family === family)
      && (!pre || shortPretraining(e.card?.pretraining ?? "") === pre));
    const val = (e: LeaderboardEntry): number => {
      switch (sort.key) {
        case "rank": return e.rank;
        case "params": return e.card?.num_params ?? Infinity;
        case "gflops": return e.card?.gflops ?? Infinity;
        case "throughput": return e.card?.throughput ?? -Infinity;
        default: return e.metrics[sort.key] ?? -Infinity;
      }
    };
    r = [...r].sort((a, b) => (val(a) - val(b)) * (sort.desc ? -1 : 1));
    return r;
  }, [entries, family, pre, sort]);

  if (!entries.length) {
    return <Empty title="No published entries yet">Train a model with the starter notebook and be the first on the board.</Empty>;
  }

  const th = (key: SortKey, label: string, cls = "num", defaultDesc = true) => (
    <th className={`sortable ${cls}`} aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}
        onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : defaultDesc }))}>
      <span className="row" style={{ gap: 4, display: "inline-flex", flexWrap: "nowrap" }}>
        {label}{sort.key === key && <Icon name={sort.desc ? "down" : "up"} size={12} />}
      </span>
    </th>
  );
  const sortedByRank = sort.key === "rank" && !sort.desc && !family && !pre;

  return (
    <div>
      <div className="toolbar">
        {families.length > 1 && (
          <div className="row" style={{ gap: 6 }}>
            <span className="xs muted">Family</span>
            {families.map((f) => (
              <button key={f} className="chip" aria-pressed={family === f} onClick={() => setFamily(family === f ? null : f)}>{f}</button>
            ))}
          </div>
        )}
        {pretrainings.length > 1 && (
          <div className="row" style={{ gap: 6 }}>
            <span className="xs muted">Pretraining</span>
            {pretrainings.map((p) => (
              <button key={p} className="chip" aria-pressed={pre === p} onClick={() => setPre(pre === p ? null : p)}>{p}</button>
            ))}
          </div>
        )}
        <span className="grow" />
        <span className="xs muted">Shaded bands = statistically tied tiers (95% bootstrap CI)</span>
      </div>
      <div className="table-wrap">
        <table className="lb">
          <thead>
            <tr>
              {th("rank", "#", "", false)}
              <th>Model</th>
              <th className="num">{bm.primary_metric_label}</th>
              <th style={{ minWidth: 120 }}>95% CI</th>
              {extra.map((m) => metricInfo[m] && th(m, metricInfo[m].label, "num", metricInfo[m].higher_is_better))}
              {th("params", "Params", "num", false)}
              {th("gflops", "GFLOPs", "num", false)}
              {th("throughput", "img/s", "num", true)}
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => (
              <tr key={e.submission_id} className={sortedByRank && e.tier % 2 === 1 ? "tier-alt" : undefined}>
                <td className={`rank ${e.rank <= 3 ? "top" : ""}`}>
                  <div className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
                    {e.rank}
                  </div>
                </td>
                <td className="model">
                  <Link to={`/submissions/${e.submission_id}`}>{e.model_name}</Link>
                  <div className="meta">
                    <span className={`tier-pill ${e.tier === 1 ? "t1" : ""}`} title={`Tier ${e.tier}`}>T{e.tier}</span>
                    <FamilyChip family={e.card?.family} />
                    <span className="chip">{shortPretraining(e.card?.pretraining ?? "")}</span>
                    {e.verified && <Verified compact />}
                    <span className="xs muted">by <Link to={`/users/${e.user.id}`} className="muted">{e.user.name}</Link></span>
                  </div>
                </td>
                <td className="num score">{fmtMetric(e.score, metricUnit(bm.metrics, bm.primary_metric))}</td>
                <td>
                  <CIBar value={e.score} lo={e.ci_low} hi={e.ci_high} domain={domain} />
                  {e.ci_low != null && (
                    <div className="xs muted tnum">{fmtMetric(e.ci_low, metricUnit(bm.metrics, bm.primary_metric))}–{fmtMetric(e.ci_high, metricUnit(bm.metrics, bm.primary_metric))}</div>
                  )}
                </td>
                {extra.map((m) => metricInfo[m] && (
                  <td key={m} className="num">{fmtMetric(e.metrics[m], metricInfo[m].unit)}</td>
                ))}
                <td className="num">{compact(e.card?.num_params)}</td>
                <td className="num">{e.card?.gflops != null ? e.card.gflops.toFixed(e.card.gflops < 10 ? 2 : 1) : "—"}</td>
                <td className="num">{e.card?.throughput != null ? compact(e.card.throughput) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="xs muted" style={{ marginTop: 10 }}>
        One row per (researcher, model); repeated submissions of the same model keep their best score.
        {bm.metrics.find((m) => m.name === "ece") && " ECE: lower is better."} Throughput measured at the model's input size, fp16, batch 128 (segmentation: batch 8) on the hardware in its card.
      </p>
    </div>
  );
}
