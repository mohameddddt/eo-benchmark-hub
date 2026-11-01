import { useQueries } from "@tanstack/react-query";
import { useMemo } from "react";
import { Link } from "react-router-dom";
import { RampLegend } from "../components/charts/Charts";
import { rampFill, useCellInk } from "../components/charts/core";
import { FamilyChip, Skeleton } from "../components/ui/ui";
import { api, useBenchmarks } from "../lib/api";
import { compact, fmtMetric, shortPretraining } from "../lib/format";
import type { LeaderboardEntry } from "../lib/types";

/** Cross-benchmark view: does a backbone that wins one benchmark win the others? */
export default function Models() {
  const bms = useBenchmarks();
  const boards = useQueries({
    queries: (bms.data ?? []).map((b) => ({ queryKey: ["leaderboard", b.slug], queryFn: () => api.leaderboard(b.slug) })),
  });
  const ink = useCellInk();
  const ready = bms.data && boards.every((b) => b.data);

  const rows = useMemo(() => {
    if (!ready) return [];
    const byName = new Map<string, { name: string; cells: Record<string, { e: LeaderboardEntry; pct: number; n: number }>; family?: string; pre?: string; params?: number | null }>();
    bms.data!.forEach((b, i) => {
      const board = boards[i].data!;
      board.forEach((e) => {
        const key = e.model_name;
        const row = byName.get(key) ?? { name: key, cells: {}, family: e.card?.family, pre: e.card?.pretraining, params: e.card?.num_params };
        // percentile within board: 1 = best, 0 = worst
        row.cells[b.slug] = { e, pct: board.length > 1 ? 1 - (e.rank - 1) / (board.length - 1) : 1, n: board.length };
        byName.set(key, row);
      });
    });
    return [...byName.values()]
      .map((r) => {
        const cells = Object.values(r.cells);
        return { ...r, coverage: cells.length, meanPct: cells.reduce((a, c) => a + c.pct, 0) / cells.length };
      })
      .sort((a, b) => b.coverage - a.coverage || b.meanPct - a.meanPct);
  }, [ready, bms.data, boards]);

  return (
    <div className="container">
      <div className="page-head">
        <h1>Models across benchmarks</h1>
        <p className="lead">The same recipe, evaluated on every track. Cells show each model's score and rank; colour is its rank percentile within that benchmark (darker = nearer the top), so you can see which backbones transfer from 10 m Sentinel-2 to 25 cm aerial imagery — and which only win one board.</p>
      </div>
      {!ready ? <Skeleton h={480} style={{ marginTop: 28 }} /> : (
        <>
          <div className="row between" style={{ margin: "28px 0 12px" }}>
            <span className="small muted">Sorted by coverage, then mean rank percentile. Only published entries.</span>
            <RampLegend lo="last" hi="first" />
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th className="num">Params</th>
                  {bms.data!.map((b) => <th key={b.slug} className="num"><Link to={`/benchmarks/${b.slug}`} style={{ color: "inherit" }}>{b.name}</Link><div className="xs" style={{ fontWeight: 450 }}>{b.primary_metric_label}</div></th>)}
                  <th className="num">Mean percentile</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.name}>
                    <td><div style={{ fontWeight: 600 }}>{r.name}</div>
                      <div className="row" style={{ gap: 6, marginTop: 3 }}><FamilyChip family={r.family} /><span className="chip">{shortPretraining(r.pre ?? "")}</span></div></td>
                    <td className="num">{compact(r.params)}</td>
                    {bms.data!.map((b) => {
                      const c = r.cells[b.slug];
                      if (!c) return <td key={b.slug} className="num muted">—</td>;
                      return (
                        <td key={b.slug} className="num" style={{ padding: 4 }}>
                          <Link to={`/submissions/${c.e.submission_id}`} style={{
                            display: "block", padding: "6px 8px", borderRadius: 6, background: rampFill(c.pct * 0.9), color: ink(c.pct * 0.9), textDecoration: "none",
                          }} title={`Rank ${c.e.rank} of ${c.n}`}>
                            <b>{fmtMetric(c.e.score)}</b><div className="xs" style={{ opacity: 0.85 }}>#{c.e.rank} of {c.n}</div>
                          </Link>
                        </td>
                      );
                    })}
                    <td className="num"><b>{(r.meanPct * 100).toFixed(0)}</b><span className="muted xs"> / 100</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
