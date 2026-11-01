import { useSearchParams, Link, useParams } from "react-router-dom";
import Analysis from "../components/benchmark/Analysis";
import LeaderboardTable from "../components/benchmark/LeaderboardTable";
import { DatasetTab, ProtocolTab, SubmitTab } from "../components/benchmark/Tabs";
import { ErrorState, Icon, PageSkeleton, Skeleton } from "../components/ui/ui";
import { useBenchmark, useBenchmarks, useLeaderboard } from "../lib/api";
import { TASK_LABEL, fmtInt, sampleUnit } from "../lib/format";
import { BenchmarkCard } from "./Home";

export function BenchmarkList() {
  const { data, error } = useBenchmarks();
  if (error) return <ErrorState error={error} />;
  return (
    <div className="container">
      <div className="page-head">
        <h1>Benchmarks</h1>
        <p className="lead">Each benchmark fixes a dataset, a split with server-only test labels, a versioned evaluator and a primary metric. Pick one to see its leaderboard, analysis and submission format.</p>
      </div>
      <div className="grid g2" style={{ marginTop: 28 }}>
        {data ? data.map((b) => <BenchmarkCard key={b.slug} b={b} />) : [0, 1, 2, 3].map((i) => <Skeleton key={i} h={330} />)}
      </div>
    </div>
  );
}

const TABS = [
  ["leaderboard", "Leaderboard"], ["analysis", "Analysis"], ["dataset", "Dataset"], ["submit", "Submit"], ["protocol", "Protocol"],
] as const;

export function BenchmarkPage() {
  const { slug = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "leaderboard";
  const bm = useBenchmark(slug);
  const board = useLeaderboard(slug);
  if (bm.error) return <ErrorState error={bm.error} />;
  if (!bm.data) return <PageSkeleton />;
  const b = bm.data;
  const entries = board.data ?? [];
  return (
    <div className="container">
      <div className="page-head">
        <div className="crumbs"><Link to="/benchmarks">Benchmarks</Link> / {TASK_LABEL[b.task_type]}</div>
        <div className="row between" style={{ alignItems: "end" }}>
          <div>
            <h1>{b.name}</h1>
            <p className="lead" style={{ marginBottom: 0 }}>{b.tagline}</p>
          </div>
          <div className="row">
            {b.starter_url && <a className="btn secondary sm" href={b.starter_url} target="_blank" rel="noreferrer"><Icon name="kaggle" size={14} />Reference notebook</a>}
            {b.paper_url && <a className="btn secondary sm" href={b.paper_url} target="_blank" rel="noreferrer"><Icon name="book" size={14} />Paper</a>}
            <button className="btn sm" onClick={() => setParams({ tab: "submit" })}><Icon name="upload" size={14} />Submit</button>
          </div>
        </div>
        <div className="meta-row">
          <span className="chip">{b.modality}</span>
          <span className="chip">{b.n_classes} classes</span>
          <span className="chip">{fmtInt(b.stats?.splits?.test_hidden)} hidden {sampleUnit(b.task_type)}</span>
          <span className="chip">primary: {b.primary_metric_label}</span>
          <span className="chip mono">{b.evaluator_version}</span>
        </div>
      </div>
      <nav className="tabs" aria-label="Benchmark sections">
        {TABS.map(([k, label]) => (
          <button key={k} className={tab === k ? "active" : ""} onClick={() => setParams(k === "leaderboard" ? {} : { tab: k })}>
            {label}{k === "leaderboard" && entries.length ? <span className="muted" style={{ marginLeft: 6, fontWeight: 500 }}>{entries.length}</span> : null}
          </button>
        ))}
      </nav>
      {tab === "leaderboard" && (board.data ? <LeaderboardTable bm={b} entries={entries} /> : <Skeleton h={400} />)}
      {tab === "analysis" && (board.data ? <Analysis bm={b} entries={entries} /> : <Skeleton h={400} />)}
      {tab === "dataset" && <DatasetTab bm={b} entries={entries} />}
      {tab === "submit" && <SubmitTab bm={b} />}
      {tab === "protocol" && <ProtocolTab bm={b} />}
    </div>
  );
}
