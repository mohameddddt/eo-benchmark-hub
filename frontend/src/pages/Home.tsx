import { Link } from "react-router-dom";
import { notable } from "../components/benchmark/Analysis";
import Scatter, { type ScatterPoint } from "../components/charts/Scatter";
import { Icon, Legend, Skeleton, familyStyle, FAMILIES } from "../components/ui/ui";
import { media, useAsset, useBenchmarks, useLeaderboard, useStats } from "../lib/api";
import { TASK_LABEL, fmtMetric, pretty, timeAgo } from "../lib/format";
import type { BenchmarkSummary, ClassificationAssets, SegmentationAssets } from "../lib/types";

function useMosaic() {
  const es = useAsset<ClassificationAssets>("eurosat");
  const rs = useAsset<ClassificationAssets>("resisc45");
  const lc = useAsset<SegmentationAssets>("landcoverai");
  const tiles: { src: string; label: string }[] = [];
  const pick = (slug: string, a: ClassificationAssets | null | undefined, n: number, offset = 0) => {
    if (!a) return;
    Object.entries(a.classes).slice(offset, offset + n).forEach(([c, imgs]) => tiles.push({ src: media(slug, imgs[0]), label: pretty(c) }));
  };
  pick("resisc45", rs.data, 8, 3);
  pick("eurosat", es.data, 10);
  lc.data?.samples.slice(0, 3).forEach((s) => tiles.push({ src: media("landcoverai", s.image), label: "LandCover.ai tile" }));
  pick("resisc45", rs.data, 3, 20);
  // interleave so sources mix across the grid
  return tiles.map((t, i) => ({ t, k: (i * 7) % tiles.length })).sort((a, b) => a.k - b.k).map((x) => x.t).slice(0, 24);
}

export function BenchmarkCard({ b }: { b: BenchmarkSummary }) {
  const slug = b.slug === "eurosat-c" ? "eurosat" : b.slug;
  const cls = useAsset<ClassificationAssets>(b.task_type === "semantic_segmentation" ? undefined : slug);
  const seg = useAsset<SegmentationAssets>(b.task_type === "semantic_segmentation" ? slug : undefined);
  let strip: string[] = [];
  if (b.slug === "eurosat-c") strip = ["clean", "haze_2", "clouds_2", "gaussian_noise_3", "band_misregistration_3"].map((c) => media("eurosat", `corruptions/0_${c}.jpg`));
  else if (cls.data) strip = Object.values(cls.data.classes).slice(0, 5).map((v) => media(slug, v[0]));
  else if (seg.data) strip = seg.data.samples.slice(0, 5).map((s) => media(slug, s.image));
  return (
    <Link to={`/benchmarks/${b.slug}`} className="card bm-card">
      <div className="strip">{strip.length ? strip.map((s) => <img key={s} src={s} alt="" loading="lazy" />) : null}</div>
      <div className="body">
        <div className="row between" style={{ gap: 8 }}>
          <span className="chip">{TASK_LABEL[b.task_type]}</span>
          <span className="xs muted">{b.n_entries} entries</span>
        </div>
        <div>
          <h3 style={{ fontSize: 18 }}>{b.name}</h3>
          <div className="soft small" style={{ marginTop: 4 }}>{b.tagline}</div>
        </div>
        <div className="xs muted">{b.modality}</div>
        <div className="leader">
          {b.top ? (
            <>
              <div><div className="xs muted">Leader</div><div style={{ fontWeight: 600, fontSize: 14 }}>{b.top.model_name}</div></div>
              <div style={{ textAlign: "right" }}><div className="xs muted">{b.primary_metric_label}</div>
                <div style={{ fontWeight: 700, fontSize: 20 }}>{fmtMetric(b.top.score, "ratio")}</div></div>
            </>
          ) : <span className="xs muted">{b.ready ? "No entries yet" : "Ground truth not installed"}</span>}
        </div>
      </div>
    </Link>
  );
}

function Headline() {
  const clean = useLeaderboard("eurosat");
  const robust = useLeaderboard("eurosat-c");
  if (!clean.data?.length || !robust.data?.length) return null;
  const byName = Object.fromEntries(clean.data.map((e) => [e.model_name, e]));
  const pairs = robust.data.filter((r) => byName[r.model_name]).map((r) => ({ r, c: byName[r.model_name] }));
  if (pairs.length < 3) return null;
  const topClean = [...pairs].sort((a, b) => b.c.score - a.c.score)[0];
  const topRobust = pairs[0];
  const worstKeep = [...pairs].sort((a, b) => a.r.metrics.relative_robustness - b.r.metrics.relative_robustness)[0];
  const pts: ScatterPoint[] = pairs.map(({ r, c }) => {
    const st = familyStyle(r.card?.family);
    return { id: r.submission_id, x: c.score, y: r.score, yLo: r.ci_low, yHi: r.ci_high, label: r.model_name, color: st.color, shape: st.shape,
      href: `/submissions/${r.submission_id}`,
      tip: <><div className="tt-title">{r.model_name}</div><div className="tt-row">Clean EuroSAT <b>{(c.score * 100).toFixed(2)}%</b></div>
        <div className="tt-row">EuroSAT-C mCA <b>{(r.score * 100).toFixed(2)}%</b></div></> };
  });
  const spread = Math.max(...pairs.map((p) => p.c.score)) - Math.min(...pairs.map((p) => p.c.score));
  const rSpread = Math.max(...pairs.map((p) => p.r.score)) - Math.min(...pairs.map((p) => p.r.score));
  const fams = [...new Set(pairs.map((p) => p.r.card?.family).filter((f): f is string => !!f && !!FAMILIES[f]))];
  return (
    <section className="section">
      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.5fr)", gap: 32, alignItems: "center" }}>
        <div>
          <div className="eyebrow">Finding from the baselines</div>
          <h2 style={{ fontSize: 28, marginTop: 8 }}>Clean accuracy hides a {(rSpread / Math.max(spread, 1e-6)).toFixed(0)}× wider robustness gap</h2>
          <p className="soft" style={{ marginTop: 14 }}>
            On clean EuroSAT, {pairs.length} reference models span just {(spread * 100).toFixed(1)} points. Under the 21 EuroSAT-C corruptions they span
            {" "}{(rSpread * 100).toFixed(1)}. The most robust model is <b>{topRobust.r.model_name}</b> ({(topRobust.r.score * 100).toFixed(1)}% mCA);
            {topClean.r.model_name !== topRobust.r.model_name
              ? <> the clean leader, <b>{topClean.r.model_name}</b>, ranks #{topClean.r.rank} under corruption.</>
              : <> it also leads on clean images.</>}
            {" "}<b>{worstKeep.r.model_name}</b> keeps only {(worstKeep.r.metrics.relative_robustness * 100).toFixed(0)}% of its clean accuracy.
          </p>
          <Link to="/benchmarks/eurosat-c?tab=analysis" className="btn secondary" style={{ marginTop: 8 }}>Explore EuroSAT-C <Icon name="arrow" size={15} /></Link>
        </div>
        <div className="card">
          <Scatter points={pts} xLabel="clean EuroSAT accuracy %" yLabel="EuroSAT-C mCA %" height={330}
                   xFormat={(v) => (v * 100).toFixed(1)} yFormat={(v) => (v * 100).toFixed(0)} label="Clean accuracy versus robustness"
                   labelIds={notable(pts)} />
          <Legend items={fams.map((f) => ({ label: f, ...familyStyle(f) }))} />
        </div>
      </div>
    </section>
  );
}

const FEATURES = [
  { icon: "lock", title: "No leaky test sets", text: "The original EuroSAT ids were filenames like AnnualCrop_1 — reading the label off the id scored 100%. split_v2 ids are content hashes: reproducible, label-free." },
  { icon: "target", title: "Rankings with error bars", text: "Every score carries a 95% bootstrap interval. Entries whose intervals overlap the leader share a tier, so noise doesn't masquerade as progress." },
  { icon: "cloud", title: "Robustness, not just accuracy", text: "EuroSAT-C re-scores the hidden split under haze, clouds, sensor noise, compression, coarser GSD and band misregistration." },
  { icon: "gauge", title: "Cost on the same axis", text: "Parameters, GFLOPs and measured T4 throughput are part of every model card, and the Pareto frontier shows what accuracy costs." },
];

export default function Home() {
  const stats = useStats();
  const bms = useBenchmarks();
  const mosaic = useMosaic();
  return (
    <div className="container">
      <section className="hero">
        <div>
          <div className="eyebrow">Earth-observation model benchmarking</div>
          <h1 style={{ marginTop: 12 }}>Benchmarks that tell you <em>more than one number</em>.</h1>
          <p className="lead">
            Upload predictions, get scored on hidden test splits with confidence intervals, robustness under real
            sensor corruptions, calibration and efficiency — evaluated identically for every model.
          </p>
          <div className="row">
            <Link to="/benchmarks" className="btn lg">Explore leaderboards <Icon name="arrow" size={16} /></Link>
            <Link to="/docs" className="btn secondary lg">How it works</Link>
          </div>
        </div>
        <div className="mosaic" aria-hidden={mosaic.length === 0}>
          {mosaic.length ? mosaic.map((t, i) => (
            <figure key={t.src + i} style={{ animationDelay: `${i * 25}ms` }}>
              <img src={t.src} alt={t.label} loading={i < 12 ? "eager" : "lazy"} /><figcaption>{t.label}</figcaption>
            </figure>
          )) : Array.from({ length: 24 }).map((_, i) => <figure key={i} />)}
        </div>
      </section>

      <section className="grid g4 stats-grid">
        {[
          ["Benchmarks", stats.data?.benchmarks, `${stats.data?.task_types ?? "–"} task types`],
          ["Models on the boards", stats.data?.models, stats.data ? `${stats.data.researchers} research group${stats.data.researchers === 1 ? "" : "s"}` : "–"],
          ["Evaluations run", stats.data?.evaluations, "every one on a hidden split"],
          ["Corruption variants", 21, "7 EO failure modes × 3 severities"],
        ].map(([label, value, sub]) => (
          <div className="card" key={label as string}>
            <div className="stat-label">{label}</div>
            <div className="stat-value">{value ?? <Skeleton h={30} w={60} />}</div>
            <div className="stat-sub">{sub}</div>
          </div>
        ))}
      </section>

      <section className="section">
        <div className="section-head">
          <div><h2>Benchmarks</h2><p>{bms.data ? `${bms.data.length} tracks across ${new Set(bms.data.map((b) => b.task_type)).size} task types` : "Tracks"}, from 10 m Sentinel-2 to 25 cm aerial orthophotos.</p></div>
          <Link to="/benchmarks" className="btn ghost">All benchmarks <Icon name="arrow" size={15} /></Link>
        </div>
        <div className="grid bm-grid">
          {bms.data ? bms.data.map((b) => <BenchmarkCard key={b.slug} b={b} />)
            : Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} h={330} />)}
        </div>
      </section>

      <Headline />

      <section className="section">
        <div className="section-head"><div><h2>What a benchmark should guarantee</h2>
          <p>The first version of this hub had none of these. Each one is now enforced by the evaluator, not by convention.</p></div></div>
        <div className="grid g4">
          {FEATURES.map((f) => (
            <div className="card flat" key={f.title}>
              <div className="feature-icon"><Icon name={f.icon} /></div>
              <h3>{f.title}</h3>
              <p className="soft small" style={{ marginTop: 6, marginBottom: 0 }}>{f.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="section grid g2" style={{ alignItems: "start" }}>
        <div>
          <h2 style={{ marginBottom: 18 }}>From notebook to leaderboard</h2>
          <div className="steps" style={{ gridTemplateColumns: "1fr 1fr" }}>
            <div><h3>Train anywhere</h3><p className="soft small">Fork a reference Kaggle notebook — same split code the server uses.</p></div>
            <div><h3>Export predictions</h3><p className="soft small">One CSV (or a ZIP of masks) for the hidden ids. Add probabilities to unlock calibration.</p></div>
            <div><h3>Upload</h3><p className="soft small">Web drag-and-drop or one <code>curl</code> with an API token. Validated before it costs quota.</p></div>
            <div><h3>Publish</h3><p className="soft small">Scores, CIs and diagnostics in seconds. Add a model card to go public.</p></div>
          </div>
        </div>
        <div className="card">
          <div className="card-title"><h3>Latest results</h3><Link to="/benchmarks" className="small">View boards</Link></div>
          {stats.data ? (
            <table style={{ fontSize: 13.5 }}>
              <tbody>
                {stats.data.recent.slice(0, 7).map((r) => (
                  <tr key={r.submission_id}>
                    <td><Link to={`/submissions/${r.submission_id}`} style={{ color: "var(--ink)", fontWeight: 600 }}>{r.model_name}</Link>
                      <div className="xs muted">{r.benchmark_name} · {r.user_name}</div></td>
                    <td className="num"><b>{fmtMetric(r.score)}</b><div className="xs muted">{r.metric_label}</div></td>
                    <td className="r xs muted nowrap">{timeAgo(r.evaluated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <Skeleton h={240} />}
        </div>
      </section>
    </div>
  );
}
