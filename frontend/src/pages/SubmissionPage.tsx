import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { BarList, Heatmap, LineChart, RampLegend, Reliability } from "../components/charts/Charts";
import { ErrorState, Icon, PageSkeleton, StatusBadge, Verified } from "../components/ui/ui";
import { STATIC, api, useBenchmark } from "../lib/api";
import { useAuth } from "../lib/auth";
import { compact, fileSize, fmtDate, fmtMetric, pretty } from "../lib/format";
import type { BenchmarkDetail, ModelCard, Submission } from "../lib/types";

const RUNNING = ["UPLOADED", "QUEUED", "VALIDATING", "EVALUATING"];
const KPI: Record<string, string[]> = {
  scene_classification: ["f1_macro", "ece", "top2_accuracy"],
  scene_classification_robustness: ["clean_accuracy", "relative_robustness", "worst_corruption_accuracy"],
  semantic_segmentation: ["pixel_accuracy", "mean_f1", "fw_iou"],
  multilabel_classification: ["map_micro", "f1_macro", "subset_accuracy"],
};

function Progress({ status }: { status: string }) {
  const steps = ["QUEUED", "VALIDATING", "EVALUATING", "COMPLETED"];
  const at = Math.max(0, steps.indexOf(status === "UPLOADED" ? "QUEUED" : status));
  return (
    <div className="card" aria-live="polite">
      <div className="row" style={{ gap: 0 }}>
        {steps.map((s, i) => (
          <div key={s} style={{ flex: 1 }}>
            <div style={{ height: 4, borderRadius: 2, background: i <= at ? "var(--accent)" : "var(--surface-3)", marginRight: 4 }} />
            <div className="xs" style={{ marginTop: 6, color: i <= at ? "var(--ink)" : "var(--ink-3)", fontWeight: i === at ? 650 : 450 }}>{pretty(s.toLowerCase())}</div>
          </div>
        ))}
      </div>
      <p className="soft small" style={{ margin: "12px 0 0" }}>Evaluating against the hidden split — this page updates automatically.</p>
    </div>
  );
}

function Kpis({ sub, bm }: { sub: Submission; bm: BenchmarkDetail }) {
  const info = Object.fromEntries(bm.metrics.map((m) => [m.name, m]));
  const res = Object.fromEntries(sub.results.map((r) => [r.metric_name, r]));
  const prim = res[bm.primary_metric];
  const others = (KPI[bm.task_type] ?? []).filter((k) => res[k]);
  return (
    <div className="grid g4">
      {prim && (
        <div className="card" style={{ borderColor: "color-mix(in srgb, var(--accent) 40%, var(--border))" }}>
          <div className="stat-label">{info[prim.metric_name]?.label} <span className="badge info">primary</span></div>
          <div className="hero-figure" style={{ fontSize: 40, marginTop: 6 }}>{fmtMetric(prim.value, info[prim.metric_name]?.unit)}<span style={{ fontSize: 20, color: "var(--ink-3)" }}>{info[prim.metric_name]?.unit === "ratio" ? "%" : ""}</span></div>
          {prim.ci_low != null && <div className="stat-sub tnum">95% CI {fmtMetric(prim.ci_low)} – {fmtMetric(prim.ci_high)}</div>}
        </div>
      )}
      {others.map((k) => (
        <div className="card" key={k}>
          <div className="stat-label">{info[k]?.label}</div>
          <div className="stat-value">{fmtMetric(res[k].value, info[k]?.unit)}{info[k]?.unit === "ratio" ? <span style={{ fontSize: 16, color: "var(--ink-3)" }}>%</span> : null}</div>
          <div className="stat-sub">{info[k]?.higher_is_better ? "higher is better" : "lower is better"}</div>
        </div>
      ))}
    </div>
  );
}

function Confusion({ cm, unitLabel }: { cm: { labels: string[]; matrix: number[][] }; unitLabel: string }) {
  const [mode, setMode] = useState<"recall" | "count">("recall");
  const rowSums = cm.matrix.map((r) => r.reduce((a, b) => a + b, 0) || 1);
  const vals = cm.matrix.map((r, i) => r.map((v) => (mode === "recall" ? v / rowSums[i] : v)));
  const max = mode === "recall" ? 1 : Math.max(...cm.matrix.flat());
  const many = cm.labels.length > 15;
  return (
    <div className="card">
      <div className="card-title">
        <div><h3>Confusion matrix</h3><div className="sub">Rows: true class · columns: predicted</div></div>
        <div className="row">
          <RampLegend lo="0" hi={mode === "recall" ? "100%" : compact(max)} />
          <div className="seg" role="group" aria-label="Normalisation">
            <button aria-pressed={mode === "recall"} onClick={() => setMode("recall")}>Row %</button>
            <button aria-pressed={mode === "count"} onClick={() => setMode("count")}>{unitLabel}</button>
          </div>
        </div>
      </div>
      <Heatmap rows={cm.labels.map(pretty)} cols={cm.labels.map(pretty)} values={vals} domain={[0, max]}
               format={(v) => (mode === "recall" ? (v * 100).toFixed(0) : compact(v, 0))} cellH={many ? 16 : 32}
               rowLabelW={many ? 150 : 170} colLabelH={many ? 120 : 100} label="Confusion matrix"
               tipFor={(i, j) => (<><div className="tt-title">{pretty(cm.labels[i])} → {pretty(cm.labels[j])}</div>
                 <div className="tt-row">{unitLabel} <b>{cm.matrix[i][j].toLocaleString()}</b></div>
                 <div className="tt-row">Share of true class <b>{((cm.matrix[i][j] / rowSums[i]) * 100).toFixed(2)}%</b></div></>)} />
    </div>
  );
}

function Diagnostics({ sub, bm }: { sub: Submission; bm: BenchmarkDetail }) {
  const d = sub.details ?? {};
  const ece = sub.results.find((r) => r.metric_name === "ece")?.value;
  if (bm.task_type === "scene_classification_robustness") {
    const corr: string[] = bm.config?.corruptions ?? Object.keys(d.per_corruption ?? {});
    const clean = sub.results.find((r) => r.metric_name === "clean_accuracy")?.value ?? 1;
    const colors = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)", "var(--s7)"];
    return (
      <div className="grid g2">
        <div className="card">
          <div className="card-title"><div><h3>Accuracy vs severity</h3><div className="sub">0 = clean</div></div></div>
          <LineChart label="Accuracy by severity" xLabel="severity" yFormat={(v) => (v * 100).toFixed(0)} height={260}
                     series={corr.map((c, i) => ({ name: pretty(c), color: colors[i % 7],
                       points: [[0, clean], [1, d.per_corruption[c].s1], [2, d.per_corruption[c].s2], [3, d.per_corruption[c].s3]] as [number, number][] }))} />
          <div className="legend">{corr.map((c, i) => <span key={c}><i style={{ width: 12, height: 2, background: colors[i % 7], display: "inline-block" }} />{pretty(c)}</span>)}</div>
        </div>
        <div className="card">
          <div className="card-title"><div><h3>Mean accuracy per corruption</h3><div className="sub">Averaged over 3 severities</div></div></div>
          <BarList label="Accuracy per corruption" format={(v) => (v * 100).toFixed(1)}
                   items={[...corr].sort((a, b) => d.per_corruption[a].mean - d.per_corruption[b].mean).map((c) => ({ label: pretty(c), value: d.per_corruption[c].mean, sub: d.corruption_descriptions?.[c] }))} />
          {d.per_class && <>
            <h3 style={{ margin: "18px 0 8px", fontSize: 14 }}>Accuracy drop by class</h3>
            <BarList label="Accuracy drop per class" color="var(--s8)" max={Math.max(...Object.values<any>(d.per_class).map((v) => v.clean - v.corrupted), 0.01)}
                     format={(v) => `−${(v * 100).toFixed(1)}`}
                     items={Object.entries<any>(d.per_class).sort((a, b) => (b[1].clean - b[1].corrupted) - (a[1].clean - a[1].corrupted)).slice(0, 6)
                       .map(([c, v]) => ({ label: pretty(c), value: v.clean - v.corrupted, sub: `clean ${(v.clean * 100).toFixed(1)}% → ${(v.corrupted * 100).toFixed(1)}%` }))} />
          </>}
        </div>
      </div>
    );
  }
  const isSeg = bm.task_type === "semantic_segmentation";
  const pc = Object.entries<any>(d.per_class ?? {});
  return (
    <div className="stack" style={{ ["--gap" as any]: "16px" }}>
      <div className="grid g2">
        <div className="card">
          <div className="card-title"><div><h3>{isSeg ? "Per-class IoU" : "Per-class error (100 − F1)"}</h3><div className="sub">Weakest first{isSeg ? "" : " · zero-based so small gaps stay visible"}</div></div></div>
          {isSeg ? (
            <BarList label="Per-class IoU" format={(v) => (v * 100).toFixed(1)}
                     items={[...pc].sort((a, b) => (a[1].iou ?? 0) - (b[1].iou ?? 0)).map(([c, v]) => ({
                       label: pretty(c), value: v.iou, sub: `${(v.pixel_share * 100).toFixed(1)}% of pixels` }))} />
          ) : (
            <BarList label="Per-class error" color="var(--s8)" format={(v) => (v * 100).toFixed(2)}
                     max={Math.max(0.001, ...pc.map(([, v]) => 1 - v.f1))}
                     items={[...pc].sort((a, b) => a[1].f1 - b[1].f1).slice(0, 15).map(([c, v]) => ({
                       label: pretty(c), value: 1 - v.f1,
                       sub: `F1 ${(v.f1 * 100).toFixed(2)} · precision ${(v.precision * 100).toFixed(1)} · recall ${(v.recall * 100).toFixed(1)} · n=${v.support}`,
                     }))} />
          )}
          {pc.length > 15 && <p className="xs muted" style={{ marginTop: 8 }}>Showing the 15 weakest of {pc.length} classes.</p>}
        </div>
        {d.reliability ? (
          <div className="card">
            <div className="card-title"><div><h3>Calibration</h3><div className="sub">Does a 90% confidence mean 90% correct?</div></div></div>
            <Reliability bins={d.reliability} ece={ece ?? 0} />
          </div>
        ) : !isSeg ? (
          <div className="card"><h3>Calibration</h3><p className="soft small" style={{ marginTop: 8 }}>This submission has labels only. Add <code>prob_&lt;class&gt;</code> columns to get ECE, NLL, Brier score and a reliability diagram.</p></div>
        ) : <TrainingCurves card={sub.model_card} isSeg />}
      </div>
      {d.confusion_matrix && <Confusion cm={d.confusion_matrix} unitLabel={isSeg ? "Pixels" : "Count"} />}
    </div>
  );
}

function TrainingCurves({ card, isSeg = false }: { card: ModelCard | null; isSeg?: boolean }) {
  const log = card?.training_log;
  if (!log?.length) return null;
  const valKey = isSeg ? "val_miou" : "val_acc";
  return (
    <div className="card">
      <div className="card-title"><div><h3>Training curves</h3><div className="sub">From the training run · best epoch {card?.extra?.best_epoch ?? "—"} kept (by validation)</div></div></div>
      <div className="grid g2" style={{ gap: 12 }}>
        <div><div className="xs muted" style={{ marginBottom: 4 }}>{isSeg ? "Validation mIoU %" : "Accuracy %"}</div>
          <LineChart label="Validation metric by epoch" height={200} xLabel="epoch" yFormat={(v) => (v * 100).toFixed(isSeg ? 0 : 1)}
                     series={[
                       ...(isSeg ? [] : [{ name: "train", color: "var(--s2)", points: log.map((r) => [r.epoch, r.train_acc] as [number, number]) }]),
                       { name: "validation", color: "var(--s1)", points: log.map((r) => [r.epoch, r[valKey]] as [number, number]) },
                     ]} />
        </div>
        <div><div className="xs muted" style={{ marginBottom: 4 }}>Loss</div>
          <LineChart label="Loss by epoch" height={200} xLabel="epoch" yFormat={(v) => v.toFixed(2)}
                     series={[
                       { name: "train", color: "var(--s2)", points: log.map((r) => [r.epoch, r.train_loss] as [number, number]) },
                       ...(isSeg ? [] : [{ name: "validation", color: "var(--s1)", points: log.map((r) => [r.epoch, r.val_loss] as [number, number]) }]),
                     ]} />
        </div>
      </div>
      <div className="legend">
        {!isSeg && <span><i style={{ width: 12, height: 2, background: "var(--s2)", display: "inline-block" }} />train</span>}
        <span><i style={{ width: 12, height: 2, background: "var(--s1)", display: "inline-block" }} />validation</span>
        {isSeg && <span><i style={{ width: 12, height: 2, background: "var(--s2)", display: "inline-block" }} />train loss</span>}
      </div>
    </div>
  );
}

const CARD_FIELDS: [keyof ModelCard, string, "text" | "number" | "url" | "area"][] = [
  ["architecture", "Architecture *", "text"], ["family", "Family (CNN, ViT, …)", "text"], ["backbone", "Backbone", "text"],
  ["pretraining", "Pretraining", "text"], ["num_params", "Parameters", "number"], ["gflops", "GFLOPs / image", "number"],
  ["throughput", "Throughput (img/s)", "number"], ["throughput_setup", "Throughput setup", "text"], ["input_size", "Input size (px)", "number"],
  ["epochs", "Epochs", "number"], ["optimizer", "Optimizer & schedule", "text"], ["augmentation", "Augmentation", "text"],
  ["hardware", "Hardware", "text"], ["training_time", "Training time", "text"], ["framework", "Framework", "text"],
  ["notebook_url", "Notebook URL", "url"], ["repo_url", "Code URL", "url"], ["paper_url", "Paper URL", "url"], ["notes", "Notes", "area"],
];

function CardView({ card }: { card: ModelCard }) {
  const rows: [string, React.ReactNode][] = [
    ["Architecture", card.architecture], ["Family", card.family], ["Backbone", <span className="mono">{card.backbone}</span>],
    ["Pretraining", card.pretraining], ["Parameters", card.num_params ? `${compact(card.num_params, 2)} (${card.num_params.toLocaleString()})` : ""],
    ["Compute", card.gflops ? `${card.gflops.toFixed(2)} GFLOPs / image` : ""],
    ["Throughput", card.throughput ? `${Math.round(card.throughput).toLocaleString()} img/s · ${card.throughput_setup}` : ""],
    ["Input size", card.input_size ? `${card.input_size} px` : ""], ["Epochs", card.epochs ?? ""], ["Optimizer", card.optimizer],
    ["Augmentation", card.augmentation], ["Hardware", card.hardware], ["Training time", card.training_time], ["Framework", card.framework],
    ["Input normalisation", card.extra?.norm ? `${card.extra.norm}${card.extra.norm_probe_knn_acc ? " (chosen by frozen-feature kNN probe on train/val)" : ""}` : ""],
  ];
  return (
    <div className="card">
      <div className="card-title"><h3>Model card</h3>
        <div className="row">
          {card.notebook_url && <a className="btn secondary sm" href={card.notebook_url} target="_blank" rel="noreferrer"><Icon name="kaggle" size={14} />Notebook</a>}
          {card.repo_url && <a className="btn secondary sm" href={card.repo_url} target="_blank" rel="noreferrer"><Icon name="external" size={14} />Code</a>}
          {card.paper_url && <a className="btn secondary sm" href={card.paper_url} target="_blank" rel="noreferrer"><Icon name="book" size={14} />Paper</a>}
        </div>
      </div>
      <dl className="kv">{rows.filter(([, v]) => v !== "" && v != null).map(([k, v]) => <FragmentKV key={k} k={k} v={v} />)}</dl>
      {card.notes && <p className="soft small" style={{ marginTop: 14 }}>{card.notes}</p>}
    </div>
  );
}
const FragmentKV = ({ k, v }: { k: string; v: React.ReactNode }) => <><dt>{k}</dt><dd>{v}</dd></>;

function CardEditor({ sub, onSaved }: { sub: Submission; onSaved: () => void }) {
  const init = sub.model_card ?? ({} as ModelCard);
  const [card, setCard] = useState<Record<string, any>>(Object.fromEntries(CARD_FIELDS.map(([k]) => [k, (init as any)[k] ?? ""])));
  const save = useMutation({
    mutationFn: () => api.saveModelCard(sub.id, Object.fromEntries(Object.entries(card).map(([k, v]) => {
      const t = CARD_FIELDS.find((f) => f[0] === k)?.[2];
      return [k, t === "number" ? (v === "" ? null : Number(v)) : v];
    }))),
    onSuccess: onSaved,
  });
  return (
    <div className="card">
      <div className="card-title"><h3>Model card</h3><span className="sub">Required to publish · shown on the leaderboard</span></div>
      <form className="form-grid" onSubmit={(e) => { e.preventDefault(); save.mutate(); }}>
        {CARD_FIELDS.map(([k, label, t]) => (
          <label key={k} className={`field ${t === "area" ? "full" : ""}`}><span>{label}</span>
            {t === "area"
              ? <textarea rows={3} value={card[k]} onChange={(e) => setCard({ ...card, [k]: e.target.value })} />
              : <input type={t === "number" ? "number" : t === "url" ? "url" : "text"} step="any" value={card[k]}
                       onChange={(e) => setCard({ ...card, [k]: e.target.value })} required={k === "architecture"} />}
          </label>
        ))}
        <div className="full row">
          <button className="btn" disabled={save.isPending}>{save.isPending ? "Saving…" : "Save model card"}</button>
          {save.isSuccess && <span className="small" style={{ color: "var(--good-ink)" }}>Saved</span>}
          {save.error && <span className="small" style={{ color: "var(--bad-ink)" }}>{(save.error as Error).message}</span>}
        </div>
      </form>
    </div>
  );
}

export default function SubmissionPage() {
  const { id = "" } = useParams();
  const { user } = useAuth();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const [editing, setEditing] = useState(false);
  const q = useQuery({
    queryKey: ["submission", id], queryFn: () => api.submission(id),
    refetchInterval: (query) => (query.state.data && RUNNING.includes(query.state.data.status) ? 1500 : false),
  });
  const bm = useBenchmark(q.data?.benchmark.slug ?? "");
  const refresh = () => { qc.invalidateQueries({ queryKey: ["submission", id] }); qc.invalidateQueries({ queryKey: ["leaderboard"] }); };
  const act = useMutation({
    mutationFn: async (a: "publish" | "unpublish" | "delete") => (a === "delete" ? api.remove(id) : a === "publish" ? api.publish(id) : api.unpublish(id)),
    onSuccess: (_, a) => (a === "delete" ? navigate("/account") : refresh()),
  });
  if (q.error) return <ErrorState error={q.error} />;
  if (!q.data || (q.data.status === "COMPLETED" && !bm.data)) return <PageSkeleton />;
  const sub = q.data;
  const owner = !STATIC && user?.id === sub.user.id;
  const running = RUNNING.includes(sub.status);

  return (
    <div className="container">
      <div className="page-head">
        <div className="crumbs"><Link to="/benchmarks">Benchmarks</Link> / <Link to={`/benchmarks/${sub.benchmark.slug}`}>{sub.benchmark.name}</Link></div>
        <div className="row between" style={{ alignItems: "start" }}>
          <div>
            <h1>{sub.model_name}</h1>
            <div className="row" style={{ marginTop: 10, gap: 8 }}>
              <StatusBadge status={sub.status} />
              {sub.verified && <Verified />}
              {sub.is_published ? <span className="badge info">public</span> : <span className="badge neutral"><Icon name="lock" size={11} />private</span>}
              {sub.rank && <span className="badge neutral">#{sub.rank} of {sub.n_ranked} on {sub.benchmark.name}</span>}
              <span className="small muted">by <Link to={`/users/${sub.user.id}`}>{sub.user.name}</Link>{sub.user.affiliation ? ` · ${sub.user.affiliation}` : ""} · {fmtDate(sub.created_at)}</span>
            </div>
          </div>
          {owner && sub.status === "COMPLETED" && (
            <div className="row">
              {sub.is_published
                ? <button className="btn secondary sm" onClick={() => act.mutate("unpublish")}>Unpublish</button>
                : <button className="btn sm" onClick={() => act.mutate("publish")} disabled={act.isPending}>Publish to leaderboard</button>}
              <button className="btn secondary sm" onClick={() => setEditing(!editing)}>{editing ? "Close editor" : "Edit model card"}</button>
              {!sub.is_published && <button className="btn danger sm" onClick={() => confirm("Delete this submission?") && act.mutate("delete")}>Delete</button>}
            </div>
          )}
        </div>
        {sub.description && <p className="soft" style={{ marginTop: 12 }}>{sub.description}</p>}
        {act.error && <div className="alert bad" style={{ marginTop: 12 }}>{(act.error as Error).message}</div>}
      </div>

      <div className="stack" style={{ ["--gap" as any]: "16px", marginTop: 24 }}>
        {running && <Progress status={sub.status} />}
        {sub.error_report?.errors && (
          <div className="alert bad">
            <b>{sub.status === "REJECTED" ? "The file was rejected — it did not count against your quota." : "Evaluation failed."}</b>
            <ul>{sub.error_report.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
          </div>
        )}
        {sub.status === "COMPLETED" && bm.data && (
          <>
            <Kpis sub={sub} bm={bm.data} />
            {owner && (editing || !sub.model_card) && <CardEditor sub={sub} onSaved={() => { refresh(); setEditing(false); }} />}
            <Diagnostics sub={sub} bm={bm.data} />
            {bm.data.task_type !== "semantic_segmentation" && <TrainingCurves card={sub.model_card} />}
            <div className="grid g2" style={{ alignItems: "start" }}>
              {sub.model_card ? <CardView card={sub.model_card} /> : <div className="card"><h3>Model card</h3><p className="soft small" style={{ marginTop: 8 }}>No model card yet.</p></div>}
              <div className="stack" style={{ ["--gap" as any]: "16px" }}>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Metric</th><th className="num">Value</th><th className="num">95% CI</th></tr></thead>
                    <tbody>
                      {bm.data.metrics.filter((m) => sub.results.some((r) => r.metric_name === m.name)).map((m) => {
                        const r = sub.results.find((x) => x.metric_name === m.name)!;
                        return (
                          <tr key={m.name} title={m.description}>
                            <td>{m.label}{m.name === bm.data!.primary_metric && <span className="badge info" style={{ marginLeft: 6 }}>primary</span>}</td>
                            <td className="num"><b>{fmtMetric(r.value, m.unit, m.unit === "ratio" ? 2 : 4)}</b></td>
                            <td className="num muted">{r.ci_low != null ? `${fmtMetric(r.ci_low, m.unit)} – ${fmtMetric(r.ci_high, m.unit)}` : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="card">
                  <h3 style={{ marginBottom: 10 }}>Provenance</h3>
                  <dl className="kv small">
                    <dt>File</dt><dd className="mono">{sub.original_filename} · {fileSize(sub.file_size)}</dd>
                    <dt>SHA-256</dt><dd className="mono">{sub.checksum.slice(0, 32)}…</dd>
                    <dt>Evaluator</dt><dd className="mono">{sub.evaluator_version}</dd>
                    <dt>Scored on</dt><dd>{sub.details?.num_samples?.toLocaleString()} hidden {bm.data.task_type === "semantic_segmentation" ? "tiles" : "images"}{sub.details?.num_variants ? ` × ${sub.details.num_variants + 1} variants` : ""}</dd>
                    <dt>Evaluated</dt><dd>{fmtDate(sub.evaluated_at)}</dd>
                  </dl>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
