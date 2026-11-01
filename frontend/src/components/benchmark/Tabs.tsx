import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { STATIC, api, media, useAsset } from "../../lib/api";
import { useAuth } from "../../lib/auth";
import { fmtInt, pretty, sampleUnit } from "../../lib/format";
import type { BenchmarkDetail, ClassificationAssets, LeaderboardEntry, SegmentationAssets } from "../../lib/types";
import { BarList } from "../charts/Charts";
import { CodeBlock, Icon } from "../ui/ui";

const SPLIT_ORDER = ["train", "val", "test_public", "test_hidden"];
const SPLIT_LABEL: Record<string, string> = { train: "Train", val: "Validation", test_public: "Test · public", test_hidden: "Test · hidden" };

// ================================================================ Dataset
export function DatasetTab({ bm, entries }: { bm: BenchmarkDetail; entries: LeaderboardEntry[] }) {
  const isSeg = bm.task_type === "semantic_segmentation";
  const isRobust = bm.task_type === "scene_classification_robustness";
  const assetSlug = (bm.config?.ground_truth as string) || bm.slug;
  const cls = useAsset<ClassificationAssets>(isSeg ? undefined : assetSlug);
  const seg = useAsset<SegmentationAssets>(isSeg ? bm.slug : undefined);
  const st = bm.stats ?? {};
  const splits = st.splits ?? {};
  const total = Object.values(splits).reduce((a, b) => a + b, 0) || 1;
  const unit = sampleUnit(bm.task_type);

  return (
    <div className="stack" style={{ ["--gap" as any]: "16px" }}>
      <div className="grid g2" style={{ gridTemplateColumns: "minmax(0,1.4fr) minmax(0,1fr)" }}>
        <div className="card">
          <h3 style={{ marginBottom: 10 }}>About the data</h3>
          <p className="soft">{bm.description}</p>
          <dl className="kv small" style={{ marginTop: 14 }}>
            <dt>Modality</dt><dd>{bm.modality}</dd>
            <dt>Classes</dt><dd>{bm.n_classes}</dd>
            <dt>License</dt><dd>{bm.license}</dd>
            <dt>Source</dt><dd><a href={bm.dataset_url} target="_blank" rel="noreferrer">{bm.dataset_url.replace(/^https?:\/\//, "").slice(0, 48)}</a></dd>
            <dt>Citation</dt><dd className="soft">{bm.citation} {bm.paper_url && <a href={bm.paper_url} target="_blank" rel="noreferrer">paper ↗</a>}</dd>
          </dl>
        </div>
        <div className="card">
          <h3 style={{ marginBottom: 12 }}>Split {bm.split_version}</h3>
          {isRobust ? (
            <p className="soft small">Uses the EuroSAT {bm.split_version} hidden split ({fmtInt(splits.test_hidden)} images), each scored clean and under
              {" "}{(bm.config?.corruptions?.length ?? 0) * (bm.config?.severities?.length ?? 0)} corruption variants — {fmtInt(st.evaluated_images)} predictions per submission.</p>
          ) : null}
          <div style={{ display: "flex", height: 12, borderRadius: 6, overflow: "hidden", gap: 2, margin: "6px 0 12px" }}>
            {SPLIT_ORDER.filter((s) => splits[s]).map((s, i) => (
              <div key={s} title={`${SPLIT_LABEL[s]}: ${splits[s]}`} style={{ width: `${(splits[s] / total) * 100}%`, background: `var(--q${[600, 400, 300, 200][i]})` }} />
            ))}
          </div>
          <table style={{ fontSize: 13 }}>
            <tbody>
              {SPLIT_ORDER.filter((s) => splits[s] != null).map((s) => (
                <tr key={s}><td>{SPLIT_LABEL[s]}{s === "test_hidden" && <span className="badge neutral" style={{ marginLeft: 6 }}><Icon name="lock" size={11} />labels server-only</span>}</td>
                  <td className="num">{fmtInt(splits[s])} {unit}</td><td className="num muted">{((splits[s] / total) * 100).toFixed(0)}%</td></tr>
              ))}
            </tbody>
          </table>
          <dl className="kv small" style={{ marginTop: 14 }}>
            <dt>Fingerprint</dt><dd className="mono">{st.fingerprint ?? "—"}</dd>
            {st.duplicates_dropped ? <><dt>Deduplicated</dt><dd>{st.duplicates_dropped} byte-identical images removed</dd></> : null}
          </dl>
          <a className="btn secondary sm" style={{ marginTop: 14 }} href={media(assetSlug, `split_manifest_${bm.split_version}.csv`)} download>
            Public manifest (ids + splits)
          </a>
        </div>
      </div>

      {isRobust && <CorruptionGallery bm={bm} />}

      {cls.data && !isRobust && (
        <div className="card">
          <div className="card-title"><h3>Classes</h3><span className="sub">Random training examples</span></div>
          {bm.n_classes <= 12 ? (
            <div>
              {Object.entries(cls.data.classes).map(([c, imgs]) => (
                <div className="class-row" key={c}>
                  <div><div style={{ fontWeight: 600 }}>{pretty(c)}</div>
                    {st.class_counts?.[c] && <div className="xs muted">{fmtInt(st.class_counts[c])} images</div>}</div>
                  <div className="imgs">{imgs.map((p) => <img key={p} src={media(assetSlug, p)} alt={c} loading="lazy" />)}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="thumbs" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(110px, 1fr))" }}>
              {Object.entries(cls.data.classes).map(([c, imgs]) => (
                <figure className="thumb" key={c}><img src={media(assetSlug, imgs[0])} alt={c} loading="lazy" /><figcaption>{pretty(c)}</figcaption></figure>
              ))}
            </div>
          )}
        </div>
      )}

      {cls.data && cls.data.gallery.length > 0 && !isRobust && <Mistakes bm={bm} assets={cls.data} entries={entries} />}

      {cls.data?.label_frequency && (
        <div className="card">
          <div className="card-title"><h3>Label frequency</h3>
            <span className="sub">Share of training patches carrying each label · {cls.data.mean_labels_per_patch?.toFixed(2)} labels per patch on average — why macro mAP, not accuracy, is primary</span></div>
          <BarList items={Object.entries(cls.data.label_frequency).sort((a, b) => b[1] - a[1]).map(([c, v]) => ({ label: c, value: v }))}
                   format={(v) => `${(v * 100).toFixed(1)}%`} max={Math.max(...Object.values(cls.data.label_frequency))}
                   label="Label frequency" />
        </div>
      )}

      {isSeg && st.hidden_pixel_share && (
        <div className="card">
          <div className="card-title"><h3>Class balance</h3><span className="sub">Share of hidden-split pixels — why mIoU, not pixel accuracy, is primary</span></div>
          <BarList items={Object.entries(st.hidden_pixel_share).map(([c, v]) => ({ label: pretty(c), value: v }))}
                   format={(v) => `${(v * 100).toFixed(1)}%`} max={Math.max(...Object.values(st.hidden_pixel_share))}
                   label="Pixel share per class" />
          {seg.data && <p className="xs muted" style={{ marginTop: 10 }}>See the Analysis tab for tile-level predictions.</p>}
        </div>
      )}
    </div>
  );
}

function Mistakes({ bm, assets, entries }: { bm: BenchmarkDetail; assets: ClassificationAssets; entries: LeaderboardEntry[] }) {
  const mistakes = assets.gallery.filter((g) => g.kind === "mistake").slice(0, 12);
  const names = assets.model_names ?? {};
  const best = entries[0];
  const bestId = Object.entries(names).find(([, n]) => n === best?.model_name)?.[0];
  if (!mistakes.length) return null;
  return (
    <div className="card">
      <div className="card-title">
        <div><h3>Where the models disagree with the labels</h3>
          <div className="sub">Most confident mistakes of the strongest baseline on the <b>public</b> test split (hidden-split images are never shown). Hover for every model's vote.</div></div>
      </div>
      <div className="thumbs" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(124px, 1fr))", gap: 14 }}>
        {mistakes.map((g) => {
          const votes = Object.entries(g.predictions);
          const p = (bestId && g.predictions[bestId]) || votes[0]?.[1];
          const agree = votes.filter(([, v]) => v.label === g.label).length;
          return (
            <figure className="thumb" key={g.sample_id} title={votes.map(([id, v]) => `${names[id] ?? id}: ${v.label} (${(v.confidence * 100).toFixed(0)}%)`).join("\n")}>
              <img src={media(bm.slug, g.image)} alt={`Labelled ${g.label}`} loading="lazy" />
              <figcaption>
                <div><span className="muted">label</span> {pretty(g.label)}</div>
                <div><span className="muted">pred</span> <b>{pretty(p?.label ?? "")}</b> {p && <span className="muted">{(p.confidence * 100).toFixed(0)}%</span>}</div>
                <div className="muted">{agree}/{votes.length} models agree with label</div>
              </figcaption>
            </figure>
          );
        })}
      </div>
    </div>
  );
}

function CorruptionGallery({ bm }: { bm: BenchmarkDetail }) {
  const corr: string[] = bm.config?.corruptions ?? [];
  const info: Record<string, { description: string; params: number[] }> = bm.config?.corruption_info ?? {};
  const [ex, setEx] = useState(0);
  return (
    <div className="card">
      <div className="card-title">
        <div><h3>The 21 corruptions</h3><div className="sub">Deterministic per image — regenerate them with <code>eobench_eval.corruptions.corrupt()</code></div></div>
        <div className="seg" role="group" aria-label="Example image">
          {[0, 1].map((i) => <button key={i} aria-pressed={ex === i} onClick={() => setEx(i)}>Example {i + 1}</button>)}
        </div>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ minWidth: 560 }}>
          <thead><tr><th>Corruption</th><th>Clean</th><th>Severity 1</th><th>Severity 2</th><th>Severity 3</th></tr></thead>
          <tbody>
            {corr.map((c) => (
              <tr key={c}>
                <td style={{ maxWidth: 220 }}><div style={{ fontWeight: 600 }}>{pretty(c)}</div><div className="xs muted">{info[c]?.description}</div></td>
                <td><img src={media("eurosat", `corruptions/${ex}_clean.jpg`)} alt="clean" width={72} height={72} style={{ borderRadius: 6 }} loading="lazy" /></td>
                {[1, 2, 3].map((s) => (
                  <td key={s}><img src={media("eurosat", `corruptions/${ex}_${c}_${s}.jpg`)} alt={`${c} severity ${s}`} width={72} height={72} style={{ borderRadius: 6 }} loading="lazy" />
                    <div className="xs muted tnum">{info[c]?.params?.[s - 1]}</div></td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ================================================================ Submit
export function SubmitTab({ bm }: { bm: BenchmarkDetail }) {
  const { user } = useAuth();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [desc, setDesc] = useState("");
  const [over, setOver] = useState(false);
  const spec = bm.format_spec;
  const mut = useMutation({
    mutationFn: () => api.submit(bm.slug, file!, name || file!.name.replace(/\.[^.]+$/, ""), desc),
    onSuccess: (s) => { qc.invalidateQueries({ queryKey: ["mine"] }); navigate(`/submissions/${s.id}`); },
  });
  const isSeg = bm.task_type === "semantic_segmentation";
  const py = bm.task_type === "multilabel_classification"
    ? `import csv
# probs: (n_patches, ${bm.n_classes}) sigmoid outputs, columns in canonical class order
with open("predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id"] + [f"prob_{j:02d}" for j in range(${bm.n_classes})])
    for sid, p in zip(ids, probs):
        w.writerow([sid] + [f"{v:.6f}" for v in p])`
    : isSeg
    ? `import io, zipfile\nfrom PIL import Image\n\n# preds: {tile_id: HxW uint8 array of class indices 0..${bm.n_classes - 1}}\nwith zipfile.ZipFile("predictions.zip", "w") as z:\n    for tile_id, mask in preds.items():\n        buf = io.BytesIO()\n        Image.fromarray(mask, mode="L").save(buf, format="PNG")\n        z.writestr(f"{tile_id}.png", buf.getvalue())`
    : bm.task_type === "scene_classification_robustness"
      ? `from eobench_eval.corruptions import all_variants, corrupt, variant_id\n\nrows = []\nfor sid, img in hidden_images.items():          # uint8 HxWx3\n    rows.append((sid, predict(img)))\n    for c, s in all_variants():\n        rows.append((variant_id(sid, c, s), predict(corrupt(img, sid, c, s))))\n\nimport csv\nwith open("predictions.csv", "w", newline="") as f:\n    w = csv.writer(f); w.writerow(["sample_id", "predicted_label"]); w.writerows(rows)`
      : `import csv\nCLASSES = [${(bm.classes ?? []).slice(0, 3).map((c) => `"${c}"`).join(", ")}, ...]  # canonical order\n\nwith open("predictions.csv", "w", newline="") as f:\n    w = csv.writer(f)\n    w.writerow(["sample_id", "predicted_label"] + [f"prob_{c}" for c in CLASSES])\n    for sid, probs in zip(ids, softmax_outputs):   # probs: len(CLASSES)\n        w.writerow([sid, CLASSES[probs.argmax()]] + [f"{p:.6f}" for p in probs])`;
  const curl = `curl -X POST ${location.origin}${import.meta.env.BASE_URL}api/benchmarks/${bm.slug}/submissions \\\n  -H "Authorization: Bearer $EOB_TOKEN" \\\n  -F "file=@predictions.${isSeg ? "zip" : "csv"}" -F "model_name=My model"`;

  return (
    <div className="grid g2" style={{ gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)", alignItems: "start" }}>
      <div className="stack" style={{ ["--gap" as any]: "16px" }}>
        <div className="card">
          <h3 style={{ marginBottom: 6 }}>Upload predictions</h3>
          <p className="soft small">Scored against the hidden split in seconds. Up to {bm.weekly_quota} scored submissions per week; rejected files don't count. Results stay private until you publish them with a model card.</p>
          {STATIC ? (
            <div className="alert info">This is a read-only snapshot. <a href="https://github.com/mohameddddt/eo-benchmark-hub#quickstart">Run the hub locally</a> to submit.</div>
          ) : !user ? (
            <div className="alert info"><Link to="/login">Sign in</Link> to submit predictions.</div>
          ) : !bm.ready ? (
            <div className="alert bad">This benchmark's hidden labels are not installed on this server yet.</div>
          ) : (
            <form className="stack" style={{ ["--gap" as any]: "14px" }} onSubmit={(e) => { e.preventDefault(); if (file) mut.mutate(); }}>
              <div className={`dropzone ${over ? "over" : ""}`} onClick={() => input.current?.click()}
                   onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
                   onDrop={(e) => { e.preventDefault(); setOver(false); setFile(e.dataTransfer.files[0] ?? null); }}
                   role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && input.current?.click()}>
                <Icon name="upload" size={22} />
                <div style={{ fontWeight: 600, marginTop: 6 }}>{file ? file.name : `Drop your .${spec.file_type ?? "csv"} here`}</div>
                <div className="xs muted">{file ? `${(file.size / 1024).toFixed(0)} KB · click to change` : "or click to browse"}</div>
                <input ref={input} type="file" hidden accept={`.${spec.file_type ?? "csv"}`} onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              </div>
              <label className="field"><span>Model name</span><input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. ConvNeXt-T + mixup" maxLength={120} /></label>
              <label className="field"><span>Notes (optional)</span><textarea rows={2} value={desc} onChange={(e) => setDesc(e.target.value)} maxLength={2000} /></label>
              {mut.error && <div className="alert bad">{(mut.error as Error).message}</div>}
              <button className="btn" disabled={!file || mut.isPending}>{mut.isPending ? "Uploading…" : "Upload and evaluate"}</button>
            </form>
          )}
        </div>
        <div className="card">
          <h3 style={{ marginBottom: 10 }}>Submit from Kaggle or Colab</h3>
          <p className="soft small">Create an API token in your <Link to="/account">account</Link>, then:</p>
          <CodeBlock code={curl} lang="bash" />
        </div>
      </div>
      <div className="stack" style={{ ["--gap" as any]: "16px" }}>
        <div className="card">
          <h3 style={{ marginBottom: 10 }}>File format</h3>
          <dl className="kv small" style={{ marginBottom: 12 }}>
            <dt>Type</dt><dd><code>.{spec.file_type}</code></dd>
            <dt>Layout</dt><dd><code>{spec.header}</code></dd>
            <dt>Rows</dt><dd>{spec.rows}</dd>
          </dl>
          {spec.columns && (
            <table style={{ fontSize: 13, marginBottom: 12 }}>
              <tbody>{spec.columns.map((c) => <tr key={c.name}><td className="mono nowrap">{c.name}</td><td className="soft">{c.description}</td></tr>)}</tbody>
            </table>
          )}
          {spec.example && <CodeBlock code={spec.example} />}
        </div>
        <div className="card">
          <h3 style={{ marginBottom: 10 }}>Writing the file</h3>
          <CodeBlock code={py} lang="python" />
          {bm.starter_url && (
            <a className="btn secondary sm" style={{ marginTop: 12 }} href={bm.starter_url} target="_blank" rel="noreferrer">
              <Icon name="kaggle" size={14} />Reference Kaggle notebook
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

// ================================================================ Protocol
export function ProtocolTab({ bm }: { bm: BenchmarkDetail }) {
  return (
    <div className="grid g2" style={{ gridTemplateColumns: "minmax(0,1.3fr) minmax(0,1fr)", alignItems: "start" }}>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Metric</th><th>Definition</th><th>Better</th></tr></thead>
          <tbody>
            {bm.metrics.map((m) => (
              <tr key={m.name}>
                <td className="nowrap"><b>{m.label}</b>{m.name === bm.primary_metric && <span className="badge info" style={{ marginLeft: 6 }}>primary</span>}
                  <div className="xs muted mono">{m.name}</div></td>
                <td className="soft small">{m.description}</td>
                <td className="small">{m.higher_is_better ? "higher" : "lower"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card">
        <h3 style={{ marginBottom: 10 }}>Ranking rules</h3>
        <ul className="soft small" style={{ paddingLeft: 18, margin: 0, display: "grid", gap: 8 }}>
          <li>Ranked by <b>{bm.primary_metric_label}</b> on the hidden split; ties broken by submission time.</li>
          <li><b>Tiers:</b> the leader's 95% bootstrap interval opens tier 1; entries whose interval reaches it join. Same tier ⇒ not distinguishable at this test-set size.</li>
          <li>One row per (researcher, model name) — resubmitting the same model keeps its best score, so probing the test set doesn't stack the board.</li>
          <li>Publishing requires a model card. <b>Verified</b> means maintainers reproduced the score from the linked notebook.</li>
        </ul>
        <dl className="kv small" style={{ marginTop: 16 }}>
          <dt>Evaluator</dt><dd className="mono">{bm.evaluator_version}</dd>
          <dt>Split</dt><dd className="mono">{bm.split_version} · {bm.stats?.fingerprint}</dd>
          <dt>Quota</dt><dd>{bm.weekly_quota} scored submissions / week</dd>
        </dl>
        <Link to="/docs#protocol" className="btn secondary sm" style={{ marginTop: 14 }}>Full protocol <Icon name="arrow" size={14} /></Link>
      </div>
    </div>
  );
}
