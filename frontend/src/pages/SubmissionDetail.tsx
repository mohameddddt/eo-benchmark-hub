import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";

const RUNNING = new Set(["UPLOADED", "QUEUED", "VALIDATING", "EVALUATING"]);
const pillClass = (s: string) => (s === "COMPLETED" ? "ok" : RUNNING.has(s) ? "warn" : "bad");

const CARD_FIELDS: [string, string][] = [
  ["architecture", "Architecture"], ["backbone", "Backbone"], ["num_params", "Parameters"],
  ["input_size", "Input size"], ["pretraining", "Pretraining"], ["hardware", "Hardware"],
  ["training_time", "Training time"], ["notebook_url", "Notebook URL"], ["notes", "Notes"],
];

export default function SubmissionDetail() {
  const { id } = useParams();
  const [sub, setSub] = useState<any>(null);
  const [card, setCard] = useState<Record<string, string>>({});
  const [msg, setMsg] = useState("");
  const timer = useRef<number>();

  async function refresh() {
    if (!id) return;
    const s = await api.submission(id).catch(() => null);
    if (s) setSub(s);
    if (s && RUNNING.has(s.status)) timer.current = window.setTimeout(refresh, 2000);
  }

  useEffect(() => {
    refresh();
    return () => window.clearTimeout(timer.current);
  }, [id]);

  if (!sub) return <p className="hint">Loading…</p>;

  const primary = sub.results.find((r: any) => r.extra) ?? sub.results[0];
  const perClass = primary?.extra?.per_class ?? null;

  async function saveCard() {
    setMsg("");
    await api.saveModelCard(sub.id, card);
    setMsg("Model card saved");
    refresh();
  }
  async function publish() {
    setMsg("");
    try {
      await api.publish(sub.id);
      setMsg("Published to the leaderboard");
      refresh();
    } catch (e: any) {
      setMsg(e.message);
    }
  }

  return (
    <>
      <h1 className="row">
        Submission <code>{sub.id.slice(0, 8)}</code>
        <span className={`pill ${pillClass(sub.status)}`}>{sub.status}</span>
        {sub.is_published && <span className="pill ok">PUBLISHED</span>}
      </h1>
      <p className="sub mono">
        {sub.benchmark_slug} · {sub.original_filename} · sha256 {sub.checksum.slice(0, 16)}
      </p>

      {RUNNING.has(sub.status) && (
        <div className="card hint">Evaluating — this page refreshes automatically.</div>
      )}

      {sub.error_report && (
        <div className="card">
          <strong>What went wrong</strong>
          {(sub.error_report.errors ?? []).map((e: string, i: number) => (
            <div className="error" key={i}>✗ {e}</div>
          ))}
        </div>
      )}

      {sub.status === "COMPLETED" && (
        <>
          <div className="card">
            <div className="row" style={{ gap: 40 }}>
              {sub.results.filter((r: any) => !r.metric_name.includes("per_")).map((r: any) => (
                <div key={r.metric_name}>
                  <div className="score">{r.value.toFixed(4)}</div>
                  <div className="score-label">{r.metric_name}</div>
                </div>
              ))}
            </div>
          </div>

          {perClass && (
            <>
              <h2>Per-class F1</h2>
              <table>
                <thead>
                  <tr><th>Class</th><th>F1</th><th></th><th>Support</th></tr>
                </thead>
                <tbody>
                  {Object.entries(perClass).map(([cls, m]: any) => (
                    <tr key={cls}>
                      <td>{cls}</td>
                      <td className="num">{m.f1.toFixed(4)}</td>
                      <td><div className="bar-track"><div className="bar-fill" style={{ width: `${m.f1 * 100}%` }} /></div></td>
                      <td className="num">{m.support}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}

          <h2>Model card</h2>
          <div className="card">
            <p className="hint">Required before publishing to the leaderboard.</p>
            {CARD_FIELDS.map(([key, label]) => (
              <div key={key}>
                <label htmlFor={key}>{label}</label>
                {key === "notes" ? (
                  <textarea id={key} rows={3} value={card[key] ?? ""}
                            onChange={(e) => setCard({ ...card, [key]: e.target.value })} />
                ) : (
                  <input id={key} value={card[key] ?? ""}
                         onChange={(e) => setCard({ ...card, [key]: e.target.value })} />
                )}
              </div>
            ))}
            <div className="row" style={{ marginTop: 14 }}>
              <button className="quiet" type="button" onClick={saveCard}>Save model card</button>
              {!sub.is_published && <button type="button" onClick={publish}>Publish to leaderboard</button>}
            </div>
            {msg && <div className="hint" style={{ marginTop: 8 }}>{msg}</div>}
          </div>
        </>
      )}
    </>
  );
}
