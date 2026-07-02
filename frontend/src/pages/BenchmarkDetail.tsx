import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, getToken } from "../api";

export default function BenchmarkDetail() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const [bm, setBm] = useState<any>(null);
  const [board, setBoard] = useState<any[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!slug) return;
    api.benchmark(slug).then(setBm).catch((e) => setError(e.message));
    api.leaderboard(slug).then(setBoard).catch(() => {});
  }, [slug]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file || !slug) return;
    setBusy(true);
    setError("");
    try {
      const sub = await api.submit(slug, file);
      navigate(`/submissions/${sub.id}`);
    } catch (err: any) {
      setError(err.message);
      setBusy(false);
    }
  }

  if (!bm) return error ? <div className="error">{error}</div> : <p className="hint">Loading…</p>;

  return (
    <>
      <h1>{bm.name}</h1>
      <p className="sub">{bm.description}</p>
      <div className="row hint mono">
        <span>task: {bm.task_type}</span>
        <span>split: {bm.split_version}</span>
        <span>primary metric: {bm.primary_metric}</span>
        <a href={bm.dataset_url} target="_blank" rel="noreferrer">dataset ↗</a>
      </div>

      <h2>Submit predictions</h2>
      <div className="card">
        {getToken() ? (
          <form onSubmit={onSubmit}>
            <label htmlFor="file">Prediction file (CSV: sample_id,predicted_label)</label>
            <input id="file" type="file" accept=".csv,.json,.zip,.geojson"
                   onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <div style={{ marginTop: 12 }}>
              <button disabled={!file || busy}>{busy ? "Uploading…" : "Upload and evaluate"}</button>
            </div>
            {error && <div className="error">{error}</div>}
          </form>
        ) : (
          <p className="hint"><Link to="/login">Sign in</Link> to submit predictions.</p>
        )}
      </div>

      <h2>Leaderboard</h2>
      {board.length === 0 ? (
        <div className="card hint">No published results yet. Yours could be the first entry.</div>
      ) : (
        <table>
          <thead>
            <tr><th>#</th><th>Researcher</th><th>{bm.primary_metric}</th><th>Date</th></tr>
          </thead>
          <tbody>
            {board.map((row) => (
              <tr key={row.submission_id}>
                <td className="num">{row.rank}</td>
                <td>{row.user_name}</td>
                <td className="num">{row.score.toFixed(4)}</td>
                <td className="num">{new Date(row.created_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
