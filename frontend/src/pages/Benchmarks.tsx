import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

export default function Benchmarks() {
  const [benchmarks, setBenchmarks] = useState<any[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    api.benchmarks().then(setBenchmarks).catch((e) => setError(e.message));
  }, []);

  return (
    <>
      <h1>Benchmarks</h1>
      <p className="sub">Upload predictions from your model, get scored on the hidden split.</p>
      {error && <div className="error">{error}</div>}
      {benchmarks.length === 0 && !error && (
        <div className="card hint">
          No benchmarks yet. Run <code>make seed</code> to register EuroSAT.
        </div>
      )}
      {benchmarks.map((b) => (
        <Link key={b.slug} to={`/benchmarks/${b.slug}`} style={{ color: "inherit", textDecoration: "none" }}>
          <div className="card">
            <div className="row">
              <strong>{b.name}</strong>
              <span className="pill ok">{b.task_type}</span>
              <span className="hint mono" style={{ marginLeft: "auto" }}>
                split {b.split_version} · {b.primary_metric}
              </span>
            </div>
            <p className="hint" style={{ margin: "8px 0 0" }}>{b.description}</p>
          </div>
        </Link>
      ))}
    </>
  );
}
