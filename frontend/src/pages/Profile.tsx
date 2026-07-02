import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

export default function Profile() {
  const [me, setMe] = useState<any>(null);
  const [tokens, setTokens] = useState<any[]>([]);
  const [subs, setSubs] = useState<any[]>([]);
  const [tokenName, setTokenName] = useState("colab");
  const [newToken, setNewToken] = useState("");

  function load() {
    api.me().then(setMe).catch(() => {});
    api.listTokens().then(setTokens).catch(() => {});
    api.mySubmissions().then(setSubs).catch(() => {});
  }
  useEffect(load, []);

  async function createToken(e: FormEvent) {
    e.preventDefault();
    const t = await api.createToken(tokenName);
    setNewToken(t.token);
    load();
  }

  return (
    <>
      <h1>{me ? me.name : "Profile"}</h1>
      <p className="sub">{me?.email}</p>

      <h2>API tokens</h2>
      <div className="card">
        <p className="hint">Use these to submit from Colab or the eobench CLI.</p>
        <form onSubmit={createToken} className="row">
          <input style={{ maxWidth: 240 }} value={tokenName}
                 onChange={(e) => setTokenName(e.target.value)} aria-label="Token name" />
          <button>Create token</button>
        </form>
        {newToken && (
          <div className="token-reveal">
            {newToken}
            <div className="hint">Copy it now — it won't be shown again.</div>
          </div>
        )}
        {tokens.length > 0 && (
          <table style={{ marginTop: 14 }}>
            <thead><tr><th>Name</th><th>Prefix</th><th>Last used</th><th></th></tr></thead>
            <tbody>
              {tokens.map((t) => (
                <tr key={t.id}>
                  <td>{t.name}</td>
                  <td className="num">{t.prefix}…</td>
                  <td className="num">{t.last_used_at ? new Date(t.last_used_at).toLocaleString() : "never"}</td>
                  <td><button className="quiet" onClick={() => api.revokeToken(t.id).then(load)}>Revoke</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <h2>My submissions</h2>
      {subs.length === 0 ? (
        <div className="card hint">Nothing yet. Pick a <Link to="/">benchmark</Link> and upload predictions.</div>
      ) : (
        <table>
          <thead><tr><th>Benchmark</th><th>File</th><th>Status</th><th>Date</th></tr></thead>
          <tbody>
            {subs.map((s) => (
              <tr key={s.id}>
                <td>{s.benchmark_slug}</td>
                <td><Link to={`/submissions/${s.id}`}>{s.original_filename}</Link></td>
                <td><span className={`pill ${s.status === "COMPLETED" ? "ok" : ["REJECTED", "FAILED"].includes(s.status) ? "bad" : "warn"}`}>{s.status}</span></td>
                <td className="num">{new Date(s.created_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}
