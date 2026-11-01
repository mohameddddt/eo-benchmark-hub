import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, Navigate, useLocation, useNavigate, useParams } from "react-router-dom";
import { Empty, ErrorState, Icon, PageSkeleton, StatusBadge, Verified } from "../components/ui/ui";
import { STATIC, api, useUser } from "../lib/api";
import { useAuth } from "../lib/auth";
import { fmtDate, fmtMetric, timeAgo } from "../lib/format";

export function Login() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [form, setForm] = useState({ email: "", password: "", name: "", affiliation: "" });
  const { signIn, user } = useAuth();
  const navigate = useNavigate();
  const loc = useLocation();
  const next = new URLSearchParams(loc.search).get("next") ?? "/account";
  const mut = useMutation({
    mutationFn: async () => {
      if (mode === "register") await api.register({ ...form, affiliation: form.affiliation || undefined });
      return api.login({ email: form.email, password: form.password });
    },
    onSuccess: ({ access_token }) => { signIn(access_token); navigate(next); },
  });
  if (STATIC) return <Navigate to="/" replace />;
  if (user) return <Navigate to={next} replace />;
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [k]: e.target.value });
  return (
    <div className="container" style={{ maxWidth: 440, paddingTop: 56 }}>
      <div className="card">
        <h1 style={{ fontSize: 26 }}>{mode === "login" ? "Sign in" : "Create an account"}</h1>
        <p className="soft small" style={{ margin: "6px 0 20px" }}>
          {mode === "login" ? "New here? " : "Already registered? "}
          <a href="#" onClick={(e) => { e.preventDefault(); setMode(mode === "login" ? "register" : "login"); mut.reset(); }}>
            {mode === "login" ? "Create an account" : "Sign in instead"}</a>
        </p>
        <form className="stack" style={{ ["--gap" as any]: "14px" }} onSubmit={(e) => { e.preventDefault(); mut.mutate(); }}>
          {mode === "register" && <>
            <label className="field"><span>Name</span><input value={form.name} onChange={set("name")} required autoComplete="name" /></label>
            <label className="field"><span>Affiliation (optional)</span><input value={form.affiliation} onChange={set("affiliation")} /></label>
          </>}
          <label className="field"><span>Email</span><input type="email" value={form.email} onChange={set("email")} required autoComplete="email" /></label>
          <label className="field"><span>Password {mode === "register" && "(8+ characters)"}</span>
            <input type="password" minLength={8} value={form.password} onChange={set("password")} required autoComplete={mode === "login" ? "current-password" : "new-password"} /></label>
          {mut.error && <div className="alert bad">{(mut.error as Error).message}</div>}
          <button className="btn" disabled={mut.isPending}>{mut.isPending ? "…" : mode === "login" ? "Sign in" : "Create account"}</button>
        </form>
      </div>
    </div>
  );
}

export function Account() {
  const { user, loading, signOut } = useAuth();
  const qc = useQueryClient();
  const navigate = useNavigate();
  const subs = useQuery({ queryKey: ["mine"], queryFn: api.mySubmissions, enabled: !!user,
    refetchInterval: (q) => (q.state.data?.some((s) => ["QUEUED", "VALIDATING", "EVALUATING"].includes(s.status)) ? 2000 : false) });
  const tokens = useQuery({ queryKey: ["tokens"], queryFn: api.listTokens, enabled: !!user });
  const [profile, setProfile] = useState({ name: "", affiliation: "", website: "", bio: "" });
  const [tokenName, setTokenName] = useState("kaggle");
  const [newToken, setNewToken] = useState("");
  useEffect(() => { if (user) setProfile({ name: user.name, affiliation: user.affiliation ?? "", website: user.website ?? "", bio: user.bio ?? "" }); }, [user]);
  const saveProfile = useMutation({ mutationFn: () => api.updateMe(profile), onSuccess: () => qc.invalidateQueries({ queryKey: ["me"] }) });
  const createToken = useMutation({ mutationFn: () => api.createToken(tokenName), onSuccess: (t) => { setNewToken(t.token); qc.invalidateQueries({ queryKey: ["tokens"] }); } });
  const revoke = useMutation({ mutationFn: api.revokeToken, onSuccess: () => qc.invalidateQueries({ queryKey: ["tokens"] }) });

  if (STATIC) return <Navigate to="/" replace />;
  if (loading) return <PageSkeleton />;
  if (!user) return <Navigate to="/login?next=/account" replace />;
  const primary = (s: NonNullable<typeof subs.data>[number]) => s.results.find((r) => r.metric_name === s.benchmark.primary_metric);

  return (
    <div className="container">
      <div className="page-head row between">
        <div><h1>{user.name}</h1><p className="lead" style={{ marginTop: 4 }}>{user.email}{user.role !== "user" && <span className="badge info" style={{ marginLeft: 8 }}>{user.role}</span>}</p></div>
        <div className="row">
          <Link to={`/users/${user.id}`} className="btn secondary sm">Public profile</Link>
          <button className="btn ghost sm" onClick={() => { signOut(); navigate("/"); }}>Sign out</button>
        </div>
      </div>

      <section className="section" style={{ marginTop: 32 }}>
        <div className="section-head"><h2>My submissions</h2><Link to="/benchmarks" className="btn sm"><Icon name="upload" size={14} />New submission</Link></div>
        {subs.data?.length === 0 ? <Empty title="Nothing submitted yet">Pick a <Link to="/benchmarks">benchmark</Link>, then use its Submit tab.</Empty> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Model</th><th>Benchmark</th><th>Status</th><th className="num">Score</th><th>Visibility</th><th className="r">Submitted</th></tr></thead>
              <tbody>
                {(subs.data ?? []).map((s) => (
                  <tr key={s.id}>
                    <td><Link to={`/submissions/${s.id}`} style={{ fontWeight: 600, color: "var(--ink)" }}>{s.model_name}</Link><div className="xs muted mono">{s.original_filename}</div></td>
                    <td>{s.benchmark.name}</td>
                    <td><StatusBadge status={s.status} /></td>
                    <td className="num">{primary(s) ? fmtMetric(primary(s)!.value) : "—"}</td>
                    <td>{s.is_published ? <span className="badge info">public</span> : <span className="badge neutral">private</span>} {s.verified && <Verified compact />}</td>
                    <td className="r small muted">{timeAgo(s.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="section grid g2" style={{ alignItems: "start" }}>
        <div className="card">
          <h3 style={{ marginBottom: 14 }}>Profile</h3>
          <form className="stack" style={{ ["--gap" as any]: "12px" }} onSubmit={(e) => { e.preventDefault(); saveProfile.mutate(); }}>
            <label className="field"><span>Name</span><input value={profile.name} onChange={(e) => setProfile({ ...profile, name: e.target.value })} required /></label>
            <label className="field"><span>Affiliation</span><input value={profile.affiliation} onChange={(e) => setProfile({ ...profile, affiliation: e.target.value })} /></label>
            <label className="field"><span>Website</span><input type="url" value={profile.website} onChange={(e) => setProfile({ ...profile, website: e.target.value })} /></label>
            <label className="field"><span>Bio</span><textarea rows={3} value={profile.bio} onChange={(e) => setProfile({ ...profile, bio: e.target.value })} /></label>
            <div className="row"><button className="btn sm" disabled={saveProfile.isPending}>Save profile</button>{saveProfile.isSuccess && <span className="small" style={{ color: "var(--good-ink)" }}>Saved</span>}</div>
          </form>
        </div>
        <div className="card">
          <h3 style={{ marginBottom: 6 }}>API tokens</h3>
          <p className="soft small">Submit from Kaggle, Colab or CI. Tokens are stored hashed — copy a new one now, it is shown once.</p>
          <form className="row" onSubmit={(e) => { e.preventDefault(); createToken.mutate(); }}>
            <input value={tokenName} onChange={(e) => setTokenName(e.target.value)} aria-label="Token name" style={{ flex: 1, minWidth: 140 }} />
            <button className="btn sm">Create token</button>
          </form>
          {newToken && <pre className="code" style={{ marginTop: 12, whiteSpace: "pre-wrap", wordBreak: "break-all" }}>{newToken}</pre>}
          {!!tokens.data?.length && (
            <table style={{ marginTop: 14, fontSize: 13 }}>
              <thead><tr><th>Name</th><th>Prefix</th><th>Last used</th><th /></tr></thead>
              <tbody>{tokens.data.map((t) => (
                <tr key={t.id}><td>{t.name}</td><td className="mono">{t.prefix}…</td><td className="muted">{t.last_used_at ? timeAgo(t.last_used_at) : "never"}</td>
                  <td className="r"><button className="btn danger sm" onClick={() => revoke.mutate(t.id)}>Revoke</button></td></tr>
              ))}</tbody>
            </table>
          )}
        </div>
      </section>
    </div>
  );
}

export function UserPage() {
  const { id = "" } = useParams();
  const { data, error } = useUser(id);
  if (error) return <ErrorState error={error} />;
  if (!data) return <PageSkeleton />;
  return (
    <div className="container">
      <div className="page-head">
        <div className="crumbs">Researchers</div>
        <h1>{data.name}</h1>
        <p className="lead">{[data.affiliation, `joined ${fmtDate(data.created_at)}`].filter(Boolean).join(" · ")}
          {data.website && <> · <a href={data.website} target="_blank" rel="noreferrer">{data.website.replace(/^https?:\/\//, "")}</a></>}</p>
        {data.bio && <p className="soft" style={{ maxWidth: 720 }}>{data.bio}</p>}
      </div>
      <section className="section" style={{ marginTop: 28 }}>
        <h2 style={{ marginBottom: 14 }}>Leaderboard entries <span className="muted" style={{ fontWeight: 500 }}>{data.entries.length}</span></h2>
        {data.entries.length === 0 ? <Empty title="No published entries yet" /> : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Benchmark</th><th>Model</th><th className="num">Score</th><th className="num">Rank</th></tr></thead>
              <tbody>{data.entries.map((e) => (
                <tr key={e.submission_id}>
                  <td><Link to={`/benchmarks/${e.benchmark_slug}`}>{e.benchmark_name}</Link></td>
                  <td><Link to={`/submissions/${e.submission_id}`} style={{ color: "var(--ink)", fontWeight: 600 }}>{e.model_name}</Link> {e.verified && <Verified compact />}</td>
                  <td className="num"><b>{fmtMetric(e.score)}</b> <span className="xs muted">{e.metric_label}</span></td>
                  <td className="num">#{e.rank} <span className="muted">of {e.of}</span></td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export function NotFound() {
  return (
    <div className="container page-head" style={{ textAlign: "center", paddingTop: 96 }}>
      <div className="hero-figure muted">404</div>
      <h1 style={{ marginTop: 12 }}>Nothing at these coordinates</h1>
      <p className="lead" style={{ margin: "10px auto 24px" }}>The page you're looking for doesn't exist.</p>
      <Link to="/" className="btn">Back to the hub</Link>
    </div>
  );
}
