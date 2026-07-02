import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, setToken } from "../api";

export default function Login() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "register") await api.register({ email, password, name });
      const { access_token } = await api.login({ email, password });
      setToken(access_token);
      navigate("/");
    } catch (err: any) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card" style={{ maxWidth: 420, margin: "40px auto" }}>
      <h1>{mode === "login" ? "Sign in" : "Create account"}</h1>
      <p className="sub">
        {mode === "login" ? "New here? " : "Already registered? "}
        <a href="#" onClick={(e) => { e.preventDefault(); setMode(mode === "login" ? "register" : "login"); }}>
          {mode === "login" ? "Create an account" : "Sign in instead"}
        </a>
      </p>
      <form onSubmit={onSubmit}>
        {mode === "register" && (
          <>
            <label htmlFor="name">Name</label>
            <input id="name" value={name} onChange={(e) => setName(e.target.value)} required />
          </>
        )}
        <label htmlFor="email">Email</label>
        <input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        <label htmlFor="password">Password (8+ characters)</label>
        <input id="password" type="password" minLength={8} value={password}
               onChange={(e) => setPassword(e.target.value)} required />
        <div style={{ marginTop: 16 }}>
          <button disabled={busy}>{mode === "login" ? "Sign in" : "Create account"}</button>
        </div>
        {error && <div className="error">{error}</div>}
      </form>
    </div>
  );
}
