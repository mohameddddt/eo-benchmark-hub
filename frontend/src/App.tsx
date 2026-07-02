import { Link, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { getToken, setToken } from "./api";
import BenchmarkDetail from "./pages/BenchmarkDetail";
import Benchmarks from "./pages/Benchmarks";
import Login from "./pages/Login";
import Profile from "./pages/Profile";
import SubmissionDetail from "./pages/SubmissionDetail";

function TopBar() {
  const navigate = useNavigate();
  const authed = Boolean(getToken());
  return (
    <header className="topbar">
      <Link to="/" className="brand">EO<span>·</span>BENCH</Link>
      <nav>
        <Link to="/">Benchmarks</Link>
        {authed && <Link to="/profile">My submissions</Link>}
        {authed ? (
          <a href="#" onClick={(e) => { e.preventDefault(); setToken(null); navigate("/login"); }}>
            Sign out
          </a>
        ) : (
          <Link to="/login">Sign in</Link>
        )}
      </nav>
    </header>
  );
}

function RequireAuth({ children }: { children: JSX.Element }) {
  return getToken() ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <>
      <TopBar />
      <main>
        <Routes>
          <Route path="/" element={<Benchmarks />} />
          <Route path="/login" element={<Login />} />
          <Route path="/benchmarks/:slug" element={<BenchmarkDetail />} />
          <Route path="/submissions/:id" element={<RequireAuth><SubmissionDetail /></RequireAuth>} />
          <Route path="/profile" element={<RequireAuth><Profile /></RequireAuth>} />
        </Routes>
      </main>
    </>
  );
}
