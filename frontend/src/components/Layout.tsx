import { useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { STATIC } from "../lib/api";
import { useAuth } from "../lib/auth";
import { BrandMark, Icon } from "./ui/ui";

function ThemeToggle() {
  const [theme, setTheme] = useState<string | undefined>(document.documentElement.dataset.theme);
  const effective = theme ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  return (
    <button className="btn ghost icon-btn" aria-label={`Switch to ${effective === "dark" ? "light" : "dark"} mode`}
            onClick={() => {
              const next = effective === "dark" ? "light" : "dark";
              document.documentElement.dataset.theme = next;
              try { localStorage.setItem("eob_theme", next); } catch { /* ignore */ }
              setTheme(next);
            }}>
      <Icon name={effective === "dark" ? "sun" : "moon"} />
    </button>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  const { user } = useAuth();
  const { pathname } = useLocation();
  useEffect(() => { window.scrollTo(0, 0); }, [pathname]);
  return (
    <>
      {STATIC && (
        <div className="static-banner">
          Read-only snapshot — every score here was produced by the hub's evaluators.{" "}
          <a href="https://github.com/mohameddddt/eo-benchmark-hub#quickstart">Run it locally</a> to submit models.
        </div>
      )}
      <header className="header">
        <div className="container header-inner">
          <Link to="/" className="brand"><BrandMark /><span className="brand-text">EO-Bench Hub</span></Link>
          <nav className="nav" aria-label="Main">
            <NavLink to="/benchmarks">Benchmarks</NavLink>
            <NavLink to="/models">Models</NavLink>
            <NavLink to="/docs">Docs</NavLink>
          </nav>
          <div className="header-right">
            <ThemeToggle />
            {!STATIC && (user
              ? <Link to="/account" className="btn secondary sm"><Icon name="user" size={15} />{user.name.split(" ")[0]}</Link>
              : <Link to="/login" className="btn sm">Sign in</Link>)}
          </div>
        </div>
      </header>
      <main>{children}</main>
      <footer className="footer">
        <div className="container row between">
          <div className="row" style={{ gap: 10 }}><BrandMark /><span>EO-Bench Hub · open benchmarking for Earth observation</span></div>
          <div className="row">
            <Link to="/docs">Documentation</Link>
            <Link to="/docs#protocol">Evaluation protocol</Link>
            {!STATIC && <a href="/api/docs">API</a>}
            <a href="https://github.com/mohameddddt/eo-benchmark-hub">GitHub</a>
          </div>
        </div>
      </footer>
    </>
  );
}
