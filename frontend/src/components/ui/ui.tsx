import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import type { Shape } from "../charts/core";
import type { Status } from "../../lib/types";

// ---------------------------------------------------------------- icons (stroke, 18px)
const P: Record<string, ReactNode> = {
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  moon: <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />,
  check: <path d="M20 6 9 17l-5-5" />,
  shield: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /><path d="m9 12 2 2 4-4" /></>,
  upload: <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m17 8-5-5-5 5M12 3v12" /></>,
  external: <><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><path d="M15 3h6v6M10 14 21 3" /></>,
  copy: <><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" /></>,
  target: <><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1" /></>,
  cloud: <path d="M17.5 19H9a7 7 0 1 1 6.7-9h1.8a4.5 4.5 0 1 1 0 9z" />,
  gauge: <><path d="M12 14 16 10" /><path d="M3.3 19a10 10 0 1 1 17.4 0" /></>,
  lock: <><rect x="4" y="11" width="16" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" /></>,
  layers: <><path d="m12 2 10 5-10 5L2 7z" /><path d="m2 17 10 5 10-5M2 12l10 5 10-5" /></>,
  arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
  book: <><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z" /><path d="M4 19.5V21h16" /></>,
  user: <><circle cx="12" cy="8" r="4" /><path d="M4 21a8 8 0 0 1 16 0" /></>,
  kaggle: <><path d="M7 3v18M17 3l-7 9 7 9" /></>,
  grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
  sort: <path d="m7 15 5 5 5-5M7 9l5-5 5 5" />,
  down: <path d="m6 9 6 6 6-6" />,
  up: <path d="m18 15-6-6-6 6" />,
};

export function Icon({ name, size = 18, className }: { name: keyof typeof P | string; size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
         strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={className} style={{ flex: "none" }}>
      {P[name]}
    </svg>
  );
}

export function BrandMark() {
  return (
    <svg className="brand-mark" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="var(--brand)" />
      <circle cx="16" cy="16" r="7" fill="none" stroke="#fff" strokeWidth="2.5" />
      <path d="M4 20 L28 12" stroke="#86b6ef" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="22.5" cy="13.8" r="2.4" fill="#fff" />
    </svg>
  );
}

// ---------------------------------------------------------------- families → fixed colour + shape
// Colour follows the entity (architecture family), never its rank. Four
// families exceed the three all-pairs-safe slots, so shape is the secondary
// encoding on every scatter.
export const FAMILIES: Record<string, { color: string; shape: Shape }> = {
  CNN: { color: "var(--s1)", shape: "circle" },
  "Efficient CNN": { color: "var(--s2)", shape: "square" },
  ViT: { color: "var(--s3)", shape: "triangle" },
  Transformer: { color: "var(--s3)", shape: "triangle" },
  ConvNeXt: { color: "var(--s4)", shape: "diamond" },
};
export const familyStyle = (f?: string) => FAMILIES[f ?? ""] ?? { color: "var(--ink-3)", shape: "circle" as Shape };

export function FamilyChip({ family }: { family?: string }) {
  if (!family) return null;
  const s = familyStyle(family);
  return <span className="chip"><span className="dot" style={{ background: s.color, borderRadius: s.shape === "square" ? 2 : "50%" }} />{family}</span>;
}

export function Verified({ compact = false }: { compact?: boolean }) {
  return (
    <span className="badge good" title="Reproduced by the maintainers from a public notebook">
      <Icon name="shield" size={12} />{!compact && "Verified"}
    </span>
  );
}

const STATUS_CLASS: Record<Status, string> = {
  UPLOADED: "warn", QUEUED: "warn", VALIDATING: "warn", EVALUATING: "warn",
  COMPLETED: "good", REJECTED: "bad", FAILED: "bad",
};
export function StatusBadge({ status }: { status: Status }) {
  return <span className={`badge ${STATUS_CLASS[status]}`}>{status === "COMPLETED" ? <Icon name="check" size={11} /> : null}{status.toLowerCase()}</span>;
}

export function CodeBlock({ code, lang }: { code: string; lang?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="code-wrap">
      <pre className="code" data-lang={lang}><code>{code}</code></pre>
      <button className="btn ghost sm copy" onClick={() => {
        navigator.clipboard?.writeText(code).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); });
      }} aria-label="Copy code">
        <Icon name={copied ? "check" : "copy"} size={14} />{copied ? "Copied" : ""}
      </button>
    </div>
  );
}

export function Skeleton({ h = 18, w = "100%", style }: { h?: number; w?: number | string; style?: React.CSSProperties }) {
  return <div className="skeleton" style={{ height: h, width: w, ...style }} />;
}

export function PageSkeleton() {
  return (
    <div className="container page-head stack" style={{ ["--gap" as any]: "14px" }}>
      <Skeleton h={14} w={180} /><Skeleton h={38} w="50%" /><Skeleton h={16} w="70%" />
      <Skeleton h={320} style={{ marginTop: 32 }} />
    </div>
  );
}

export function ErrorState({ error }: { error: unknown }) {
  return (
    <div className="container page-head">
      <div className="alert bad">{error instanceof Error ? error.message : "Something went wrong."}</div>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="card flat" style={{ textAlign: "center", padding: "40px 20px" }}>
      <h3>{title}</h3>
      {children && <div className="soft small" style={{ marginTop: 6 }}>{children}</div>}
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; shape?: Shape }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label}>
          <svg width="12" height="12" viewBox="0 0 12 12">
            {i.shape === "square" ? <rect x="1.5" y="1.5" width="9" height="9" rx="1.5" fill={i.color} />
              : i.shape === "triangle" ? <path d="M6 1 L11 10 L1 10 Z" fill={i.color} />
              : i.shape === "diamond" ? <path d="M6 0.5 L11.5 6 L6 11.5 L0.5 6 Z" fill={i.color} />
              : <circle cx="6" cy="6" r="5" fill={i.color} />}
          </svg>
          {i.label}
        </span>
      ))}
    </div>
  );
}

export function SubmissionLink({ id, children }: { id: string; children: ReactNode }) {
  return <Link to={`/submissions/${id}`}>{children}</Link>;
}
