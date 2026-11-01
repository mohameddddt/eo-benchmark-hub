import type { MetricInfo, TaskType } from "./types";

/** Ratio metrics as percentages (98.37), value metrics as plain decimals. */
export function fmtMetric(v: number | null | undefined, unit = "ratio", digits?: number): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (unit === "ratio") return (v * 100).toFixed(digits ?? 2);
  return v.toFixed(digits ?? 3);
}

export const metricUnit = (metrics: MetricInfo[] | undefined, name: string) =>
  metrics?.find((m) => m.name === name)?.unit ?? "ratio";

export function compact(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined) return "—";
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(digits) + "B";
  if (a >= 1e6) return (n / 1e6).toFixed(digits) + "M";
  if (a >= 1e3) return (n / 1e3).toFixed(digits) + "K";
  return n.toFixed(a < 10 && n % 1 ? digits : 0);
}

export const fmtInt = (n: number | null | undefined) => (n === null || n === undefined ? "—" : n.toLocaleString("en-US"));

export function fmtDate(s: string | null | undefined): string {
  if (!s) return "—";
  return new Date(s).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

export function timeAgo(s: string | null | undefined): string {
  if (!s) return "";
  const sec = (Date.now() - new Date(s).getTime()) / 1000;
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)} min ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} h ago`;
  return `${Math.floor(sec / 86400)} d ago`;
}

export const fileSize = (b: number) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);

export const TASK_LABEL: Record<TaskType, string> = {
  scene_classification: "Scene classification",
  scene_classification_robustness: "Robustness",
  semantic_segmentation: "Semantic segmentation",
  multilabel_classification: "Multi-label classification",
};

/** The secondary metrics worth a leaderboard column, per task. */
export const BOARD_METRICS: Record<TaskType, string[]> = {
  scene_classification: ["f1_macro", "ece"],
  scene_classification_robustness: ["clean_accuracy", "relative_robustness", "worst_corruption_accuracy"],
  semantic_segmentation: ["pixel_accuracy", "mean_f1"],
  multilabel_classification: ["map_micro", "f1_macro"],
};

export const pretty = (s: string) => s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export const shortPretraining = (p: string) =>
  !p || p === "none" ? "From scratch" : p.startsWith("SSL4EO") ? "SSL4EO-S12" : p.replace(" → 1k", "");
