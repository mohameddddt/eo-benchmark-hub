import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type {
  ApiToken, BenchmarkDetail, BenchmarkSummary, LeaderboardEntry, Me, PublicUser, Stats, Submission,
} from "./types";

declare const __STATIC__: boolean;
export const STATIC = __STATIC__;
export const BASE = import.meta.env.BASE_URL; // "/" or "/eo-benchmark-hub/"

const TOKEN_KEY = "eob_session";

export function getToken(): string | null {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}
export function setToken(t: string | null) {
  try { t ? localStorage.setItem(TOKEN_KEY, t) : localStorage.removeItem(TOKEN_KEY); } catch { /* private mode */ }
}

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export const media = (slug: string, path: string) => `${BASE}media/${slug}/${path}`;

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  if (STATIC) {
    if (options.method && options.method !== "GET") {
      throw new ApiError("This is a read-only snapshot of EO-Bench Hub. Run the server locally to submit.", 405);
    }
    const res = await fetch(`${BASE}data${path.split("?")[0]}.json`);
    if (!res.ok) throw new ApiError("Not found in this snapshot", res.status);
    return res.json();
  }
  const headers: Record<string, string> = { ...(options.headers as Record<string, string>) };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (options.body && !(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const res = await fetch(`${BASE}api${path}`, { ...options, headers });
  if (res.status === 204) return null as T;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = (data as any).detail;
    const msg = typeof d === "string" ? d
      : Array.isArray(d) ? d.map((e: any) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ")
      : `Request failed (${res.status})`;
    if (res.status === 401 && token && path !== "/auth/login") setToken(null);
    throw new ApiError(msg, res.status);
  }
  return data as T;
}

const json = (method: string, body?: unknown): RequestInit => ({ method, body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  register: (b: { email: string; password: string; name: string; affiliation?: string }) => request<Me>("/auth/register", json("POST", b)),
  login: (b: { email: string; password: string }) => request<{ access_token: string }>("/auth/login", json("POST", b)),
  me: () => request<Me>("/auth/me"),
  updateMe: (b: Partial<Me>) => request<Me>("/auth/me", json("PATCH", b)),
  createToken: (name: string) => request<{ token: string; prefix: string }>("/auth/tokens", json("POST", { name })),
  listTokens: () => request<ApiToken[]>("/auth/tokens"),
  revokeToken: (id: string) => request<null>(`/auth/tokens/${id}`, { method: "DELETE" }),
  stats: () => request<Stats>("/stats"),
  benchmarks: () => request<BenchmarkSummary[]>("/benchmarks"),
  benchmark: (slug: string) => request<BenchmarkDetail>(`/benchmarks/${slug}`),
  leaderboard: (slug: string) => request<LeaderboardEntry[]>(`/benchmarks/${slug}/leaderboard`),
  submission: (id: string) => request<Submission>(`/submissions/${id}`),
  mySubmissions: () => request<Submission[]>("/users/me/submissions"),
  user: (id: string) => request<PublicUser>(`/users/${id}`),
  submit: (slug: string, file: File, modelName: string, description: string) => {
    const form = new FormData();
    form.append("file", file);
    form.append("model_name", modelName);
    form.append("description", description);
    return request<Submission>(`/benchmarks/${slug}/submissions`, { method: "POST", body: form });
  },
  updateSubmission: (id: string, b: { model_name?: string; description?: string }) => request<Submission>(`/submissions/${id}`, json("PATCH", b)),
  saveModelCard: (id: string, b: object) => request<Submission>(`/submissions/${id}/model-card`, json("PUT", b)),
  publish: (id: string) => request<Submission>(`/submissions/${id}/publish`, { method: "POST" }),
  unpublish: (id: string) => request<Submission>(`/submissions/${id}/unpublish`, { method: "POST" }),
  remove: (id: string) => request<null>(`/submissions/${id}`, { method: "DELETE" }),
};

// ---- query hooks ----
export const useStats = () => useQuery({ queryKey: ["stats"], queryFn: api.stats });
export const useBenchmarks = () => useQuery({ queryKey: ["benchmarks"], queryFn: api.benchmarks });
export const useBenchmark = (slug: string) => useQuery({ queryKey: ["benchmark", slug], queryFn: () => api.benchmark(slug), enabled: !!slug });
export const useLeaderboard = (slug: string, enabled = true) =>
  useQuery({ queryKey: ["leaderboard", slug], queryFn: () => api.leaderboard(slug), enabled, placeholderData: keepPreviousData });
export const useUser = (id: string) => useQuery({ queryKey: ["user", id], queryFn: () => api.user(id) });

export function useAsset<T>(slug: string | undefined) {
  return useQuery({
    queryKey: ["assets", slug],
    enabled: !!slug,
    staleTime: Infinity,
    retry: false,
    queryFn: async (): Promise<T | null> => {
      const r = await fetch(media(slug!, "assets.json"));
      return r.ok ? r.json() : null;
    },
  });
}
