const BASE = "/api";

export function getToken(): string | null {
  return localStorage.getItem("eob_session");
}
export function setToken(t: string | null) {
  t ? localStorage.setItem("eob_session", t) : localStorage.removeItem("eob_session");
}

async function request(path: string, options: RequestInit = {}) {
  const headers: Record<string, string> = { ...(options.headers as any) };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(BASE + path, { ...options, headers });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data));
  }
  return data;
}

export const api = {
  register: (body: object) => request("/auth/register", { method: "POST", body: JSON.stringify(body) }),
  login: (body: object) => request("/auth/login", { method: "POST", body: JSON.stringify(body) }),
  me: () => request("/auth/me"),
  createToken: (name: string) => request("/auth/tokens", { method: "POST", body: JSON.stringify({ name }) }),
  listTokens: () => request("/auth/tokens"),
  revokeToken: (id: string) => request(`/auth/tokens/${id}`, { method: "DELETE" }),
  benchmarks: () => request("/benchmarks"),
  benchmark: (slug: string) => request(`/benchmarks/${slug}`),
  leaderboard: (slug: string) => request(`/benchmarks/${slug}/leaderboard`),
  submit: (slug: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/benchmarks/${slug}/submissions`, { method: "POST", body: form });
  },
  submission: (id: string) => request(`/submissions/${id}`),
  mySubmissions: () => request("/users/me/submissions"),
  saveModelCard: (id: string, body: object) =>
    request(`/submissions/${id}/model-card`, { method: "PUT", body: JSON.stringify(body) }),
  publish: (id: string) => request(`/submissions/${id}/publish`, { method: "POST" }),
};
