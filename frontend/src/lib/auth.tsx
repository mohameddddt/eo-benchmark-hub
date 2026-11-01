import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createContext, useContext, useState, type ReactNode } from "react";
import { STATIC, api, getToken, setToken } from "./api";
import type { Me } from "./types";

interface AuthState {
  user: Me | null;
  loading: boolean;
  signIn: (token: string) => void;
  signOut: () => void;
}

const Ctx = createContext<AuthState>({ user: null, loading: false, signIn: () => {}, signOut: () => {} });

export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [token, setTok] = useState(getToken());
  const me = useQuery({ queryKey: ["me", token], queryFn: api.me, enabled: !!token && !STATIC, retry: false });
  const value: AuthState = {
    user: me.data ?? null,
    loading: !!token && me.isLoading,
    signIn: (t) => { setToken(t); setTok(t); qc.invalidateQueries(); },
    signOut: () => { setToken(null); setTok(null); qc.clear(); },
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useAuth = () => useContext(Ctx);
