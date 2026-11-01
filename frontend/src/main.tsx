import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { AuthProvider } from "./lib/auth";
import { Account, Login, NotFound, UserPage } from "./pages/Account";
import { BenchmarkList, BenchmarkPage } from "./pages/Benchmarks";
import Docs from "./pages/Docs";
import Home from "./pages/Home";
import Models from "./pages/Models";
import SubmissionPage from "./pages/SubmissionPage";
import "./styles.css";

const qc = new QueryClient({
  defaultOptions: { queries: { staleTime: 30_000, refetchOnWindowFocus: false, retry: 1 } },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, "") || "/"}>
        <AuthProvider>
          <Layout>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/benchmarks" element={<BenchmarkList />} />
              <Route path="/benchmarks/:slug" element={<BenchmarkPage />} />
              <Route path="/submissions/:id" element={<SubmissionPage />} />
              <Route path="/models" element={<Models />} />
              <Route path="/docs" element={<Docs />} />
              <Route path="/login" element={<Login />} />
              <Route path="/account" element={<Account />} />
              <Route path="/users/:id" element={<UserPage />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Layout>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
