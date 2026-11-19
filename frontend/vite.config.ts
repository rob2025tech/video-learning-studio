import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

// Vite's ProxyServer extends node:events' EventEmitter, which is untyped here
// because @types/node is not a dependency. Describe only the event we use.
type ProxyEventSource = {
  on(event: "proxyReq", listener: (proxyReq: { removeHeader(name: string): void }) => void): void;
};

export default defineConfig(({ mode }) => {
  const envDir = new URL(".", import.meta.url).pathname;
  const env = loadEnv(mode, envDir);
  const apiTarget = env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": {
          target: apiTarget,
          // Rewrites Host to the target so the M1 allowlist (127.0.0.1/localhost
          // with the exact API port) accepts the proxied request.
          changeOrigin: true,
          configure(proxy) {
            (proxy as unknown as ProxyEventSource).on("proxyReq", (proxyReq) => {
              // The browser's Origin is the Vite server (:5173), which the M1
              // policy refuses. A proxied request is a loopback client, and the
              // policy explicitly allows a missing Origin. Sec-Fetch-Site is
              // forwarded untouched: same-origin passes, cross-site stays refused.
              proxyReq.removeHeader("origin");
            });
          },
        },
      },
    },
    test: {
      environment: "jsdom",
    },
  };
});