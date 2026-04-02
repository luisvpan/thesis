/**
 * Punto de entrada: API Elysia en desarrollo; en producción sirve también el build de Vite (React).
 * Desarrollo: solo API → usa el proxy de Vite (`/api` → este servidor).
 * Producción: `NODE_ENV=production` tras `bun run build` en la raíz del monorepo.
 */
import { Elysia } from "elysia";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { app } from "./server";

const port = Number(process.env.PORT ?? 3000);
const isProd = process.env.NODE_ENV === "production";
const distDir = join(import.meta.dir, "../../frontend/dist");

function createProdServer() {
  return (
    new Elysia()
      .use(app)
      // Rutas que no sean /api: archivos estáticos o SPA (React Router).
      .get("*", ({ request }) => {
        const url = new URL(request.url);
        const pathname = url.pathname;
        if (pathname.startsWith("/api")) {
          return new Response(
            JSON.stringify({ success: false, error: "Not found" }),
            {
              status: 404,
              headers: { "Content-Type": "application/json" },
            },
          );
        }
        if (pathname === "/") {
          return new Response(Bun.file(join(distDir, "index.html")), {
            headers: { "Content-Type": "text/html" },
          });
        }
        const rel = pathname.slice(1);
        const filePath = join(distDir, rel);
        if (existsSync(filePath)) {
          return new Response(Bun.file(filePath));
        }
        return new Response(Bun.file(join(distDir, "index.html")), {
          headers: { "Content-Type": "text/html" },
        });
      })
  );
}

const server = isProd ? createProdServer() : app;

server.listen(port);

console.log(
  `[@dataflow/http-api] http://localhost:${port} (${isProd ? "API + React estático" : "solo API — usa Vite con proxy /api"})`,
);
