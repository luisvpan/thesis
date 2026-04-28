/**
 * Salud del relé FastAPI (CV IDE); en dev suele llegar por proxy Vite `/api` → `:8765`.
 */

export type HealthData = {
  status: string;
  version: string;
  uptime: number;
};

function healthUrl(): string {
  const base =
    typeof import.meta.env.VITE_API_URL === "string" &&
    import.meta.env.VITE_API_URL.length > 0
      ? import.meta.env.VITE_API_URL.replace(/\/$/, "")
      : "";
  return base ? `${base}/api/v1/health` : "/api/v1/health";
}

export async function fetchHealth(): Promise<HealthData> {
  const res = await fetch(healthUrl(), { headers: { Accept: "application/json" } });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  const data = (await res.json()) as HealthData;
  return data;
}
