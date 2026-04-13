/**
 * Pushes True Markets MCP data to the backend.
 * The frontend can call the TM MCP API (via browser/proxy),
 * then pushes the results to POST /api/tm/push for the
 * recommendation engine to use.
 *
 * This is called manually or on a timer from the page.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

// These match the True Markets MCP data API structure
interface TMPushData {
  price: number;
  sentiment: string;
  summary: string;
  trending: Array<{ symbol: string; trending_ratio: number }>;
  surging: Array<{ symbol: string; price_change_pct_1h: number; price_change_pct_24h: number }>;
}

export async function pushTMData(data: TMPushData): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/tm/push`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Fetch TM data from the MCP proxy endpoints and push to backend.
 * The actual MCP calls happen via the Claude extension / browser context.
 * For now this fetches from our backend's cached TM data.
 */
export async function fetchAndPushTMData(): Promise<boolean> {
  // This would be called by Claude or a browser extension that has MCP access
  // For now, it's a no-op that returns the current state
  try {
    const res = await fetch(`${API_BASE}/tm/data`);
    if (res.ok) {
      const data = await res.json();
      return data.live;
    }
  } catch {}
  return false;
}
