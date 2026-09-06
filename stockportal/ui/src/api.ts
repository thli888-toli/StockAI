import type { WatchlistItem } from "./types";


type UserInfo = {
  token?: string;
  user_id: string;
  openid: string;
  nickname: string;
  avatar: string;
};

type AddResult = WatchlistItem | { already_exists: boolean; message: string };
type RefreshResult = WatchlistItem | { already_generated: boolean; message: string };


const TOKEN_KEY = "stockportal_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}


async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init?.body) headers.set("Content-Type", "application/json");
  const response = await fetch(url, { ...init, headers });
  if (response.status === 401) {
    setToken(null);
    throw new Error("未登录，请先登录");
  }
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}


export const api = {
  login: (nickname: string) =>
    request<UserInfo>("/api/login", {
      method: "POST",
      body: JSON.stringify({ nickname })
    }),
  me: () => request<UserInfo>("/api/me"),
  listWatchlist: (market: string = "a") =>
    request<WatchlistItem[]>(`/api/watchlist?market=${encodeURIComponent(market)}`),
  addWatchlist: (query: string, market: string = "a") =>
    request<AddResult>("/api/watchlist", {
      method: "POST",
      body: JSON.stringify({ query, market })
    }),
  refreshWatchlist: (symbol: string, market: string = "a") =>
    request<RefreshResult>(
      `/api/watchlist/${encodeURIComponent(symbol)}/refresh?market=${encodeURIComponent(market)}`,
      {
        method: "POST"
      }
    ),
  removeWatchlist: (symbol: string, market: string = "a") =>
    request<{ deleted: boolean }>(
      `/api/watchlist/${encodeURIComponent(symbol)}?market=${encodeURIComponent(market)}`,
      {
        method: "DELETE"
      }
    ),
  updateTags: (symbol: string, tags: string[], market: string = "a") =>
    request<WatchlistItem>(
      `/api/watchlist/${encodeURIComponent(symbol)}/tags?market=${encodeURIComponent(market)}`,
      {
      method: "PUT",
        body: JSON.stringify({ tags, market })
      }
    ),
  saveChart: (symbol: string, period: string, market: string = "a") =>
    request<{ id: number; period: string; label: string; saved_at: string }>(
      `/api/watchlist/${encodeURIComponent(symbol)}/chart/save?market=${encodeURIComponent(market)}`,
      { method: "POST", body: JSON.stringify({ period, market }) }
    ),
  listChartSnapshots: (symbol: string, market: string = "a") =>
    request<{ id: number; period: string; label: string; saved_at: string }[]>(
      `/api/watchlist/${encodeURIComponent(symbol)}/charts?market=${encodeURIComponent(market)}`
    ),
  getChartSnapshot: (symbol: string, id: number, market: string = "a") =>
    request<Record<string, unknown> & { payload: unknown }>(
      `/api/watchlist/${encodeURIComponent(symbol)}/charts/${id}?market=${encodeURIComponent(market)}`
    ),
  deleteChartSnapshot: (symbol: string, id: number, market: string = "a") =>
    request<{ deleted: boolean }>(
      `/api/watchlist/${encodeURIComponent(symbol)}/charts/${id}?market=${encodeURIComponent(market)}`,
      { method: "DELETE" }
    )
};
