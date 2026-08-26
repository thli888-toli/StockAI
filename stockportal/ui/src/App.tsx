import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, getToken, setToken } from "./api";
import StockChart from "./StockChart";
import type { WatchlistItem } from "./types";


type ModalState = { kind: "report" | "error" | "chart"; item: WatchlistItem } | null;
type User = { user_id: string; openid: string; nickname: string; avatar: string };


export default function App() {
  const [symbol, setSymbol] = useState("");
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [modal, setModal] = useState<ModalState>(null);
  const [user, setUser] = useState<User | null>(null);
  const [loginNickname, setLoginNickname] = useState("");
  const pollRef = useRef<number | null>(null);

  const stopPolling = () => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const startPolling = () => {
    if (pollRef.current !== null) return;
    pollRef.current = window.setInterval(() => {
      refreshList();
    }, 2000);
  };

  const refreshList = useCallback(async () => {
    try {
      const list = await api.listWatchlist();
      setItems(list);
    } catch (refreshError) {
      const message = refreshError instanceof Error ? refreshError.message : String(refreshError);
      setError(message);
      if (message.includes("未登录")) setUser(null);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const restore = async () => {
      if (!getToken()) {
        setUser(null);
        return;
      }
      try {
        const me = await api.me();
        if (!cancelled) {
          setUser(me);
          await refreshList();
        }
      } catch {
        if (!cancelled) setUser(null);
      }
    };
    restore();
    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [refreshList]);

  useEffect(() => {
    const anyRunning = items.some(
      (item) => item.status === "running" || item.status === "queued"
    );
    if (anyRunning) startPolling();
    else stopPolling();
  }, [items]);

  const submit = async () => {
    const code = symbol.trim();
    if (!/^\d{6}$/.test(code)) {
      setError("请输入 6 位 A 股代码，例如 600519。");
      return;
    }
    setError("");
    setLoading(true);
    try {
      const item = await api.addWatchlist(code);
      if ("already_exists" in item) {
        setError(item.message);
        return;
      }
      if (item.status === "failed" && item.error) {
        setError(item.error);
      }
      setItems((prev) => [item, ...prev.filter((existing) => existing.symbol !== item.symbol)]);
      setSymbol("");
      await refreshList();
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : String(submitError));
    } finally {
      setLoading(false);
    }
  };

  const login = async () => {
    setError("");
    try {
      const info = await api.login(loginNickname.trim() || "微信用户");
      if (info.token) setToken(info.token);
      setUser(info);
      await refreshList();
    } catch (loginError) {
      setError(loginError instanceof Error ? loginError.message : String(loginError));
    }
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    setItems([]);
    stopPolling();
  };

  const refreshItem = async (code: string) => {
    setError("");
    try {
      const item = await api.refreshWatchlist(code);
      if ("already_generated" in item) {
        setError(item.message);
        return;
      }
      if (item.status === "failed" && item.error) {
        setError(item.error);
      }
      setItems((prev) => prev.map((existing) => (existing.symbol === item.symbol ? item : existing)));
      await refreshList();
    } catch (refreshError) {
      setError(refreshError instanceof Error ? refreshError.message : String(refreshError));
    }
  };

  const removeItem = async (code: string) => {
    setError("");
    try {
      await api.removeWatchlist(code);
      setItems((prev) => prev.filter((existing) => existing.symbol !== code));
    } catch (removeError) {
      setError(removeError instanceof Error ? removeError.message : String(removeError));
    }
  };

  const rawReport = modal?.kind === "report"
    ? (modal.item.outputs?.report as string | undefined)
    : undefined;
  const report = (() => {
    if (!rawReport) return undefined;
    try {
      const parsed = JSON.parse(rawReport);
      if (parsed && typeof parsed.report === "string") return parsed.report;
    } catch {
      // Fall through to treat the value as plain Markdown.
    }
    return rawReport;
  })();

  return (
    <main className="app">
      <h1>A 股分析门户</h1>

      <section className="card">
        {user ? (
          <div className="controls">
            <span>当前用户：{user.nickname}</span>
            <button onClick={logout}>退出登录</button>
          </div>
        ) : (
          <div className="controls">
            <input
              value={loginNickname}
              onChange={(event) => setLoginNickname(event.target.value)}
              placeholder="昵称"
            />
            <button onClick={login}>微信登录（模拟）</button>
          </div>
        )}
      </section>

      {!user && <p>请先登录后使用。</p>}

      {user && (
        <>
      <section className="card">
        <div className="controls">
          <input
            value={symbol}
            onChange={(event) => setSymbol(event.target.value)}
            placeholder="600519"
            maxLength={6}
          />
          <button onClick={submit} disabled={loading}>
            {loading ? "提交中..." : "分析"}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="card">
        <h2>监控列表</h2>
        {items.length === 0 ? (
          <p>暂无监控股票。</p>
        ) : (
          <ul className="watchlist">
            {items.map((item) => (
              <li key={item.symbol} className="watchlist-row">
                <div className="watchlist-main">
                  <strong>{item.symbol}</strong>
                  {item.company_name && <span>{item.company_name}</span>}
                  {item.industry && <span className="muted">{item.industry}</span>}
                  <span className={`status status-${item.status}`}>
                    {item.status === "queued"
                      ? "排队中"
                      : item.status === "running"
                        ? "分析中"
                        : item.status === "completed"
                          ? "已完成"
                          : "失败"}
                  </span>
                </div>
                <div className="watchlist-actions">
                  {(item.status === "running" || item.status === "queued") && (
                    <span className="muted">
                      {item.status === "queued" ? "排队中..." : "分析中..."}
                    </span>
                  )}
                  {item.status === "completed" && (
                    <button onClick={() => setModal({ kind: "report", item })}>
                      查看结果
                    </button>
                  )}
                  {item.status === "failed" && (
                    <button onClick={() => setModal({ kind: "error", item })}>
                      错误
                    </button>
                  )}
                  <button onClick={() => setModal({ kind: "chart", item })}>
                    图表
                  </button>
                  <button onClick={() => refreshItem(item.symbol)}>刷新</button>
                  <button onClick={() => removeItem(item.symbol)}>删除</button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
        </>
      )}

      <p className="disclaimer">
        本报告仅供参考研究，不构成投资建议。
      </p>

      {modal && (
        <div className="modal-overlay" onClick={() => setModal(null)}>
          <div className="modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h2>
                {modal.kind === "report"
                  ? `${modal.item.symbol} 分析结果`
                  : modal.kind === "error"
                    ? `${modal.item.symbol} 错误`
                    : `${modal.item.symbol} K 线图`}
              </h2>
              <button className="modal-close" onClick={() => setModal(null)}>
                ×
              </button>
            </div>
            <div className="modal-body">
              {modal.kind === "report" ? (
                typeof report === "string" ? (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
                ) : (
                  <p>暂无报告。</p>
                )
              ) : modal.kind === "chart" ? (
                <StockChart symbol={modal.item.symbol} />
              ) : (
                <pre className="error-text">{modal.item.error || "未知错误"}</pre>
              )}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
