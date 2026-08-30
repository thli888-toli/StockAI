import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, getToken, setToken } from "./api";
import KLineChartView, { type ChartPayload, type Period } from "./KLineChartView";
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
  const [tagDrafts, setTagDrafts] = useState<Record<string, string>>({});
  const [tagQuery, setTagQuery] = useState("");
  const [tagSort, setTagSort] = useState<"none" | "asc" | "desc">("none");
  const pollRef = useRef<number | null>(null);
  const [chartSnapshots, setChartSnapshots] = useState<
    { id: number; period: string; label: string; saved_at: string }[]
  >([]);
  const [historyPayload, setHistoryPayload] = useState<ChartPayload | null>(null);
  const [historyError, setHistoryError] = useState("");
  const [selectedSnapshotId, setSelectedSnapshotId] = useState<string>("");
  const [chartPeriod, setChartPeriod] = useState<Period>("daily");
  const [chartSaving, setChartSaving] = useState(false);
  const [chartSaveMessage, setChartSaveMessage] = useState("");

  useEffect(() => {
    if (!modal || modal.kind !== "chart") return;
    let cancelled = false;
    setChartSnapshots([]);
    setHistoryPayload(null);
    setHistoryError("");
    setSelectedSnapshotId("");
    api
      .listChartSnapshots(modal.item.symbol)
      .then((list) => {
        if (!cancelled) setChartSnapshots(list);
      })
      .catch((err) => {
        if (!cancelled) {
          setHistoryError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [modal]);

  const loadHistory = async (value: string) => {
    setSelectedSnapshotId(value);
    setHistoryError("");
    if (!value) {
      setHistoryPayload(null);
      return;
    }
    const snapshotId = Number(value);
    try {
      const snapshot = await api.getChartSnapshot(modal!.item.symbol, snapshotId);
      setHistoryPayload(snapshot.payload as ChartPayload);
    } catch (err) {
      setHistoryPayload(null);
      setHistoryError(err instanceof Error ? err.message : String(err));
    }
  };

  const deleteHistory = async () => {
    if (!selectedSnapshotId) return;
    const snapshotId = Number(selectedSnapshotId);
    setHistoryError("");
    try {
      await api.deleteChartSnapshot(modal!.item.symbol, snapshotId);
      setChartSnapshots((prev) => prev.filter((item) => item.id !== snapshotId));
      if (historyPayload && snapshotId === Number(selectedSnapshotId)) {
        setHistoryPayload(null);
      }
      setSelectedSnapshotId("");
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : String(err));
    }
  };

  const saveCurrentChart = async () => {
    if (!modal || modal.kind !== "chart") return;
    setChartSaveMessage("");
    setChartSaving(true);
    try {
      await api.saveChart(modal.item.symbol, chartPeriod);
      setChartSaveMessage("已保存");
      const list = await api.listChartSnapshots(modal.item.symbol);
      setChartSnapshots(list);
    } catch (err) {
      setHistoryError(err instanceof Error ? err.message : String(err));
    } finally {
      setChartSaving(false);
    }
  };

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

  const saveTags = async (item: WatchlistItem, nextTags: string[]) => {
    setError("");
    try {
      const updated = await api.updateTags(item.symbol, nextTags);
      setItems((prev) =>
        prev.map((existing) => (existing.symbol === updated.symbol ? updated : existing))
      );
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : String(saveError));
    }
  };

  const addTag = (item: WatchlistItem) => {
    const draft = (tagDrafts[item.symbol] ?? "").trim();
    if (!draft) return;
    const next = [
      ...new Set([
        ...(item.tags || []),
        ...draft
          .split(/[,，、\s]+/)
          .map((tag) => tag.trim())
          .filter(Boolean)
      ])
    ];
    setTagDrafts((prev) => ({ ...prev, [item.symbol]: "" }));
    void saveTags(item, next);
  };

  const removeTag = (item: WatchlistItem, tag: string) => {
    void saveTags(item, (item.tags || []).filter((existing) => existing !== tag));
  };

  const visibleItems = useMemo(() => {
    let list = items;
    const query = tagQuery.trim().toLowerCase();
    if (query) {
      const wanted = query.split(/[,，、\s]+/).filter(Boolean);
      list = list.filter((item) => {
        const tags = (item.tags || []).map((tag) => tag.toLowerCase());
        return wanted.every((word) => tags.some((tag) => tag.includes(word)));
      });
    }
    if (tagSort !== "none") {
      const key = (item: WatchlistItem) => (item.tags || []).join(",").toLowerCase();
      list = [...list].sort((a, b) => {
        const compared = key(a).localeCompare(key(b));
        return tagSort === "asc" ? compared : -compared;
      });
    }
    return list;
  }, [items, tagQuery, tagSort]);

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
        <div className="watchlist-toolbar">
          <input
            value={tagQuery}
            onChange={(event) => setTagQuery(event.target.value)}
            placeholder="按标签搜索（多个标签用逗号分隔）"
          />
          <select
            value={tagSort}
            onChange={(event) => setTagSort(event.target.value as "none" | "asc" | "desc")}
          >
            <option value="none">默认排序</option>
            <option value="asc">标签 A→Z</option>
            <option value="desc">标签 Z→A</option>
          </select>
        </div>
        {items.length === 0 ? (
          <p>暂无监控股票。</p>
        ) : visibleItems.length === 0 ? (
          <p>没有匹配的标签。</p>
        ) : (
          <ul className="watchlist">
            {visibleItems.map((item) => (
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
                <div className="watchlist-tags">
                  {(item.tags || []).map((tag) => (
                    <span key={tag} className="tag">
                      {tag}
                      <button
                        className="tag-remove"
                        onClick={() => removeTag(item, tag)}
                        aria-label={`删除标签 ${tag}`}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                  <input
                    className="tag-input"
                    value={tagDrafts[item.symbol] ?? ""}
                    onChange={(event) =>
                      setTagDrafts((prev) => ({
                        ...prev,
                        [item.symbol]: event.target.value
                      }))
                    }
                    onKeyDown={(event) => {
                      if (event.key === "Enter") addTag(item);
                    }}
                    onBlur={() => {
                      if ((tagDrafts[item.symbol] ?? "").trim()) addTag(item);
                    }}
                    placeholder="+ 标签"
                  />
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
          <div
            className={`modal ${historyPayload ? "modal-compare" : ""}`}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <h2>
                {modal.kind === "report"
                  ? `${modal.item.symbol} 分析结果`
                  : modal.kind === "error"
                    ? `${modal.item.symbol} 错误`
                    : `${modal.item.symbol}${modal.item.company_name ? ` ${modal.item.company_name}` : ""} K 线图`}
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
                <>
                  <div className="chart-history-controls">
                    {historyPayload && (
                      <>
                        {(["daily", "weekly", "monthly"] as Period[]).map((item) => (
                          <button
                            key={item}
                            className={chartPeriod === item ? "active" : ""}
                            onClick={() => setChartPeriod(item)}
                          >
                            {item === "daily" ? "日K" : item === "weekly" ? "周K" : "月K"}
                          </button>
                        ))}
                        <button
                          className="save-chart"
                          onClick={saveCurrentChart}
                          disabled={chartSaving}
                        >
                          {chartSaving ? "保存中..." : "保存此图表"}
                        </button>
                        {chartSaveMessage && (
                          <span className="save-message">{chartSaveMessage}</span>
                        )}
                      </>
                    )}
                    <label htmlFor="history-chart-select">历史图表</label>
                    <select
                      id="history-chart-select"
                      value={selectedSnapshotId}
                      onChange={(event) => loadHistory(event.target.value)}
                    >
                      <option value="">选择历史快照</option>
                      {chartSnapshots.map((snapshot) => (
                        <option key={snapshot.id} value={snapshot.id}>
                          {snapshot.label}（
                          {snapshot.period === "daily"
                            ? "日K"
                            : snapshot.period === "weekly"
                              ? "周K"
                              : "月K"}
                          ）
                        </option>
                      ))}
                    </select>
                    {selectedSnapshotId && (
                      <button className="delete-history" onClick={deleteHistory}>
                        删除历史快照
                      </button>
                    )}
                  </div>
                  {historyError && <p className="error">{historyError}</p>}
                  <div className={historyPayload ? "chart-compare" : ""}>
                    <div className={historyPayload ? "chart-pane" : ""}>
                      {historyPayload ? (
                        <>
                          <h3>当前 K线图</h3>
                          <StockChart
                            symbol={modal.item.symbol}
                            period={chartPeriod}
                            onPeriodChange={setChartPeriod}
                            hideHeader
                          />
                        </>
                      ) : (
                        <StockChart symbol={modal.item.symbol} />
                      )}
                    </div>
                    {historyPayload && (
                      <div className="chart-pane">
                        <h3>历史 K线图</h3>
                        <div className="chart-page">
                          <KLineChartView payload={historyPayload} />
                        </div>
                      </div>
                    )}
                  </div>
                </>
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
