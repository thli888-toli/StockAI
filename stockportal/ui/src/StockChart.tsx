import { useEffect, useState } from "react";
import { api, getToken } from "./api";
import KLineChartView, {
  type ChartPayload,
  type FundamentalInfo,
  type Period,
} from "./KLineChartView";


const PERIOD_LABELS: Record<Period, string> = {
  daily: "日K",
  weekly: "周K",
  monthly: "月K",
};


function directionCn(direction: string): string {
  return direction === "up" ? "上行" : direction === "down" ? "下行" : "中性";
}

function overallCn(overall: string): string {
  return overall === "bullish" ? "偏多" : overall === "bearish" ? "偏空" : "中性";
}


type StockChartProps = {
  symbol: string;
  period?: Period;
  onPeriodChange?: (period: Period) => void;
  hideHeader?: boolean;
};


export default function StockChart({
  symbol,
  period: controlledPeriod,
  onPeriodChange,
  hideHeader = false,
}: StockChartProps) {
  const [internalPeriod, setInternalPeriod] = useState<Period>("daily");
  const period = controlledPeriod ?? internalPeriod;
  const setPeriod = onPeriodChange ?? setInternalPeriod;
  const [error, setError] = useState("");
  const [data, setData] = useState<ChartPayload | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState("");
  const [overview, setOverview] = useState<{
    weekly?: string;
    monthly?: string;
    llm?: { overall: string; text: string } | null;
    fundamental?: FundamentalInfo;
  }>({});

  useEffect(() => {
    let cancelled = false;
    setError("");
    const load = async () => {
      const headers = new Headers();
      const token = getToken();
      if (token) headers.set("Authorization", `Bearer ${token}`);
      const response = await fetch(
        `/api/watchlist/${encodeURIComponent(symbol)}/chart?period=${period}`,
        { headers }
      );
      if (!response.ok) {
        let detail = "";
        try {
          detail = (await response.json()).detail || "";
        } catch {
          detail = response.statusText;
        }
        throw new Error(detail || `${response.status} ${response.statusText}`);
      }
      const payload = (await response.json()) as ChartPayload;
      if (cancelled) return;
      setData(payload);
      setOverview({
        weekly: payload.signals?.["1w"],
        monthly: payload.signals?.["1mo"],
        llm: payload.llm_summary,
        fundamental: payload.fundamental,
      });
    };

    load().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : String(loadError));
      }
    });

    return () => {
      cancelled = true;
    };
  }, [symbol, period]);

  const save = async () => {
    setError("");
    setSaveMessage("");
    setSaving(true);
    try {
      await api.saveChart(symbol, period);
      setSaveMessage("已保存");
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : String(saveError));
    } finally {
      setSaving(false);
    }
  };

  const fundamentalInfo = overview.fundamental;

  if (hideHeader) {
    return (
      <div className="chart-page">
        {error && <p className="error">{error}</p>}
        {data && <KLineChartView payload={data} />}
      </div>
    );
  }

  return (
    <div className="chart-page">
      <div className="controls">
        {(["daily", "weekly", "monthly"] as Period[]).map((item) => (
          <button
            key={item}
            className={period === item ? "active" : ""}
            onClick={() => setPeriod(item)}
          >
            {PERIOD_LABELS[item]}
          </button>
        ))}
        <button
          className="save-chart"
          onClick={save}
          disabled={saving || !data}
        >
          {saving ? "保存中..." : "保存此图表"}
        </button>
        {saveMessage && <span className="save-message">{saveMessage}</span>}
      </div>
      {error && <p className="error">{error}</p>}
      <div className="chart-overview">
        {overview.weekly && <span className="overview-badge">周线趋势：{directionCn(overview.weekly)}</span>}
        {overview.monthly && <span className="overview-badge">月线趋势：{directionCn(overview.monthly)}</span>}
        {overview.llm && (
          <span className="overview-badge">
            LLM 观点：{overallCn(overview.llm.overall)} · {overview.llm.text}
          </span>
        )}
        {fundamentalInfo && fundamentalInfo.mid_price != null && (
          <span className="overview-badge">
            估值中枢：{fundamentalInfo.mid_price} 元
            {fundamentalInfo.current_price != null &&
              ` · 当前价 ${fundamentalInfo.current_price} 元`}
            {fundamentalInfo.deviation_pct != null &&
              `（偏离 ${fundamentalInfo.deviation_pct >= 0 ? "+" : ""}${fundamentalInfo.deviation_pct.toFixed(1)}%）`}
            {fundamentalInfo.verdict && ` · ${fundamentalInfo.verdict}`}
          </span>
        )}
      </div>
      {data && <KLineChartView payload={data} />}
    </div>
  );
}
