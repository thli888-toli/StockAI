import { useEffect, useRef, useState } from "react";
import {
  CandlestickSeries,
  HistogramSeries,
  LineSeries,
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type Time,
} from "lightweight-charts";
import { getToken } from "./api";


type Period = "daily" | "weekly" | "monthly";

type FundamentalInfo = {
  mid_price: number;
  low: number | null;
  high: number | null;
  current_price: number | null;
  deviation_pct: number | null;
  verdict: string | null;
} | null;

type ChartPayload = {
  symbol: string;
  period: Period;
  candles: { time: string; open: number; high: number; low: number; close: number; volume: number }[];
  ma: Record<string, { time: string; value: number }[]>;
  macd: {
    dif: { time: string; value: number }[];
    dea: { time: string; value: number }[];
    histogram: { time: string; value: number }[];
  };
  signals: Record<string, "up" | "down" | "flat">;
  levels: {
    daily: { support: number; resistance: number };
    weekly: { support: number; resistance: number };
    monthly: { support: number; resistance: number };
  };
  llm_summary: { overall: "bullish" | "bearish" | "neutral"; text: string } | null;
  fundamental: FundamentalInfo;
};


const PERIOD_LABELS: Record<Period, string> = {
  daily: "日K",
  weekly: "周K",
  monthly: "月K",
};

const MA_COLORS = ["#f5a623", "#4a90e2", "#9013fe", "#7ed321"];


function directionCn(direction: string): string {
  return direction === "up" ? "上行" : direction === "down" ? "下行" : "中性";
}

function overallCn(overall: string): string {
  return overall === "bullish" ? "偏多" : overall === "bearish" ? "偏空" : "中性";
}


export default function StockChart({ symbol }: { symbol: string }) {
  const [period, setPeriod] = useState<Period>("daily");
  const [error, setError] = useState("");
  const priceRef = useRef<HTMLDivElement>(null);
  const macdRef = useRef<HTMLDivElement>(null);
  const syncingRef = useRef(false);
  const [overview, setOverview] = useState<{
    weekly?: string;
    monthly?: string;
    llm?: { overall: string; text: string } | null;
    fundamental?: FundamentalInfo;
  }>({});
  const [legend, setLegend] = useState<
    { label: string; value: string; color: string }[]
  >([]);

  useEffect(() => {
    let cancelled = false;
    let priceChart: IChartApi | null = null;
    let macdChart: IChartApi | null = null;

    const load = async () => {
      setError("");
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
      const data = (await response.json()) as ChartPayload;
      if (cancelled || !priceRef.current || !macdRef.current) return;
      setOverview({
        weekly: data.signals?.["1w"],
        monthly: data.signals?.["1mo"],
        llm: data.llm_summary,
        fundamental: data.fundamental,
      });

      priceRef.current.innerHTML = "";
      macdRef.current.innerHTML = "";

      priceChart = createChart(priceRef.current, {
        autoSize: true,
        layout: { background: { color: "#ffffff" }, textColor: "#333333" },
        grid: { vertLines: { color: "#e7e9ee" }, horzLines: { color: "#e7e9ee" } },
        rightPriceScale: { borderColor: "#d1d4dc" },
        timeScale: { borderColor: "#d1d4dc" },
      });
      const candleSeries = priceChart.addSeries(CandlestickSeries, {
        upColor: "#e04c4c",
        downColor: "#2f9e44",
        borderUpColor: "#e04c4c",
        borderDownColor: "#2f9e44",
        wickUpColor: "#e04c4c",
        wickDownColor: "#2f9e44",
        lastValueVisible: false,
      });
      candleSeries.setData(
        data.candles.map((candle) => ({
          time: candle.time as Time,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        }))
      );
      candleSeries.createPriceLine({
        price: data.levels[period].support,
        color: "#2f9e44",
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: false,
        title: "支撑",
      });
      candleSeries.createPriceLine({
        price: data.levels[period].resistance,
        color: "#e04c4c",
        lineWidth: 1,
        lineStyle: 2,
        axisLabelVisible: false,
        title: "阻力",
      });
      if (data.fundamental?.mid_price != null) {
        const deviation = data.fundamental.deviation_pct;
        const deviationText =
          deviation == null
            ? ""
            : `（${deviation >= 0 ? "+" : ""}${deviation.toFixed(1)}%）`;
        candleSeries.createPriceLine({
          price: data.fundamental.mid_price,
          color: "#9013fe",
          lineWidth: 2,
          lineStyle: 2,
          axisLabelVisible: false,
          title: `中枢 ${data.fundamental.mid_price}${deviationText}`,
        });
      }

      const lastCandle = data.candles[data.candles.length - 1];
      const legendItems: { label: string; value: string; color: string }[] = [];
      if (lastCandle) {
        legendItems.push({
          label: "当前价",
          value: String(lastCandle.close),
          color: "#333333",
        });
      }
      legendItems.push({
        label: "支撑",
        value: String(data.levels[period].support),
        color: "#2f9e44",
      });
      legendItems.push({
        label: "阻力",
        value: String(data.levels[period].resistance),
        color: "#e04c4c",
      });
      if (data.fundamental?.mid_price != null) {
        const deviation = data.fundamental.deviation_pct;
        const deviationText =
          deviation == null
            ? ""
            : `（${deviation >= 0 ? "+" : ""}${deviation.toFixed(1)}%）`;
        legendItems.push({
          label: "中枢",
          value: `${data.fundamental.mid_price}${deviationText}`,
          color: "#9013fe",
        });
      }
      setLegend(legendItems);
      if (lastCandle) {
        const shapes = {
          up: "arrowUp",
          down: "arrowDown",
          flat: "circle",
        } as const;
        const positions = ["aboveBar", "belowBar"] as const;
        const horizons = ["5d", "15d"] as const;
        createSeriesMarkers(
          candleSeries,
          horizons.map((horizon, index) => {
            const direction = data.signals?.[horizon] ?? "flat";
            return {
              time: lastCandle.time as Time,
              position: positions[index],
              shape: shapes[direction],
              color: direction === "up" ? "#e04c4c" : direction === "down" ? "#2f9e44" : "#888888",
              text: `${horizon}${direction === "up" ? "↑" : direction === "down" ? "↓" : "→"}`,
            };
          })
        );
      }

      Object.entries(data.ma).forEach(([_, points], index) => {
        const lineSeries = priceChart!.addSeries(LineSeries, {
          color: MA_COLORS[index % MA_COLORS.length],
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        });
        lineSeries.setData(points.map((point) => ({ time: point.time as Time, value: point.value })));
      });
      priceChart.timeScale().fitContent();

      macdChart = createChart(macdRef.current, {
        autoSize: true,
        layout: { background: { color: "#ffffff" }, textColor: "#333333" },
        grid: { vertLines: { color: "#e7e9ee" }, horzLines: { color: "#e7e9ee" } },
        rightPriceScale: { borderColor: "#d1d4dc" },
        timeScale: { borderColor: "#d1d4dc" },
      });
      const histogramSeries = macdChart.addSeries(HistogramSeries);
      histogramSeries.setData(
        data.macd.histogram.map((point) => ({
          time: point.time as Time,
          value: point.value,
          color: point.value >= 0 ? "#e04c4c" : "#2f9e44",
        }))
      );
      const difSeries = macdChart.addSeries(LineSeries, {
        color: "#4a90e2",
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      difSeries.setData(data.macd.dif.map((point) => ({ time: point.time as Time, value: point.value })));
      const deaSeries = macdChart.addSeries(LineSeries, {
        color: "#f5a623",
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      deaSeries.setData(data.macd.dea.map((point) => ({ time: point.time as Time, value: point.value })));
      macdChart.timeScale().fitContent();

      const syncTimeScale = (source: IChartApi, target: IChartApi) => {
        source.timeScale().subscribeVisibleLogicalRangeChange((range) => {
          if (range && !syncingRef.current) {
            syncingRef.current = true;
            target.timeScale().setVisibleLogicalRange(range);
            syncingRef.current = false;
          }
        });
      };
      syncTimeScale(priceChart, macdChart);
      syncTimeScale(macdChart, priceChart);
    };

    load().catch((loadError) => {
      if (!cancelled) setError(loadError instanceof Error ? loadError.message : String(loadError));
    });

    return () => {
      cancelled = true;
      if (priceChart) priceChart.remove();
      if (macdChart) macdChart.remove();
    };
  }, [symbol, period]);

  const fundamentalInfo = overview.fundamental;

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
      <div className="chart-with-legend">
        <div className="chart-stack">
          <div ref={priceRef} className="chart price-chart" />
          <div ref={macdRef} className="chart macd-chart" />
        </div>
        {legend.length > 0 && (
          <div className="chart-legend">
            {legend.map((item) => (
              <div key={item.label} className="chart-legend-item">
                <span
                  className="legend-dot"
                  style={{ backgroundColor: item.color }}
                />
                <span className="legend-label">{item.label}</span>
                <span className="legend-value">{item.value}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
