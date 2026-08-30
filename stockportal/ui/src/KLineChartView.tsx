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


export type Period = "daily" | "weekly" | "monthly";

export type FundamentalInfo = {
  mid_price: number;
  low: number | null;
  high: number | null;
  current_price: number | null;
  deviation_pct: number | null;
  verdict: string | null;
} | null;

export type ChartPayload = {
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


const MA_COLORS = ["#f5a623", "#4a90e2", "#9013fe", "#7ed321"];


export default function KLineChartView({ payload }: { payload: ChartPayload }) {
  const priceRef = useRef<HTMLDivElement>(null);
  const macdRef = useRef<HTMLDivElement>(null);
  const syncingRef = useRef(false);
  const [legend, setLegend] = useState<
    { label: string; value: string; color: string }[]
  >([]);

  useEffect(() => {
    if (!priceRef.current || !macdRef.current) return;
    let priceChart: IChartApi | null = null;
    let macdChart: IChartApi | null = null;

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
      payload.candles.map((candle) => ({
        time: candle.time as Time,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      }))
    );
    const lastCandle = payload.candles[payload.candles.length - 1];
    if (lastCandle) {
      candleSeries.createPriceLine({
        price: lastCandle.close,
        color: "#000000",
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: false,
        title: "当前价",
      });
    }
    candleSeries.createPriceLine({
      price: payload.levels[payload.period].support,
      color: "#2f9e44",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: false,
      title: "支撑",
    });
    candleSeries.createPriceLine({
      price: payload.levels[payload.period].resistance,
      color: "#e04c4c",
      lineWidth: 1,
      lineStyle: 2,
      axisLabelVisible: false,
      title: "阻力",
    });
    if (payload.fundamental?.mid_price != null) {
      const deviation = payload.fundamental.deviation_pct;
      const deviationText =
        deviation == null
          ? ""
          : `（${deviation >= 0 ? "+" : ""}${deviation.toFixed(1)}%）`;
      candleSeries.createPriceLine({
        price: payload.fundamental.mid_price,
        color: "#9013fe",
        lineWidth: 2,
        lineStyle: 2,
        axisLabelVisible: false,
        title: `中枢 ${payload.fundamental.mid_price}${deviationText}`,
      });
    }

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
      value: String(payload.levels[payload.period].support),
      color: "#2f9e44",
    });
    legendItems.push({
      label: "阻力",
      value: String(payload.levels[payload.period].resistance),
      color: "#e04c4c",
    });
    if (payload.fundamental?.mid_price != null) {
      const deviation = payload.fundamental.deviation_pct;
      const deviationText =
        deviation == null
          ? ""
          : `（${deviation >= 0 ? "+" : ""}${deviation.toFixed(1)}%）`;
      legendItems.push({
        label: "中枢",
        value: `${payload.fundamental.mid_price}${deviationText}`,
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
          const direction = payload.signals?.[horizon] ?? "flat";
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

    Object.entries(payload.ma).forEach(([_, points], index) => {
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
      payload.macd.histogram.map((point) => ({
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
    difSeries.setData(payload.macd.dif.map((point) => ({ time: point.time as Time, value: point.value })));
    const deaSeries = macdChart.addSeries(LineSeries, {
      color: "#f5a623",
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    deaSeries.setData(payload.macd.dea.map((point) => ({ time: point.time as Time, value: point.value })));
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

    return () => {
      if (priceChart) priceChart.remove();
      if (macdChart) macdChart.remove();
    };
  }, [payload]);

  return (
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
  );
}
