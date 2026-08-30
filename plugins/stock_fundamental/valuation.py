"""Multi-method fair-value estimation built on fundamental metrics.

Three methods are combined with equal weight when available:
  1. Relative valuation  - target multiples from historical percentiles or
     industry medians applied to TTM per-share metrics.
  2. DCF                - two-stage free cash flow model.
  3. Dividend discount  - Gordon growth model.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any


DEFAULT_DISCOUNT_RATE = 0.10
DEFAULT_TERMINAL_GROWTH = 0.02
DEFAULT_FORECAST_YEARS = 5
DEFAULT_TARGET_PERCENTILE = 0.50
DEFAULT_FALLBACK_GROWTH = 0.05
SENSITIVITY_RATES = (0.08, 0.12)
VERDICT_BAND = 0.10
MAX_GROWTH = 0.30
MIN_DIVIDEND_YIELD = 0.01
METHOD_WEIGHTS = {"relative": 1.0, "dcf": 0.8, "ddm": 0.5}
OUTLIER_LOW_FACTOR = 0.2
OUTLIER_HIGH_FACTOR = 5.0
CYCLICAL_KEYWORDS = (
    "证券",
    "券商",
    "银行",
    "保险",
    "有色",
    "钢铁",
    "煤炭",
    "航运",
    "地产",
    "航空",
)
PEG_FACTOR = 1.0
MIN_PEG_GROWTH = 0.15
TARGET_BAND = 0.15
METRIC_OUTLIER_FACTOR = 4.0
HISTORY_WEIGHT = 0.8
HISTORY_CAP = 3.0
PEER_INDUSTRY_BAND = (0.6, 1.6)
GROWTH_LEADER_THRESHOLD = 0.30
RESTRUCTURING_PB_RATIO = 10.0
RESTRUCTURING_PS_RATIO = 10.0
RESTRUCTURING_MAX_EPS = 0.5
RESTRUCTURING_MIN_BPS = 1.0
RESTRUCTURING_MIN_SPS = 1.0
PS_PEER_DIVERGENCE_FACTOR = 2.5
CYCLICAL_BOOM_PE_RATIO = 0.5
LEADER_HISTORY_WEIGHT = 0.5
LEADER_HISTORY_CAP_FACTOR = 1.5
PEER_OWN_PREMIUM_FACTOR = 2.0
PEER_MISMATCH_KEYWORDS = ("不匹配", "应替换为")


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _restructuring_in_progress(metrics: dict[str, Any]) -> bool:
    """Detect restructuring/asset-injection shells without flagging profitable
    high-multiple growth names.

    A genuine shell usually has not consolidated the injected assets yet, so
    its per-share book value and revenue are tiny in *absolute* terms. Merely
    having a high PB/PS is not enough: profitable semiconductor equipment and
    chip-design names routinely trade at PB 10-15x and PS 20-40x without any
    restructuring. We therefore first require negligible absolute fundamentals
    or negligible earnings before looking at the PB/PS ratios.
    """
    current_price = _num(metrics.get("current_price"))
    bps = _num(metrics.get("bps"))
    sps_ttm = _num(metrics.get("sps_ttm"))
    eps_ttm = _num(metrics.get("eps_ttm"))
    growth = _num(metrics.get("forecast_growth"))
    if current_price is None or current_price <= 0:
        return False
    if bps is None or bps <= 0 or sps_ttm is None or sps_ttm <= 0:
        return False
    if growth is not None and growth >= GROWTH_LEADER_THRESHOLD:
        return False
    # Absolute shell: book and revenue per share are both near zero.
    if bps < RESTRUCTURING_MIN_BPS and sps_ttm < RESTRUCTURING_MIN_SPS:
        return True
    # Otherwise only flag when earnings are negligible AND PB/PS are extreme.
    if eps_ttm is not None and eps_ttm > RESTRUCTURING_MAX_EPS:
        return False
    pb_current = current_price / bps
    ps_current = current_price / sps_ttm
    return (
        pb_current >= RESTRUCTURING_PB_RATIO
        and ps_current >= RESTRUCTURING_PS_RATIO
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _weighted_median(pairs: list[tuple[float, float]]) -> float:
    """Weighted median: the price where cumulative weight crosses half."""
    ordered = sorted(pairs, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    half = total / 2.0
    accumulated = 0.0
    for price, weight in ordered:
        accumulated += weight
        if accumulated >= half:
            return price
    return ordered[-1][0]


def _weighted_trimmed_mean(pairs: list[tuple[float, float]]) -> float:
    """Weighted mean after dropping the lowest and highest price samples.

    This is used for the in-metric aggregation of relative valuation. A plain
    median lets a single low PB/PS peer multiple pin the result, so for normal
    (non-leader) companies we trim one observation from each tail before
    weighting. With one or two samples the trimmed set is empty, so we fall
    back to the ordinary weighted mean.
    """
    if len(pairs) <= 2:
        total_weight = sum(weight for _, weight in pairs)
        return sum(price * weight for price, weight in pairs) / total_weight
    ordered = sorted(pairs, key=lambda item: item[0])
    trimmed = ordered[1:-1]
    total_weight = sum(weight for _, weight in trimmed)
    return sum(price * weight for price, weight in trimmed) / total_weight


def _weighted_growth(
    cagr: float | None,
    forecast: float | None,
    notes: list[str],
) -> float:
    if cagr is not None and forecast is not None:
        value = 0.6 * cagr + 0.4 * forecast
        notes.append(f"增长假设采用近3年营收CAGR({cagr:.1%})与一致预期增速({forecast:.1%})加权。")
    elif cagr is not None:
        value = cagr
        notes.append(f"增长假设采用近3年营收CAGR({cagr:.1%})；缺少一致预期数据。")
    elif forecast is not None:
        value = forecast
        notes.append(f"增长假设采用一致预期增速({forecast:.1%})；缺少历史营收CAGR。")
    else:
        value = DEFAULT_FALLBACK_GROWTH
        notes.append(f"缺少历史与一致预期增速数据，按默认 {DEFAULT_FALLBACK_GROWTH:.1%} 假设。")
    return _clamp(value, 0.0, MAX_GROWTH)


def _resolve_forward_pe(
    peers: dict[str, Any],
    forecast_year: Any,
) -> float | None:
    """Return a positive forward PE for the consensus year, else nearest year."""
    pe_forward = peers.get("pe_forward") or {}
    if forecast_year is not None:
        value = _num(pe_forward.get(str(forecast_year)))
        if value is not None and value > 0:
            return value
    for year_key in sorted(
        pe_forward,
        key=lambda item: (str(item).isdigit(), int(item) if str(item).isdigit() else 10**9),
    ):
        value = _num(pe_forward[year_key])
        if value is not None and value > 0:
            return value
    return None


def _growth_leader_relative(
    metrics: dict[str, Any],
    notes: list[str],
    growth: float | None,
) -> dict[str, Any] | None:
    """Relative valuation for high-growth leaders.

    When consensus forward EPS and a comparable-company forward PE are both
    available, anchor on the market-consistent forward PE plus a conservative
    PB multiple, and treat the company's own trailing historical p50 as a
    low-weight calibration with an upper cap (a past bubble can leave the
    historical p50 far above what the market pays today). When forward data is
    unavailable, fall back to the previous own-history-p50 anchor.
    """
    historical = metrics.get("historical") or {}
    peers = metrics.get("industry_peers") or {}
    industry = metrics.get("industry_bench") or {}
    eps_ttm = _num(metrics.get("eps_ttm"))
    bps = _num(metrics.get("bps"))
    sps_ttm = _num(metrics.get("sps_ttm"))
    forward_eps = _num(metrics.get("forecast_eps"))
    forecast_year = metrics.get("forecast_year")
    current_price = _num(metrics.get("current_price"))
    forward_pe = _resolve_forward_pe(peers, forecast_year)
    has_forward = (
        forward_eps is not None
        and forward_eps > 0
        and forward_pe is not None
        and forward_pe > 0
    )

    if has_forward:
        notes.append(
            f"高成长龙头分支：一致预期增速 {growth:.1%} ≥ {GROWTH_LEADER_THRESHOLD:.0%}，"
            "采用 forward PE 为主锚，并以自身历史分位低权重校准（带上限）。"
        )
    else:
        notes.append(
            f"高成长龙头分支：一致预期增速 {growth:.1%} ≥ {GROWTH_LEADER_THRESHOLD:.0%}，"
            "缺少 forward EPS/PE，采用自身近3年历史50分位为主锚，同行/行业仅作参考。"
        )

    estimates: list[tuple[float, float, float, float, dict[str, Any]]] = []

    pb_anchor: float | None = None
    if has_forward and bps is not None and bps > 0:
        peer_pb = _num((peers.get("pb") or {}).get("median"))
        current_pb = (
            current_price / bps
            if current_price is not None and current_price > 0
            else None
        )
        candidates = [value for value in (peer_pb, current_pb) if value is not None and value > 0]
        pb_anchor = min(candidates) if candidates else None

    def add_own_history_anchor(
        key: str,
        base: float | None,
        weight: float,
        cap_multiple: float | None = None,
    ) -> None:
        if base is None or base <= 0:
            return
        hist = historical.get(key) or {}
        p50 = _num(hist.get("p50"))
        p25 = _num(hist.get("p25"))
        p75 = _num(hist.get("p75"))
        if p50 is None or p50 <= 0:
            return
        if key == "pe_ttm":
            if p25 is not None and p25 <= 0:
                notes.append("PE 历史分位不可靠（p25≤0），龙头主锚改用 PB。")
                return
        elif key == "pb":
            if p25 is not None and p25 <= 0:
                return
        multiple = p50
        if cap_multiple is not None and cap_multiple > 0 and multiple > cap_multiple:
            label = "PE-TTM" if key == "pe_ttm" else "PB"
            notes.append(
                f"{label}自身历史50分位 {multiple:.2f} 超过龙头锚上限 "
                f"{cap_multiple:.2f}，已按上限参与校准。"
            )
            multiple = cap_multiple
            p25 = None
            p75 = None
        low_mult = (
            p25 if p25 is not None and p25 > 0 else multiple * (1.0 - TARGET_BAND)
        )
        high_mult = (
            p75 if p75 is not None and p75 > 0 else multiple * (1.0 + TARGET_BAND)
        )
        low_mult = min(low_mult, multiple)
        high_mult = max(high_mult, multiple)
        label = "PE-TTM" if key == "pe_ttm" else "PB"
        name_suffix = "龙头主锚" if weight >= 1.0 else "龙头校准"
        target_source = (
            "自身历史50分位(龙头主锚)"
            if weight >= 1.0
            else "自身历史50分位(龙头校准)"
        )
        estimates.append(
            (
                base * multiple,
                base * low_mult,
                base * high_mult,
                weight,
                {
                    "metric": f"{key}_hist_leader",
                    "name": f"{label}(自身历史{name_suffix})",
                    "base": _round(base),
                    "target_multiple": _round(multiple),
                    "target_source": target_source,
                    "implied_price": _round(base * multiple),
                    "implied_low": _round(base * low_mult),
                    "implied_high": _round(base * high_mult),
                    "weight": weight,
                },
            ),
        )

    if has_forward:
        if forward_eps is not None and forward_pe is not None and forward_pe > 0:
            fwd_implied = forward_eps * forward_pe
            estimates.append(
                (
                    fwd_implied,
                    fwd_implied * (1.0 - TARGET_BAND),
                    fwd_implied * (1.0 + TARGET_BAND),
                    1.0,
                    {
                        "metric": "pe_ttm_fwd_leader",
                        "name": "PE(forward 龙头主锚)",
                        "base": _round(forward_eps),
                        "target_multiple": _round(forward_pe),
                        "target_source": "类似公司中位数(forward PE)",
                        "implied_price": _round(fwd_implied),
                        "implied_low": _round(fwd_implied * (1.0 - TARGET_BAND)),
                        "implied_high": _round(fwd_implied * (1.0 + TARGET_BAND)),
                        "weight": 1.0,
                    },
                ),
            )
        if bps is not None and bps > 0 and pb_anchor is not None:
            pb_implied = bps * pb_anchor
            estimates.append(
                (
                    pb_implied,
                    pb_implied * (1.0 - TARGET_BAND),
                    pb_implied * (1.0 + TARGET_BAND),
                    1.0,
                    {
                        "metric": "pb_peer_leader",
                        "name": "PB(龙头主锚)",
                        "base": _round(bps),
                        "target_multiple": _round(pb_anchor),
                        "target_source": "类似公司中位数/当前PB(取低)",
                        "implied_price": _round(pb_implied),
                        "implied_low": _round(pb_implied * (1.0 - TARGET_BAND)),
                        "implied_high": _round(pb_implied * (1.0 + TARGET_BAND)),
                        "weight": 1.0,
                    },
                ),
            )
        add_own_history_anchor(
            "pe_ttm",
            eps_ttm,
            LEADER_HISTORY_WEIGHT,
            forward_pe * LEADER_HISTORY_CAP_FACTOR,
        )
        add_own_history_anchor(
            "pb",
            bps,
            LEADER_HISTORY_WEIGHT,
            pb_anchor * LEADER_HISTORY_CAP_FACTOR if pb_anchor else None,
        )
    else:
        add_own_history_anchor("pe_ttm", eps_ttm, 1.0)
        add_own_history_anchor("pb", bps, 1.0)

    if not estimates:
        return None

    peer_source = (
        "手动指定类似公司中位数"
        if peers.get("source") == "manual"
        else "大模型建议类似公司中位数"
        if peers.get("source") == "llm"
        else "类似公司中位数"
    )
    reference_details: list[dict[str, Any]] = []
    for key, peer_key, base in (
        ("pe_ttm", "pe", eps_ttm),
        ("pb", "pb", bps),
        ("ps", "ps", sps_ttm),
    ):
        if base is None or base <= 0:
            continue
        label = {"pe_ttm": "PE", "pb": "PB", "ps": "PS"}[key]
        for source, multiple in (
            (peer_source, (peers.get(peer_key) or {}).get("median")),
            ("行业整体中位数", ((industry or {}).get(peer_key) or {}).get("median")),
        ):
            multiple = _num(multiple)
            if multiple is None or multiple <= 0:
                continue
            reference_details.append(
                {
                    "metric": f"{key}_{'peer' if source == peer_source else 'industry'}",
                    "name": f"{label}({'同行参考' if source == peer_source else '行业参考'})",
                    "base": _round(base),
                    "target_multiple": _round(multiple),
                    "target_source": source,
                    "implied_price": _round(base * multiple),
                    "weight": 0,
                }
            )

    if has_forward:
        mid = _weighted_trimmed_mean(
            [(estimate[0], estimate[3]) for estimate in estimates]
        )
        basis = "forward PE + PB(高成长龙头)"
    else:
        mid = float(statistics.median([estimate[0] for estimate in estimates]))
        basis = "自身历史50分位(高成长龙头)"
    low = min(estimate[1] for estimate in estimates)
    high = max(estimate[2] for estimate in estimates)

    peer_names = [
        str(item.get("name"))
        for item in (peers.get("peer_list") or [])
        if item.get("name")
    ]
    if peers.get("source") == "llm" and peers.get("reason"):
        notes.append(f"可比公司名单经大模型校验：{peers.get('reason')}")
    if peer_names:
        notes.append(
            "同行/行业仅作参考，不参与龙头主锚中枢计算："
            + "、".join(peer_names[:10])
            + ("等" if len(peer_names) > 10 else "")
            + "。"
        )

    return {
        "available": True,
        "method": "relative",
        "price": _round(mid),
        "low": _round(low),
        "high": _round(high),
        "detail": [estimate[4] for estimate in estimates] + reference_details,
        "notes": notes,
        "basis": basis,
        "peer_names": peer_names,
        "peer_count": len(peer_names) if peer_names else None,
    }


def relative_valuation(metrics: dict[str, Any]) -> dict[str, Any]:
    """Implied price from target multiples.

    Target-multiple hierarchy: comparable-company median -> own 3y historical
    p50 -> whole-industry median. PE is PEG-calibrated when consensus growth
    is available and a forward-PE estimate is added when consensus EPS exists.
    Cyclical industries drop PE-TTM and rely on PB/PS. Loss-making companies
    (EPS-TTM <= 0) anchor PB on own historical percentile and treat peer
    PB/PS as reference-only.
    """
    valuation = metrics.get("valuation") or {}
    historical = metrics.get("historical") or {}
    peers = metrics.get("industry_peers") or {}
    industry = metrics.get("industry_bench") or {}
    industry_name = str(metrics.get("industry_name") or "")
    growth = _num(metrics.get("forecast_growth"))
    eps_ttm = _num(metrics.get("eps_ttm"))
    forward_eps = _num(metrics.get("forecast_eps"))
    forecast_year = metrics.get("forecast_year")
    loss_making = eps_ttm is not None and eps_ttm <= 0
    cyclical = any(keyword and keyword in industry_name for keyword in CYCLICAL_KEYWORDS)
    notes: list[str] = []
    if cyclical:
        notes.append(
            f"行业「{industry_name or '未知'}」为周期行业，PE-TTM 不参与相对估值，优先采用 PB/PS。"
        )
    if loss_making:
        notes.append(
            "TTM 每股收益为负，PE 口径不可用；相对估值以自身历史 PB 为主锚，同行 PB/PS 仅作参考。"
        )
    peer_reason = str(peers.get("reason") or "")
    peers_low_confidence = (
        peers.get("source") in ("manual", "llm")
        and any(keyword in peer_reason for keyword in PEER_MISMATCH_KEYWORDS)
    )
    if peers_low_confidence:
        notes.append(
            "大模型校验提示可比公司名单与目标公司业务/规模不匹配，"
            "目标倍数改用自身历史分位为主锚，同行仅作参考。"
        )
    if _restructuring_in_progress(metrics):
        return {
            "available": False,
            "method": "relative",
            "restructuring_in_progress": True,
            "notes": [
                "疑似重组/资产注入中：当前报表的每股净资产与每股营收相对市值严重偏低，"
                "相对估值不适用，建议等待注入资产并表后再估值。"
            ],
        }
    growth_leader = growth is not None and growth >= GROWTH_LEADER_THRESHOLD
    if growth_leader:
        leader = _growth_leader_relative(metrics, notes, growth)
        if leader is not None:
            return leader
    notes.append("目标倍数层级：可比公司中位数 → 自身历史50分位 → 行业整体中位数。")

    per_share = {
        "pe_ttm": metrics.get("eps_ttm"),
        "pb": metrics.get("bps"),
        "ps": metrics.get("sps_ttm"),
    }
    names = {"pe_ttm": "PE-TTM", "pb": "PB", "ps": "PS"}
    # (price, low, high, weight, detail)
    estimates: list[tuple[float, float, float, float, dict[str, Any]]] = []
    reference_details: list[dict[str, Any]] = []
    for key, base_value in per_share.items():
        if key == "pe_ttm" and cyclical:
            continue
        base = _num(base_value)
        if base is None or base <= 0:
            continue
        hist = historical.get(key) or {}
        peer_key = {"pe_ttm": "pe", "pb": "pb", "ps": "ps"}[key]
        peers_item = peers.get(peer_key) or {}
        industry_item = industry.get(peer_key) or {}
        target: float | None = None
        source = ""
        peers_source = (
            "手动指定类似公司中位数"
            if peers.get("source") == "manual"
            else "大模型建议类似公司中位数"
            if peers.get("source") == "llm"
            else "类似公司中位数"
        )
        peers_median = _num(peers_item.get("median"))
        industry_median = _num(industry_item.get("median"))
        hist_p50 = _num(hist.get("p50"))
        hist_p25 = _num(hist.get("p25"))
        hist_p75 = _num(hist.get("p75"))
        pe_history_unusable = (
            key == "pe_ttm"
            and hist_p25 is not None
            and hist_p25 <= 0
        )

        # Cyclical earnings boom detection: when the company's current PE is a
        # fraction of both the peer median and its own historical p50, the
        # market is pricing a cyclical earnings peak (e.g. memory/storage at
        # the top of the cycle). TTM EPS is unsustainably high, so PE-based
        # estimates massively overstate value; fall back to PB/PS only.
        cyclical_boom = False
        if (
            key == "pe_ttm"
            and not loss_making
            and not cyclical
        ):
            current_pe = _num((valuation or {}).get("pe_ttm"))
            pe_candidates = [
                multiple
                for multiple in (peers_median, hist_p50)
                if multiple is not None and multiple > 0
            ]
            if (
                current_pe is not None
                and current_pe > 0
                and pe_candidates
                and current_pe <= CYCLICAL_BOOM_PE_RATIO * min(pe_candidates)
            ):
                cyclical_boom = True
        if cyclical_boom:
            notes.append(
                f"当前PE {current_pe:.2f} 显著低于同行与自身历史中位数"
                f"（≤ {min(pe_candidates):.2f} × {CYCLICAL_BOOM_PE_RATIO:.0%}），"
                "疑似周期景气高点盈利，PE 口径不适用，改用 PB/PS。"
            )
            continue

        # Peer PS medians are unreliable whenever they diverge from the
        # company's own PS by a wide margin: for low-margin names (server
        # assemblers) peers overstate, and for high-revenue cyclical names
        # (memory modules) peers mix in non-comparable businesses. Drop the
        # PS leg instead of feeding an unrealistic multiple into the mean.
        if (
            key == "ps"
            and not loss_making
            and not cyclical
            and peers_median is not None
            and peers_median > 0
        ):
            current_ps = _num((valuation or {}).get("ps"))
            if (
                current_ps is not None
                and current_ps > 0
                and peers_median > current_ps * PS_PEER_DIVERGENCE_FACTOR
            ):
                notes.append(
                    f"同行PS中位数 {peers_median:.2f} 与自身当前PS {current_ps:.2f} "
                    "偏离过大，弃用PS相对估值口径。"
                )
                continue

        if loss_making:
            if key == "pb" and hist_p50 is not None and hist_p50 > 0:
                implied = base * hist_p50
                multiple_low = min(
                    hist_p25
                    if hist_p25 is not None and hist_p25 > 0
                    else hist_p50 * (1.0 - TARGET_BAND),
                    hist_p50,
                )
                multiple_high = max(
                    hist_p75
                    if hist_p75 is not None and hist_p75 > 0
                    else hist_p50 * (1.0 + TARGET_BAND),
                    hist_p50,
                )
                estimates.append(
                    (
                        implied,
                        base * multiple_low,
                        base * multiple_high,
                        1.0,
                        {
                            "metric": "pb_hist",
                            "name": "PB(自身历史)",
                            "base": _round(base),
                            "target_multiple": _round(hist_p50),
                            "target_source": "自身历史50分位",
                            "implied_price": _round(implied),
                            "implied_low": _round(base * multiple_low),
                            "implied_high": _round(base * multiple_high),
                            "weight": 1.0,
                        },
                    ),
                )
                if peers_median is not None and peers_median > 0:
                    reference_details.append(
                        {
                            "metric": "pb_peer",
                            "name": "PB(同行参考)",
                            "base": _round(base),
                            "target_multiple": _round(peers_median),
                            "target_source": peers_source,
                            "implied_price": _round(base * peers_median),
                            "weight": 0,
                        }
                    )
                continue
            if key == "ps":
                if peers_median is not None and peers_median > 0:
                    reference_details.append(
                        {
                            "metric": "ps_peer",
                            "name": "PS(同行参考)",
                            "base": _round(base),
                            "target_multiple": _round(peers_median),
                            "target_source": peers_source,
                            "implied_price": _round(base * peers_median),
                            "weight": 0,
                        }
                    )
                continue

        target = None
        source = ""
        current_multiple = _num((valuation or {}).get(key))
        validated_peers = peers.get("source") in ("manual", "llm")

        # Premium-leader guard: when the company's own current PB/PS is far
        # above the peer median, the peer median understates a sustained
        # premium. Anchor on own historical p50 if available, otherwise drop
        # the leg so a too-low peer multiple cannot drag the midpoint down.
        if (
            key in ("pb", "ps")
            and not loss_making
            and not cyclical
            and peers_median is not None
            and peers_median > 0
            and current_multiple is not None
            and current_multiple > 0
            and current_multiple > peers_median * PEER_OWN_PREMIUM_FACTOR
        ):
            if hist_p50 is not None and hist_p50 > 0 and hist_p50 > peers_median:
                target = hist_p50
                source = "自身历史50分位"
                notes.append(
                    f"同行{peer_key.upper()}中位数 {peers_median:.2f} 显著低于自身当前"
                    f"{names[key]} {current_multiple:.2f}，采用自身历史50分位 "
                    f"{hist_p50:.2f} 作为{names[key]}锚。"
                )
            else:
                notes.append(
                    f"同行{peer_key.upper()}中位数 {peers_median:.2f} 显著低于自身当前"
                    f"{names[key]} {current_multiple:.2f} 且缺少自身历史分位，"
                    f"弃用{names[key]}口径。"
                )
                continue

        if target is None:
            if (
                peers_low_confidence
                and not pe_history_unusable
                and hist_p50 is not None
                and hist_p50 > 0
            ):
                target = hist_p50
                source = "自身历史50分位"
            elif industry_median is not None and industry_median > 0:
                target = industry_median
                source = "行业整体中位数"
                if peers_median is not None and peers_median > 0:
                    band_low, band_high = PEER_INDUSTRY_BAND
                    too_low = peers_median < band_low * industry_median
                    too_high = peers_median > band_high * industry_median
                    # Manually/LLM-validated peers represent a vetted comparable
                    # set, so a higher multiple is trusted instead of being
                    # downgraded to the broad industry median. A too-low
                    # multiple is still discarded (premium-leader case).
                    if (not too_low and not too_high) or (
                        validated_peers and not too_low
                    ):
                        target = peers_median
                        source = peers_source
                    else:
                        notes.append(
                            f"同行{peer_key.upper()}中位数 {peers_median:.2f} 与行业整体中位数 "
                            f"{industry_median:.2f} 偏离过大（超出 "
                            f"{band_low:.0%}–{band_high:.0%} 区间），改用行业整体+自身历史。"
                        )
            elif peers_median is not None and peers_median > 0:
                target = peers_median
                source = peers_source
            elif (
                hist_p50 is not None
                and hist_p50 > 0
                and not pe_history_unusable
            ):
                target = hist_p50
                source = "自身历史50分位"
        if target is None:
            continue
        if key == "pe_ttm":
            target = _pe_target_adjusted(target, growth, notes)
        implied = base * target
        if implied <= 0:
            continue
        hist_p25 = _num(hist.get("p25"))
        hist_p75 = _num(hist.get("p75"))
        multiple_low = (
            hist_p25
            if hist_p25 is not None and hist_p25 > 0
            else target * (1.0 - TARGET_BAND)
        )
        multiple_high = (
            hist_p75
            if hist_p75 is not None and hist_p75 > 0
            else target * (1.0 + TARGET_BAND)
        )
        multiple_low = min(multiple_low, target)
        multiple_high = max(multiple_high, target)
        estimates.append(
            (
                implied,
                base * multiple_low,
                base * multiple_high,
                1.0,
                {
                    "metric": key,
                    "name": names[key],
                    "base": _round(base),
                    "target_multiple": _round(target),
                    "target_source": source,
                    "implied_price": _round(implied),
                    "implied_low": _round(base * multiple_low),
                    "implied_high": _round(base * multiple_high),
                    "weight": 1.0,
                },
            ),
        )
        if pe_history_unusable:
            notes.append(
                "历史PE分位区间含亏损期（p25≤0），PE自身历史锚不参与估值。"
            )
        if (
            source != "自身历史50分位"
            and hist_p50 is not None
            and hist_p50 > 0
            and not pe_history_unusable
        ):
            cap_multiple = HISTORY_CAP * target
            hist_multiple_low = (
                hist_p25
                if hist_p25 is not None and hist_p25 > 0
                else hist_p50 * (1.0 - TARGET_BAND)
            )
            hist_multiple_high = (
                hist_p75
                if hist_p75 is not None and hist_p75 > 0
                else hist_p50 * (1.0 + TARGET_BAND)
            )
            hist_multiple = min(hist_p50, cap_multiple)
            hist_multiple_low = min(
                min(hist_multiple_low, hist_p50), cap_multiple
            )
            hist_multiple_high = min(
                max(hist_multiple_high, hist_p50), cap_multiple
            )
            if hist_p50 > cap_multiple:
                notes.append(
                    f"{names[key]}自身历史50分位 {hist_p50:.2f} 显著高于当前基准"
                    f"（上限 {HISTORY_CAP:.1f} 倍），已按上限参与。"
                )
            hist_price = base * hist_multiple
            estimates.append(
                (
                    hist_price,
                    base * hist_multiple_low,
                    base * hist_multiple_high,
                    HISTORY_WEIGHT,
                    {
                        "metric": f"{key}_hist",
                        "name": f"{names[key]}(历史分位)",
                        "base": _round(base),
                        "target_multiple": _round(hist_multiple),
                        "target_source": "自身历史50分位",
                        "implied_price": _round(hist_price),
                        "implied_low": _round(base * hist_multiple_low),
                        "implied_high": _round(base * hist_multiple_high),
                        "weight": HISTORY_WEIGHT,
                    },
                ),
            )
        if key == "pe_ttm" and forward_eps is not None and forward_eps > 0:
            forward_pe = _num(
                (peers.get("pe_forward") or {}).get(str(forecast_year))
            )
            if forward_pe is None or forward_pe <= 0:
                notes.append(
                    "缺少可比公司 forward PE，未计算 forward 相对估值（避免用 TTM 倍数乘 forward EPS 双重计增）。"
                )
                continue
            forward_implied = forward_eps * forward_pe
            fwd_low = forward_eps * forward_pe * (1.0 - TARGET_BAND)
            fwd_high = forward_eps * forward_pe * (1.0 + TARGET_BAND)
            estimates.append(
                (
                    forward_implied,
                    fwd_low,
                    fwd_high,
                    1.0,
                    {
                        "metric": "pe_ttm_fwd",
                        "name": "PE-TTM(forward)",
                        "base": _round(forward_eps),
                        "target_multiple": _round(forward_pe),
                        "target_source": "类似公司中位数(forward PE)",
                        "implied_price": _round(forward_implied),
                        "implied_low": _round(fwd_low),
                        "implied_high": _round(fwd_high),
                        "weight": 1.0,
                    },
                ),
            )
    if len(estimates) >= 3:
        median_price = float(
            statistics.median(estimate[0] for estimate in estimates)
        )
        dropped = [
            estimate
            for estimate in estimates
            if estimate[0] > METRIC_OUTLIER_FACTOR * median_price
        ]
        if len(estimates) - len(dropped) >= 2 and dropped:
            estimates = [
                estimate
                for estimate in estimates
                if estimate[0] <= METRIC_OUTLIER_FACTOR * median_price
            ]
            for estimate in dropped:
                detail = estimate[4]
                notes.append(
                    f"剔除异常放大口径：{detail.get('name', detail.get('metric', ''))}"
                    f"（隐含价 {_round(detail.get('implied_price'))} 元，"
                    f"超过方法中位数 {METRIC_OUTLIER_FACTOR:.0f} 倍）。"
                )
    if not estimates:
        return {
            "available": False,
            "method": "relative",
            "notes": notes + ["缺少足够的倍数或每股指标，相对估值不可用。"],
        }
    history_blends = [
        estimate[4]
        for estimate in estimates
        if estimate[3] == HISTORY_WEIGHT
    ]
    if history_blends:
        notes.append(
            "混入自身历史分位估值（权重"
            + f"{HISTORY_WEIGHT:.0%}"
            + "）："
            + "、".join(
                f"{detail.get('name', detail.get('metric', ''))} "
                f"{_round(detail.get('implied_price'))} 元"
                for detail in history_blends
            )
            + "。"
        )
    primary_estimates = [
        estimate[4]
        for estimate in estimates
        if estimate[3] >= 1.0
    ] or [estimate[4] for estimate in estimates]
    source_counts = Counter(item["target_source"] for item in primary_estimates)
    primary_source = source_counts.most_common(1)[0][0] if source_counts else ""
    if primary_source == "自身历史50分位":
        if peers_low_confidence:
            notes.append(
                "同行名单经大模型校验与目标公司不匹配，目标倍数采用自身历史分位。"
            )
        elif not loss_making:
            notes.append(
                "可比公司与行业整体数据不可用，目标倍数回退至自身近3年历史50分位。"
            )
    elif primary_source == "行业整体中位数":
        notes.append("可比公司数据不可用或样本不足，目标倍数回退至行业整体中位数。")
    peer_names: list[str] = []
    if primary_source in (
        "类似公司中位数",
        "手动指定类似公司中位数",
        "大模型建议类似公司中位数",
    ):
        peer_names = [
            str(item.get("name"))
            for item in (peers.get("peer_list") or [])
            if item.get("name")
        ]
        if "手动" in primary_source:
            label = "手动指定类似公司"
        elif "大模型" in primary_source:
            label = "大模型建议类似公司"
        else:
            label = "类似公司"
        if peers.get("source") == "llm" and peers.get("reason"):
            notes.append(f"可比公司名单经大模型校验：{peers.get('reason')}")
        notes.append(
            f"目标倍数采用{label}中位数（共{len(peer_names)}家：{'、'.join(peer_names[:10])}"
            + ("等" if len(peer_names) > 10 else "")
            + "）。"
        )
    return {
        "available": True,
        "method": "relative",
        "price": _round(
            _weighted_trimmed_mean(
                [(estimate[0], estimate[3]) for estimate in estimates]
            )
        ),
        "low": _round(
            _weighted_trimmed_mean(
                [(estimate[1], estimate[3]) for estimate in estimates]
            )
        ),
        "high": _round(
            _weighted_trimmed_mean(
                [(estimate[2], estimate[3]) for estimate in estimates]
            )
        ),
        "detail": [estimate[4] for estimate in estimates] + reference_details,
        "notes": notes,
        "basis": primary_source,
        "peer_names": peer_names,
        "peer_count": len(peer_names) if peer_names else None,
    }


def _pe_target_adjusted(
    raw_target: float,
    growth: float | None,
    notes: list[str],
) -> float:
    """Calibrate a PE target with consensus growth (g% * PEG_FACTOR)."""
    if growth is None or growth <= 0:
        return raw_target
    if growth < MIN_PEG_GROWTH:
        notes.append(
            f"一致预期增速 {growth:.1%} 低于 {MIN_PEG_GROWTH:.0%}，"
            "PEG 校准不适用，目标倍数保持行业/同行口径。"
        )
        return raw_target
    peg_target = growth * 100.0 * PEG_FACTOR
    adjusted = _clamp(peg_target, raw_target * 0.6, raw_target * 1.5)
    if abs(adjusted - raw_target) > 1e-9:
        notes.append(
            f"PE 目标倍数按 PEG 校准：{raw_target:.2f} → {adjusted:.2f}"
            f"（一致预期增速 {growth:.1%}）。"
        )
    return adjusted


def _discounted_cash_flow(
    fcf_per_share: float,
    growth: float,
    discount_rate: float,
    terminal_growth: float,
    years: int,
) -> float:
    if discount_rate <= terminal_growth:
        raise ValueError("discount rate must exceed terminal growth")
    present_value = 0.0
    fcf = fcf_per_share
    for year in range(1, years + 1):
        fcf *= 1.0 + growth
        present_value += fcf / ((1.0 + discount_rate) ** year)
    terminal_value = fcf * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
    present_value += terminal_value / ((1.0 + discount_rate) ** years)
    return present_value


def dcf_valuation(metrics: dict[str, Any]) -> dict[str, Any]:
    """Two-stage DCF with 5 explicit years and a terminal value."""
    fcf = _num(metrics.get("fcf"))
    total_shares = _num(metrics.get("total_shares"))
    if fcf is None or total_shares is None or total_shares <= 0:
        return {
            "available": False,
            "method": "dcf",
            "notes": ["缺少自由现金流或股本数据，DCF 不可用。"],
        }
    if fcf <= 0:
        return {
            "available": False,
            "method": "dcf",
            "notes": ["自由现金流≤0（亏损或高资本开支），DCF 模型不适用。"],
        }
    notes: list[str] = []
    growth = _weighted_growth(
        _num(metrics.get("revenue_growth_cagr")),
        _num(metrics.get("forecast_growth")),
        notes,
    )
    fcf_per_share = fcf / total_shares
    base = _discounted_cash_flow(
        fcf_per_share,
        growth,
        DEFAULT_DISCOUNT_RATE,
        DEFAULT_TERMINAL_GROWTH,
        DEFAULT_FORECAST_YEARS,
    )
    prices = {rate: _discounted_cash_flow(fcf_per_share, growth, rate, DEFAULT_TERMINAL_GROWTH, DEFAULT_FORECAST_YEARS) for rate in SENSITIVITY_RATES}
    notes.append(
        f"折现率 {DEFAULT_DISCOUNT_RATE:.0%}，永续增速 {DEFAULT_TERMINAL_GROWTH:.0%}，"
        f"显性期 {DEFAULT_FORECAST_YEARS} 年；敏感性折现率 {'/'.join(f'{r:.0%}' for r in SENSITIVITY_RATES)}。"
    )
    return {
        "available": True,
        "method": "dcf",
        "price": _round(base),
        "low": _round(min(prices.values())),
        "high": _round(max(prices.values())),
        "fcf_per_share": _round(fcf_per_share),
        "growth": _round(growth, 4),
        "discount_rate": DEFAULT_DISCOUNT_RATE,
        "terminal_growth": DEFAULT_TERMINAL_GROWTH,
        "years": DEFAULT_FORECAST_YEARS,
        "sensitivity": {
            f"{rate:.0%}": _round(price) for rate, price in sorted(prices.items())
        },
        "notes": notes,
    }


def ddm_valuation(metrics: dict[str, Any]) -> dict[str, Any]:
    """Gordon growth dividend discount model."""
    dps = _num(metrics.get("dps"))
    roe = _num(metrics.get("roe"))
    if dps is None or dps <= 0:
        return {
            "available": False,
            "method": "ddm",
            "notes": ["缺少每股股息数据，股息折现不可用。"],
        }
    if roe is None or roe <= 0:
        return {
            "available": False,
            "method": "ddm",
            "notes": ["ROE 为负或缺失，可持续增长率无法计算，股息折现不适用。"],
        }
    payout = _num(metrics.get("payout_ratio"))
    if payout is None:
        return {
            "available": False,
            "method": "ddm",
            "notes": ["缺少分红率数据，股息折现不适用。"],
        }
    current_price = _num(metrics.get("current_price"))
    if current_price is not None and current_price > 0:
        dividend_yield = dps / current_price
        if dividend_yield < MIN_DIVIDEND_YIELD:
            return {
                "available": False,
                "method": "ddm",
                "notes": [
                    f"股息率 {dividend_yield:.2%} 低于 {MIN_DIVIDEND_YIELD:.0%}，"
                    "股息折现不适用。"
                ],
            }
    notes: list[str] = []
    payout = _clamp(payout, 0.0, 1.0)
    growth = _clamp(roe * (1.0 - payout), 0.0, 0.10)
    if DEFAULT_DISCOUNT_RATE <= growth:
        return {
            "available": False,
            "method": "ddm",
            "notes": ["股息增长率不低于折现率，戈登模型不适用。"],
        }
    price = dps * (1.0 + growth) / (DEFAULT_DISCOUNT_RATE - growth)
    sensitivity: dict[float, float] = {}
    for rate in SENSITIVITY_RATES:
        if rate > growth + 0.01:
            sensitivity[rate] = dps * (1.0 + growth) / (rate - growth)
    all_prices = [price] + list(sensitivity.values())
    notes.append(
        f"增长假设 g=ROE×(1-分红率)={growth:.1%}，折现率 {DEFAULT_DISCOUNT_RATE:.0%}。"
    )
    return {
        "available": True,
        "method": "ddm",
        "price": _round(price),
        "low": _round(min(all_prices)),
        "high": _round(max(all_prices)),
        "dps": _round(dps),
        "growth": _round(growth, 4),
        "payout_ratio": _round(payout, 4),
        "discount_rate": DEFAULT_DISCOUNT_RATE,
        "sensitivity": {
            f"{rate:.0%}": _round(value) for rate, value in sorted(sensitivity.items())
        },
        "notes": notes,
    }


DISCLAIMER = "以上为基于历史财务数据与公开预期的估算区间，不构成投资建议或收益保证。"


def estimate_fair_value(metrics: dict[str, Any]) -> dict[str, Any]:
    """Combine available valuation methods into a low/mid/high range."""
    methods = [
        relative_valuation(metrics),
        dcf_valuation(metrics),
        ddm_valuation(metrics),
    ]
    available = [method for method in methods if method.get("available")]
    if not available:
        restructuring = next(
            (method for method in methods if method.get("restructuring_in_progress")),
            None,
        )
        if restructuring is not None:
            current_price = _num(metrics.get("current_price"))
            return {
                "fair_value_range": {
                    "low": None,
                    "mid": None,
                    "high": None,
                },
                "per_method": {
                    method["method"]: dict(method) for method in methods
                },
                "verdict": {
                    "label": "重组/注入中",
                    "current_price": _round(current_price),
                    "margin": None,
                    "text": "疑似重组/资产注入中，相对估值不适用，暂不给出估值中枢。",
                },
                "available_methods": [],
                "excluded_methods": [
                    {
                        "method": method["method"],
                        "reason": next(iter(method.get("notes") or []), "不适用"),
                    }
                    for method in methods
                    if not method.get("available")
                ],
                "assumptions": {
                    "discount_rate": DEFAULT_DISCOUNT_RATE,
                    "terminal_growth": DEFAULT_TERMINAL_GROWTH,
                    "forecast_years": DEFAULT_FORECAST_YEARS,
                    "target_percentile": DEFAULT_TARGET_PERCENTILE,
                    "method_weights": dict(METHOD_WEIGHTS),
                    "outlier_band": [OUTLIER_LOW_FACTOR, OUTLIER_HIGH_FACTOR],
                },
                "disclaimer": DISCLAIMER,
            }
        raise ValueError("没有足够的财务数据计算合理股价估值")

    prices = [float(method["price"]) for method in available]
    median_price = float(statistics.median(prices))
    kept = [
        method
        for method in available
        if OUTLIER_LOW_FACTOR * median_price
        <= float(method["price"])
        <= OUTLIER_HIGH_FACTOR * median_price
    ]
    excluded = [method for method in available if method not in kept]
    if not kept:
        kept = available
        excluded = []

    weighted_pairs = [
        (method, METHOD_WEIGHTS.get(method["method"], 1.0)) for method in kept
    ]
    low = _weighted_median(
        [(float(method.get("low") or method["price"]), weight) for method, weight in weighted_pairs]
    )
    high = _weighted_median(
        [(float(method.get("high") or method["price"]), weight) for method, weight in weighted_pairs]
    )
    mid = _weighted_median(
        [(float(method["price"]), weight) for method, weight in weighted_pairs]
    )
    low = min(low, mid)
    high = max(high, mid)

    current_price = _num(metrics.get("current_price"))
    if current_price is None or current_price <= 0:
        verdict: dict[str, Any] = {
            "label": "数据不足",
            "text": "缺少当前股价，无法给出低估/合理/高估判断。",
        }
    else:
        margin = (current_price - mid) / mid
        if margin < -VERDICT_BAND:
            label = "低估"
        elif margin > VERDICT_BAND:
            label = "高估"
        else:
            label = "合理"
        verdict = {
            "label": label,
            "current_price": _round(current_price),
            "margin": _round(margin, 4),
            "text": (
                f"当前股价 {_round(current_price)} 元相对估值中枢 {_round(mid)} 元"
                f"偏离 {margin:+.1%}，判断为{label}。"
            ),
        }

    per_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        per_method[method["method"]] = dict(method)

    return {
        "fair_value_range": {
            "low": _round(low),
            "mid": _round(mid),
            "high": _round(high),
        },
        "per_method": per_method,
        "verdict": verdict,
        "available_methods": [method["method"] for method in kept],
        "excluded_methods": [
            {
                "method": method["method"],
                "reason": next(iter(method.get("notes") or []), "与其他方法偏差过大"),
            }
            for method in excluded
        ],
        "assumptions": {
            "discount_rate": DEFAULT_DISCOUNT_RATE,
            "terminal_growth": DEFAULT_TERMINAL_GROWTH,
            "forecast_years": DEFAULT_FORECAST_YEARS,
            "target_percentile": DEFAULT_TARGET_PERCENTILE,
            "method_weights": dict(METHOD_WEIGHTS),
            "outlier_band": [OUTLIER_LOW_FACTOR, OUTLIER_HIGH_FACTOR],
        },
        "disclaimer": DISCLAIMER,
    }
