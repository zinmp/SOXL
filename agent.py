"""
떨사오팔 금융 에이전트
- Claude API를 통해 백테스트 결과를 분석하고 전략 개선안을 제안
- 매주 1회 자동 실행 or 수동 실행
- 결과는 agent_log.json에 저장
"""
import json
import os
from datetime import datetime, timedelta
from typing import Iterator

import anthropic
import pandas as pd

from backtest import run_backtest
from config import STRATEGIES, EVAL_WINDOW
from data import get_data
from indicators import calc_indicators
from tuner import calc_error_stats, load_log as load_comparison_log

AGENT_LOG_FILE = "agent_log.json"
ANTHROPIC_API_KEY_FILE = "api_key.txt"


# ── API 키 관리 ────────────────────────────────────────────────
def get_api_key() -> str:
    # 환경변수 우선
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    # 파일에서 읽기
    if os.path.exists(ANTHROPIC_API_KEY_FILE):
        with open(ANTHROPIC_API_KEY_FILE, "r") as f:
            return f.read().strip()
    return ""


def save_api_key(key: str):
    with open(ANTHROPIC_API_KEY_FILE, "w") as f:
        f.write(key.strip())


# ── 에이전트 로그 ──────────────────────────────────────────────
def load_agent_log() -> list:
    if not os.path.exists(AGENT_LOG_FILE):
        return []
    with open(AGENT_LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_agent_log(log: list):
    with open(AGENT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)


def should_run_weekly() -> bool:
    """마지막 실행 후 7일 이상 지났으면 True"""
    log = load_agent_log()
    if not log:
        return True
    last_run = log[-1].get("실행시각", "2000-01-01")
    try:
        last_dt = datetime.strptime(last_run[:16], "%Y-%m-%d %H:%M")
        return datetime.now() - last_dt >= timedelta(days=7)
    except Exception:
        return True


# ── 후보 파라미터 백테스트 ────────────────────────────────────
def _annual_stats(df: pd.DataFrame, params: dict) -> dict:
    """전략 파라미터로 연도별 백테스트 실행 후 요약 통계 반환"""
    years = sorted(df.index.year.unique())
    annual = {}
    for year in years:
        mask = df.index.year == year
        if mask.sum() < 20:
            continue
        prices = df.loc[mask, "close"]
        ret, mdd = run_backtest(
            prices=prices,
            splits=params["splits"],
            buy_pct=params["buy_pct"],
            sell_pct=params["sell_pct"],
            stop_loss_days=params["stop_loss_days"],
            buy_on_stop=params.get("buy_on_stop", True),
            double_buy_threshold=params.get("double_buy_threshold"),
            surge_sell_threshold=params.get("surge_sell_threshold"),
        )
        annual[year] = {"ret": ret, "mdd": mdd}

    rets = [v["ret"] for v in annual.values()]
    mdds = [v["mdd"] for v in annual.values()]
    if not rets:
        return {"annual": {}, "avg_ret": 0, "avg_mdd": 0, "min_mdd": 0, "std_ret": 0, "sharpe": 0}

    import statistics
    avg_ret = round(sum(rets) / len(rets), 2)
    avg_mdd = round(sum(mdds) / len(mdds), 2)
    min_mdd = round(min(mdds), 2)
    std_ret = round(statistics.stdev(rets) if len(rets) > 1 else 0, 2)
    sharpe  = round(avg_ret / std_ret if std_ret > 0 else 0, 3)
    return {
        "annual":   annual,
        "avg_ret":  avg_ret,
        "avg_mdd":  avg_mdd,
        "min_mdd":  min_mdd,
        "std_ret":  std_ret,
        "sharpe":   sharpe,
    }


def build_candidate_backtests(df: pd.DataFrame) -> dict:
    """
    전략별 후보 파라미터 백테스트.
    목표: 수익률 유지/향상 + MDD 감소를 동시에 달성하는 파레토 개선 탐색.

    후보 설계 원칙:
    - sell_pct 상향: 더 높은 목표가 → 수익↑ 가능, MDD는 보유시간 증가로 소폭 악화 가능
    - buy_pct 강화: 더 큰 하락에만 매수 → 매수 빈도↓, 진입가 유리 → MDD↓ 수익은 유지
    - stop 단축: 손절 빨리 → MDD↓, 수익은 소폭 감소 가능
    - sell상향+buy강화: 수익↑ + 진입 선별 → 파레토 개선 가능성 최대
    - sell상향+stop단축: 수익↑ + 보유기간 제한 → MDD↓
    - 3종 복합: buy강화+sell상향+stop단축 조합
    """
    results = {}
    for name, p in STRATEGIES.items():
        buy  = p["buy_pct"]
        sell = p["sell_pct"]
        stop = p["stop_loss_days"]
        sp   = p["splits"]

        candidates = {
            "현재(기준)":              {"splits": sp, "buy_pct": buy,       "sell_pct": sell,       "stop_loss_days": stop},
            "sell상향(1.5x)":          {"splits": sp, "buy_pct": buy,       "sell_pct": sell*1.5,   "stop_loss_days": stop},
            "sell상향(2x)":            {"splits": sp, "buy_pct": buy,       "sell_pct": sell*2,     "stop_loss_days": stop},
            "buy강화(-2x)":            {"splits": sp, "buy_pct": buy*2,     "sell_pct": sell,       "stop_loss_days": stop},
            "buy강화(-3x)":            {"splits": sp, "buy_pct": buy*3,     "sell_pct": sell,       "stop_loss_days": stop},
            "stop단축(-2일)":          {"splits": sp, "buy_pct": buy,       "sell_pct": sell,       "stop_loss_days": max(5, stop-2)},
            "sell상향+buy강화":        {"splits": sp, "buy_pct": buy*2,     "sell_pct": sell*1.5,   "stop_loss_days": stop},
            "sell상향+stop단축":       {"splits": sp, "buy_pct": buy,       "sell_pct": sell*1.5,   "stop_loss_days": max(5, stop-2)},
            "buy강화+stop단축":        {"splits": sp, "buy_pct": buy*2,     "sell_pct": sell,       "stop_loss_days": max(5, stop-2)},
            "3종복합(수익+MDD최적화)": {"splits": sp, "buy_pct": buy*2,     "sell_pct": sell*1.5,   "stop_loss_days": max(5, stop-2)},
            # ── 새로운 다채로운 전략 ─────────────────────────────
            # 급락 시 추가 1티어 매수: 평소 1개 + 급락일 1개 더
            "급락2티어(-3%)":          {"splits": sp, "buy_pct": buy,       "sell_pct": sell,       "stop_loss_days": stop,           "double_buy_threshold": -0.03},
            "급락2티어(-5%)":          {"splits": sp, "buy_pct": buy,       "sell_pct": sell,       "stop_loss_days": stop,           "double_buy_threshold": -0.05},
            "급락2티어+sell상향":      {"splits": sp, "buy_pct": buy,       "sell_pct": sell*1.5,   "stop_loss_days": stop,           "double_buy_threshold": -0.03},
            # 급등일 수익 포지션 전량 이익실현: 큰 상승 오면 안 놓침
            "급등이익실현(+3%)":       {"splits": sp, "buy_pct": buy,       "sell_pct": sell,       "stop_loss_days": stop,           "surge_sell_threshold": 0.03},
            "급등이익실현(+5%)":       {"splits": sp, "buy_pct": buy,       "sell_pct": sell,       "stop_loss_days": stop,           "surge_sell_threshold": 0.05},
            # 급락매수+급등실현 복합: 낙폭 크면 더 담고, 급등 오면 빠르게 회수
            "급락2티어+급등실현(복합)": {"splits": sp, "buy_pct": buy,      "sell_pct": sell,       "stop_loss_days": stop,           "double_buy_threshold": -0.03, "surge_sell_threshold": 0.03},
        }

        # 현재 기준값 계산
        base_stats = _annual_stats(df, candidates["현재(기준)"])
        base_avg_ret = base_stats["avg_ret"]
        base_min_mdd = base_stats["min_mdd"]

        base_avg_mdd = base_stats["avg_mdd"]

        strat_results = {}
        for label, cparams in candidates.items():
            stats = _annual_stats(df, cparams)
            # MDD 개선 = 절대값이 줄어드는 것 = 음수이므로 숫자가 0에 가까워져야 함
            # 예: -10% > -12% → True → 개선
            ret_ok     = stats["avg_ret"] >= base_avg_ret - 1.0
            avg_mdd_ok = stats["avg_mdd"] > base_avg_mdd   # 평균 MDD 절대값 감소 (덜 음수)
            min_mdd_ok = stats["min_mdd"] > base_min_mdd   # 최악 MDD 절대값 감소 (덜 음수)
            mdd_ok     = avg_mdd_ok and min_mdd_ok          # 둘 다 개선되어야 통과

            if label == "현재(기준)":
                pareto = "기준"
            elif ret_ok and mdd_ok:
                pareto = "✅ 파레토개선"
            elif ret_ok and not mdd_ok:
                avg_tag = "평균MDD악화" if not avg_mdd_ok else "최악MDD악화"
                pareto = f"⚠️ {avg_tag}"
            elif not ret_ok and mdd_ok:
                pareto = "❌ 수익하락"
            else:
                pareto = "❌ 둘다악화"

            strat_results[label] = {
                "params":   {k: round(v, 6) if isinstance(v, float) else v
                             for k, v in cparams.items() if k != "splits"},
                "avg_ret":  stats["avg_ret"],
                "avg_mdd":  stats["avg_mdd"],
                "min_mdd":  stats["min_mdd"],
                "std_ret":  stats["std_ret"],
                "sharpe":   stats["sharpe"],
                "pareto":   pareto,
            }
        results[name] = strat_results
    return results


# ── 백테스트 요약 생성 ─────────────────────────────────────────
def build_backtest_summary(df: pd.DataFrame) -> dict:
    """
    에이전트에게 넘겨줄 백테스트 데이터 패키지 생성
    - 연도별 전략 성과
    - 오차 통계
    - 현재 파라미터
    """
    ind = calc_indicators(df)
    years = sorted(df.index.year.unique())

    annual = {}
    for year in years:
        mask = df.index.year == year
        if mask.sum() < 20:
            continue
        row = {}
        for name, params in STRATEGIES.items():
            prices = df.loc[mask, "close"]
            ret, mdd = run_backtest(
                prices=prices,
                splits=params["splits"],
                buy_pct=params["buy_pct"],
                sell_pct=params["sell_pct"],
                stop_loss_days=params["stop_loss_days"],
                buy_on_stop=params["buy_on_stop"],
            )
            row[name] = {"ret": ret, "mdd": mdd}
        annual[year] = row

    comp_log = load_comparison_log()
    error_stats = calc_error_stats(comp_log)

    # 급락일(-10% 이하) 연도별 카운트
    daily_ret = df["close"].pct_change()
    crash_days_by_year = {}
    for year in years:
        year_ret = daily_ret[df.index.year == year]
        crash_days_by_year[year] = int((year_ret <= -0.10).sum())

    # 현재 계좌 사이클 기간 동안 전략별 시뮬레이션
    scenario_data = {}
    _account_file = "account.json"
    if os.path.exists(_account_file):
        try:
            with open(_account_file, "r", encoding="utf-8") as _f:
                _acct = json.load(_f)
            _cycle_start = pd.Timestamp(_acct.get("cycle_start_date", str(df.index[-1].date())))
            _cycle_df = df[df.index >= _cycle_start]
            if len(_cycle_df) >= 2:
                for _name, _p in STRATEGIES.items():
                    _ret, _mdd = run_backtest(
                        prices=_cycle_df["close"],
                        splits=_p["splits"], buy_pct=_p["buy_pct"],
                        sell_pct=_p["sell_pct"], stop_loss_days=_p["stop_loss_days"],
                        buy_on_stop=_p.get("buy_on_stop", True),
                    )
                    scenario_data[_name] = {
                        "ret":          _ret,
                        "mdd":          _mdd,
                        "days":         len(_cycle_df),
                        "cycle_start":  str(_cycle_start.date()),
                        "active":       _name == _acct.get("strategy", "Pro3"),
                    }
        except Exception:
            pass

    return {
        "current_params": {
            name: {
                "splits":          params["splits"],
                "stop_loss_days":  params["stop_loss_days"],
                "buy_pct":         params["buy_pct"],
                "sell_pct":        params["sell_pct"],
                "crash_buy":       params.get("crash_buy", {}),
            }
            for name, params in STRATEGIES.items()
        },
        "annual_backtest":      annual,
        "crash_days_by_year":   crash_days_by_year,
        "scenario_comparison":  scenario_data,
        "eval_window":          EVAL_WINDOW,
        "error_stats":          error_stats,
        "comparison_count":     len(comp_log),
        "data_range": {
            "start": str(df.index[0].date()),
            "end":   str(df.index[-1].date()),
        },
    }


# ── 시장 뉴스 수집 ────────────────────────────────────────────
def fetch_market_context() -> str:
    """Yahoo Finance RSS로 SOXL/반도체 관련 최신 뉴스 수집"""
    import urllib.request
    import xml.etree.ElementTree as ET

    feeds = [
        ("SOXL", "https://finance.yahoo.com/rss/headline?s=SOXL"),
        ("SMH",  "https://finance.yahoo.com/rss/headline?s=SMH"),
        ("NVDA", "https://finance.yahoo.com/rss/headline?s=NVDA"),
    ]
    headlines = []
    for label, url in feeds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                root = ET.fromstring(resp.read())
            for item in root.findall(".//item")[:3]:
                title = item.findtext("title", "").strip()
                pub   = item.findtext("pubDate", "")[:16].strip()
                if title:
                    headlines.append(f"[{label}] {title} ({pub})")
        except Exception:
            pass

    if not headlines:
        return "⚠️ 시장 뉴스 조회 실패 (인터넷 연결 또는 Yahoo Finance 서비스 불가)"
    return "\n".join(headlines)


# ── 급락 추가매수 섹션 빌더 ──────────────────────────────────
def _build_crash_buy_section(summary: dict) -> str:
    """급락 관련 데이터를 에이전트 메시지에 추가할 텍스트로 변환"""
    crash_days = summary.get("crash_days_by_year", {})
    if not crash_days:
        return ""

    lines = ["\n## 연도별 -10% 이상 급락일 수"]
    for year, count in sorted(crash_days.items()):
        lines.append(f"- {year}: {count}일")

    # 현재 crash_buy 활성화 여부 표시
    enabled_strategies = [
        name for name, params in summary["current_params"].items()
        if params.get("crash_buy", {}).get("enabled", False)
    ]
    if enabled_strategies:
        lines.append(f"\n## 급락 추가매수 활성화 전략: {', '.join(enabled_strategies)}")
        for name in enabled_strategies:
            cb = summary["current_params"][name]["crash_buy"]
            lines.append(f"- {name}: threshold={cb['threshold']*100:.0f}%, alloc={cb['alloc']*100:.0f}%, "
                         f"sell={cb['sell_pct']*100:.1f}%, stop={cb['stop_loss_days']}일, "
                         f"max={cb['max_concurrent']}개")
    else:
        lines.append("\n## 급락 추가매수: 현재 모든 전략에서 비활성화 상태")

    return "\n".join(lines) + "\n"


def _build_scenario_section(summary: dict) -> str:
    """현재 사이클 시나리오 비교 데이터를 에이전트 메시지에 삽입할 텍스트로 변환"""
    sc = summary.get("scenario_comparison", {})
    if not sc:
        return ""
    lines = ["\n## 현재 사이클 시나리오 비교 (사이클 시작~현재 각 전략 시뮬레이션)"]
    for name, d in sc.items():
        active_tag = " ← **현재 운용 중**" if d.get("active") else ""
        lines.append(
            f"- {name}{active_tag}: 사이클수익 {d['ret']:+.2f}%, "
            f"사이클MDD {d['mdd']:.2f}% "
            f"({d['days']}영업일, {d['cycle_start']}~)"
        )
    # 최고 성과 전략 찾기
    best = max(sc.items(), key=lambda x: x[1]["ret"])
    active_name = next((n for n, d in sc.items() if d.get("active")), None)
    if active_name and best[0] != active_name:
        diff = best[1]["ret"] - sc[active_name]["ret"]
        lines.append(
            f"\n💡 이 기간 최고 성과: **{best[0]}** ({best[1]['ret']:+.2f}%) "
            f"— 현재 운용 중인 {active_name} 대비 **{diff:+.2f}%p** 차이. "
            f"다음 사이클 전략 전환을 검토하세요."
        )
    return "\n".join(lines) + "\n"


# ── 에이전트 프롬프트 구성 ────────────────────────────────────
SYSTEM_PROMPT = """당신은 세계 최고 수준의 퀀트 트레이딩 전략가입니다. 헤지펀드에서 15년간 알고리즘 전략을 운용했으며, 데이터로 증명되지 않는 주장은 절대 하지 않습니다. 매크로 환경 분석, 기술적 분석, 퀀트 백테스트를 결합한 복합 판단이 당신의 강점입니다.

당신의 임무: SOXL 떨사오팔 전략의 파라미터를 개선해서 **수익률은 유지하거나 높이면서 MDD는 줄이는** 결과를 만들어내는 것입니다. 단순 파라미터 조정을 넘어, 시장 환경에 맞는 다채로운 전략 변형을 과감하게 제안하세요.

## 절대 원칙

> **수익률 희생은 실패다.** 목표는 "수익률 ≥ 현재, MDD 절대값 < 현재"를 동시에 달성하는 파레토 개선입니다.
> 수익률이 소폭(1%p 이내) 감소하면서 MDD 절대값이 **3%p 이상** 줄어드는 경우만 예외적으로 허용하되, 반드시 명시하세요.

## MDD 방향 정의 (절대 혼동 금지)

**MDD는 항상 음수입니다.** 개선 = 숫자가 0에 가까워지는 것.
- ✅ 개선: -12% → -10% (절대값 감소)
- ❌ 악화: -12% → -15% (절대값 증가)
- -10%는 -12%보다 **좋은** MDD입니다 (|-10%| < |-12%|)

## 전략 구조

'떨사오팔'은 주가 하락 시 매수, 상승 시 매도하는 역추세 전략:
- Pro1 (보수형): 피라미딩 splits [5/10/15/20/25/25%], 빠른 목표가
- Pro2 (균형형): 중앙집중 splits [10/15/20/25/20/10%], 중간 목표가
- Pro3 (공격형): 균등 splits [1/6×6], 높은 목표가

핵심 파라미터:
- buy_pct: 전일 종가 대비 하락률 트리거 (음수, LOC)
- sell_pct: 매수가 대비 익절 목표 (양수)
- stop_loss_days: 최대 보유 영업일 (초과 시 강제 손절)
- LOC 주문: 하루 최대 1티어 매수 (원칙), 가장 낮은 번호 빈 슬롯 사용
- 단리 적용: 슬롯 규모는 초기 시드 기준 고정

## 추가 전략 파라미터 (적극 활용하세요)

### double_buy_threshold (급락 2티어 매수)
- 일간 수익률 ≤ 이 값이면 당일 티어를 1개 추가 매수 (총 2티어/일)
- 예: -0.03 → -3% 이상 급락일에 2티어 매수
- 탐색 범위: -0.03 ~ -0.07
- 시장 하락 리스크가 클 때는 -0.05 이하로 보수적으로 설정

### surge_sell_threshold (급등 전량 이익실현)
- 일간 수익률 ≥ 이 값이면 수익 중인 모든 슬롯 즉시 청산
- 예: 0.03 → +3% 이상 급등일에 수익 포지션 전량 매도
- 탐색 범위: +0.02 ~ +0.06
- 상승 모멘텀 강할 때는 0.04~0.06으로 높게, 약할 때는 0.02~0.03으로 낮게

## 시장 뉴스 및 거시환경 분석 의무사항

제공된 뉴스를 반드시 전략 제안에 반영하세요:
- 반도체 하락 리스크(관세, 규제, 매크로 충격): double_buy_threshold를 -0.05 이하로 보수적으로, surge_sell을 낮게(+0.02~0.03) 설정해 빠른 회전
- 상승 모멘텀(AI 붐, 실적 서프라이즈): buy_pct를 예민하게 조정(-0.001~-0.002), surge_sell 기준을 높게(+0.04~0.06)
- 변동성 확대: stop_loss_days 단축(7~8일), sell_pct 낮춰 빠른 회전
- 명확한 신호 없으면 "뉴스 중립, 기술적 분석만으로 판단"이라고 명시

## 현재 사이클 성과 분석 의무사항

사이클 시나리오 데이터가 제공되면 반드시:
1. 현재 운용 중인 전략이 동 기간 다른 전략 대비 어떻게 성과를 냈는지 수치로 비교
2. 다음 사이클 전략 전환이 필요한지 데이터로 판단
3. "이번 사이클 Pro3는 Pro2 대비 X%p 수익 차이. 현재 시장 환경 고려 시 다음 사이클은 Pro2 권고" 형태로 구체적으로 명시

## 파레토 개선 탐색 방법론

1. **1단계**: 평균수익률 ≥ 현재 (또는 -1%p 이내 예외)
2. **2단계**: 평균MDD > 현재 AND 최악MDD > 현재 (둘 다 0에 가까워져야 함)
3. **3단계**: 1·2 동시 만족 후보 중 샤프비율 최고 선택
4. 파레토 개선 후보 없으면 "현재 파라미터 유지 권고" — 억지로 제안 금지

## 다채로운 전략 제안 원칙

단순 파라미터 미세조정을 넘어 아래를 적극 검토하세요:
- double_buy_threshold + surge_sell_threshold 복합 전략
- splits 구조 변경 (역피라미딩 [0.25,0.25,0.20,0.15,0.10,0.05] 등)
- 각 제안에 "이 전략이 유리한 시장 환경: ..." 명시
- 현재 뉴스 환경과 제안 전략의 정합성 설명

## 분석 의무사항

### 1. 시장 상황 진단 (최우선, 맨 앞에)
뉴스 기반으로 SOXL/반도체 방향성 리스크를 1-3문장으로 진단하고, 이것이 전략 선택에 어떤 영향을 주는지 명시.

### 2. 데이터 기반 주장만
"~할 것 같습니다" 불허. "백테스트 결과 수익 X%→Y%, MDD X%→Y%로 파레토 개선"처럼 수치로 증명.

### 3. 후보 필터링 결과 공개
각 전략별 파레토 통과/탈락 후보를 명시. 탈락 이유 구체적으로.

### 4. 최종 추천 JSON 필수
분석 맨 마지막에 반드시 JSON 블록 출력. 없으면 분석 무효.
파레토 개선 없는 전략은 현재 파라미터 그대로. double_buy_threshold / surge_sell_threshold는 권장할 때만 포함.

```json
{
  "Pro1": {"buy_pct": -0.00X, "sell_pct": 0.00X, "stop_loss_days": X},
  "Pro2": {"buy_pct": -0.00X, "sell_pct": 0.0XX, "stop_loss_days": X, "double_buy_threshold": -0.03},
  "Pro3": {"buy_pct": -0.00X, "sell_pct": 0.0XX, "stop_loss_days": X, "surge_sell_threshold": 0.03}
}
```

### 5. 투자자 설득
수치로 증명해서 확신하고 파라미터를 바꿀 수 있게 설득. 과적합 경고는 한 문장만.

6. 한국어로 답변하세요"""


def run_agent_analysis(df: pd.DataFrame, api_key: str) -> Iterator[str]:
    """
    Claude API를 통해 전략 분석 수행.
    - 후보 파라미터 백테스트를 미리 계산해서 Claude에게 넘김
    - Claude는 숫자를 보고 최선의 파라미터를 선택 + 반드시 JSON 출력
    """
    # 1. 현재 전략 요약
    summary = build_backtest_summary(df)
    crash_buy_section = _build_crash_buy_section(summary)

    # 2. 시장 뉴스 수집
    yield "⏳ 최신 시장 뉴스 수집 중...\n\n"
    market_news = fetch_market_context()

    # 3. 후보 파라미터 백테스트 (핵심: Claude가 숫자를 보고 판단)
    yield "⏳ 후보 파라미터 백테스트 계산 중 (17가지 × 3전략)...\n\n"
    candidates = build_candidate_backtests(df)

    # 후보 요약 테이블 생성 (Claude에게 전달할 텍스트)
    candidate_summary_lines = []
    for name, cands in candidates.items():
        candidate_summary_lines.append(f"\n### {name} 후보 비교표 (판정 기준: 수익≥현재-1%p AND 최악MDD개선)")
        candidate_summary_lines.append("| 파레토판정 | 후보 | buy_pct | sell_pct | stop일 | double_buy | surge_sell | 평균수익% | 평균MDD% | 최악MDD% | 샤프비율 |")
        candidate_summary_lines.append("|-----------|------|---------|----------|--------|-----------|-----------|----------|---------|---------|---------|")
        for label, r in cands.items():
            p = r["params"]
            dbt = f"{p.get('double_buy_threshold','') * 100:.0f}%" if p.get('double_buy_threshold') is not None else "-"
            sst = f"+{p.get('surge_sell_threshold','') * 100:.0f}%" if p.get('surge_sell_threshold') is not None else "-"
            candidate_summary_lines.append(
                f"| {r.get('pareto','?')} | {label} | {p['buy_pct']*100:.3f}% | {p['sell_pct']*100:.3f}% | "
                f"{p['stop_loss_days']}일 | {dbt} | {sst} | {r['avg_ret']:+.2f}% | {r['avg_mdd']:.2f}% | "
                f"{r['min_mdd']:.2f}% | {r['sharpe']:.3f} |"
            )

    candidate_text = "\n".join(candidate_summary_lines)

    user_message = f"""아래는 SOXL 떨사오팔 전략의 완전한 분석 데이터입니다. 당신은 이 숫자를 직접 분석해 최적 파라미터를 결정해야 합니다.

## ⚠️ 운용자 지시사항 (최우선 원칙)
이전 파라미터 변경 시도에서 다음과 같은 결과가 나왔습니다:
- Pro1: 수익 +22.97% → +20.04% (**수익 -2.94%p 하락**), 평균MDD -12.15% → -12.41% (**MDD 절대값 증가 = 악화**)
- Pro2: 수익 +31.74% → +30.26% (**수익 -1.48%p 하락**), MDD 소폭 개선
- Pro3: 수익 +34.86% → +33.75% (**수익 -1.11%p 하락**), MDD 소폭 개선

**이 결과는 실패입니다.** 이유:
1. 수익률이 모든 전략에서 하락했습니다
2. Pro1은 MDD 절대값이 오히려 커졌습니다 (-12.15% → -12.41%, 즉 손실이 더 커진 것)

**MDD 개선의 정의**: MDD 절대값이 줄어드는 것 = 숫자가 0에 가까워지는 것
- ✅ 개선: -12.15% → -10.50% (절대값 감소)
- ❌ 악화: -12.15% → -12.41% (절대값 증가)

당신의 임무는 **수익률은 유지하거나 높이면서, MDD 절대값도 줄이는** 파레토 개선을 찾는 것입니다.

## 현재 파라미터
```json
{json.dumps({n: {k: v for k, v in p.items() if k != 'crash_buy'} for n, p in summary['current_params'].items()}, ensure_ascii=False, indent=2)}
```

## 현재 파라미터 기준 연도별 백테스트 성과
```json
{json.dumps(summary['annual_backtest'], ensure_ascii=False, indent=2)}
```

## 오차 통계 (사이트 vs 백테스트)
- 비교 데이터: {summary['comparison_count']}건 | 전체 RMSE: {summary['error_stats'].get('overall_rmse', 'N/A')}%p
{crash_buy_section}
## 데이터 범위: {summary['data_range']['start']} ~ {summary['data_range']['end']}

## 최신 시장 뉴스 (오늘 기준)
{market_news}
**위 뉴스를 참고해서 현재 반도체/SOXL 시장의 방향성 리스크를 판단하고, 전략 제안에 반영하세요.**
(예: 관세 리스크가 크면 급락 2티어 매수를 보수적으로 설정, 상승 모멘텀이면 surge_sell 기준을 높이는 방향)

{_build_scenario_section(summary)}

---

## 후보 파라미터 백테스트 결과 (파이썬이 직접 계산한 수치)

{candidate_text}

---

## 분석 요청

아래 순서로 분석하세요:

### 0. 시장 상황 진단 (최우선)
- 제공된 뉴스를 기반으로 현재 SOXL/반도체 방향성 리스크를 진단
- 이것이 전략 선택에 어떤 영향을 주는지 명시 (강세/약세/중립)

### 1. 현재 사이클 복기 (사이클 데이터 제공 시)
- 현재 운용 중인 전략이 동 기간 시뮬레이션 대비 우위/열위인지
- 다음 사이클 전략 전환 권고 (데이터 기반)

### 2. 현재 전략 성과 진단
- 연도별 데이터 기반 각 전략의 강약점 수치로 설명
- MDD가 특히 큰 연도와 그 원인

### 3. 파레토 개선 후보 분석
- 각 전략별로 "✅ 파레토개선" 판정 후보만 집중 분석
- 수익률 유지/개선 + MDD 감소 폭을 현재 대비 수치로 명시
- "❌ 수익하락" 후보는 채택 불가 명시 후 스킵

### 4. 전략별 최종 추천
- 파레토 개선 후보 중 샤프비율 최고 선정
- 개선 폭: "수익 X%→Y%, 최악MDD X%→Y%, 샤프 X→Y"로 표기
- 파레토 개선 없으면 "현재 유지 권고"
- double_buy_threshold / surge_sell_threshold 활용 후보가 우수하면 적극 추천

### 5. 리스크 & 과적합 경고
- 핵심 리스크 1~2개만 (간결하게)

### 5. 최종 추천 파라미터 (JSON 필수)
반드시 아래 형식 그대로 출력하세요.
- double_buy_threshold, surge_sell_threshold는 해당 기능을 추천할 때만 포함 (불필요하면 생략)
- 파레토 개선 후보 없는 전략은 현재 파라미터 그대로

```json
{{
  "Pro1": {{
    "buy_pct": -0.001,
    "sell_pct": 0.002,
    "stop_loss_days": 10
  }},
  "Pro2": {{
    "buy_pct": -0.002,
    "sell_pct": 0.018,
    "stop_loss_days": 10,
    "double_buy_threshold": -0.03
  }},
  "Pro3": {{
    "buy_pct": -0.003,
    "sell_pct": 0.025,
    "stop_loss_days": 10,
    "surge_sell_threshold": 0.03
  }}
}}
```"""

    client = anthropic.Anthropic(api_key=api_key)

    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def extract_param_suggestions(text: str) -> list[dict]:
    """
    에이전트 분석 텍스트에서 JSON 파라미터 블록을 모두 추출.
    Pro1/Pro2/Pro3 키를 가진 dict만 유효한 제안으로 반환.
    Returns: [{"label": str, "params": {Pro1:{...}, Pro2:{...}, Pro3:{...}}}, ...]
    """
    import re

    suggestions = []
    # ```json ... ``` 블록 추출
    blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    # 코드블록 없이 중괄호만 있는 경우도 처리
    blocks += re.findall(r'(?<!\w)(\{[^{}]*"Pro[123]"[^{}]*\{.*?\}.*?\})', text, re.DOTALL)

    seen = set()
    for raw in blocks:
        try:
            obj = json.loads(raw)
        except Exception:
            # 작은따옴표 등 허용
            try:
                import ast
                obj = ast.literal_eval(raw)
            except Exception:
                continue

        if not isinstance(obj, dict):
            continue
        # Pro1/Pro2/Pro3 를 키로 갖는 블록인지 확인
        if not any(k in obj for k in ["Pro1", "Pro2", "Pro3"]):
            continue

        # 각 전략에 필요한 키가 있는지 확인
        valid = {}
        for name in ["Pro1", "Pro2", "Pro3"]:
            p = obj.get(name, {})
            if isinstance(p, dict) and any(k in p for k in ["buy_pct", "sell_pct", "stop_loss_days", "splits",
                                                              "double_buy_threshold", "surge_sell_threshold"]):
                valid[name] = p

        if not valid:
            continue

        key = json.dumps(valid, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)

        # 라벨: 블록 앞 50자 컨텍스트에서 추출
        idx = text.find(raw[:30])
        ctx = text[max(0, idx-80):idx].strip().split("\n")
        label = next((l.strip("# ").strip() for l in reversed(ctx) if l.strip()), f"제안 {len(suggestions)+1}")
        label = label[:40]

        suggestions.append({"label": label, "params": valid})

    return suggestions


def run_error_minimization(df: pd.DataFrame, api_key: str) -> Iterator[str]:
    """
    오차 최소화에 특화된 에이전트 분석.
    비교 로그의 전략별 오차 패턴을 분석하고
    buy_pct / sell_pct / stop_loss_days 조정 방향을 구체적으로 제안.
    """
    comp_log = load_comparison_log()
    error_stats = calc_error_stats(comp_log)

    # 구간별 오차 상세 (최근 20건)
    recent_errors = []
    for entry in comp_log[-20:]:
        for period in entry.get("구간별비교", []):
            row = {
                "기준일":   entry["기준일"],
                "구간":     period.get("구간", ""),
                "유사도":   period.get("유사도", 0),
            }
            for name in ["Pro1", "Pro2", "Pro3"]:
                s = period.get("사이트", {}).get(name, {})
                o = period.get("우리", {}).get(name, {})
                if s.get("ret") is not None and o.get("ret") is not None:
                    row[f"{name}_사이트"] = s["ret"]
                    row[f"{name}_우리"]   = o["ret"]
                    row[f"{name}_오차"]   = round(o["ret"] - s["ret"], 2)
            recent_errors.append(row)

    system = """당신은 알고리즘 트레이딩 백테스트 오차 분석 전문가입니다.
주어진 오차 데이터를 보고 우리 백테스트 엔진의 파라미터를 어떻게 조정하면
사이트 수치와의 오차(RMSE)를 줄일 수 있는지 구체적으로 분석합니다.

분석 규칙:
1. 오차의 방향(bias)이 일관되면 체계적 원인이 있음 - 이를 먼저 파악하세요
2. 오차가 특정 전략에 집중되는지 확인하세요
3. 파라미터 조정 제안은 반드시 현재값 → 제안값 형식으로 JSON 포함
4. EVAL_WINDOW가 아닌 전략 파라미터(buy_pct, sell_pct, stop_loss_days)도 조정 대상
5. 한국어로 답변하세요"""

    user_msg = f"""우리 백테스트 vs 사이트 수치 오차를 최소화하기 위한 분석입니다.

## 현재 파라미터
```json
{json.dumps({n: {k: v for k, v in p.items() if k != 'crash_buy'} for n, p in STRATEGIES.items()}, ensure_ascii=False, indent=2)}
```

## 전략별 오차 통계 ({error_stats.get('total_records', 0)}건 기반)
```json
{json.dumps(error_stats.get('by_strategy', {}), ensure_ascii=False, indent=2)}
```
- 전체 RMSE: {error_stats.get('overall_rmse', 'N/A')}%p
- 전체 평균 절대오차: {error_stats.get('overall_mean_abs', 'N/A')}%p

## 최근 20건 구간별 오차 상세
```json
{json.dumps(recent_errors, ensure_ascii=False, indent=2)}
```

## EVAL_WINDOW: {EVAL_WINDOW}영업일

---

다음을 분석해주세요:

1. **오차 패턴 분석**: 어떤 전략에서 오차가 크고, 오차가 한쪽 방향으로 치우쳐 있는지
2. **원인 가설**: 오차가 발생하는 구조적 원인 (예: 손절 처리 방식, 매수 타이밍 차이 등)
3. **파라미터 조정 제안**: 오차를 줄이기 위한 구체적인 파라미터 변경안
   - 형식: 현재값 → 제안값, 이유 포함
   - JSON으로도 제공 (config.py에 바로 적용 가능한 형태)
4. **우선순위**: 어떤 전략부터 조정하면 전체 RMSE 감소 효과가 가장 클지"""

    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def run_and_save(df: pd.DataFrame, api_key: str) -> dict:
    """에이전트 실행 후 전체 텍스트 반환 + 로그 저장 (저장 전용)"""
    summary = build_backtest_summary(df)
    full_text = ""
    for chunk in run_agent_analysis(df, api_key):
        full_text += chunk

    entry = {
        "실행시각": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "분석내용": full_text,
        "당시요약": {
            "비교데이터수":    summary["comparison_count"],
            "전체RMSE":       summary["error_stats"].get("overall_rmse"),
            "데이터범위종료": summary["data_range"]["end"],
        },
    }
    log = load_agent_log()
    log.append(entry)
    save_agent_log(log)
    return entry
