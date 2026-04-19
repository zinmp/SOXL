# [MOBILE] 모바일 최적화 버전
"""
떨사오팔 Pro 레이더 - Streamlit GUI (모바일/아이폰 최적화)
실행: streamlit run app_mobile.py

[모바일 개선 사항]
  - layout="centered" 적용
  - 컬럼 수 축소 (5컬럼→2+3, 6컬럼→3+3, 4컬럼→2+2)
  - 탭 이름 축약 (아이폰 화면 잘림 방지)
  - expander 적극 활용 (초기 화면 최소화)
  - 리스크 경고 → expander로 접기
  - 주문표 핵심 컬럼만 기본 표시
  - 모든 버튼 use_container_width=True
"""
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from backtest import run_backtest
from config import STRATEGIES, EVAL_WINDOW, BASELINE_STRATEGIES, TRANSACTION_COST, PRICE_STOP_LOSS_PCT
from data import get_data
from indicators import calc_indicators, FEATURE_COLS
from recommender import recommend
from parser import parse_site_text
from tuner import calc_error_stats, load_log, load_tuning_log, run_tuning
from agent import (
    get_api_key, save_api_key, load_agent_log,
    run_agent_analysis, run_error_minimization, should_run_weekly,
    extract_param_suggestions,
)
from settings import load_params, save_params, update_strategies
from storage import load_account, save_account

_BASE = Path(__file__).parent
COMPARISON_FILE = str(_BASE / "comparison_log.json")
THRESHOLD_DIFF  = 2.0


# ══════════════════════════════════════════════════════════════
# 전략 추적 유틸
# ══════════════════════════════════════════════════════════════
def _update_strategy_tracking(acct: dict, rec: str) -> dict:
    today = date.today().isoformat()
    current = acct.get("strategy", "Pro3")
    if rec == current:
        if acct.get("strategy_mismatch_since") is not None:
            acct["strategy_mismatch_since"] = None
            acct["filter_streak"] = 0
        return acct
    if acct.get("strategy_mismatch_since") is None:
        acct["strategy_mismatch_since"] = today
        acct["filter_streak"] = 1
    else:
        last_date = acct.get("_last_check_date")
        if last_date != today:
            acct["filter_streak"] = acct.get("filter_streak", 0) + 1
    acct["_last_check_date"] = today
    return acct


def _get_transition_threshold(current: str, rec: str) -> int:
    rank = {"Pro1": 1, "Pro2": 2, "Pro3": 3}
    diff = rank.get(rec, 2) - rank.get(current, 2)
    return 5 if diff > 0 else (3 if diff < 0 else 2)


# ══════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="SOXL 데이터 로딩 중...")
def load_data_cached():
    df = get_data()
    ind = calc_indicators(df)
    return df, ind


def load_comparison_log() -> list:
    if not os.path.exists(COMPARISON_FILE):
        return []
    with open(COMPARISON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_comparison_log(log: list):
    with open(COMPARISON_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)


def color_diff(diff: float) -> str:
    if abs(diff) >= THRESHOLD_DIFF:
        return f"🔴 {diff:+.2f}%p"
    return f"🟢 {diff:+.2f}%p"


# ══════════════════════════════════════════════════════════════
# 앱 설정 — 모바일: centered 레이아웃
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="떨사오팔 Pro",
    page_icon="📡",
    layout="centered",          # [MOBILE] wide → centered
    initial_sidebar_state="collapsed",
)

# ── 리스크 경고 → expander로 접기 (매번 전면 노출 제거) ─────────
with st.expander("⚠️ 투자 위험 고지 (클릭하여 확인)"):
    st.warning(
        "⚠️ **SOXL은 3배 레버리지 ETF**입니다. 2022년 최대 낙폭 약 -90%. "
        "이 앱은 분석 도구이며 투자 권유가 아닙니다. 모든 투자 판단과 손실은 본인 책임입니다.",
        icon="⚠️",
    )

# ── 탭 정의 — 모바일: 이름 축약 ──────────────────────────────────
tab_today, tab_account, tab_backtest, tab_agent, tab_input, tab_error = st.tabs([
    "📡 오늘",      # 오늘 추천
    "💼 계좌",      # 내 계좌
    "📊 통계",      # 백테스트 통계
    "🤖 AI",        # 금융 에이전트
    "📥 입력",      # 데이터 입력
    "🔍 오차",      # 오차 & 조정
])

# 기존 코드 변수명 호환
tab1 = tab_today
tab2 = tab_input
tab3 = tab_error
tab4 = tab_backtest
tab5 = tab_agent
tab6 = tab_account


# ══════════════════════════════════════════════════════════════
# TAB 1 : 오늘 추천 (모바일 최적화)
# ══════════════════════════════════════════════════════════════
with tab1:
    st.title("📡 오늘 추천")
    st.caption(
        "데이터: Yahoo Finance — SOXL 일봉 종가 자동 다운로드. "
        "장 마감 후 약 15~30분 뒤 반영."
    )

    # [MOBILE] 날짜 + 갱신 버튼을 2컬럼으로
    col_date, col_btn = st.columns([3, 1])
    with col_date:
        target_date_input = st.date_input("기준일", value=date.today(), max_value=date.today())
    with col_btn:
        st.write("")
        if st.button("🔄 갱신", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    try:
        df, ind = load_data_cached()

        target_ts   = pd.Timestamp(target_date_input)
        valid_dates = ind[FEATURE_COLS].dropna().index
        if target_ts not in valid_dates:
            target_ts = valid_dates[valid_dates <= target_ts][-1]
            st.info(f"영업일 기준 {target_ts.date()} 사용")

        with st.spinner("계산 중..."):
            result = recommend(
                df, target_date=target_ts, verbose=False,
                transaction_cost=TRANSACTION_COST,
                price_stop_loss_pct=PRICE_STOP_LOSS_PCT,
            )

        ind_data = result["indicators"]
        rec      = result["recommended"]
        params   = result["params"]
        scores   = result["scores"]
        similar  = result["similar"]
        uptrend  = ind_data["uptrend"]

        target_idx = df.index.get_loc(target_ts)
        if target_idx == 0:
            st.error("오류: 전일 종가 데이터 없음")
            st.stop()

        prev_close = float(df["close"].iloc[target_idx - 1])
        buy_limit  = round(prev_close * (1 + params["buy_pct"]), 4)
        sell_tgt   = round(buy_limit  * (1 + params["sell_pct"]), 4)

        bdays     = pd.bdate_range(start=target_ts + pd.Timedelta(days=1), periods=params["stop_loss_days"])
        stop_date = bdays[-1].date() if len(bdays) > 0 else "-"

        _acct     = load_account()
        seed      = float(_acct.get("seed", 10000))
        _filled   = len([h for h in _acct.get("holdings", []) if h.get("active", True)])
        today_ret = float(df["close"].iloc[target_idx]) / prev_close - 1

        # ──────────────────────────────────────────────────────
        # [0] 전략 불일치 배너 (매수 신호보다 위에 표시)
        # ──────────────────────────────────────────────────────
        recommended_strategy = rec
        _acct = _update_strategy_tracking(_acct, recommended_strategy)
        save_account(_acct)

        mismatch_since = _acct.get("strategy_mismatch_since")
        streak = _acct.get("filter_streak", 0)
        switch_date = _acct.get("strategy_switch_date")

        in_lockout = False
        if switch_date:
            from pandas.tseries.offsets import BDay
            switch_dt = pd.Timestamp(switch_date)
            lockout_end = switch_dt + 5 * BDay()
            in_lockout = pd.Timestamp.today() < lockout_end

        if mismatch_since and not in_lockout:
            threshold = _get_transition_threshold(_acct.get("strategy", "Pro3"), recommended_strategy)
            days_left = max(0, threshold - streak)

            if days_left == 0:
                st.error(f"🔄 전략 전환 권고: {_acct.get('strategy')} → {recommended_strategy} | {streak}일 연속 확인 완료")
            else:
                st.warning(f"⚠️ {streak}일째 불일치 | {_acct.get('strategy')} → {recommended_strategy} | {days_left}일 더 필요")

            with st.expander("📋 전환 가이드 보기"):
                st.markdown(f"**현재:** {_acct.get('strategy')}  →  **추천:** {recommended_strategy}")
                st.markdown("---")
                st.markdown("**A. 자연 만료 (권장)**")
                st.markdown("신규 매수 중단 → 기존 포지션 자연 청산 → 새 사이클")
                st.markdown("---")
                st.markdown("**B. 즉시 전환**")
                st.markdown("수익 포지션 우선 청산 → 전량 청산 → 새 사이클")

        # ──────────────────────────────────────────────────────
        # [1] 히어로 섹션: 매수 신호 크게 표시
        # ──────────────────────────────────────────────────────
        st.subheader(f"📌 기준일: {result['date']}")

        buy_signal_possible = _filled < len(params["splits"])
        buy_condition_met   = today_ret <= params["buy_pct"]

        # [MOBILE] 매수 신호를 큰 마크다운 텍스트로 강조
        if not buy_signal_possible:
            st.markdown(
                "<div style='background:#ffebee;border-left:5px solid #e53935;"
                "padding:16px;border-radius:8px;font-size:1.3em;font-weight:bold'>"
                "🔒 모든 슬롯 채워짐 — 오늘 매수 없음</div>",
                unsafe_allow_html=True,
            )
        elif buy_condition_met:
            st.markdown(
                f"<div style='background:#e8f5e9;border-left:5px solid #43a047;"
                f"padding:16px;border-radius:8px;font-size:1.3em;font-weight:bold'>"
                f"🟢 매수 신호 있음!<br>"
                f"<span style='font-size:0.8em;font-weight:normal'>"
                f"오늘 수익률 {today_ret*100:+.2f}% ≤ 기준 {params['buy_pct']*100:+.2f}%"
                f"</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#e3f2fd;border-left:5px solid #1e88e5;"
                f"padding:16px;border-radius:8px;font-size:1.2em;font-weight:bold'>"
                f"🔵 매수 조건 미충족<br>"
                f"<span style='font-size:0.8em;font-weight:normal'>"
                f"수익률 {today_ret*100:+.2f}% > 기준 {params['buy_pct']*100:+.2f}%"
                f"</span></div>",
                unsafe_allow_html=True,
            )

        st.write("")

        # [MOBILE] 5컬럼 → 2+3 배치
        hrow1_c1, hrow1_c2 = st.columns(2)
        hrow1_c1.metric("🎯 추천 전략", rec)
        hrow1_c2.metric("💰 시드", f"${seed:,.0f}")

        hrow2_c1, hrow2_c2, hrow2_c3 = st.columns(3)
        hrow2_c1.metric("전일 종가", f"${prev_close:.4f}")
        hrow2_c2.metric("LOC 한도", f"${buy_limit:.4f}")
        hrow2_c3.metric("매도 목표", f"${sell_tgt:.4f}")

        # ──────────────────────────────────────────────────────
        # [2] 오늘 주문표 — 모바일: 핵심 3컬럼만 기본 표시
        # ──────────────────────────────────────────────────────
        if buy_signal_possible:
            st.subheader("📋 오늘 주문 (LOC)")
            _next_alloc = params["splits"][_filled]

            # [MOBILE] 핵심 컬럼만 기본 노출 (티어/금액/LOC한도)
            single_row_mobile = [{
                "티어":     _filled + 1,
                "금액":     f"${seed * _next_alloc:,.0f}",
                "LOC 한도": f"${buy_limit:.4f}",
            }]
            st.dataframe(pd.DataFrame(single_row_mobile), use_container_width=True, hide_index=True)

            # 상세 정보는 expander로
            with st.expander("🔎 주문 상세 보기"):
                single_row_full = [{
                    "티어":     _filled + 1,
                    "비율":     f"{_next_alloc*100:.1f}%",
                    "금액":     f"${seed * _next_alloc:,.0f}",
                    "LOC 한도": f"${buy_limit:.4f}",
                    "매도 목표": f"${sell_tgt:.4f}",
                    "손절 날짜": str(stop_date),
                }]
                st.dataframe(pd.DataFrame(single_row_full), use_container_width=True, hide_index=True)

            st.caption(
                f"LOC 최대 1티어/일 — 현재 {_filled}개 보유, 다음 진입: {_filled+1}번 슬롯 "
                f"| 시드 변경은 **계좌** 탭에서"
            )

        with st.expander("📋 전체 슬롯 구조 보기"):
            rows = []
            for i, alloc in enumerate(params["splits"], 1):
                rows.append({
                    "티어":     i,
                    "비율":     f"{alloc*100:.1f}%",
                    "금액":     f"${seed * alloc:,.0f}",
                    "LOC 한도": f"${buy_limit:.4f}",
                    "매도 목표": f"${sell_tgt:.4f}",
                    "손절 날짜": str(stop_date),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # ──────────────────────────────────────────────────────
        # [3] 추천 전략 파라미터 (접이식)
        # ──────────────────────────────────────────────────────
        with st.expander(f"⚙️ 추천 전략 파라미터: {rec}"):
            # [MOBILE] 4컬럼 → 2+2
            pm1, pm2 = st.columns(2)
            pm1.metric("손절일(시간)",   f"{params['stop_loss_days']}영업일")
            pm2.metric("손절(가격)",     "비활성")
            pm3, pm4 = st.columns(2)
            pm3.metric("매수 기준",      f"{params['buy_pct']*100:+.3f}%")
            pm4.metric("매도 기준",      f"{params['sell_pct']*100:+.3f}%")
            splits_str = "  /  ".join(f"{x*100:.1f}%" for x in params["splits"])
            st.caption(f"분할 비율: {splits_str}  |  거래비용: 편도 {TRANSACTION_COST*100:.1f}%")

        # ──────────────────────────────────────────────────────
        # [4] 지표 상세 (접이식)
        # ──────────────────────────────────────────────────────
        with st.expander("📌 기술 지표 상세"):
            # [MOBILE] 6컬럼 → 3+3
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric("정배열",       "✅" if uptrend else "❌")
            ic2.metric("기울기 20MA",  f"{ind_data['slope']:+.2f}%")
            ic3.metric("이격도",       f"{ind_data['deviation']:+.2f}%")
            ic4, ic5, ic6 = st.columns(3)
            ic4.metric("RSI(14)",      f"{ind_data['rsi']:.2f}")
            ic5.metric("ROC(12)",      f"{ind_data['roc']:+.2f}%")
            ic6.metric("변동성(20일)", f"{ind_data['volatility']:.4f}")

        # ──────────────────────────────────────────────────────
        # [5] 전략 점수 (접이식)
        # ──────────────────────────────────────────────────────
        with st.expander("📊 전략별 점수"):
            sc1, sc2, sc3 = st.columns(3)
            for col, (name, score) in zip([sc1, sc2, sc3], scores.items()):
                excl  = uptrend and STRATEGIES[name].get("exclude_uptrend", False)
                label = f"{'🏆 ' if name == rec else ''}{name}{'  [정배열 제외]' if excl else ''}"
                col.metric(label, f"{score:.3f}")
            st.caption(
                "점수 공식: Σ[(유사도 가중치) × 수익률 × max(0, 1+MDD/100)²]  "
                "— MDD 제곱 페널티 적용"
            )

        # ──────────────────────────────────────────────────────
        # [6] 유사 구간 Top-3 (접이식)
        # ──────────────────────────────────────────────────────
        if similar:
            with st.expander("🔍 추천 근거: 유사 구간 Top-3", expanded=True):
                st.caption(
                    "현재 시장과 지표가 가장 유사했던 과거 3개 구간의 전략별 성과입니다. "
                    f"거래비용 {TRANSACTION_COST*100:.1f}% 반영. 가격 손절 비활성."
                )
                sim_rows = []
                for past_date, sim_pct, perf in similar:
                    row = {"날짜": str(past_date.date()), "유사도": f"{sim_pct:.1f}%"}
                    for name in ["Pro1", "Pro2", "Pro3"]:
                        if name in perf:
                            r, m = perf[name]
                            row[f"{name} 수익"] = f"{r:+.1f}%"
                            row[f"{name} MDD"]  = f"{m:.1f}%"
                    sim_rows.append(row)
                st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

        st.info(
            "ℹ️ 유사도 계산 방식이 사이트와 완전히 일치하지 않을 수 있습니다. "
            "'입력' 탭에서 데이터를 꾸준히 입력하면 오차가 누적 분석됩니다."
        )

        # ──────────────────────────────────────────────────────
        # [7] 현재 사이클 전략 시나리오 비교 (접이식)
        # ──────────────────────────────────────────────────────
        st.divider()
        with st.expander("📈 현재 사이클 전략 시나리오 비교"):
            try:
                _cycle_start = pd.Timestamp(_acct.get("cycle_start_date", str(date.today())))
                _cycle_df    = df[df.index >= _cycle_start]
                _active_strat = _acct.get("strategy", "Pro3")

                if len(_cycle_df) < 2:
                    st.info("사이클 데이터 부족 (최소 2일 필요)")
                else:
                    _scenario_rows = []
                    for _sname, _sp in STRATEGIES.items():
                        _ret, _mdd = run_backtest(
                            prices=_cycle_df["close"],
                            splits=_sp["splits"],
                            buy_pct=_sp["buy_pct"],
                            sell_pct=_sp["sell_pct"],
                            stop_loss_days=_sp["stop_loss_days"],
                            buy_on_stop=_sp["buy_on_stop"],
                            transaction_cost=TRANSACTION_COST,
                            price_stop_loss_pct=PRICE_STOP_LOSS_PCT,
                        )
                        _scenario_rows.append({
                            "전략":       _sname,
                            "상태":       "▶ 진행중" if _sname == _active_strat else "〇 가상",
                            "사이클 수익": f"{_ret:+.2f}%",
                            "사이클 MDD":  f"{_mdd:.2f}%",
                            "경과 영업일": len(_cycle_df),
                        })
                    st.dataframe(pd.DataFrame(_scenario_rows), use_container_width=True, hide_index=True)
                    st.caption(f"동일 기간 각 전략 백테스트 (거래비용 {TRANSACTION_COST*100:.1f}% 반영)")
            except Exception as _e_sc:
                st.warning(f"시나리오 비교 오류: {_e_sc}")

    except Exception as e:
        st.error(f"오류: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════
# TAB 2 : 사이트 데이터 붙여넣기 (오차 비교용 — 거래비용 0)
# ══════════════════════════════════════════════════════════════
with tab2:
    st.title("📥 데이터 입력")

    input_mode = st.radio("입력 방식", ["📋 텍스트 붙여넣기", "📊 엑셀 업로드"], horizontal=True)

    if input_mode == "📊 엑셀 업로드":
        st.caption(
            "열마다 하루치 데이터가 있는 엑셀 파일을 업로드하세요. "
            "1행: 날짜, 2행~: 사이트에서 복사한 내용 그대로."
        )
        uploaded = st.file_uploader("엑셀 파일 (.xlsx)", type=["xlsx"])

        if uploaded:
            import openpyxl
            wb = openpyxl.load_workbook(uploaded, data_only=True)
            ws = wb.active

            cols_data = {}
            for col in ws.iter_cols():
                col_values = [str(cell.value).strip() if cell.value is not None else "" for cell in col]
                col_values = [v for v in col_values if v and v != "None"]
                if not col_values:
                    continue
                date_key  = col_values[0]
                text_body = "\n".join(col_values)
                cols_data[date_key] = text_body

            if not cols_data:
                st.error("데이터를 읽지 못했습니다. 파일 구조를 확인해주세요.")
            else:
                st.info(f"{len(cols_data)}개 날짜 컬럼 감지: {', '.join(list(cols_data.keys())[:5])}")

                if st.button("🔬 전체 파싱 & 저장", type="primary", use_container_width=True):
                    df_data, _ = load_data_cached()
                    comparison_log = load_comparison_log()
                    existing_dates = {e.get("기준일") for e in comparison_log}

                    saved, skipped, failed = 0, 0, 0
                    progress = st.progress(0)
                    keys = list(cols_data.keys())

                    for idx, (date_key, text_body) in enumerate(cols_data.items()):
                        progress.progress((idx + 1) / len(keys))
                        try:
                            parsed  = parse_site_text(text_body)
                            ref_date = parsed.get("ref_date") or date_key

                            if ref_date in existing_dates:
                                skipped += 1
                                continue
                            if not parsed.get("periods"):
                                failed += 1
                                continue

                            our_results = []
                            all_diffs   = []
                            for p in parsed["periods"]:
                                eval_start_ts = pd.Timestamp(p["eval_start"])
                                eval_end_ts   = pd.Timestamp(p["eval_end"])
                                mask   = (df_data.index >= eval_start_ts) & (df_data.index <= eval_end_ts)
                                prices = df_data.loc[mask, "close"]
                                if len(prices) < 5:
                                    our_results.append({})
                                    continue
                                our_perf = {}
                                for name, strat_params in STRATEGIES.items():
                                    ret, mdd = run_backtest(
                                        prices=prices,
                                        splits=strat_params["splits"],
                                        buy_pct=strat_params["buy_pct"],
                                        sell_pct=strat_params["sell_pct"],
                                        stop_loss_days=strat_params["stop_loss_days"],
                                        buy_on_stop=strat_params["buy_on_stop"],
                                    )
                                    our_perf[name] = {"ret": ret, "mdd": mdd}
                                    site_r = p.get(name, {}).get("ret", 0)
                                    all_diffs.append(abs(ret - site_r))
                                our_results.append(our_perf)

                            avg_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0
                            max_diff = max(all_diffs) if all_diffs else 0
                            log_entry = {
                                "기준일":         ref_date,
                                "입력시각":       pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                                "사이트추천전략": parsed.get("recommended", "-"),
                                "사이트점수":     parsed.get("scores", {}),
                                "지표": {
                                    "uptrend":    parsed.get("uptrend"),
                                    "slope":      parsed.get("slope"),
                                    "deviation":  parsed.get("deviation"),
                                    "rsi":        parsed.get("rsi"),
                                    "roc":        parsed.get("roc"),
                                    "volatility": parsed.get("volatility"),
                                },
                                "구간별비교": [
                                    {
                                        "구간":   p["eval_start"] + "~" + p["eval_end"],
                                        "유사도": p["similarity"],
                                        "사이트": {k: p.get(k, {}) for k in ["Pro1", "Pro2", "Pro3"]},
                                        "우리":   our_results[i] if i < len(our_results) else {},
                                    }
                                    for i, p in enumerate(parsed["periods"])
                                ],
                                "평균수익률차이": round(avg_diff, 3),
                                "최대수익률차이": round(max_diff, 3),
                                "재검증필요":     max_diff >= THRESHOLD_DIFF,
                            }
                            comparison_log.append(log_entry)
                            existing_dates.add(ref_date)
                            saved += 1
                        except Exception:
                            failed += 1

                    save_comparison_log(comparison_log)
                    progress.progress(1.0)
                    st.success(f"✅ 저장 {saved}건 | 중복 스킵 {skipped}건 | 파싱 실패 {failed}건")

        st.stop()

    st.caption("사이트 텍스트를 전체 복사해서 붙여넣으면 자동 파싱 후 백테스트와 비교합니다.")

    pasted = st.text_area(
        "사이트 전체 텍스트 붙여넣기",
        height=250,
        placeholder="📌 추천 기준일: 2026-04-05\n...\n정액 매수 X",
    )

    col_run, col_clear = st.columns([2, 1])
    run_compare = col_run.button("🔬 파싱 & 비교", type="primary", use_container_width=True)
    if col_clear.button("🗑️ 초기화", use_container_width=True):
        st.rerun()

    if run_compare and pasted.strip():
        try:
            parsed = parse_site_text(pasted)

            st.subheader("📋 파싱 확인")
            # [MOBILE] 3컬럼 유지 (3개는 모바일에서도 OK)
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("기준일",    parsed.get("ref_date", "파싱 실패"))
            rc2.metric("정배열",    "✅" if parsed.get("uptrend") else "❌")
            rc3.metric("추천 전략", parsed.get("recommended", "-"))

            # [MOBILE] 5컬럼 → 3+2 배치
            with st.expander("📌 지표 상세"):
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("기울기",  f"{parsed.get('slope', 0):+.2f}%")
                ic2.metric("이격도",  f"{parsed.get('deviation', 0):+.2f}%")
                ic3.metric("RSI",     f"{parsed.get('rsi', 0):.2f}")
                ic4, ic5 = st.columns(2)
                ic4.metric("ROC",     f"{parsed.get('roc', 0):+.2f}%")
                ic5.metric("변동성",  f"{parsed.get('volatility', 0):.4f}")

            scores_parsed = parsed.get("scores", {})
            if scores_parsed:
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Pro1 점수", f"{scores_parsed.get('Pro1', 0):.3f}")
                sc2.metric("Pro2 점수", f"{scores_parsed.get('Pro2', 0):.3f}")
                sc3.metric("Pro3 점수", f"{scores_parsed.get('Pro3', 0):.3f}")

            if not parsed.get("periods"):
                st.error("유사 구간 파싱 실패. 텍스트를 확인해주세요.")
                st.stop()

            st.divider()

            df_cmp, _ = load_data_cached()
            comparison_log = load_comparison_log()
            our_results = []
            all_diffs   = []

            st.subheader("📊 백테스트 비교 (거래비용 0 — 사이트 기준)")
            for i, p in enumerate(parsed["periods"]):
                eval_start_ts = pd.Timestamp(p["eval_start"])
                eval_end_ts   = pd.Timestamp(p["eval_end"])

                mask   = (df_cmp.index >= eval_start_ts) & (df_cmp.index <= eval_end_ts)
                prices = df_cmp.loc[mask, "close"]

                if len(prices) < 5:
                    st.warning(f"구간 {i+1}: 데이터 부족 ({len(prices)}일)")
                    our_results.append({})
                    continue

                our_perf = {}
                for name, strat_params in STRATEGIES.items():
                    ret, mdd = run_backtest(
                        prices=prices,
                        splits=strat_params["splits"],
                        buy_pct=strat_params["buy_pct"],
                        sell_pct=strat_params["sell_pct"],
                        stop_loss_days=strat_params["stop_loss_days"],
                        buy_on_stop=strat_params["buy_on_stop"],
                    )
                    our_perf[name] = {"ret": ret, "mdd": mdd}
                our_results.append(our_perf)

                st.markdown(f"**구간 {i+1}: {p['eval_start']} ~ {p['eval_end']}** (유사도 {p['similarity']:.2f}%)")
                diff_rows = []
                for name in ["Pro1", "Pro2", "Pro3"]:
                    site_r = p.get(name, {}).get("ret", 0)
                    site_m = p.get(name, {}).get("mdd", 0)
                    our_r  = our_perf.get(name, {}).get("ret", 0)
                    our_m  = our_perf.get(name, {}).get("mdd", 0)
                    r_diff = our_r - site_r
                    m_diff = our_m - site_m
                    all_diffs.append(abs(r_diff))
                    diff_rows.append({
                        "전략":        name,
                        "사이트 수익": f"{site_r:+.1f}%",
                        "우리 수익":   f"{our_r:+.1f}%",
                        "수익 차이":   color_diff(r_diff),
                        "사이트 MDD":  f"{site_m:.1f}%",
                        "우리 MDD":    f"{our_m:.1f}%",
                        "MDD 차이":    color_diff(m_diff),
                    })
                st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)

            avg_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0
            max_diff = max(all_diffs) if all_diffs else 0

            log_entry = {
                "기준일":         parsed.get("ref_date", "-"),
                "입력시각":       pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "사이트추천전략": parsed.get("recommended", "-"),
                "사이트점수":     parsed.get("scores", {}),
                "지표": {
                    "uptrend":    parsed.get("uptrend"),
                    "slope":      parsed.get("slope"),
                    "deviation":  parsed.get("deviation"),
                    "rsi":        parsed.get("rsi"),
                    "roc":        parsed.get("roc"),
                    "volatility": parsed.get("volatility"),
                },
                "구간별비교": [
                    {
                        "구간":   p["eval_start"] + "~" + p["eval_end"],
                        "유사도": p["similarity"],
                        "사이트": {k: p.get(k, {}) for k in ["Pro1", "Pro2", "Pro3"]},
                        "우리":   our_results[i] if i < len(our_results) else {},
                    }
                    for i, p in enumerate(parsed["periods"])
                ],
                "평균수익률차이": round(avg_diff, 3),
                "최대수익률차이": round(max_diff, 3),
                "재검증필요":     max_diff >= THRESHOLD_DIFF,
            }
            comparison_log.append(log_entry)
            save_comparison_log(comparison_log)

            if max_diff >= THRESHOLD_DIFF:
                st.error(f"⚠️ 최대 차이 {max_diff:.2f}%p — 오차 탭에서 자동 조정을 실행하세요.")
            else:
                st.success(f"✅ 저장 완료 | 최대 차이 {max_diff:.2f}%p (기준 {THRESHOLD_DIFF}%p 이내)")

        except Exception as e:
            st.error(f"파싱 오류: {e}")
            st.exception(e)

    elif run_compare:
        st.warning("텍스트를 먼저 붙여넣어주세요.")


# ══════════════════════════════════════════════════════════════
# TAB 3 : 오차 이력 & 자동 조정
# ══════════════════════════════════════════════════════════════
with tab3:
    st.title("🔍 오차 & 조정")

    log_t3   = load_comparison_log()
    stats_t3 = calc_error_stats(log_t3)

    if not log_t3:
        st.info("아직 비교 데이터가 없습니다. '입력' 탭에서 사이트 데이터를 입력해주세요.")
    else:
        st.subheader("📈 누적 오차 현황")

        # [MOBILE] 3컬럼 유지 (OK)
        ov1, ov2, ov3 = st.columns(3)
        ov1.metric("비교 데이터 수",  f"{stats_t3['total_records']}건")
        ov2.metric("전체 RMSE",        f"{stats_t3.get('overall_rmse', 0):.3f}%p")
        ov3.metric("평균 절대오차",    f"{stats_t3.get('overall_mean_abs', 0):.3f}%p")

        n = stats_t3["total_records"]
        if n < 10:
            st.warning(f"⚠️ 표본 {n}건 — 통계적으로 불충분합니다. 30건 이상 쌓인 후 해석하세요.")
        elif n < 30:
            st.info(f"📊 표본 {n}건 — 경향은 보이지만, 30건 이상에서 신뢰도가 높아집니다.")
        else:
            st.success(f"✅ 표본 {n}건 — 충분한 데이터입니다.")

        if stats_t3.get("need_tuning"):
            st.warning("⚠️ RMSE ≥ 2.0%p — 자동 조정이 권장됩니다.")
        else:
            st.success("✅ 오차 수준 양호 (RMSE < 2.0%p)")

        if len(log_t3) >= 3:
            rmse_trend = []
            for k in range(3, len(log_t3) + 1):
                sub = log_t3[:k]
                s = calc_error_stats(sub)
                rmse_trend.append({"누적 건수": k, "RMSE": s.get("overall_rmse", 0), "평균절대오차": s.get("overall_mean_abs", 0)})
            trend_df = pd.DataFrame(rmse_trend)
            st.subheader("📉 오차 추이 (누적)")
            st.line_chart(trend_df.set_index("누적 건수")[["RMSE", "평균절대오차"]])
            st.caption("건수가 늘어날수록 곡선이 안정되면 파라미터가 수렴하고 있는 것입니다.")

        by_s = stats_t3.get("by_strategy", {})
        if by_s:
            with st.expander("전략별 오차 상세"):
                err_rows = []
                for name, e in by_s.items():
                    confidence = "🟢 양호" if e["rmse"] < 2.0 else ("🟡 주의" if e["rmse"] < 4.0 else "🔴 높음")
                    err_rows.append({
                        "전략":           name,
                        "샘플 수":        e["count"],
                        "평균 절대오차":  f"{e['mean_abs']:.3f}%p",
                        "최대 오차":      f"{e['max_abs']:.3f}%p",
                        "RMSE":           f"{e['rmse']:.3f}%p",
                        "편향 (bias)":    f"{e['bias']:+.3f}%p",
                        "상태":           confidence,
                    })
                st.dataframe(pd.DataFrame(err_rows), use_container_width=True, hide_index=True)
                st.caption("편향(bias) 양수 = 우리 계산이 사이트보다 높게 나오는 경향 | 30건 미만은 참고용")

        st.divider()

        st.subheader("🤖 자동 파라미터 조정")

        tuning_log_t3 = load_tuning_log()
        last_tune_t3  = tuning_log_t3[-1] if tuning_log_t3 else None

        if last_tune_t3:
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("마지막 조정",   last_tune_t3.get("실행시각", "-"))
            lc2.metric("EVAL_WINDOW",  f"{last_tune_t3.get('새_EVAL_WINDOW', '-')}일")
            lc3.metric("조정 후 RMSE", f"{last_tune_t3.get('전체RMSE_후', '-')}")

        st.caption(
            f"현재 EVAL_WINDOW = **{EVAL_WINDOW}** 영업일  |  "
            "조정 범위: 15~28일  |  결과는 params.json에 저장됩니다"
        )

        if st.button("🔧 지금 자동 조정 실행", type="primary", use_container_width=True):
            with st.spinner("EVAL_WINDOW 탐색 중 (15~28일)..."):
                try:
                    df_tune, _ = load_data_cached()
                    tune_out   = run_tuning(df_tune)

                    if tune_out["status"] == "데이터 부족":
                        st.warning(tune_out["message"])
                    else:
                        tr = tune_out["tune_result"]
                        st.write("**EVAL_WINDOW별 RMSE:**")
                        window_df = pd.DataFrame([
                            {"EVAL_WINDOW": w, "RMSE": r,
                             "": "◀ 최적" if w == tr["best_window"] else ""}
                            for w, r in sorted(tr["all_results"].items())
                        ])
                        st.dataframe(window_df, use_container_width=True, hide_index=True)

                        if tune_out["improved"]:
                            st.success(
                                f"✅ {tune_out['message']}  |  "
                                f"RMSE {tr['best_rmse']:.3f}%p"
                            )
                            st.info("params.json이 업데이트됐습니다. 앱을 재시작하면 반영됩니다.")
                        else:
                            st.info(tune_out["message"])
                except Exception as e:
                    st.error(f"오류: {e}")
                    st.exception(e)

        st.divider()

        with st.expander("📊 기준 전략 vs 현재 전략 성과 비교"):
            st.caption(
                "오차(RMSE)는 항상 **기준 파라미터** 기준으로 계산됩니다. "
                "에이전트 제안을 적용해도 오차 추적이 왜곡되지 않습니다."
            )

            _changed_keys = ["buy_pct", "sell_pct", "stop_loss_days"]
            _any_changed = any(
                STRATEGIES[n].get(k) != BASELINE_STRATEGIES[n].get(k)
                for n in ["Pro1", "Pro2", "Pro3"] for k in _changed_keys
            )

            if not _any_changed:
                st.info("현재 전략 = 기준 전략 (변경 없음) — 에이전트 제안 적용 시 여기서 비교됩니다.")
            else:
                if st.button("📊 기준 vs 현재 연도별 비교 계산", key="cmp_baseline_btn", use_container_width=True):
                    try:
                        _df_b  = load_data_cached()[0]
                        _years_b = sorted(_df_b.index.year.unique())
                        _brows = []
                        _prog_b = st.progress(0)
                        for _bi, _yr in enumerate(_years_b):
                            _mask_b = _df_b.index.year == _yr
                            if _mask_b.sum() < 5:
                                continue
                            _px = _df_b.loc[_mask_b, "close"]
                            _row_b = {"연도": _yr}
                            for _nm in ["Pro1", "Pro2", "Pro3"]:
                                _bp = BASELINE_STRATEGIES[_nm]
                                _cp = STRATEGIES[_nm]
                                _br, _bm = run_backtest(
                                    prices=_px, splits=_bp["splits"],
                                    buy_pct=_bp["buy_pct"], sell_pct=_bp["sell_pct"],
                                    stop_loss_days=_bp["stop_loss_days"],
                                    buy_on_stop=_bp.get("buy_on_stop", True),
                                )
                                _cr, _cm = run_backtest(
                                    prices=_px, splits=_cp["splits"],
                                    buy_pct=_cp["buy_pct"], sell_pct=_cp["sell_pct"],
                                    stop_loss_days=_cp["stop_loss_days"],
                                    buy_on_stop=_cp.get("buy_on_stop", True),
                                )
                                _row_b[f"{_nm} 기준수익"] = f"{_br:+.1f}%"
                                _row_b[f"{_nm} 현재수익"] = f"{_cr:+.1f}%"
                                _row_b[f"{_nm} 수익변화"] = f"{_cr-_br:+.1f}%p"
                                _row_b[f"{_nm} MDD변화"]  = f"{_cm-_bm:+.1f}%p"
                            _brows.append(_row_b)
                            _prog_b.progress((_bi+1)/len(_years_b))
                        _bdf = pd.DataFrame(_brows)
                        st.dataframe(_bdf, use_container_width=True, hide_index=True, height=400)
                        st.markdown("**평균 요약**")
                        _sum_cols = st.columns(3)
                        for _ci, _nm in enumerate(["Pro1", "Pro2", "Pro3"]):
                            _avg_r = _bdf[f"{_nm} 수익변화"].str.replace("%p", "").astype(float).mean()
                            _avg_m = _bdf[f"{_nm} MDD변화"].str.replace("%p", "").astype(float).mean()
                            _sum_cols[_ci].metric(_nm, f"수익 {_avg_r:+.2f}%p", f"MDD {_avg_m:+.2f}%p (양수=개선)")
                    except Exception as _e_b:
                        st.error(f"비교 오류: {_e_b}")

        st.divider()

        st.subheader("📋 비교 이력")
        summary_rows_t3 = []
        for entry in log_t3:
            summary_rows_t3.append({
                "기준일":    entry["기준일"],
                "입력시각":  entry.get("입력시각", "-"),
                "추천 전략": entry.get("사이트추천전략", "-"),
                "평균차이":  f"{entry.get('평균수익률차이', 0):.2f}%p",
                "최대차이":  f"{entry.get('최대수익률차이', 0):.2f}%p",
                "상태":      "🔴 재검증" if entry.get("재검증필요") else "🟢 정상",
            })
        st.dataframe(pd.DataFrame(summary_rows_t3), use_container_width=True, hide_index=True)

        if log_t3:
            st.subheader("상세 보기")
            sel_t3 = st.selectbox(
                "날짜 선택",
                range(len(log_t3)),
                format_func=lambda i: f"{log_t3[i]['기준일']}  ({log_t3[i].get('사이트추천전략','-')})  최대차이 {log_t3[i].get('최대수익률차이',0):.2f}%p",
            )
            for pd_item in log_t3[sel_t3].get("구간별비교", []):
                with st.expander(f"구간: {pd_item['구간']}  (유사도 {pd_item['유사도']:.2f}%)"):
                    rows_t3 = []
                    for name in ["Pro1", "Pro2", "Pro3"]:
                        s = pd_item.get("사이트", {}).get(name, {})
                        o = pd_item.get("우리", {}).get(name, {})
                        rows_t3.append({
                            "전략":       name,
                            "사이트 수익": f"{s.get('ret', 0):+.1f}%",
                            "우리 수익":   f"{o.get('ret', 0):+.1f}%",
                            "차이":        color_diff(o.get("ret", 0) - s.get("ret", 0)),
                            "사이트 MDD":  f"{s.get('mdd', 0):.1f}%",
                            "우리 MDD":    f"{o.get('mdd', 0):.1f}%",
                        })
                    st.dataframe(pd.DataFrame(rows_t3), use_container_width=True, hide_index=True)

        if st.button("🗑️ 비교 이력 전체 삭제", type="secondary", use_container_width=True):
            save_comparison_log([])
            st.rerun()


# ══════════════════════════════════════════════════════════════
# TAB 4 : 백테스트 통계
# ══════════════════════════════════════════════════════════════
with tab4:
    st.title("📊 백테스트 통계")
    st.caption(
        f"거래비용 편도 **{TRANSACTION_COST*100:.1f}%** 반영 (가격 손절 비활성 — 수익률 극대화 모드)"
    )

    mode = st.radio("기간 단위", ["연도별", "분기별"], horizontal=True)

    if st.button("📈 통계 계산", type="primary", use_container_width=True):
        try:
            df_t4, ind_t4 = load_data_cached()

            if mode == "연도별":
                periods  = df_t4.index.year.unique()
                get_mask = lambda p: df_t4.index.year == p
                label_fn = str
            else:
                df_t4 = df_t4.copy()
                df_t4["_q"] = df_t4.index.to_period("Q")
                periods  = sorted(df_t4["_q"].unique())
                get_mask = lambda p: df_t4["_q"] == p
                label_fn = lambda p: str(p).replace("Q", " Q")

            rows_t4  = []
            progress_t4 = st.progress(0)
            for i, period in enumerate(periods):
                mask = get_mask(period)
                if mask.sum() < 5:
                    continue
                up_days = ind_t4.loc[df_t4.index[mask], "uptrend"].sum()
                up_pct  = int(round(up_days / mask.sum() * 100))
                row = {"기간": label_fn(period), "정배열": f"{up_pct}%"}
                for name, strat_params in STRATEGIES.items():
                    prices = df_t4.loc[mask, "close"]
                    ret, mdd = run_backtest(
                        prices=prices,
                        splits=strat_params["splits"],
                        buy_pct=strat_params["buy_pct"],
                        sell_pct=strat_params["sell_pct"],
                        stop_loss_days=strat_params["stop_loss_days"],
                        buy_on_stop=strat_params["buy_on_stop"],
                        crash_buy=strat_params.get("crash_buy"),
                        transaction_cost=TRANSACTION_COST,
                        price_stop_loss_pct=PRICE_STOP_LOSS_PCT,
                    )
                    row[f"{name} 수익"] = f"{ret:+.1f}%"
                    row[f"{name} MDD"]  = f"{mdd:.1f}%"
                rows_t4.append(row)
                progress_t4.progress((i + 1) / len(periods))

            st.dataframe(pd.DataFrame(rows_t4), use_container_width=True, hide_index=True, height=600)

        except Exception as e:
            st.error(f"오류: {e}")
            st.exception(e)
    else:
        st.info("'통계 계산' 버튼을 눌러주세요. 전체 계산에 수십 초가 소요됩니다.")

    st.divider()

    # ── 파라미터 비교 백테스트 ──────────────────────────────────
    st.subheader("🔬 파라미터 비교 백테스트")
    st.caption("에이전트 제안값을 불러오거나 직접 입력해서 현재값과 나란히 비교합니다.")

    agent_log_all = load_agent_log()
    all_suggestions = []
    for log_i, entry in enumerate(agent_log_all):
        text = entry.get("분석내용", "")
        for s in extract_param_suggestions(text):
            all_suggestions.append((log_i + 1, entry.get("실행시각", ""), s["label"], s["params"]))

    if all_suggestions:
        st.markdown("#### 🤖 에이전트 제안 불러오기")
        suggestion_labels = [f"#{idx} [{ts}] {label}" for idx, ts, label, _ in all_suggestions]
        sel_sug = st.selectbox("제안 선택", range(len(all_suggestions)),
                               format_func=lambda i: suggestion_labels[i], key="sug_sel")
        sel_params = all_suggestions[sel_sug][3]

        prev_key = "sug_sel_prev"
        if st.session_state.get(prev_key) != sel_sug:
            st.session_state[prev_key] = sel_sug
            for name, sk_buy, sk_sell, sk_stop, sk_dbt, sk_sst in [
                ("Pro1", "cp1b", "cp1s", "cp1stop", "cp1dbt", "cp1sst"),
                ("Pro2", "cp2b", "cp2s", "cp2stop", "cp2dbt", "cp2sst"),
                ("Pro3", "cp3b", "cp3s", "cp3stop", "cp3dbt", "cp3sst"),
            ]:
                p = sel_params.get(name, {})
                if "buy_pct"              in p: st.session_state[sk_buy]  = float(p["buy_pct"])  * 100
                if "sell_pct"             in p: st.session_state[sk_sell] = float(p["sell_pct"]) * 100
                if "stop_loss_days"       in p: st.session_state[sk_stop] = int(p["stop_loss_days"])
                if "double_buy_threshold" in p: st.session_state[sk_dbt]  = float(p["double_buy_threshold"]) * 100
                if "surge_sell_threshold" in p: st.session_state[sk_sst]  = float(p["surge_sell_threshold"]) * 100
            st.rerun()

        st.caption("선택한 제안의 파라미터가 아래 입력란에 자동 반영됩니다. 수정도 가능합니다.")

        def _get(name, key, pct_scale=False):
            v = sel_params.get(name, {}).get(key)
            if v is None:
                v = STRATEGIES[name].get(key, 0)
            return float(v) * 100 if pct_scale else v
    else:
        st.info("에이전트 분석을 먼저 실행하면 제안값을 자동으로 불러올 수 있습니다.")
        sel_params = {}

        def _get(name, key, pct_scale=False):
            v = STRATEGIES[name].get(key, 0)
            return float(v) * 100 if pct_scale else v

    st.markdown("#### ✏️ 비교 파라미터")
    # [MOBILE] 3컬럼 → 탭으로 분리 (아이폰에서 3컬럼 입력이 매우 좁음)
    pro_tab1, pro_tab2, pro_tab3 = st.tabs(["Pro1", "Pro2", "Pro3"])

    with pro_tab1:
        p1_buy  = st.number_input("buy_pct (%)",              value=_get("Pro1", "buy_pct", True),  step=0.001, key="cp1b", format="%.3f")
        p1_sell = st.number_input("sell_pct (%)",             value=_get("Pro1", "sell_pct", True), step=0.001, key="cp1s", format="%.3f")
        p1_stop = st.number_input("stop_loss_days",           value=int(_get("Pro1", "stop_loss_days")), step=1, key="cp1stop")
        p1_dbt  = st.number_input("급락2티어 트리거(%) 0=비활성",   value=_get("Pro1", "double_buy_threshold", True), step=0.1, key="cp1dbt", format="%.1f")
        p1_sst  = st.number_input("급등이익실현 트리거(%) 0=비활성", value=_get("Pro1", "surge_sell_threshold", True), step=0.1, key="cp1sst", format="%.1f")

    with pro_tab2:
        p2_buy  = st.number_input("buy_pct (%)",              value=_get("Pro2", "buy_pct", True),  step=0.001, key="cp2b", format="%.3f")
        p2_sell = st.number_input("sell_pct (%)",             value=_get("Pro2", "sell_pct", True), step=0.001, key="cp2s", format="%.3f")
        p2_stop = st.number_input("stop_loss_days",           value=int(_get("Pro2", "stop_loss_days")), step=1, key="cp2stop")
        p2_dbt  = st.number_input("급락2티어 트리거(%) 0=비활성",   value=_get("Pro2", "double_buy_threshold", True), step=0.1, key="cp2dbt", format="%.1f")
        p2_sst  = st.number_input("급등이익실현 트리거(%) 0=비활성", value=_get("Pro2", "surge_sell_threshold", True), step=0.1, key="cp2sst", format="%.1f")

    with pro_tab3:
        p3_buy  = st.number_input("buy_pct (%)",              value=_get("Pro3", "buy_pct", True),  step=0.001, key="cp3b", format="%.3f")
        p3_sell = st.number_input("sell_pct (%)",             value=_get("Pro3", "sell_pct", True), step=0.001, key="cp3s", format="%.3f")
        p3_stop = st.number_input("stop_loss_days",           value=int(_get("Pro3", "stop_loss_days")), step=1, key="cp3stop")
        p3_dbt  = st.number_input("급락2티어 트리거(%) 0=비활성",   value=_get("Pro3", "double_buy_threshold", True), step=0.1, key="cp3dbt", format="%.1f")
        p3_sst  = st.number_input("급등이익실현 트리거(%) 0=비활성", value=_get("Pro3", "surge_sell_threshold", True), step=0.1, key="cp3sst", format="%.1f")

    run_cmp      = st.button("📊 비교 백테스트 실행", type="primary", use_container_width=True)
    apply_config = st.button("✅ params.json에 적용", use_container_width=True,
                             help="비교 파라미터를 params.json에 저장합니다 (재시작 후 반영)")

    if apply_config:
        try:
            params_data = load_params()
            params_data["BASELINE_STRATEGIES"] = {
                nm: {k: v for k, v in strat.items()}
                for nm, strat in params_data.get("STRATEGIES", {}).items()
            }
            for nm, p_buy, p_sell, p_stop in [
                ("Pro1", p1_buy, p1_sell, int(p1_stop)),
                ("Pro2", p2_buy, p2_sell, int(p2_stop)),
                ("Pro3", p3_buy, p3_sell, int(p3_stop)),
            ]:
                if nm in params_data["STRATEGIES"]:
                    params_data["STRATEGIES"][nm]["buy_pct"]        = round(p_buy / 100, 6)
                    params_data["STRATEGIES"][nm]["sell_pct"]       = round(p_sell / 100, 6)
                    params_data["STRATEGIES"][nm]["stop_loss_days"] = p_stop
            save_params(params_data)
            st.success("✅ params.json 적용 완료! 앱을 재시작하면 반영됩니다.")
        except Exception as _e_apply:
            st.error(f"적용 오류: {_e_apply}")

    def _to_opt_pct(val):
        return None if val == 0.0 else val / 100

    if run_cmp:
        candidate = {
            "Pro1": {**STRATEGIES["Pro1"], "buy_pct": p1_buy/100, "sell_pct": p1_sell/100, "stop_loss_days": int(p1_stop),
                     "double_buy_threshold": _to_opt_pct(p1_dbt), "surge_sell_threshold": _to_opt_pct(p1_sst)},
            "Pro2": {**STRATEGIES["Pro2"], "buy_pct": p2_buy/100, "sell_pct": p2_sell/100, "stop_loss_days": int(p2_stop),
                     "double_buy_threshold": _to_opt_pct(p2_dbt), "surge_sell_threshold": _to_opt_pct(p2_sst)},
            "Pro3": {**STRATEGIES["Pro3"], "buy_pct": p3_buy/100, "sell_pct": p3_sell/100, "stop_loss_days": int(p3_stop),
                     "double_buy_threshold": _to_opt_pct(p3_dbt), "surge_sell_threshold": _to_opt_pct(p3_sst)},
        }
        try:
            df_cmp2, _ = load_data_cached()
            years_cmp  = sorted(df_cmp2.index.year.unique())
            cmp_rows   = []
            progress_cmp = st.progress(0)
            for i, year in enumerate(years_cmp):
                mask = df_cmp2.index.year == year
                if mask.sum() < 5:
                    continue
                prices = df_cmp2.loc[mask, "close"]
                row = {"연도": year}
                for name in ["Pro1", "Pro2", "Pro3"]:
                    cur_r, cur_m = run_backtest(
                        prices=prices,
                        splits=STRATEGIES[name]["splits"],
                        buy_pct=STRATEGIES[name]["buy_pct"],
                        sell_pct=STRATEGIES[name]["sell_pct"],
                        stop_loss_days=STRATEGIES[name]["stop_loss_days"],
                        buy_on_stop=STRATEGIES[name]["buy_on_stop"],
                        transaction_cost=TRANSACTION_COST,
                        price_stop_loss_pct=PRICE_STOP_LOSS_PCT,
                    )
                    new_r, new_m = run_backtest(
                        prices=prices,
                        splits=candidate[name]["splits"],
                        buy_pct=candidate[name]["buy_pct"],
                        sell_pct=candidate[name]["sell_pct"],
                        stop_loss_days=candidate[name]["stop_loss_days"],
                        buy_on_stop=candidate[name]["buy_on_stop"],
                        double_buy_threshold=candidate[name].get("double_buy_threshold"),
                        surge_sell_threshold=candidate[name].get("surge_sell_threshold"),
                        transaction_cost=TRANSACTION_COST,
                        price_stop_loss_pct=PRICE_STOP_LOSS_PCT,
                    )
                    row[f"{name} 현재수익"]  = f"{cur_r:+.1f}%"
                    row[f"{name} 제안수익"]  = f"{new_r:+.1f}%"
                    row[f"{name} 차이"]      = f"{new_r-cur_r:+.1f}%p"
                    row[f"{name} 현재MDD"]   = f"{cur_m:.1f}%"
                    row[f"{name} 제안MDD"]   = f"{new_m:.1f}%"
                    row[f"{name} MDD차이"]   = f"{new_m-cur_m:+.1f}%p"
                cmp_rows.append(row)
                progress_cmp.progress((i + 1) / len(years_cmp))

            st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True, height=550)

            st.markdown("**평균 비교 요약**")
            sum_rows_cmp = []
            for name in ["Pro1", "Pro2", "Pro3"]:
                cur_r_vals = [float(r[f"{name} 현재수익"].replace("%", "").replace("+", "")) for r in cmp_rows]
                new_r_vals = [float(r[f"{name} 제안수익"].replace("%", "").replace("+", "")) for r in cmp_rows]
                cur_m_vals = [float(r[f"{name} 현재MDD"].replace("%", "")) for r in cmp_rows]
                new_m_vals = [float(r[f"{name} 제안MDD"].replace("%", "")) for r in cmp_rows]
                sum_rows_cmp.append({
                    "전략":          name,
                    "현재 평균수익": f"{np.mean(cur_r_vals):+.2f}%",
                    "제안 평균수익": f"{np.mean(new_r_vals):+.2f}%",
                    "수익 변화":     f"{np.mean(new_r_vals)-np.mean(cur_r_vals):+.2f}%p",
                    "현재 평균MDD":  f"{np.mean(cur_m_vals):.2f}%",
                    "제안 평균MDD":  f"{np.mean(new_m_vals):.2f}%",
                    "MDD 변화":      f"{np.mean(new_m_vals)-np.mean(cur_m_vals):+.2f}%p",
                })
            st.dataframe(pd.DataFrame(sum_rows_cmp), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"비교 백테스트 오류: {e}")
            st.exception(e)


# ══════════════════════════════════════════════════════════════
# TAB 5 : 금융 에이전트
# ══════════════════════════════════════════════════════════════
with tab5:
    st.title("🤖 AI 에이전트")
    st.caption("Claude AI가 백테스트 데이터를 분석하고 전략 개선안을 제안합니다.")

    with st.expander("⚙️ Anthropic API 키 설정", expanded=not bool(get_api_key())):
        st.markdown("API 키는 [console.anthropic.com](https://console.anthropic.com)에서 발급받으세요.")
        key_input = st.text_input("API 키", value=get_api_key(), type="password", placeholder="sk-ant-...")
        if st.button("저장", key="save_key", use_container_width=True):
            if key_input.startswith("sk-ant-"):
                save_api_key(key_input)
                st.success("저장됐습니다.")
                st.rerun()
            else:
                st.error("올바른 API 키 형식이 아닙니다 (sk-ant-로 시작)")

    st.divider()

    api_key = get_api_key()
    if not api_key:
        st.warning("API 키를 먼저 설정해주세요.")
        st.stop()

    agent_log = load_agent_log()

    if should_run_weekly() and agent_log:
        st.info("📅 마지막 분석 후 7일이 지났습니다. 새 분석을 실행하세요.")

    if agent_log:
        st.subheader("📋 분석 이력")
        log_rows_t5 = []
        for i, entry in enumerate(agent_log):
            n_sug = len(extract_param_suggestions(entry.get("분석내용", "")))
            log_rows_t5.append({
                "#":          i + 1,
                "실행시각":   entry.get("실행시각", "-"),
                "비교데이터": f"{entry.get('당시요약', {}).get('비교데이터수', '-')}건",
                "JSON 제안":  f"{'✅' if n_sug else '❌'} {n_sug}개",
            })
        st.dataframe(pd.DataFrame(log_rows_t5), use_container_width=True, hide_index=True)

        sel_idx_t5 = st.selectbox(
            "과거 분석 보기",
            range(len(agent_log)),
            format_func=lambda i: f"#{i+1}  {agent_log[i].get('실행시각', '-')}",
            index=len(agent_log) - 1,
        )
        sel_text_t5 = agent_log[sel_idx_t5].get("분석내용", "")
        with st.expander(f"분석 내용 #{sel_idx_t5 + 1}", expanded=False):
            st.markdown(sel_text_t5)

        sel_suggestions_t5 = extract_param_suggestions(sel_text_t5)
        if sel_suggestions_t5:
            st.success(f"✅ 이 분석에서 {len(sel_suggestions_t5)}개의 JSON 파라미터 제안이 감지됐습니다. → 통계 탭에서 선택해 적용하세요.")
        else:
            st.caption("이 분석에서는 JSON 파라미터 제안이 감지되지 않았습니다.")

        st.divider()

    st.subheader("🔍 새 분석 실행")

    agent_mode = st.radio(
        "분석 모드",
        ["📊 종합 전략 분석", "🎯 오차 최소화 분석"],
        horizontal=True,
        help="종합: 연도별 성과·리스크·파라미터 개선안 / 오차 최소화: 사이트 수치와의 차이를 줄이는 파라미터 조정에 집중",
    )

    col_run_t5, col_del_t5 = st.columns([3, 1])
    run_agent = col_run_t5.button("🚀 분석 시작", type="primary", use_container_width=True)
    if col_del_t5.button("🗑️ 이력 삭제", use_container_width=True):
        from agent import save_agent_log
        save_agent_log([])
        st.rerun()

    if run_agent:
        try:
            df_agent, _ = load_data_cached()

            st.subheader("📝 분석 결과")
            result_container = st.empty()
            full_text_t5 = ""

            mode_label = "오차 최소화" if "오차" in agent_mode else "종합 전략"
            with st.spinner(f"Claude가 [{mode_label}] 분석 중입니다..."):
                gen = run_error_minimization(df_agent, api_key) if "오차" in agent_mode else run_agent_analysis(df_agent, api_key)
                for chunk in gen:
                    full_text_t5 += chunk
                    result_container.markdown(full_text_t5 + "▌")

            result_container.markdown(full_text_t5)

            from agent import build_backtest_summary, save_agent_log, load_agent_log as _load_al
            summary_t5 = build_backtest_summary(df_agent)
            entry_t5 = {
                "실행시각": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "분석내용": full_text_t5,
                "당시요약": {
                    "비교데이터수":    summary_t5["comparison_count"],
                    "전체RMSE":       summary_t5["error_stats"].get("overall_rmse"),
                    "데이터범위종료": summary_t5["data_range"]["end"],
                },
            }
            cur_log_t5 = _load_al()
            cur_log_t5.append(entry_t5)
            save_agent_log(cur_log_t5)
            st.success("✅ 분석 완료 및 저장됨")

        except Exception as e:
            if "authentication" in str(e).lower() or "api_key" in str(e).lower():
                st.error("API 키가 유효하지 않습니다. 설정을 확인해주세요.")
            else:
                st.error(f"오류: {e}")
                st.exception(e)


# ══════════════════════════════════════════════════════════════
# TAB 6 : 내 계좌 (모바일 최적화)
# ══════════════════════════════════════════════════════════════
with tab6:
    st.title("💼 내 계좌")

    acct_t6      = load_account()
    strat_params = STRATEGIES.get(acct_t6["strategy"], STRATEGIES["Pro3"])

    # ── 계좌 설정 카드 — [MOBILE] 5컬럼 → 2+3 ──────────────────
    with st.container(border=True):
        st.markdown("#### ⚙️ 계좌 설정")
        cs1, cs2 = st.columns(2)
        cs1.metric("💰 시드", f"${acct_t6['seed']:,.0f}")
        cs2.metric("📋 전략", acct_t6["strategy"])
        cs3, cs4, cs5 = st.columns(3)
        cs3.metric("🔄 사이클", f"제 {acct_t6['cycle_number']}회차")
        cs4.metric("📅 시작일", acct_t6["cycle_start_date"])
        active_cnt = sum(1 for h in acct_t6["holdings"] if h.get("active", True))
        cs5.metric("📦 매수 티어", f"{active_cnt} / {len(strat_params['splits'])}")

    with st.expander("⚙️ 설정 변경 / 새 사이클 시작"):
        st.caption("전략은 새 사이클 시작 시에만 변경할 수 있습니다.")
        # [MOBILE] 3컬럼 → 세로 배치
        new_seed     = st.number_input("시드 ($)", value=float(acct_t6["seed"]), step=100.0, min_value=1000.0)
        new_strategy = st.selectbox("전략", ["Pro1", "Pro2", "Pro3"], index=["Pro1", "Pro2", "Pro3"].index(acct_t6["strategy"]))
        new_start    = st.date_input("사이클 시작일", value=pd.Timestamp(acct_t6["cycle_start_date"]).date())

        if st.button("💾 시드만 저장", use_container_width=True):
            acct_t6["seed"] = int(new_seed)
            save_account(acct_t6)
            st.success("시드 저장됨 — 오늘 추천 탭에 자동 반영됩니다.")
            st.rerun()
        if st.button("🚀 새 사이클 시작 (전략·포지션 초기화)", type="primary", use_container_width=True):
            _worst_mdd_info = ""
            try:
                _df_risk, _ = load_data_cached()
                _years_risk = sorted(_df_risk.index.year.unique())
                _mdds_risk = []
                for _yr in _years_risk:
                    _mx = _df_risk.index.year == _yr
                    if _mx.sum() < 5:
                        continue
                    _, _mdd_r = run_backtest(
                        prices=_df_risk.loc[_mx, "close"],
                        splits=STRATEGIES[new_strategy]["splits"],
                        buy_pct=STRATEGIES[new_strategy]["buy_pct"],
                        sell_pct=STRATEGIES[new_strategy]["sell_pct"],
                        stop_loss_days=STRATEGIES[new_strategy]["stop_loss_days"],
                        buy_on_stop=STRATEGIES[new_strategy]["buy_on_stop"],
                        transaction_cost=TRANSACTION_COST,
                        price_stop_loss_pct=PRICE_STOP_LOSS_PCT,
                    )
                    _mdds_risk.append(_mdd_r)
                if _mdds_risk:
                    _worst = min(_mdds_risk)
                    _worst_dollar = float(new_seed) * _worst / 100
                    _worst_mdd_info = (
                        f"\n\n⚠️ **{new_strategy} 역사적 최악 MDD: {_worst:.1f}%** "
                        f"(시드 ${new_seed:,.0f} 기준 최대 예상 손실: **${_worst_dollar:+,.0f}**)"
                    )
            except Exception:
                pass

            acct_t6["seed"]                   = int(new_seed)
            acct_t6["strategy"]               = new_strategy
            acct_t6["cycle_number"]           += 1
            acct_t6["cycle_start_date"]       = str(new_start)
            acct_t6["holdings"]               = []
            acct_t6["initial_asset"]          = float(new_seed)
            acct_t6["strategy_switch_date"]   = date.today().isoformat()
            acct_t6["strategy_mismatch_since"] = None
            acct_t6["filter_streak"]          = 0
            save_account(acct_t6)
            st.success(f"✅ 제 {acct_t6['cycle_number']}회차 시작! 전략: {new_strategy}{_worst_mdd_info}")
            st.rerun()

    st.divider()

    # ── 자산 현황 — [MOBILE] 5컬럼 → 2+3 ───────────────────────
    try:
        df_cur_t6 = load_data_cached()[0]
        cur_price = float(df_cur_t6["close"].iloc[-1])
        cur_date  = str(df_cur_t6.index[-1].date())
    except Exception:
        cur_price = 0.0
        cur_date  = "-"

    stock_value = sum(
        h.get("quantity", 0) * cur_price
        for h in acct_t6["holdings"] if h.get("active", True)
    )
    invested_cost = sum(
        h.get("quantity", 0) * h.get("entry_price", 0)
        for h in acct_t6["holdings"] if h.get("active", True)
    )
    total_asset = acct_t6["seed"] - invested_cost + stock_value
    cash        = acct_t6["seed"] - invested_cost
    invest_pct  = (stock_value / acct_t6["seed"] * 100) if acct_t6["seed"] > 0 else 0
    pnl_pct     = (total_asset / acct_t6["seed"] - 1) * 100

    with st.container(border=True):
        st.markdown(
            f"#### 💰 자산 현황 <span style='color:gray;font-size:0.8em'>({cur_date} 기준)</span>",
            unsafe_allow_html=True,
        )
        # [MOBILE] 5컬럼 → 2+3
        am1, am2 = st.columns(2)
        am1.metric("총자산",   f"${total_asset:,.0f}", f"{pnl_pct:+.2f}%")
        am2.metric("SOXL 현재가", f"${cur_price:.2f}")
        am3, am4, am5 = st.columns(3)
        am3.metric("주식평가", f"${stock_value:,.2f}")
        am4.metric("예수금",   f"${cash:,.0f}")
        am5.metric("투자비중", f"{invest_pct:.1f}%")

    st.divider()

    # ── 보유 현황 ────────────────────────────────────────────────
    st.markdown("#### 📦 보유 현황 (티어별)")

    splits_t6      = strat_params["splits"]
    sell_pct_val   = strat_params["sell_pct"]
    stop_days_val  = strat_params["stop_loss_days"]

    # 기존 데이터 하위 호환: slot_type 없으면 "normal" 처리
    for _h in acct_t6["holdings"]:
        _h.setdefault("slot_type", "normal")

    active_holdings = [h for h in acct_t6["holdings"] if h.get("active", True)]
    holding_by_tier = {h["tier"]: h for h in active_holdings}

    def _build_tier_rows(filter_slot_type: str) -> list:
        rows = []
        for i, alloc in enumerate(splits_t6, 1):
            alloc_amt = acct_t6["seed"] * alloc
            h = holding_by_tier.get(i)
            if h and h.get("slot_type", "normal") == filter_slot_type:
                entry_p    = h["entry_price"]
                qty        = h["quantity"]
                buy_date   = pd.Timestamp(h["entry_date"])
                bdays_held = len(pd.bdate_range(start=buy_date, end=pd.Timestamp(cur_date))) - 1 if cur_date != "-" else 0
                days_left  = stop_days_val - bdays_held
                sell_tgt   = round(entry_p * (1 + sell_pct_val), 4)
                pnl_t      = (cur_price - entry_p) / entry_p * 100 if cur_price > 0 else 0
                pnl_dollar = qty * (cur_price - entry_p)

                if days_left <= 0:
                    dday_str = "🔴 손절일 경과!"
                elif days_left == 1:
                    dday_str = "🟠 D-1 (내일 손절)"
                elif days_left <= 3:
                    dday_str = f"🟡 D-{days_left}"
                else:
                    dday_str = f"D-{days_left}"

                rows.append({
                    "티어":     i,
                    "비율":     f"{alloc*100:.1f}%",
                    "할당금":   f"${alloc_amt:,.0f}",
                    "매수일":   h["entry_date"],
                    "매수가":   f"${entry_p:.4f}",
                    "수량":     qty,
                    "매도목표": f"${sell_tgt:.4f}",
                    "손절까지": dday_str,
                    "현재손익": f"{pnl_t:+.1f}% (${pnl_dollar:+,.0f})",
                })
        return rows

    normal_rows = _build_tier_rows("normal")
    crash_rows  = _build_tier_rows("crash")

    # 전체 티어 표 (slot_type 무관, 빈 슬롯 포함)
    tier_rows_t6 = []
    for i, alloc in enumerate(splits_t6, 1):
        alloc_amt = acct_t6["seed"] * alloc
        h = holding_by_tier.get(i)
        if h:
            entry_p    = h["entry_price"]
            qty        = h["quantity"]
            buy_date   = pd.Timestamp(h["entry_date"])
            bdays_held = len(pd.bdate_range(start=buy_date, end=pd.Timestamp(cur_date))) - 1 if cur_date != "-" else 0
            days_left  = stop_days_val - bdays_held
            sell_tgt   = round(entry_p * (1 + sell_pct_val), 4)
            pnl_t      = (cur_price - entry_p) / entry_p * 100 if cur_price > 0 else 0
            pnl_dollar = qty * (cur_price - entry_p)

            if days_left <= 0:
                dday_str = "🔴 손절일 경과!"
            elif days_left == 1:
                dday_str = "🟠 D-1 (내일 손절)"
            elif days_left <= 3:
                dday_str = f"🟡 D-{days_left}"
            else:
                dday_str = f"D-{days_left}"

            tier_rows_t6.append({
                "티어":     i,
                "타입":     "⚡크래시" if h.get("slot_type") == "crash" else "📦일반",
                "비율":     f"{alloc*100:.1f}%",
                "할당금":   f"${alloc_amt:,.0f}",
                "매수일":   h["entry_date"],
                "매수가":   f"${entry_p:.4f}",
                "수량":     qty,
                "매도목표": f"${sell_tgt:.4f}",
                "손절까지": dday_str,
                "현재손익": f"{pnl_t:+.1f}% (${pnl_dollar:+,.0f})",
            })
        else:
            tier_rows_t6.append({
                "티어":     i,
                "타입":     "-",
                "비율":     f"{alloc*100:.1f}%",
                "할당금":   f"${alloc_amt:,.0f}",
                "매수일":   "-",
                "매수가":   "-",
                "수량":     "-",
                "매도목표": "-",
                "손절까지": "-",
                "현재손익": "-",
            })

    st.dataframe(pd.DataFrame(tier_rows_t6), use_container_width=True, hide_index=True)

    # 슬롯 타입별 분리 섹션
    if normal_rows:
        st.markdown("**📦 일반 슬롯**")
        normal_display = [{k: v for k, v in r.items() if k in ("티어", "매수가", "수량", "매도목표", "손절까지", "현재손익")} for r in normal_rows]
        st.dataframe(pd.DataFrame(normal_display), use_container_width=True, hide_index=True)

    if crash_rows:
        st.markdown("**⚡ 크래시 슬롯**")
        crash_display = [{k: v for k, v in r.items() if k in ("티어", "매수가", "수량", "매도목표", "손절까지", "현재손익")} for r in crash_rows]
        st.dataframe(pd.DataFrame(crash_display), use_container_width=True, hide_index=True)

    # ── 매수 / 매도 기록 — [MOBILE] 2컬럼 → 세로 배치 ──────────
    with st.expander("🛒 매수 기록"):
        available_tiers = [i for i in range(1, len(splits_t6)+1) if i not in holding_by_tier]
        if not available_tiers:
            st.info("모든 티어가 채워졌습니다.")
        else:
            slot_type_label = st.radio("매수 타입", ["일반", "크래시"], horizontal=True, key="b_slot_type")
            slot_type = "crash" if slot_type_label == "크래시" else "normal"
            # [MOBILE] 4컬럼 → 2+2
            bt1, bt2 = st.columns(2)
            b_tier  = bt1.selectbox("티어", available_tiers, key="b_tier")
            b_price = bt2.number_input("매수가 ($)", min_value=0.01, step=0.01, key="b_price")
            bt3, bt4 = st.columns(2)
            b_qty   = bt3.number_input("수량", min_value=1, step=1, key="b_qty")
            b_date  = bt4.date_input("매수일", value=date.today(), key="b_date")
            if st.button("✅ 매수 기록 저장", use_container_width=True, key="save_buy"):
                acct_t6["holdings"].append({
                    "slot_type":          slot_type,
                    "tier":               b_tier,
                    "entry_price":        float(b_price),
                    "entry_date":         str(b_date),
                    "quantity":           int(b_qty),
                    "active":             True,
                    "strategy_at_entry":  acct_t6.get("strategy", "Pro3"),
                })
                save_account(acct_t6)
                st.success(f"티어 {b_tier} 매수 기록 완료 ({slot_type_label})")
                st.rerun()

    with st.expander("💸 매도 / 손절 기록"):
        active_tiers_t6 = [h["tier"] for h in acct_t6["holdings"] if h.get("active", True)]
        if not active_tiers_t6:
            st.info("보유 중인 포지션이 없습니다.")
        else:
            s_tier  = st.selectbox("청산 티어", active_tiers_t6, key="s_tier")
            s_price = st.number_input("매도가 ($)", min_value=0.01, step=0.01, key="s_price")
            s_type  = st.radio("구분", ["익절", "손절"], horizontal=True, key="s_type")
            if st.button("✅ 매도 기록 저장", use_container_width=True, key="save_sell"):
                h_idx = next(i for i, h in enumerate(acct_t6["holdings"]) if h["tier"] == s_tier and h.get("active", True))
                h_old = acct_t6["holdings"][h_idx]
                pnl_amt = (float(s_price) - h_old["entry_price"]) * h_old["quantity"]
                acct_t6["profit_log"].append({
                    "date":    str(date.today()),
                    "tier":    s_tier,
                    "strategy": acct_t6["strategy"],
                    "entry":   h_old["entry_price"],
                    "exit":    float(s_price),
                    "qty":     h_old["quantity"],
                    "pnl":     round(pnl_amt, 2),
                    "pnl_pct": round((float(s_price)/h_old["entry_price"] - 1)*100, 2),
                    "type":    s_type,
                })
                acct_t6["holdings"][h_idx]["active"] = False
                acct_t6["seed"] = round(acct_t6["seed"] + pnl_amt, 2)
                save_account(acct_t6)
                st.success(f"티어 {s_tier} 매도 완료 | 손익: ${pnl_amt:+,.2f}")
                st.rerun()

    st.divider()

    # ── 손익 기록 ────────────────────────────────────────────────
    st.markdown("#### 📈 손익 기록")
    if not acct_t6["profit_log"]:
        st.info("아직 기록된 거래가 없습니다.")
    else:
        plog_rows_t6 = []
        total_pnl = 0.0
        for p in acct_t6["profit_log"]:
            plog_rows_t6.append({
                "날짜":   p["date"],
                "티어":   p.get("tier", "-"),
                "전략":   p.get("strategy", "-"),
                "매수가": f"${p['entry']:.4f}",
                "매도가": f"${p['exit']:.4f}",
                "수량":   p["qty"],
                "손익금": f"${p['pnl']:+,.2f}",
                "수익률": f"{p['pnl_pct']:+.2f}%",
                "구분":   p.get("type", "-"),
            })
            total_pnl += p["pnl"]
        st.dataframe(pd.DataFrame(plog_rows_t6), use_container_width=True, hide_index=True)

        pnl_color = "green" if total_pnl >= 0 else "red"
        st.markdown(
            f"<div style='text-align:right; font-size:1.1em; color:{pnl_color};'>"
            f"누적 실현 손익: <b>${total_pnl:+,.2f}</b></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 사이클 시나리오 비교 ─────────────────────────────────────
    with st.expander("📈 사이클 시나리오 비교"):
        st.caption("동일 사이클 기간 동안 각 전략을 운용했다면 어떻게 됐는지 백테스트 시뮬레이션입니다.")
        try:
            _df_sc6 = load_data_cached()[0]
            _cs6    = pd.Timestamp(acct_t6.get("cycle_start_date", str(date.today())))
            _cdf6   = _df_sc6[_df_sc6.index >= _cs6]
            if len(_cdf6) < 2:
                st.info("사이클 데이터 부족 (최소 2일 필요) — 사이클 시작일을 확인하세요.")
            else:
                _seed6   = float(acct_t6.get("seed", 10000))
                _active6 = acct_t6.get("strategy", "Pro3")
                _sc6_rows = []
                for _sn6, _sp6 in STRATEGIES.items():
                    _r6, _m6 = run_backtest(
                        prices=_cdf6["close"],
                        splits=_sp6["splits"], buy_pct=_sp6["buy_pct"],
                        sell_pct=_sp6["sell_pct"], stop_loss_days=_sp6["stop_loss_days"],
                        buy_on_stop=_sp6.get("buy_on_stop", True),
                        transaction_cost=TRANSACTION_COST,
                        price_stop_loss_pct=PRICE_STOP_LOSS_PCT,
                    )
                    _dollar6 = _seed6 * (1 + _r6 / 100)
                    _sc6_rows.append({
                        "전략":        _sn6,
                        "상태":        "▶ 진행중" if _sn6 == _active6 else "〇 가상",
                        "사이클 수익": f"{_r6:+.2f}%",
                        "사이클 MDD":  f"{_m6:.2f}%",
                        "자산 (시뮬)": f"${_dollar6:,.0f}",
                        "손익 (시뮬)": f"${_dollar6 - _seed6:+,.0f}",
                        "경과 영업일": len(_cdf6),
                    })
                _sc6_df = pd.DataFrame(_sc6_rows)
                st.dataframe(_sc6_df, use_container_width=True, hide_index=True)

                _best6 = max(_sc6_rows, key=lambda x: float(x["사이클 수익"].replace("%", "").replace("+", "")))
                _cur6  = next(r for r in _sc6_rows if r["전략"] == _active6)
                _diff6 = float(_best6["사이클 수익"].replace("%", "").replace("+", "")) - \
                         float(_cur6["사이클 수익"].replace("%", "").replace("+", ""))

                if _best6["전략"] != _active6 and _diff6 > 0.1:
                    st.info(
                        f"💡 이 기간 최고 성과: **{_best6['전략']}** ({_best6['사이클 수익']}) "
                        f"— 현재 운용 중인 {_active6} 대비 **{_diff6:+.2f}%p** 차이"
                    )
                else:
                    st.success(f"✅ 현재 운용 중인 **{_active6}**가 이 기간 최고 또는 동등 성과입니다.")
        except Exception as _e6:
            st.error(f"시나리오 비교 오류: {_e6}")

    st.divider()

    # ── 자산 수익률 요약 — [MOBILE] 4컬럼 → 2+2 ────────────────
    with st.container(border=True):
        st.markdown("#### 💹 자산 수익률 요약")
        init = acct_t6.get("initial_asset", acct_t6["seed"])
        cur  = total_asset
        gain = cur - init
        gain_pct = (gain / init * 100) if init > 0 else 0.0

        # [MOBILE] 4컬럼 → 2+2
        rr1, rr2 = st.columns(2)
        rr1.metric("초기 자산", f"${init:,.0f}")
        rr2.metric("현재 자산", f"${cur:,.0f}")
        rr3, rr4 = st.columns(2)
        rr3.metric("수익금", f"${gain:+,.0f}", delta_color="normal" if gain >= 0 else "inverse")
        rr4.metric("수익률", f"{gain_pct:+.2f}%", delta_color="normal" if gain >= 0 else "inverse")

        new_init = st.number_input("초기 자산 수정 ($)", value=float(init), step=100.0, key="init_asset")
        if st.button("💾 초기 자산 저장", key="save_init", use_container_width=True):
            acct_t6["initial_asset"] = float(new_init)
            save_account(acct_t6)
            st.success("저장됨")
            st.rerun()
