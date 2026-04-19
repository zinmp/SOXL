"""
떨사오팔 전략 백테스트 엔진

메커니즘:
  - 6개 슬롯(티어), 하루 최대 1개 슬롯 매수 (LOC 기준)
  - 매수 조건: 오늘 종가 / 어제 종가 - 1 <= buy_pct
  - 매도 조건: 현재가 / 매수가 - 1 >= sell_pct
  - 시간 손절: 매수일로부터 stop_loss_days 영업일 경과 시 매도
  - 가격 손절: 진입가 대비 price_stop_loss_pct 이하 하락 시 즉시 매도 [퀀트 추가]
  - 손절일에도 매수 가능 (buy_on_stop=True)
  - 단리: 각 슬롯 규모는 초기 시드의 고정 비율
  - 거래비용: 매수·매도 시 편도 transaction_cost 차감 [퀀트 추가]

[퀀트 수정 사항]
  1. transaction_cost: 편도 거래비용 (수수료+슬리피지). 기본 0.0 (하위호환)
     사이트 비교용은 0.0, 실전 추천은 config.TRANSACTION_COST(0.001) 사용
  2. price_stop_loss_pct: 가격 기반 손절 (-0.15 = -15%).
     10영업일 시간 손절보다 빠른 급락 시 자본 보호.
     기본 None (비활성, 하위호환).
"""
import numpy as np
import pandas as pd
from typing import Tuple


def run_backtest(
    prices: pd.Series,
    splits: list,
    buy_pct: float,
    sell_pct: float,
    stop_loss_days: int,
    buy_on_stop: bool = True,
    crash_buy: dict = None,
    double_buy_threshold: float = None,
    surge_sell_threshold: float = None,
    transaction_cost: float = 0.0,
    price_stop_loss_pct: float = None,
) -> Tuple[float, float]:
    """
    prices                : 평가 구간의 종가 시리즈
    splits                : 각 슬롯의 자금 비율 리스트 (합=1)
    buy_pct               : 매수 트리거 (예: -0.001 = -0.1%)
    sell_pct              : 매도 목표 (예: 0.02 = +2%)
    stop_loss_days        : 시간 손절일 (영업일)
    buy_on_stop           : 손절일에도 매수 허용 여부
    crash_buy             : 급락 추가매수 파라미터 dict
    double_buy_threshold  : 이 이하로 급락하면 1일 2티어 매수 허용
    surge_sell_threshold  : 이 이상 급등하면 수익 포지션 전량 이익실현
    transaction_cost      : 편도 거래비용 비율 (기본 0.0 = 비활성)
                            매수 시 effective_entry = price × (1 + tc)
                            매도 시 effective_exit  = price × (1 - tc)
    price_stop_loss_pct   : 가격 기반 손절 비율 (기본 None = 비활성)
                            effective_entry 대비 이 비율 이하 하락 시 즉시 청산

    Returns
    -------
    (total_return_pct, mdd_pct)
    """
    prices = prices.values

    tc = transaction_cost

    # 슬롯 상태: entry는 거래비용 포함한 실효 매수가
    slots = [{'active': False, 'entry': 0.0, 'days': 0, 'alloc': s} for s in splits]

    cb = crash_buy if (crash_buy and crash_buy.get('enabled')) else None
    crash_slots: list = []

    seed         = 1.0
    realized_pnl = 0.0
    peak_value   = seed
    mdd          = 0.0

    def portfolio_value(current_price: float) -> float:
        """현재 포트폴리오 가치 (미실현: 거래비용 미반영 mark-to-market)"""
        unrealized = sum(
            s['alloc'] * (current_price / s['entry'] - 1)
            for s in slots if s['active']
        )
        if cb:
            unrealized += sum(
                s['alloc'] * (current_price / s['entry'] - 1)
                for s in crash_slots
            )
        return seed + realized_pnl + unrealized

    for i, price in enumerate(prices):
        if i == 0:
            prev_price = price
            continue

        prev_price = prices[i - 1]
        daily_ret  = price / prev_price - 1

        # ── 1. 일반 슬롯 매도 처리 ─────────────────────────────
        surge_day = surge_sell_threshold is not None and daily_ret >= surge_sell_threshold
        effective_exit = price * (1 - tc)

        for s in slots:
            if not s['active']:
                continue
            profit_ratio = effective_exit / s['entry'] - 1
            price_stop   = (
                price_stop_loss_pct is not None
                and (price / s['entry'] - 1) <= price_stop_loss_pct
            )
            sell_triggered = (
                profit_ratio >= sell_pct
                or s['days'] >= stop_loss_days
                or price_stop
                or (surge_day and profit_ratio > 0)
            )
            if sell_triggered:
                realized_pnl += s['alloc'] * profit_ratio
                s['active'] = False
                s['days']   = 0

        # ── 2. 급락 슬롯 매도 처리 ─────────────────────────────
        if cb:
            closed = []
            for s in crash_slots:
                profit_ratio = effective_exit / s['entry'] - 1
                if profit_ratio >= cb['sell_pct'] or s['days'] >= cb['stop_loss_days']:
                    realized_pnl += s['alloc'] * profit_ratio
                    closed.append(s)
            for s in closed:
                crash_slots.remove(s)

        # ── 3. 보유 슬롯 일수 증가 ─────────────────────────────
        for s in slots:
            if s['active']:
                s['days'] += 1
        for s in crash_slots:
            s['days'] += 1

        # ── 4. 일반 매수 처리 (하루 1슬롯) ─────────────────────
        if daily_ret <= buy_pct:
            for s in slots:
                if not s['active']:
                    s['active'] = True
                    s['entry']  = price * (1 + tc)  # 거래비용 반영한 실효 매수가
                    s['days']   = 1
                    break

        # ── 4b. 급락 시 추가 1티어 매수 ────────────────────────
        if double_buy_threshold is not None and daily_ret <= double_buy_threshold:
            for s in slots:
                if not s['active']:
                    s['active'] = True
                    s['entry']  = price * (1 + tc)
                    s['days']   = 1
                    break

        # ── 5. 급락 추가매수 처리 (일반 매수와 독립) ────────────
        if cb and daily_ret <= cb['threshold']:
            if len(crash_slots) < cb['max_concurrent']:
                crash_slots.append({
                    'entry': price * (1 + tc),
                    'days':  1,
                    'alloc': cb['alloc'],
                })

        # ── 6. MDD 추적 ────────────────────────────────────────
        current_val = portfolio_value(price)
        if current_val > peak_value:
            peak_value = current_val
        dd = (current_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

    # ── 평가 기간 종료: 남은 포지션 강제 청산 ──────────────────
    final_price      = prices[-1]
    final_exit       = final_price * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
            s['active'] = False
    for s in crash_slots:
        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
    crash_slots.clear()

    total_return_pct = realized_pnl * 100
    mdd_pct          = mdd * 100

    return round(total_return_pct, 2), round(mdd_pct, 2)
