"""
자연 청산 방식 동적 전략 전환 백테스트

핵심 아이디어:
- 기존 "N일 강제 청산" 방식과 달리 포지션이 자연적으로 모두 청산될 때까지 운용
- 모든 슬롯(일반 + crash) 비활성화 시 → 그 시점 지표로 다음 전략 선택
- 복리: 청산 후 자산이 다음 사이클의 seed

실행: python natural_close_backtest.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json
from data import get_data
from indicators import calc_indicators

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params.json'), 'r') as f:
    params_json = json.load(f)

STRATEGIES = params_json['STRATEGIES']
BACKTEST_START = '2015-01-01'
INITIAL_CAPITAL = 10000.0


def get_strategy_params(name: str) -> dict:
    p = STRATEGIES[name]
    return dict(
        splits=p['splits'],
        buy_pct=p['buy_pct'],
        sell_pct=p['sell_pct'],
        stop_loss_days=p['stop_loss_days'],
        buy_on_stop=p.get('buy_on_stop', True),
        crash_buy=p.get('crash_buy', None),
    )


def run_natural_close_backtest(
    prices: pd.Series,
    ind: pd.DataFrame,
    strategy_selector,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = 0.0,
    max_cycle_days: int = 60,
    min_cycle_days: int = 1,
    verbose: bool = False,
) -> dict:
    """
    자연 청산 방식 동적 전략 전환 백테스트

    Parameters
    ----------
    prices            : 가격 시리즈 (pd.Series, index=날짜)
    ind               : 지표 DataFrame (calc_indicators 결과)
    strategy_selector : (ind_row: pd.Series) -> str 함수
    initial_capital   : 초기 자본 ($)
    transaction_cost  : 편도 거래비용 (기본 0.0)
    max_cycle_days    : 자연 청산 대기 최대 일수 (초과 시 강제 전환, 무한루프 방지)
    min_cycle_days    : 전략 전환 최소 영업일 (너무 잦은 전환 방지)
    verbose           : 사이클별 상세 출력

    Returns
    -------
    dict: {
        'final_capital', 'total_return_pct', 'cagr_pct',
        'total_cycles', 'avg_cycle_days',
        'strategy_counts', 'strategy_pcts',
        'cycle_log': [(날짜, 전략명, 사이클 길이, 수익률%), ...]
    }
    """
    price_arr = prices.values
    dates = prices.index
    n = len(price_arr)
    tc = transaction_cost

    capital = initial_capital
    total_cycles = 0
    cycle_lengths = []
    strategy_counts = {'Pro1': 0, 'Pro2': 0, 'Pro3': 0}
    cycle_log = []

    peak_value_global = capital
    mdd_global = 0.0

    pos = 0  # 현재 날짜 인덱스

    while pos < n - 2:
        # ── 1. 현재 날짜 지표로 전략 선택 ────────────────────────────
        current_date = dates[pos]
        if current_date in ind.index:
            row = ind.loc[current_date]
            if not row[['uptrend', 'rsi']].isna().any():
                strat_name = strategy_selector(row)
            else:
                strat_name = 'Pro3'
        else:
            strat_name = 'Pro3'

        p = get_strategy_params(strat_name)
        splits = p['splits']
        buy_pct = p['buy_pct']
        sell_pct = p['sell_pct']
        stop_loss_days = p['stop_loss_days']
        cb_raw = p['crash_buy']
        cb = cb_raw if (cb_raw and cb_raw.get('enabled')) else None

        # ── 2. 슬롯 초기화 ───────────────────────────────────────────
        slots = [{'active': False, 'entry': 0.0, 'days': 0, 'alloc': s}
                 for s in splits]
        crash_slots = []

        # 이 사이클의 realized_pnl (비율 기준, seed=1.0)
        realized_pnl = 0.0

        cycle_start_pos = pos
        cycle_days_elapsed = 0
        had_any_position = False  # 이번 사이클에 한 번이라도 포지션 진입했는지

        # ── 3. 자연 청산될 때까지 일별 시뮬레이션 ────────────────────
        i = pos
        while i < n - 1:
            price = price_arr[i]
            if i == pos:
                i += 1
                continue

            prev_price = price_arr[i - 1]
            daily_ret = price / prev_price - 1
            effective_exit = price * (1 - tc)

            # 3a. 일반 슬롯 매도
            for s in slots:
                if not s['active']:
                    continue
                profit_ratio = effective_exit / s['entry'] - 1
                if profit_ratio >= sell_pct or s['days'] >= stop_loss_days:
                    realized_pnl += s['alloc'] * profit_ratio
                    s['active'] = False
                    s['days'] = 0

            # 3b. 급락 슬롯 매도
            if cb:
                closed = []
                for s in crash_slots:
                    profit_ratio = effective_exit / s['entry'] - 1
                    if profit_ratio >= cb['sell_pct'] or s['days'] >= cb['stop_loss_days']:
                        realized_pnl += s['alloc'] * profit_ratio
                        closed.append(s)
                for s in closed:
                    crash_slots.remove(s)

            # 3c. 보유 일수 증가
            for s in slots:
                if s['active']:
                    s['days'] += 1
            for s in crash_slots:
                s['days'] += 1

            # 3d. 일반 매수
            if daily_ret <= buy_pct:
                for s in slots:
                    if not s['active']:
                        s['active'] = True
                        s['entry'] = price * (1 + tc)
                        s['days'] = 1
                        had_any_position = True
                        break

            # 3e. 급락 추가매수
            if cb and daily_ret <= cb['threshold']:
                if len(crash_slots) < cb['max_concurrent']:
                    crash_slots.append({
                        'entry': price * (1 + tc),
                        'days': 1,
                        'alloc': cb['alloc'],
                    })
                    had_any_position = True

            # 3f. MDD 추적 (포트폴리오 가치)
            unrealized = sum(
                s['alloc'] * (price / s['entry'] - 1)
                for s in slots if s['active']
            )
            if cb:
                unrealized += sum(
                    s['alloc'] * (price / s['entry'] - 1)
                    for s in crash_slots
                )
            cur_val = capital * (1 + realized_pnl + unrealized)
            if cur_val > peak_value_global:
                peak_value_global = cur_val
            dd = (cur_val - peak_value_global) / peak_value_global
            if dd < mdd_global:
                mdd_global = dd

            cycle_days_elapsed = i - cycle_start_pos

            # ── 자연 청산 조건 체크 ─────────────────────────────────
            # 조건: 1) 한 번이라도 포지션 진입했고 2) 현재 모든 슬롯 비활성
            all_cleared = (
                not any(s['active'] for s in slots)
                and len(crash_slots) == 0
            )

            if had_any_position and all_cleared:
                # 자연 청산 완료 → 사이클 종료
                i += 1
                break

            # 최대 사이클 일수 초과 → 강제 전환 (포지션 강제 청산)
            if cycle_days_elapsed >= max_cycle_days:
                # 강제 청산
                final_price = price
                final_exit = final_price * (1 - tc)
                for s in slots:
                    if s['active']:
                        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
                        s['active'] = False
                for s in crash_slots:
                    realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
                crash_slots.clear()
                i += 1
                break

            i += 1

        else:
            # 데이터 끝: 남은 포지션 강제 청산
            if i < n:
                final_price = price_arr[i - 1]
            else:
                final_price = price_arr[-1]
            final_exit = final_price * (1 - tc)
            for s in slots:
                if s['active']:
                    realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
                    s['active'] = False
            for s in crash_slots:
                realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
            crash_slots.clear()
            i = n  # while 루프 종료 유도

        # ── 4. 복리 적용 ──────────────────────────────────────────────
        cycle_ret_pct = realized_pnl * 100.0
        capital = capital * (1 + realized_pnl)
        strategy_counts[strat_name] = strategy_counts.get(strat_name, 0) + 1
        cycle_lengths.append(cycle_days_elapsed)
        total_cycles += 1

        cycle_log.append((
            current_date.date(),
            strat_name,
            cycle_days_elapsed,
            round(cycle_ret_pct, 2),
        ))

        if verbose:
            print(f"  사이클{total_cycles:3d} [{current_date.date()}] "
                  f"{strat_name} | {cycle_days_elapsed:3d}일 | "
                  f"수익 {cycle_ret_pct:+.2f}% | 자산 ${capital:,.0f}")

        pos = i  # 다음 사이클 시작점

        if pos >= n - 1:
            break

    # ── 5. 결과 계산 ──────────────────────────────────────────────────
    if total_cycles == 0:
        return None

    years = (dates[-1] - dates[0]).days / 365.25
    if years <= 0 or capital <= 0:
        return None

    total_return_pct = (capital - initial_capital) / initial_capital * 100
    cagr = ((capital / initial_capital) ** (1 / years) - 1) * 100
    avg_cycle = np.mean(cycle_lengths) if cycle_lengths else 0

    total_strat = sum(strategy_counts.values())
    strategy_pcts = {
        k: v / total_strat * 100
        for k, v in strategy_counts.items()
    } if total_strat > 0 else {}

    return {
        'final_capital': round(capital, 2),
        'total_return_pct': round(total_return_pct, 2),
        'cagr_pct': round(cagr, 2),
        'mdd_pct': round(mdd_global * 100, 2),
        'total_cycles': total_cycles,
        'avg_cycle_days': round(avg_cycle, 1),
        'strategy_counts': strategy_counts,
        'strategy_pcts': strategy_pcts,
        'cycle_log': cycle_log,
    }


# ── 전략 선택 함수들 ────────────────────────────────────────────────

def selector_simple(ind_row: pd.Series) -> str:
    """단순 규칙: uptrend → Pro2, else → Pro3"""
    uptrend = ind_row['uptrend'] > 0.5
    return 'Pro2' if uptrend else 'Pro3'


def make_rsi3way_selector(rsi_low: float = 40, rsi_high: float = 60):
    """
    RSI 3-way 규칙:
    - uptrend=False AND rsi < rsi_low → Pro3
    - uptrend=False AND rsi >= rsi_low → Pro1
    - uptrend=True → Pro2
    """
    def selector(ind_row: pd.Series) -> str:
        uptrend = ind_row['uptrend'] > 0.5
        rsi = ind_row['rsi']
        if not uptrend:
            return 'Pro3' if rsi < rsi_low else 'Pro1'
        else:
            return 'Pro2'
    return selector


def make_slope_selector(rsi_low: float = 45, slope_threshold: float = 0.0):
    """
    slope 기반 복합 규칙:
    - uptrend=False AND rsi < rsi_low → Pro3
    - uptrend=False AND rsi >= rsi_low → Pro1
    - uptrend=True AND slope > 0 → Pro2
    - uptrend=True AND slope <= 0 → Pro3
    """
    def selector(ind_row: pd.Series) -> str:
        uptrend = ind_row['uptrend'] > 0.5
        rsi = ind_row['rsi']
        slope = ind_row['slope']
        if not uptrend:
            return 'Pro3' if rsi < rsi_low else 'Pro1'
        else:
            return 'Pro2' if slope > slope_threshold else 'Pro3'
    return selector


def run_forced_cycle_backtest(
    prices: pd.Series,
    ind: pd.DataFrame,
    strategy_selector,
    cycle_days: int = 22,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = 0.0,
) -> dict:
    """
    기존 방식: N일 강제 청산 백테스트 (비교 기준)
    experiment_runner.py의 run_dynamic_backtest와 동일 로직
    """
    price_arr = prices.values
    dates = prices.index
    n = len(price_arr)

    from backtest import run_backtest

    capital = initial_capital
    total_cycles = 0
    strategy_counts = {'Pro1': 0, 'Pro2': 0, 'Pro3': 0}

    pos = 0
    while pos < n - 1:
        current_date = dates[pos]
        if current_date in ind.index and not ind.loc[current_date, ['uptrend', 'rsi']].isna().any():
            strat_name = strategy_selector(ind.loc[current_date])
        else:
            strat_name = 'Pro3'

        strategy_counts[strat_name] = strategy_counts.get(strat_name, 0) + 1

        end_pos = min(pos + cycle_days + 1, n)
        cycle_prices = prices.iloc[pos:end_pos]

        if len(cycle_prices) < 2:
            break

        p = get_strategy_params(strat_name)
        cb = p['crash_buy']
        crash = cb if (cb and cb.get('enabled')) else None

        ret_pct, _ = run_backtest(
            prices=cycle_prices,
            splits=p['splits'],
            buy_pct=p['buy_pct'],
            sell_pct=p['sell_pct'],
            stop_loss_days=p['stop_loss_days'],
            buy_on_stop=p['buy_on_stop'],
            crash_buy=crash,
            transaction_cost=transaction_cost,
            price_stop_loss_pct=None,
        )
        capital = capital * (1 + ret_pct / 100.0)
        total_cycles += 1
        pos += cycle_days

    if total_cycles == 0:
        return None

    years = (dates[-1] - dates[0]).days / 365.25
    total_return_pct = (capital - initial_capital) / initial_capital * 100
    cagr = ((capital / initial_capital) ** (1 / years) - 1) * 100
    total_strat = sum(strategy_counts.values())
    strategy_pcts = {k: v / total_strat * 100 for k, v in strategy_counts.items()}

    return {
        'final_capital': round(capital, 2),
        'total_return_pct': round(total_return_pct, 2),
        'cagr_pct': round(cagr, 2),
        'total_cycles': total_cycles,
        'avg_cycle_days': cycle_days,
        'strategy_counts': strategy_counts,
        'strategy_pcts': strategy_pcts,
    }


def print_result(label: str, r: dict, show_cycles: bool = False):
    if r is None:
        print(f"  {label}: 실패")
        return
    pcts = r.get('strategy_pcts', {})
    mdd_str = f"MDD {r['mdd_pct']:+.1f}%" if 'mdd_pct' in r else ""
    print(f"  [{label}]")
    print(f"    최종자산:    ${r['final_capital']:>12,.0f}")
    print(f"    CAGR:        {r['cagr_pct']:>+8.1f}%")
    print(f"    총 수익률:   {r['total_return_pct']:>+8.1f}%")
    if mdd_str:
        print(f"    {mdd_str}")
    print(f"    사이클 수:   {r['total_cycles']:>5}  평균 {r['avg_cycle_days']:.1f}일/사이클")
    print(f"    전략 비율:   Pro1={pcts.get('Pro1',0):.1f}%  Pro2={pcts.get('Pro2',0):.1f}%  Pro3={pcts.get('Pro3',0):.1f}%")
    print()


def main():
    print("=" * 65)
    print("자연 청산 방식 동적 전략 전환 백테스트")
    print(f"초기자본: ${INITIAL_CAPITAL:,.0f}  |  거래비용: 0%")
    print(f"데이터 시작: {BACKTEST_START}")
    print("=" * 65)

    df = get_data()
    df_bt = df[df.index >= BACKTEST_START].copy()
    prices = df_bt['close']

    ind_full = calc_indicators(df)
    ind = ind_full[ind_full.index >= BACKTEST_START]

    print(f"\n데이터: {df_bt.index[0].date()} ~ {df_bt.index[-1].date()} ({len(df_bt)}일)")
    print()

    # ── 비교 기준: 기존 22일 강제 청산 방식 ─────────────────────────
    print("=" * 65)
    print("[ 비교 기준: 기존 22일 강제 청산 방식 ]")
    print("=" * 65)

    forced_simple = run_forced_cycle_backtest(prices, ind, selector_simple, cycle_days=22)
    print_result("22일 강제 / 단순규칙(uptrend→Pro2, else→Pro3)", forced_simple)

    forced_rsi45 = run_forced_cycle_backtest(prices, ind, make_rsi3way_selector(rsi_low=45), cycle_days=22)
    print_result("22일 강제 / RSI3way(rsi_low=45)", forced_rsi45)

    # ── 자연 청산 방식 실험 ──────────────────────────────────────────
    print("=" * 65)
    print("[ 자연 청산 방식: 단순 규칙 (uptrend→Pro2, else→Pro3) ]")
    print("=" * 65)

    nat_simple = run_natural_close_backtest(
        prices, ind, selector_simple,
        max_cycle_days=60, min_cycle_days=1, verbose=False
    )
    print_result("자연청산 / 단순규칙", nat_simple)

    # ── 자연 청산: RSI 3-way 임계값 튜닝 ────────────────────────────
    print("=" * 65)
    print("[ 자연 청산 방식: RSI 3-way 임계값 튜닝 ]")
    print("=" * 65)

    best_cagr = -999
    best_label = None
    best_result = None

    rsi_results = {}
    for rsi_low in [35, 40, 45, 50, 55, 60]:
        selector = make_rsi3way_selector(rsi_low=rsi_low)
        r = run_natural_close_backtest(
            prices, ind, selector,
            max_cycle_days=60, min_cycle_days=1
        )
        label = f"자연청산 / RSI3way rsi_low={rsi_low}"
        rsi_results[label] = r
        print_result(label, r)
        if r and r['cagr_pct'] > best_cagr:
            best_cagr = r['cagr_pct']
            best_label = label
            best_result = r

    # ── 자연 청산: slope 복합 규칙 ──────────────────────────────────
    print("=" * 65)
    print("[ 자연 청산 방식: slope 복합 규칙 ]")
    print("=" * 65)

    slope_results = {}
    for rsi_low in [40, 45, 50]:
        selector = make_slope_selector(rsi_low=rsi_low, slope_threshold=0.0)
        r = run_natural_close_backtest(
            prices, ind, selector,
            max_cycle_days=60, min_cycle_days=1
        )
        label = f"자연청산 / slope+RSI rsi_low={rsi_low}"
        slope_results[label] = r
        print_result(label, r)
        if r and r['cagr_pct'] > best_cagr:
            best_cagr = r['cagr_pct']
            best_label = label
            best_result = r

    # ── max_cycle_days 파라미터 탐색 ────────────────────────────────
    print("=" * 65)
    print("[ 자연 청산 방식: max_cycle_days 파라미터 탐색 (RSI best 기반) ]")
    print("=" * 65)

    best_rsi_low = 45  # 기본값; 위 튜닝에서 최적값으로 조정됨
    max_results = {}
    for max_days in [20, 30, 40, 60, 90]:
        selector = make_rsi3way_selector(rsi_low=best_rsi_low)
        r = run_natural_close_backtest(
            prices, ind, selector,
            max_cycle_days=max_days, min_cycle_days=1
        )
        label = f"자연청산 / RSI3way(rsi_low={best_rsi_low}) max={max_days}일"
        max_results[label] = r
        print_result(label, r)
        if r and r['cagr_pct'] > best_cagr:
            best_cagr = r['cagr_pct']
            best_label = label
            best_result = r

    # ── 최종 종합 요약 ───────────────────────────────────────────────
    print("=" * 65)
    print("[ 최종 종합 비교 요약 ]")
    print("=" * 65)
    print(f"홈페이지 기준: CAGR 41.8%  최종자산 $519,820")
    print()

    all_results = {}
    all_results["22일 강제 / 단순규칙"] = forced_simple
    all_results["22일 강제 / RSI3way(45)"] = forced_rsi45
    all_results["자연청산 / 단순규칙"] = nat_simple
    all_results.update(rsi_results)
    all_results.update(slope_results)
    all_results.update(max_results)

    valid = [(k, v) for k, v in all_results.items() if v is not None]
    sorted_results = sorted(valid, key=lambda x: x[1]['cagr_pct'], reverse=True)

    print(f"{'설정':<50} {'CAGR':>8} {'최종자산':>12} {'사이클':>6} {'평균일':>6}  전략비율(P1/P2/P3)")
    print("-" * 110)
    for label, r in sorted_results[:15]:
        pcts = r.get('strategy_pcts', {})
        print(
            f"{label:<50} "
            f"{r['cagr_pct']:>7.1f}% "
            f"${r['final_capital']:>11,.0f} "
            f"{r['total_cycles']:>6} "
            f"{r['avg_cycle_days']:>5.1f}일  "
            f"P1={pcts.get('Pro1',0):4.1f}% "
            f"P2={pcts.get('Pro2',0):4.1f}% "
            f"P3={pcts.get('Pro3',0):4.1f}%"
        )

    print()
    if best_result:
        target_cagr = 41.8
        target_capital = 519820
        exceed = best_result['final_capital'] > target_capital
        status = "초과 달성" if exceed else "미달"
        print(f"최고 CAGR 설정: [{best_label}]")
        print(f"  CAGR {best_result['cagr_pct']:.1f}%  최종자산 ${best_result['final_capital']:,.0f}")
        pcts = best_result.get('strategy_pcts', {})
        print(f"  Pro1={pcts.get('Pro1',0):.1f}%  Pro2={pcts.get('Pro2',0):.1f}%  Pro3={pcts.get('Pro3',0):.1f}%")
        print(f"  홈페이지(CAGR {target_cagr}%, ${target_capital:,}) 대비: {status}")


if __name__ == '__main__':
    main()
