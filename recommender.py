"""
추천 전략 파이프라인

흐름:
  1. 현재 날짜 지표값 계산
  2. 유사 과거 구간 Top-3 탐색
  3. 각 구간에서 Pro1/2/3 백테스트
  4. 점수 계산
  5. 정배열 여부 고려하여 최적 전략 추천
"""
import math
from typing import Optional

import pandas as pd

from config import STRATEGIES, EVAL_WINDOW, TOP_K
from indicators import calc_indicators, FEATURE_COLS
from similarity import find_similar_periods
from backtest import run_backtest
from scorer import calc_score


def recommend(
    df: pd.DataFrame,
    target_date: Optional[pd.Timestamp] = None,
    verbose: bool = True,
    transaction_cost: float = 0.0,
    price_stop_loss_pct: Optional[float] = None,
) -> dict:
    """
    Parameters
    ----------
    df                  : SOXL 일봉 데이터
    target_date         : 추천 기준일 (None이면 가장 최근 영업일)
    verbose             : 결과 출력 여부
    transaction_cost    : 편도 거래비용 (기본 0.0; 실전 추천 시 TRANSACTION_COST 전달)
    price_stop_loss_pct : 가격 손절 비율 (기본 None; 실전 추천 시 PRICE_STOP_LOSS_PCT 전달)

    Returns
    -------
    {
      'date'        : 기준일,
      'indicators'  : 지표 dict,
      'similar'     : [(date, sim, {strategy: (ret, mdd)}), ...],
      'scores'      : {'Pro1': float, 'Pro2': float, 'Pro3': float},
      'recommended' : 'Pro1' | 'Pro2' | 'Pro3',
      'params'      : 추천 전략 파라미터 dict,
    }
    """
    ind = calc_indicators(df)

    if target_date is None:
        valid_rows = ind[FEATURE_COLS].dropna()
        target_date = valid_rows.index[-1]
    else:
        target_date = pd.Timestamp(target_date)

    # ── 1. 현재 지표 ────────────────────────────────────────────
    cur = ind.loc[target_date]
    indicators = {
        'uptrend':    bool(cur['uptrend'] > 0.5),
        'slope':      round(cur['slope'] * 100, 2),
        'deviation':  round(cur['deviation'] * 100, 2),
        'rsi':        round(cur['rsi'], 2),
        'roc':        round(cur['roc'] * 100, 2),
        'volatility': round(cur['volatility'], 4),
    }

    # ── 2. 유사 구간 ─────────────────────────────────────────────
    similar_periods = find_similar_periods(ind, target_date, top_k=TOP_K)

    # ── 3. 각 유사 구간에서 모든 전략 백테스트 ───────────────────
    similar_results = []
    for past_date, sim in similar_periods:
        past_idx = df.index.get_loc(past_date)
        eval_prices = df['close'].iloc[past_idx: past_idx + EVAL_WINDOW + 1]

        if len(eval_prices) < EVAL_WINDOW:
            continue

        strat_perf = {}
        for name, params in STRATEGIES.items():
            ret, mdd = run_backtest(
                prices              = eval_prices,
                splits              = params['splits'],
                buy_pct             = params['buy_pct'],
                sell_pct            = params['sell_pct'],
                stop_loss_days      = params['stop_loss_days'],
                buy_on_stop         = params['buy_on_stop'],
                transaction_cost    = transaction_cost,
                price_stop_loss_pct = price_stop_loss_pct,
            )
            strat_perf[name] = (ret, mdd)

        similar_results.append((past_date, sim, strat_perf))

    # ── 4. 점수 계산 ─────────────────────────────────────────────
    scores = {}
    for name in STRATEGIES:
        period_data = [
            (sim, perf[name][0], perf[name][1])
            for _, sim, perf in similar_results
            if name in perf
        ]
        scores[name] = calc_score(period_data)

    # ── 5. 추천 전략 선정 (정배열 시 Pro1 제외) ───────────────────
    uptrend = indicators['uptrend']
    eligible = {
        name: score
        for name, score in scores.items()
        if not (STRATEGIES[name]['exclude_uptrend'] and uptrend)
    }
    recommended = max(eligible, key=eligible.get)

    result = {
        'date':        target_date.date(),
        'indicators':  indicators,
        'similar':     similar_results,
        'scores':      scores,
        'recommended': recommended,
        'params':      STRATEGIES[recommended],
    }

    if verbose:
        _print_result(result)

    return result


def _print_result(r: dict):
    ind = r['indicators']
    uptrend_str = 'O' if ind['uptrend'] else 'X'

    print(f"\n{'='*55}")
    print(f"[추천 기준일] {r['date']}")
    print(f"{'='*55}")
    print(f"정배열(20ma-60ma)   : {uptrend_str}")
    print(f"기울기(20ma 10일)   : {ind['slope']:+.2f}%")
    print(f"이격도(주가/20ma)   : {ind['deviation']:+.2f}%")
    print(f"RSI(14)            : {ind['rsi']:.2f}")
    print(f"ROC(12)            : {ind['roc']:+.2f}%")
    print(f"변동성(20day)       : {ind['volatility']:.4f}")

    print(f"\n[유사했던 과거 구간 Top {len(r['similar'])}]")
    for past_date, sim, perf in r['similar']:
        print(f"\n  {past_date.date()}  유사도: {sim:.2f}%")
        for name, (ret, mdd) in perf.items():
            print(f"    {name}: 수익률 {ret:+.1f}%  MDD {mdd:.1f}%")

    print(f"\n[전략별 종합 점수]")
    uptrend = r['indicators']['uptrend']
    for name, score in r['scores'].items():
        excl = " (정배열 제외)" if STRATEGIES[name]['exclude_uptrend'] and uptrend else ""
        print(f"  {name}: {score:.3f}{excl}")

    p = r['params']
    splits_str = ' / '.join(f"{x*100:.1f}%" for x in p['splits'])
    print(f"\n[추천 전략] {r['recommended']}")
    print(f"   분할 비율: {splits_str}")
    print(f"   {len(p['splits'])}분할 {p['stop_loss_days']}일 손절")
    print(f"   매수 기준: {p['buy_pct']*100:+.2f}%")
    print(f"   매도 기준: {p['sell_pct']*100:+.2f}%")
    print(f"   손절일 매수: {'O' if p['buy_on_stop'] else 'X'}")
