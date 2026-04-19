"""
떨사오팔 Pro 레이더 - 독립 실행 버전
사용법:
  python main.py                       # 오늘 기준 추천 + 주문표
  python main.py --date 2026-04-01     # 특정일 기준
  python main.py --seed 100000         # 시드 지정 ($)
  python main.py --refresh             # 데이터 강제 갱신
  python main.py --backtest-annual     # 연도별 백테스트 통계
  python main.py --backtest-quarterly  # 분기별 백테스트 통계
"""
import argparse
import sys
from datetime import datetime

import pandas as pd

from config import TICKER, STRATEGIES, EVAL_WINDOW
from data import get_data
from indicators import calc_indicators
from recommender import recommend
from order_table import print_order_table
from backtest import run_backtest


def run_annual_backtest(df: pd.DataFrame):
    """연도별 백테스트 통계 출력"""
    from tabulate import tabulate

    ind = calc_indicators(df)
    print(f"\n{'='*65}")
    print(f"[연도별] 전략별 연도별 통계 ({TICKER})")
    print(f"{'='*65}")

    years = sorted(df.index.year.unique())
    rows = []

    for year in years:
        year_idx  = df.index[df.index.year == year]
        uptrend_days = ind.loc[year_idx, 'uptrend'].sum()
        uptrend_pct  = int(round(uptrend_days / len(year_idx) * 100))

        row = {'연도': year, '정배열비율': f"{uptrend_pct}%"}
        for name, params in STRATEGIES.items():
            prices = df.loc[year_idx, 'close']
            ret, mdd = run_backtest(
                prices=prices,
                splits=params['splits'],
                buy_pct=params['buy_pct'],
                sell_pct=params['sell_pct'],
                stop_loss_days=params['stop_loss_days'],
                buy_on_stop=params['buy_on_stop'],
            )
            row[f"{name} 수익률"] = f"{ret:+.1f}%"
            row[f"{name} MDD"]    = f"{mdd:.1f}%"
        rows.append(row)

    print(tabulate(rows, headers='keys', tablefmt='simple', showindex=False))


def run_quarterly_backtest(df: pd.DataFrame):
    """분기별 백테스트 통계 출력"""
    from tabulate import tabulate

    ind = calc_indicators(df)
    print(f"\n{'='*65}")
    print(f"[분기별] 전략별 분기별 통계 ({TICKER})")
    print(f"{'='*65}")

    df['_quarter'] = df.index.to_period('Q')
    quarters = sorted(df['_quarter'].unique())
    rows = []

    for q in quarters:
        q_idx = df.index[df['_quarter'] == q]
        uptrend_days = ind.loc[q_idx, 'uptrend'].sum()
        uptrend_pct  = int(round(uptrend_days / len(q_idx) * 100))

        row = {'분기': str(q).replace('Q', ' Q'), '정배열비율': f"{uptrend_pct}%"}
        for name, params in STRATEGIES.items():
            prices = df.loc[q_idx, 'close']
            ret, mdd = run_backtest(
                prices=prices,
                splits=params['splits'],
                buy_pct=params['buy_pct'],
                sell_pct=params['sell_pct'],
                stop_loss_days=params['stop_loss_days'],
                buy_on_stop=params['buy_on_stop'],
            )
            row[f"{name} 수익률"] = f"{ret:+.1f}%"
            row[f"{name} MDD"]    = f"{mdd:.1f}%"
        rows.append(row)

    df.drop(columns=['_quarter'], inplace=True)
    print(tabulate(rows, headers='keys', tablefmt='simple', showindex=False))


def main():
    parser = argparse.ArgumentParser(description='떨사오팔 Pro 레이더')
    parser.add_argument('--date',               type=str,   default=None,    help='기준일 (YYYY-MM-DD)')
    parser.add_argument('--seed',               type=float, default=10000.0, help='투자 시드 ($)')
    parser.add_argument('--refresh',            action='store_true',         help='데이터 강제 갱신')
    parser.add_argument('--backtest-annual',    action='store_true',         help='연도별 백테스트')
    parser.add_argument('--backtest-quarterly', action='store_true',         help='분기별 백테스트')
    parser.add_argument('--order-only',         action='store_true',         help='주문표만 출력')
    parser.add_argument('--strategy',           type=str,   default=None,    help='전략 강제 지정 (Pro1|Pro2|Pro3)')
    args = parser.parse_args()

    # ── 데이터 로드 ─────────────────────────────────────────────
    df = get_data(refresh=args.refresh)

    if args.backtest_annual:
        run_annual_backtest(df)
        return

    if args.backtest_quarterly:
        run_quarterly_backtest(df)
        return

    # ── 기준일 설정 ─────────────────────────────────────────────
    if args.date:
        target_date = pd.Timestamp(args.date)
    else:
        ind = calc_indicators(df)
        from indicators import FEATURE_COLS
        valid = ind[FEATURE_COLS].dropna()
        target_date = valid.index[-1]

    # ── 추천 실행 ───────────────────────────────────────────────
    if not args.order_only:
        result = recommend(df, target_date=target_date, verbose=True)
        strategy_name = args.strategy or result['recommended']
    else:
        strategy_name = args.strategy or 'Pro3'
        result = None

    # ── 주문표 출력 ─────────────────────────────────────────────
    # 전일 종가 가져오기
    target_idx = df.index.get_loc(target_date)
    if target_idx == 0:
        print("오류: 전일 종가 데이터 없음")
        return

    prev_close = float(df['close'].iloc[target_idx - 1])

    print_order_table(
        strategy_name = strategy_name,
        prev_close    = prev_close,
        trade_date    = target_date,
        seed          = args.seed,
        calendar      = df.index,
    )


if __name__ == '__main__':
    main()
