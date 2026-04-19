"""
LOC 주문표 생성

LOC (Limit On Close) 매수 주문:
  - 장 마감 직전 limit 가격 이하로 마감할 경우에만 체결
  - limit = 전일 종가 × (1 + buy_pct)
  - 체결 가정가 = limit (보수적 추정)
  - 매도 목표 = 체결가 × (1 + sell_pct)
  - 손절 날짜 = 체결일 + stop_loss_days 영업일
"""
import pandas as pd
from tabulate import tabulate
from typing import Optional
import numpy as np

from config import STRATEGIES


def make_order_table(
    strategy_name: str,
    prev_close: float,
    trade_date: pd.Timestamp,
    seed: float,
    current_slots: Optional[list] = None,
    calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    strategy_name  : 'Pro1' | 'Pro2' | 'Pro3'
    prev_close     : 전일 종가 ($)
    trade_date     : 오늘 날짜 (주문 기준일)
    seed           : 총 투자 시드 ($)
    current_slots  : 현재 활성 슬롯 리스트 (없으면 None)
                     [{'slot': 1, 'entry': 15.2, 'date': Timestamp, 'alloc': 0.167}, ...]
    calendar       : 영업일 캘린더 (없으면 pd.bdate_range 사용)

    Returns
    -------
    주문표 DataFrame
    """
    params   = STRATEGIES[strategy_name]
    splits   = params['splits']
    buy_pct  = params['buy_pct']
    sell_pct = params['sell_pct']
    stop_days = params['stop_loss_days']

    buy_limit   = round(prev_close * (1 + buy_pct), 4)
    sell_target = round(buy_limit * (1 + sell_pct), 4)

    # 손절 날짜 계산
    if calendar is not None:
        future_dates = calendar[calendar > trade_date]
        stop_date = future_dates[stop_days - 1] if len(future_dates) >= stop_days else None
    else:
        bdays = pd.bdate_range(start=trade_date + pd.Timedelta(days=1), periods=stop_days)
        stop_date = bdays[-1] if len(bdays) > 0 else None

    stop_date_str = stop_date.strftime('%Y-%m-%d') if stop_date is not None else '-'

    # ── 신규 매수 주문표 ────────────────────────────────────────
    rows = []
    for i, alloc in enumerate(splits, start=1):
        amount = round(seed * alloc, 2)
        rows.append({
            '티어': i,
            '배분비율': f"{alloc*100:.1f}%",
            '금액($)': f"{amount:,.0f}",
            'LOC 매수 한도': f"${buy_limit:.4f}",
            '매도 목표': f"${sell_target:.4f}",
            '수익률 기준': f"+{sell_pct*100:.2f}%",
            '손절 날짜': stop_date_str,
        })

    df_new = pd.DataFrame(rows)

    # ── 활성 슬롯 현황 (이미 열린 포지션) ──────────────────────
    if current_slots:
        active_rows = []
        for slot in current_slots:
            entry      = slot['entry']
            open_date  = pd.Timestamp(slot['date'])
            alloc      = slot['alloc']
            amount     = round(seed * alloc, 2)

            if calendar is not None:
                days_held = len(calendar[(calendar >= open_date) & (calendar <= trade_date)])
            else:
                days_held = len(pd.bdate_range(start=open_date, end=trade_date))

            remaining = stop_days - days_held
            target_price = round(entry * (1 + sell_pct), 4)

            if calendar is not None:
                future = calendar[calendar > trade_date]
                sl_date = future[remaining - 1] if remaining > 0 and len(future) >= remaining else '손절대상'
            else:
                if remaining > 0:
                    sl_bdays = pd.bdate_range(start=trade_date + pd.Timedelta(days=1), periods=remaining)
                    sl_date  = sl_bdays[-1].strftime('%Y-%m-%d')
                else:
                    sl_date  = '손절대상'

            active_rows.append({
                '티어': slot['slot'],
                '매수일': open_date.strftime('%Y-%m-%d'),
                '매수가': f"${entry:.4f}",
                '금액($)': f"{amount:,.0f}",
                '매도 목표': f"${target_price:.4f}",
                '보유일수': days_held,
                '잔여 손절일': remaining if remaining > 0 else '즉시 손절',
                '손절 날짜': sl_date if isinstance(sl_date, str) else sl_date.strftime('%Y-%m-%d'),
            })

        df_active = pd.DataFrame(active_rows)
    else:
        df_active = None

    return df_new, df_active


def print_order_table(
    strategy_name: str,
    prev_close: float,
    trade_date: pd.Timestamp,
    seed: float,
    current_slots: Optional[list] = None,
    calendar: Optional[pd.DatetimeIndex] = None,
):
    params = STRATEGIES[strategy_name]
    buy_limit = round(prev_close * (1 + params['buy_pct']), 4)

    print(f"\n{'='*60}")
    print(f"[LOC 주문표]  전략: {strategy_name}  |  날짜: {trade_date.date()}")
    print(f"{'='*60}")
    print(f"  전일 종가   : ${prev_close:.4f}")
    print(f"  LOC 매수 한도: ${buy_limit:.4f}  ({params['buy_pct']*100:+.2f}%)")
    print(f"  매도 목표   : +{params['sell_pct']*100:.2f}%")
    print(f"  손절        : {params['stop_loss_days']}영업일")
    print(f"  총 시드     : ${seed:,.0f}")

    df_new, df_active = make_order_table(
        strategy_name, prev_close, trade_date, seed, current_slots, calendar
    )

    print(f"\n▶ 신규 매수 조건 (오늘 종가 ≤ ${buy_limit:.4f} 이면 체결)")
    print(tabulate(df_new, headers='keys', tablefmt='simple', showindex=False))

    if df_active is not None and len(df_active) > 0:
        print(f"\n▶ 현재 활성 포지션")
        print(tabulate(df_active, headers='keys', tablefmt='simple', showindex=False))
    print()
