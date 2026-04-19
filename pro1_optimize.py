"""
Pro1 전략 파라미터 최적화 스크립트
Sharpe = (mean_annual_ret% - 4.5) / std_annual_ret%  (연도별 수익률 기준)
TC=0.001, price_stop_loss_pct=None
"""
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, 'c:/Users/ze1po/OneDrive/바탕 화면/프로젝트/떨사오팔')
from data import get_data
from backtest import run_backtest

# ── 데이터 로드 (캐시 사용) ─────────────────────────────────────────
df = get_data(refresh=False)
YEARS = sorted(df.index.year.unique())
print(f"데이터 기간: {df.index[0].date()} ~ {df.index[-1].date()}, 연도: {YEARS}")

# ── 파라미터 공간 ────────────────────────────────────────────────────
splits_candidates = [
    [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],   # 현재 Pro1
    [0.05, 0.08, 0.12, 0.18, 0.25, 0.32],
    [0.06, 0.09, 0.13, 0.18, 0.24, 0.30],
    [0.05, 0.08, 0.13, 0.20, 0.24, 0.30],
    [0.06, 0.10, 0.14, 0.20, 0.22, 0.28],
    [0.05, 0.08, 0.12, 0.20, 0.22, 0.33],
    [0.07, 0.10, 0.14, 0.19, 0.22, 0.28],
    [0.05, 0.09, 0.13, 0.19, 0.24, 0.30],
    [0.06, 0.10, 0.15, 0.20, 0.21, 0.28],
    [0.08, 0.12, 0.15, 0.20, 0.21, 0.24],   # Pro3 [B]와 동일
]

buy_pct_candidates  = [-0.002, -0.003, -0.004, -0.005, -0.006]
sell_pct_candidates = [0.003, 0.006, 0.010, 0.015, 0.020, 0.025, 0.030]

crash_buy_candidates = [
    None,
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.10, 'sell_pct': 0.020, 'stop_loss_days': 10, 'max_concurrent': 2},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 2},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3},
    {'enabled': True, 'threshold': -0.100, 'alloc': 0.12, 'sell_pct': 0.020, 'stop_loss_days': 10, 'max_concurrent': 2},
]

TC = 0.001

# ── 평가 함수 ────────────────────────────────────────────────────────
def evaluate(splits, buy_pct, sell_pct, crash_buy=None):
    rets, mdds = [], []
    for year in YEARS:
        mask = df.index.year == year
        if mask.sum() < 5:
            continue
        ret, mdd = run_backtest(
            prices=df.loc[mask, 'close'],
            splits=splits,
            buy_pct=buy_pct,
            sell_pct=sell_pct,
            stop_loss_days=10,
            buy_on_stop=True,
            crash_buy=crash_buy,
            transaction_cost=TC,
            price_stop_loss_pct=None,
        )
        rets.append(ret)
        mdds.append(mdd)
    a = np.array(rets)
    s = np.std(a, ddof=1)
    return {
        'cum':        round(float(np.sum(a)), 1),
        'sharpe':     round((np.mean(a) - 4.5) / s if s > 0 else 0.0, 2),
        'mdd':        round(float(np.min(np.array(mdds))), 1),
        'worst_year': round(float(np.min(a)), 1),
    }

# ── 그리드 탐색 ──────────────────────────────────────────────────────
total = len(splits_candidates) * len(buy_pct_candidates) * len(sell_pct_candidates) * len(crash_buy_candidates)
print(f"\n탐색 조합 수: {total}")

results = []
done = 0
for splits, buy_pct, sell_pct, crash_buy in itertools.product(
        splits_candidates, buy_pct_candidates, sell_pct_candidates, crash_buy_candidates):
    metrics = evaluate(splits, buy_pct, sell_pct, crash_buy)
    if crash_buy is None:
        cb_str = 'None'
    else:
        cb_str = (f"thr={crash_buy['threshold']}, alloc={crash_buy['alloc']}, "
                  f"spct={crash_buy['sell_pct']}, max={crash_buy['max_concurrent']}")
    results.append({
        'splits':    str(splits),
        'buy_pct':   buy_pct,
        'sell_pct':  sell_pct,
        'crash_buy': cb_str,
        **metrics,
    })
    done += 1
    if done % 100 == 0:
        print(f"  {done}/{total} 완료...")

# ── 결과 정렬 및 출력 ────────────────────────────────────────────────
df_res = pd.DataFrame(results).sort_values('sharpe', ascending=False).reset_index(drop=True)

print("\n" + "=" * 120)
print("TOP 15 조합 (Sharpe 기준)")
print("=" * 120)

top15 = df_res.head(15)
col_widths = {
    'splits':    40,
    'buy_pct':   8,
    'sell_pct':  9,
    'crash_buy': 52,
    'cum':       8,
    'sharpe':    8,
    'mdd':       8,
    'worst_year':10,
}
header = (f"{'splits':<40} {'buy_pct':>8} {'sell_pct':>9} {'crash_buy':<52} "
          f"{'cum%':>8} {'sharpe':>8} {'mdd%':>8} {'worst%':>10}")
print(header)
print("-" * 150)
for _, row in top15.iterrows():
    print(f"{row['splits']:<40} {row['buy_pct']:>8.3f} {row['sell_pct']:>9.3f} {row['crash_buy']:<52} "
          f"{row['cum']:>8.1f} {row['sharpe']:>8.2f} {row['mdd']:>8.1f} {row['worst_year']:>10.1f}")

# ── 최종 추천 (Sharpe 최고, mdd > -70 필터) ─────────────────────────
filtered = df_res[df_res['mdd'] > -70].head(1)
if filtered.empty:
    filtered = df_res.head(1)

rec = filtered.iloc[0]
print("\n" + "=" * 120)
print("최종 추천 (Sharpe 최고, MDD > -70% 필터)")
print("=" * 120)
print(f"splits    : {rec['splits']}")
print(f"buy_pct   : {rec['buy_pct']}")
print(f"sell_pct  : {rec['sell_pct']}")
print(f"crash_buy : {rec['crash_buy']}")
print(f"cum%      : {rec['cum']}")
print(f"sharpe    : {rec['sharpe']}")
print(f"mdd%      : {rec['mdd']}")
print(f"worst_yr% : {rec['worst_year']}")

# 현재 Pro1 파라미터 결과도 출력
print("\n" + "=" * 120)
print("현재 Pro1 파라미터 결과 (기준선)")
print("=" * 120)
baseline = evaluate(
    splits=[0.05, 0.1, 0.15, 0.2, 0.25, 0.25],
    buy_pct=-0.003,
    sell_pct=0.002,
    crash_buy=None,
)
print(f"splits    : [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]")
print(f"buy_pct   : -0.003, sell_pct: 0.002, crash_buy: None")
print(f"cum%={baseline['cum']}, sharpe={baseline['sharpe']}, mdd%={baseline['mdd']}, worst_yr%={baseline['worst_year']}")
