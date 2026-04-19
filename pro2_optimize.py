"""
Pro2 파라미터 최적화 스크립트
- Sharpe = (mean_annual_ret% - 4.5) / std_annual_ret%
- TC = 0.001 (0.1% 편도)
- price_stop_loss_pct = None
"""
import sys
import itertools
import numpy as np

sys.path.insert(0, 'c:/Users/ze1po/OneDrive/바탕 화면/프로젝트/떨사오팔')
from data import get_data
from backtest import run_backtest

df = get_data(refresh=False)
YEARS = sorted(df.index.year.unique())
print(f"데이터 기간: {df.index[0].date()} ~ {df.index[-1].date()}, 연도 수: {len(YEARS)}")

# ── 파라미터 후보 정의 ─────────────────────────────────────────────────────────

splits_candidates = [
    [0.1, 0.15, 0.2, 0.25, 0.2, 0.1],      # 현재 (종형)
    [0.08, 0.12, 0.15, 0.20, 0.21, 0.24],  # Pro3 [B]와 동일
    [0.07, 0.10, 0.14, 0.19, 0.23, 0.27],
    [0.07, 0.11, 0.15, 0.20, 0.22, 0.25],
    [0.08, 0.12, 0.16, 0.20, 0.22, 0.22],
    [0.06, 0.10, 0.14, 0.20, 0.23, 0.27],
    [0.08, 0.12, 0.15, 0.19, 0.22, 0.24],
    [0.07, 0.10, 0.15, 0.20, 0.23, 0.25],
    [0.09, 0.13, 0.16, 0.20, 0.21, 0.21],
    [0.06, 0.10, 0.15, 0.20, 0.22, 0.27],
    [0.08, 0.12, 0.16, 0.21, 0.20, 0.23],
    [0.07, 0.11, 0.14, 0.20, 0.22, 0.26],
    # 추가 후보
    [0.06, 0.09, 0.13, 0.20, 0.24, 0.28],
    [0.07, 0.10, 0.14, 0.20, 0.24, 0.25],
    [0.08, 0.11, 0.15, 0.20, 0.22, 0.24],
]

buy_pct_candidates = [-0.002, -0.003, -0.004, -0.005, -0.006, -0.007]

sell_pct_candidates = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]

crash_buy_candidates = [
    None,
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.12, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 2},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.030, 'stop_loss_days': 10, 'max_concurrent': 3},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.20, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3},
]

TC = 0.001

total_combinations = (
    len(splits_candidates)
    * len(buy_pct_candidates)
    * len(sell_pct_candidates)
    * len(crash_buy_candidates)
)
print(f"총 조합 수: {total_combinations:,}")

# ── 평가 함수 ─────────────────────────────────────────────────────────────────

def evaluate(splits, buy_pct, sell_pct, crash_buy=None):
    rets = []
    mdds = []
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
    sharpe = (np.mean(a) - 4.5) / s if s > 0 else 0.0

    return {
        'cum': round(float(np.sum(a)), 1),
        'sharpe': round(sharpe, 2),
        'mdd': round(float(np.min(np.array(mdds))), 1),
        'worst_year': round(float(np.min(a)), 1),
        'mean_annual': round(float(np.mean(a)), 2),
    }


# ── 그리드 탐색 ───────────────────────────────────────────────────────────────

results = []
done = 0
report_interval = max(1, total_combinations // 20)

for splits, buy_pct, sell_pct, crash_buy in itertools.product(
    splits_candidates,
    buy_pct_candidates,
    sell_pct_candidates,
    crash_buy_candidates,
):
    metrics = evaluate(splits, buy_pct, sell_pct, crash_buy)
    metrics.update({
        'splits': splits,
        'buy_pct': buy_pct,
        'sell_pct': sell_pct,
        'crash_buy': crash_buy,
    })
    results.append(metrics)
    done += 1
    if done % report_interval == 0 or done == total_combinations:
        print(f"  진행: {done}/{total_combinations} ({100*done/total_combinations:.0f}%)")

# ── 정렬 및 출력 ──────────────────────────────────────────────────────────────

results.sort(key=lambda x: x['sharpe'], reverse=True)

print("\n" + "=" * 130)
print(f"{'#':>3}  {'splits':40} {'buy_pct':>8} {'sell_pct':>9} {'crash_buy':>35} {'cum%':>7} {'sharpe':>7} {'mdd%':>7} {'worst_yr%':>10}")
print("=" * 130)

for rank, r in enumerate(results[:15], 1):
    cb = r['crash_buy']
    if cb is None:
        cb_str = "None"
    else:
        cb_str = f"alloc={cb['alloc']},sp={cb['sell_pct']},mc={cb['max_concurrent']}"

    splits_str = str([round(v, 2) for v in r['splits']])
    print(
        f"{rank:>3}  {splits_str:40} {r['buy_pct']:>8.3f} {r['sell_pct']:>9.3f} "
        f"{cb_str:>35} {r['cum']:>7.1f} {r['sharpe']:>7.2f} {r['mdd']:>7.1f} {r['worst_year']:>10.1f}"
    )

print("=" * 130)

# ── 최종 추천 ─────────────────────────────────────────────────────────────────

best = results[0]
print("\n[최종 추천 파라미터]")
print(f"  splits      : {best['splits']}")
print(f"  buy_pct     : {best['buy_pct']}")
print(f"  sell_pct    : {best['sell_pct']}")
print(f"  crash_buy   : {best['crash_buy']}")
print(f"  cum%        : {best['cum']}")
print(f"  sharpe      : {best['sharpe']}")
print(f"  mdd%        : {best['mdd']}")
print(f"  worst_year% : {best['worst_year']}")
print(f"  mean_annual%: {best['mean_annual']}")

# ── 현재 Pro2 베이스라인 비교 ──────────────────────────────────────────────────

print("\n[현재 Pro2 베이스라인]")
baseline = evaluate(
    splits=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1],
    buy_pct=-0.002,
    sell_pct=0.018,
    crash_buy=None,
)
print(f"  cum%        : {baseline['cum']}")
print(f"  sharpe      : {baseline['sharpe']}")
print(f"  mdd%        : {baseline['mdd']}")
print(f"  worst_year% : {baseline['worst_year']}")
print(f"  mean_annual%: {baseline['mean_annual']}")
