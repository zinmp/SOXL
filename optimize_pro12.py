"""
Pro1 / Pro2 파라미터 최적화 스크립트
목표: 수익률 개선 + Sharpe 비율 향상

Pro1 특성: exclude_uptrend=True (정배열 시 매수 자제)
Pro2 특성: exclude_uptrend=False (항상 매수 가능)
Pro3 [B] 기준: splits=[0.08,0.12,0.15,0.20,0.21,0.24], buy=-0.007, sell=0.042, crash_buy

기준: Sharpe 최고 & cum 최대화
TC=0.001 (0.1% 편도)
"""
import sys
import numpy as np

sys.path.insert(0, 'c:/Users/ze1po/OneDrive/바탕 화면/프로젝트/떨사오팔')
from data import get_data
from backtest import run_backtest

df = get_data(refresh=False)
YEARS = sorted(df.index.year.unique())
RISK_FREE = 4.5  # %


def sharpe_score(splits, buy_pct, sell_pct, crash_buy=None, tc=0.001):
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
            transaction_cost=tc,
            price_stop_loss_pct=None,
        )
        rets.append(ret)
        mdds.append(mdd)
    a = np.array(rets)
    s = np.std(a, ddof=1)
    return {
        'cum': round(float(np.sum(a)), 1),
        'mean': round(float(np.mean(a)), 2),
        'sharpe': round((np.mean(a) - RISK_FREE) / s if s > 0 else 0, 2),
        'mdd': round(float(np.min(np.array(mdds))), 1),
        'worst_year': round(float(np.min(a)), 1),
    }


# ─── Pro3 [B] 기준선 ──────────────────────────────────────────────
PRO3_B = {'enabled': True, 'threshold': -0.075, 'alloc': 0.15,
          'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3}
pro3_ref = sharpe_score(
    [0.08, 0.12, 0.15, 0.20, 0.21, 0.24],
    buy_pct=-0.007, sell_pct=0.042,
    crash_buy=PRO3_B
)
print(f"Pro3 [B] 기준: cum={pro3_ref['cum']}%, sharpe={pro3_ref['sharpe']}, mdd={pro3_ref['mdd']}%\n")

# ─── 공통: 후방 가중 분할 후보 생성 ────────────────────────────────────────
def gen_splits():
    """후방 가중 6-slot 조합 (s1<=s2<=...<=s6, sum=1)"""
    for s1 in range(3, 12):         # 3~11%
        for s2 in range(s1, 22):    # s1~21%
            for s3 in range(s2, 28):
                for s4 in range(s3, 32):
                    for s5 in range(s4, 36):
                        s6 = 100 - s1 - s2 - s3 - s4 - s5
                        if s6 < s5 or s6 > 40 or s6 < 1:
                            continue
                        yield [x/100 for x in [s1, s2, s3, s4, s5, s6]]


# ═══════════════════════════════════════════════════════════════════
# Pro1 최적화
# exclude_uptrend=True → 정배열 시 보수적. 작은 buy_pct, 작은 sell_pct
# crash_buy는 독립 슬롯이라 exclude_uptrend와 병행 가능
# ═══════════════════════════════════════════════════════════════════
print("="*60)
print("Pro1 파라미터 탐색 시작...")
print("="*60)

PRO1_BUY  = [-0.002, -0.003, -0.004, -0.005, -0.006]
PRO1_SELL = [0.003, 0.006, 0.010, 0.015, 0.020, 0.025, 0.030]

# crash_buy 후보 (Pro1은 보수적 → threshold 더 낮게, alloc 작게)
PRO1_CB_VARIANTS = [
    None,
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.10, 'sell_pct': 0.020, 'stop_loss_days': 10, 'max_concurrent': 2},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 2},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3},
    {'enabled': True, 'threshold': -0.100, 'alloc': 0.12, 'sell_pct': 0.020, 'stop_loss_days': 10, 'max_concurrent': 2},
]

# 현재 Pro1 기준선
pro1_current = sharpe_score(
    [0.05, 0.1, 0.15, 0.2, 0.25, 0.25],
    buy_pct=-0.003, sell_pct=0.002, crash_buy=None
)
print(f"Pro1 현재: cum={pro1_current['cum']}%, sharpe={pro1_current['sharpe']}, mdd={pro1_current['mdd']}%")

# 고속 탐색: splits 후보를 미리 샘플링 (계산 시간 제한)
splits_pool = list(gen_splits())
# 대표 후보만 선택 (전체 탐색 시 너무 많음)
import random
random.seed(42)
if len(splits_pool) > 200:
    splits_sample = random.sample(splits_pool, 200)
else:
    splits_sample = splits_pool

# 고정 splits로 먼저 buy/sell/crash 탐색
SPLITS_CANDIDATES_P1 = [
    [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],  # 현재
    [0.05, 0.08, 0.12, 0.18, 0.25, 0.32],  # 후방가중 소폭
    [0.06, 0.09, 0.13, 0.18, 0.24, 0.30],
    [0.05, 0.08, 0.13, 0.20, 0.24, 0.30],
    [0.06, 0.10, 0.14, 0.20, 0.22, 0.28],
    [0.05, 0.08, 0.12, 0.20, 0.22, 0.33],
    [0.07, 0.10, 0.14, 0.19, 0.22, 0.28],
    [0.05, 0.09, 0.13, 0.19, 0.24, 0.30],
    [0.06, 0.10, 0.15, 0.20, 0.21, 0.28],
    [0.08, 0.12, 0.15, 0.20, 0.21, 0.24],  # Pro3[B]와 동일
]

pro1_results = []
total_p1 = len(SPLITS_CANDIDATES_P1) * len(PRO1_BUY) * len(PRO1_SELL) * len(PRO1_CB_VARIANTS)
done = 0
for splits in SPLITS_CANDIDATES_P1:
    for buy in PRO1_BUY:
        for sell in PRO1_SELL:
            for cb in PRO1_CB_VARIANTS:
                r = sharpe_score(splits, buy, sell, crash_buy=cb)
                r['splits'] = splits
                r['buy'] = buy
                r['sell'] = sell
                r['crash_buy'] = cb is not None
                pro1_results.append(r)
                done += 1

pro1_results.sort(key=lambda x: (-x['sharpe'], -x['cum']))

print(f"\nPro1 탐색 완료: {done}개 조합")
print(f"\n{'splits':<42} {'buy':>7} {'sell':>6} {'cb':>3} {'cum':>7} {'sharpe':>7} {'mdd':>7} {'worst':>7}")
print("-"*90)
seen = set()
count = 0
for r in pro1_results:
    key = (tuple(r['splits']), r['buy'], r['sell'], r['crash_buy'])
    if key in seen:
        continue
    seen.add(key)
    sp = str([round(x, 2) for x in r['splits']])
    cb_str = 'Y' if r['crash_buy'] else 'N'
    print(f"{sp:<42} {r['buy']:>7.3f} {r['sell']:>6.3f} {cb_str:>3} {r['cum']:>6.1f}% {r['sharpe']:>7.2f} {r['mdd']:>6.1f}% {r['worst_year']:>6.1f}%")
    count += 1
    if count >= 20:
        break

best_pro1 = pro1_results[0]
print(f"\nPro1 최적: splits={[round(x,2) for x in best_pro1['splits']]}, buy={best_pro1['buy']}, sell={best_pro1['sell']}, crash_buy={best_pro1['crash_buy']}")
print(f"  cum={best_pro1['cum']}%, sharpe={best_pro1['sharpe']}, mdd={best_pro1['mdd']}%, worst_year={best_pro1['worst_year']}%")
print(f"  현재 대비: cum {best_pro1['cum']-pro1_current['cum']:+.1f}%p, sharpe {best_pro1['sharpe']-pro1_current['sharpe']:+.2f}")


# ═══════════════════════════════════════════════════════════════════
# Pro2 최적화
# exclude_uptrend=False → 모든 시장 조건에서 매수. 중간 공격성.
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Pro2 파라미터 탐색 시작...")
print("="*60)

PRO2_BUY  = [-0.002, -0.003, -0.004, -0.005, -0.006, -0.007]
PRO2_SELL = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]

PRO2_CB_VARIANTS = [
    None,
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.12, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 2},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.030, 'stop_loss_days': 10, 'max_concurrent': 3},
    {'enabled': True, 'threshold': -0.075, 'alloc': 0.20, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3},
]

pro2_current = sharpe_score(
    [0.1, 0.15, 0.2, 0.25, 0.2, 0.1],
    buy_pct=-0.002, sell_pct=0.018, crash_buy=None
)
print(f"Pro2 현재: cum={pro2_current['cum']}%, sharpe={pro2_current['sharpe']}, mdd={pro2_current['mdd']}%")

SPLITS_CANDIDATES_P2 = [
    [0.1, 0.15, 0.2, 0.25, 0.2, 0.1],    # 현재 (비대칭 종형)
    [0.08, 0.12, 0.15, 0.20, 0.21, 0.24], # Pro3 [B]
    [0.07, 0.10, 0.14, 0.19, 0.23, 0.27],
    [0.07, 0.11, 0.15, 0.20, 0.22, 0.25],
    [0.08, 0.12, 0.16, 0.20, 0.22, 0.22],
    [0.06, 0.10, 0.14, 0.20, 0.23, 0.27],
    [0.08, 0.12, 0.15, 0.19, 0.22, 0.24],
    [0.07, 0.10, 0.15, 0.20, 0.23, 0.25],
    [0.09, 0.13, 0.16, 0.20, 0.21, 0.21],
    [0.06, 0.10, 0.15, 0.20, 0.22, 0.27],
]

pro2_results = []
done2 = 0
for splits in SPLITS_CANDIDATES_P2:
    for buy in PRO2_BUY:
        for sell in PRO2_SELL:
            for cb in PRO2_CB_VARIANTS:
                r = sharpe_score(splits, buy, sell, crash_buy=cb)
                r['splits'] = splits
                r['buy'] = buy
                r['sell'] = sell
                r['crash_buy'] = cb is not None
                pro2_results.append(r)
                done2 += 1

pro2_results.sort(key=lambda x: (-x['sharpe'], -x['cum']))

print(f"\nPro2 탐색 완료: {done2}개 조합")
print(f"\n{'splits':<42} {'buy':>7} {'sell':>6} {'cb':>3} {'cum':>7} {'sharpe':>7} {'mdd':>7} {'worst':>7}")
print("-"*90)
seen2 = set()
count2 = 0
for r in pro2_results:
    key = (tuple(r['splits']), r['buy'], r['sell'], r['crash_buy'])
    if key in seen2:
        continue
    seen2.add(key)
    sp = str([round(x, 2) for x in r['splits']])
    cb_str = 'Y' if r['crash_buy'] else 'N'
    print(f"{sp:<42} {r['buy']:>7.3f} {r['sell']:>6.3f} {cb_str:>3} {r['cum']:>6.1f}% {r['sharpe']:>7.2f} {r['mdd']:>6.1f}% {r['worst_year']:>6.1f}%")
    count2 += 1
    if count2 >= 20:
        break

best_pro2 = pro2_results[0]
print(f"\nPro2 최적: splits={[round(x,2) for x in best_pro2['splits']]}, buy={best_pro2['buy']}, sell={best_pro2['sell']}, crash_buy={best_pro2['crash_buy']}")
print(f"  cum={best_pro2['cum']}%, sharpe={best_pro2['sharpe']}, mdd={best_pro2['mdd']}%, worst_year={best_pro2['worst_year']}%")
print(f"  현재 대비: cum {best_pro2['cum']-pro2_current['cum']:+.1f}%p, sharpe {best_pro2['sharpe']-pro2_current['sharpe']:+.2f}")

print("\n" + "="*60)
print("=== 최종 요약 ===")
print("="*60)
print(f"Pro1 현재:  cum={pro1_current['cum']}%, sharpe={pro1_current['sharpe']}, mdd={pro1_current['mdd']}%")
print(f"Pro1 최적:  cum={best_pro1['cum']}%, sharpe={best_pro1['sharpe']}, mdd={best_pro1['mdd']}%  ← splits={[round(x,2) for x in best_pro1['splits']]}, buy={best_pro1['buy']}, sell={best_pro1['sell']}, cb={best_pro1['crash_buy']}")
print()
print(f"Pro2 현재:  cum={pro2_current['cum']}%, sharpe={pro2_current['sharpe']}, mdd={pro2_current['mdd']}%")
print(f"Pro2 최적:  cum={best_pro2['cum']}%, sharpe={best_pro2['sharpe']}, mdd={best_pro2['mdd']}%  ← splits={[round(x,2) for x in best_pro2['splits']]}, buy={best_pro2['buy']}, sell={best_pro2['sell']}, cb={best_pro2['crash_buy']}")
print()
print(f"Pro3 [B] 기준: cum={pro3_ref['cum']}%, sharpe={pro3_ref['sharpe']}, mdd={pro3_ref['mdd']}%")
