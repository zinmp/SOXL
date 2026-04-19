import numpy as np
import sys
from itertools import product

sys.path.insert(0, 'c:/Users/ze1po/OneDrive/바탕 화면/프로젝트/떨사오팔')
from data import get_data
from backtest import run_backtest

df = get_data(refresh=False)

CB = {'enabled': True, 'threshold': -0.075, 'alloc': 0.15,
      'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3}

YEARS = sorted(df.index.year.unique())

def score(splits):
    rets, mdds = [], []
    for year in YEARS:
        mask = df.index.year == year
        if mask.sum() < 5:
            continue
        ret, mdd = run_backtest(
            prices=df.loc[mask, 'close'], splits=splits,
            buy_pct=-0.007, sell_pct=0.042, stop_loss_days=10, buy_on_stop=True,
            crash_buy=CB, transaction_cost=0.001, price_stop_loss_pct=None,
        )
        rets.append(ret)
        mdds.append(mdd)
    a = np.array(rets)
    s = np.std(a, ddof=1)
    return {
        'cum': round(float(np.sum(a)), 1),
        'sharpe': round((np.mean(a) - 4.5) / s if s > 0 else 0, 2),
        'mdd': round(float(np.min(np.array(mdds))), 1),
    }

results = []

# -----------------------------------------------------------------------
# 1. 6-slot pyramiding: fine-grained search
#    First slot: 0.04~0.10 (step 0.01)
#    Each subsequent slot >= previous slot
#    Sum must equal 1.0
# -----------------------------------------------------------------------
print("Searching 6-slot pyramiding...")

# Enumerate combinations where s1 <= s2 <= s3 <= s4 <= s5 <= s6, sum=1
# Use integer cents (multiply by 100) to avoid float issues
# Range each slot: 3~35 (cent units), total=100

def gen_6slot_pyr(min_first=3, max_first=10):
    """Generate 6-slot splits summing to 100 (in cents) with pyramiding shape."""
    count = 0
    for s1 in range(min_first, max_first + 1):
        for s2 in range(s1, 30):
            for s3 in range(s2, 35):
                for s4 in range(s3, 38):
                    for s5 in range(s4, 40):
                        s6 = 100 - s1 - s2 - s3 - s4 - s5
                        if s6 < s5:
                            break
                        if s6 > 40:
                            continue
                        if s6 < 1:
                            break
                        count += 1
                        yield [s1/100, s2/100, s3/100, s4/100, s5/100, s6/100]
    print(f"  Generated {count} 6-slot candidates")

# Also try non-strict pyramiding but still increasing overall
def gen_6slot_flexible(min_first=4, max_first=9):
    """Slightly flexible - allow small dips but overall increasing."""
    count = 0
    for s1 in range(min_first, max_first + 1):
        for s2 in range(max(s1-1, s1), s1 + 8):
            for s3 in range(s2, s2 + 10):
                for s4 in range(s3, s3 + 10):
                    for s5 in range(s4 - 2, s4 + 10):
                        s6 = 100 - s1 - s2 - s3 - s4 - s5
                        if s6 < 5 or s6 > 40:
                            continue
                        if s5 <= 0 or s6 <= 0:
                            continue
                        count += 1
                        yield [s1/100, s2/100, s3/100, s4/100, s5/100, s6/100]
    print(f"  Generated {count} flexible candidates")

# -----------------------------------------------------------------------
# 2. 5-slot pyramiding (last slot = conceptually 0, but use 5 real slots)
# -----------------------------------------------------------------------
def gen_5slot_pyr(min_first=4, max_first=12):
    count = 0
    for s1 in range(min_first, max_first + 1):
        for s2 in range(s1, 35):
            for s3 in range(s2, 40):
                for s4 in range(s3, 45):
                    s5 = 100 - s1 - s2 - s3 - s4
                    if s5 < s4:
                        break
                    if s5 < 1 or s5 > 50:
                        continue
                    count += 1
                    yield [s1/100, s2/100, s3/100, s4/100, s5/100]
    print(f"  Generated {count} 5-slot candidates")

# Run searches
batch_size = 0

# 6-slot strict pyramiding
for splits in gen_6slot_pyr(3, 10):
    r = score(splits)
    results.append({'splits': splits, **r})
    batch_size += 1

print(f"6-slot strict done: {batch_size} candidates")

batch_size2 = 0
for splits in gen_5slot_pyr(4, 14):
    r = score(splits)
    results.append({'splits': splits, **r})
    batch_size2 += 1

print(f"5-slot done: {batch_size2} candidates")

# -----------------------------------------------------------------------
# Sort and report
# -----------------------------------------------------------------------
results.sort(key=lambda x: -x['sharpe'])

print("\n" + "="*70)
print("=== 세밀 탐색 결과 Top 20 (sharpe 기준) ===")
print(f"{'splits':<50} {'cum':>7} {'sharpe':>7} {'mdd':>7}")
print("-"*70)
for r in results[:20]:
    sp_str = str([round(x, 3) for x in r['splits']])
    print(f"{sp_str:<50} {r['cum']:>6.1f}% {r['sharpe']:>7.2f} {r['mdd']:>6.1f}%")

print("\n" + "="*70)
print("=== 목표 달성 후보 (cum≥630%, mdd≤-50%, sharpe≥1.7) ===")
print(f"{'splits':<50} {'cum':>7} {'sharpe':>7} {'mdd':>7}")
print("-"*70)
targets = [r for r in results if r['cum'] >= 630 and r['mdd'] >= -50 and r['sharpe'] >= 1.7]
targets.sort(key=lambda x: (-x['sharpe'], -x['cum']))
for r in targets[:20]:
    sp_str = str([round(x, 3) for x in r['splits']])
    print(f"{sp_str:<50} {r['cum']:>6.1f}% {r['sharpe']:>7.2f} {r['mdd']:>6.1f}%")

if not targets:
    print("  (없음 - 조건 완화하여 표시)")
    relaxed = [r for r in results if r['cum'] >= 600 and r['mdd'] >= -52 and r['sharpe'] >= 1.7]
    relaxed.sort(key=lambda x: (-x['sharpe'], -x['cum']))
    for r in relaxed[:10]:
        sp_str = str([round(x, 3) for x in r['splits']])
        print(f"{sp_str:<50} {r['cum']:>6.1f}% {r['sharpe']:>7.2f} {r['mdd']:>6.1f}%")

# Champion comparison
champion_result = score([1/6]*6)
print(f"\nChampion (균등): cum={champion_result['cum']}%, sharpe={champion_result['sharpe']}, mdd={champion_result['mdd']}%")

best = results[0]
print("\n" + "="*70)
print("=== 최종 추천 splits (sharpe 최고) ===")
print(f"splits: {[round(x,3) for x in best['splits']]}")
print(f"cum={best['cum']}%, sharpe={best['sharpe']}, mdd={best['mdd']}%")
print(f"Champion 대비: cum {best['cum']-champion_result['cum']:+.1f}%p, mdd {best['mdd']-champion_result['mdd']:+.1f}%p 개선")

# Also find best by cum among those with sharpe>1.7 and mdd>-50
if targets:
    best_target = max(targets, key=lambda x: x['cum'])
    print("\n=== 최종 추천 splits (목표 조건 내 최고 cum) ===")
    print(f"splits: {[round(x,3) for x in best_target['splits']]}")
    print(f"cum={best_target['cum']}%, sharpe={best_target['sharpe']}, mdd={best_target['mdd']}%")
    print(f"Champion 대비: cum {best_target['cum']-champion_result['cum']:+.1f}%p, mdd {best_target['mdd']-champion_result['mdd']:+.1f}%p 개선")

print(f"\nTotal candidates evaluated: {len(results)}")
