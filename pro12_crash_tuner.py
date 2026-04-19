"""
Pro1 & Pro2 crash_buy 파라미터 세밀 탐색 스크립트
"""
import sys
import json
import itertools
import numpy as np

sys.path.insert(0, 'c:/Users/ze1po/OneDrive/바탕 화면/프로젝트/떨사오팔')
from data import get_data
from backtest import run_backtest

df = get_data(refresh=False)
YEARS = sorted(df.index.year.unique())


def score(splits, buy_pct, sell_pct, crash_buy=None, tc=0.001):
    rets, mdds = [], []
    for year in YEARS:
        mask = df.index.year == year
        if mask.sum() < 5:
            continue
        ret, mdd = run_backtest(
            prices=df.loc[mask, 'close'],
            splits=splits, buy_pct=buy_pct, sell_pct=sell_pct,
            stop_loss_days=10, buy_on_stop=True,
            crash_buy=crash_buy,
            transaction_cost=tc, price_stop_loss_pct=None,
        )
        rets.append(ret)
        mdds.append(mdd)
    a = np.array(rets)
    s = np.std(a, ddof=1)
    return {
        'cum': round(float(np.sum(a)), 1),
        'sharpe': round((np.mean(a) - 4.5) / s if s > 0 else 0, 2),
        'mdd': round(float(np.min(np.array(mdds))), 1),
        'worst_year': round(float(np.min(a)), 1),
    }


# ─────────────────────────────────────────────
# PRO1 탐색
# ─────────────────────────────────────────────
print("=" * 60)
print("PRO1 탐색 시작")
print("=" * 60)

PRO1_SPLITS_A = [0.05, 0.08, 0.12, 0.18, 0.25, 0.32]
PRO1_SPLITS_B = [0.06, 0.10, 0.14, 0.20, 0.22, 0.28]
PRO1_BUY = [-0.003, -0.004, -0.005]
PRO1_SELL = [0.010, 0.015, 0.020, 0.025]

# ── Step 1: base params 탐색 (crash_buy 없이) ──
print("\n[Pro1] Step1: Base params 탐색 (no crash_buy)...")
pro1_base_results = []
for splits, buy, sell in itertools.product(
    [PRO1_SPLITS_A, PRO1_SPLITS_B], PRO1_BUY, PRO1_SELL
):
    r = score(splits, buy, sell, crash_buy=None)
    pro1_base_results.append({
        'splits': splits,
        'buy': buy,
        'sell': sell,
        **r,
    })

pro1_base_results.sort(key=lambda x: x['sharpe'], reverse=True)
top5_pro1 = pro1_base_results[:5]
print("  Top5 base (sharpe 기준):")
for i, r in enumerate(top5_pro1):
    print(f"    [{i+1}] splits={'A' if r['splits']==PRO1_SPLITS_A else 'B'} "
          f"buy={r['buy']} sell={r['sell']} "
          f"→ cum={r['cum']}% sharpe={r['sharpe']} mdd={r['mdd']}% worst={r['worst_year']}%")

# ── Step 2: crash_buy 세밀 탐색 ──
CB_THRESHOLD = [-0.060, -0.075, -0.090, -0.100, -0.120]
CB_ALLOC = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
CB_SELL = [0.015, 0.020, 0.025, 0.030, 0.035]
CB_MAX_CONCURRENT = [1, 2, 3]

# 상위 3개 base로 crash_buy 탐색
print("\n[Pro1] Step2: crash_buy 세밀 탐색 (상위 3개 base 기준)...")
pro1_cb_results = []
total_combos = 3 * len(CB_THRESHOLD) * len(CB_ALLOC) * len(CB_SELL) * len(CB_MAX_CONCURRENT)
print(f"  총 조합 수: {total_combos}")

count = 0
for base in top5_pro1[:3]:
    for thr, alloc, csell, maxc in itertools.product(
        CB_THRESHOLD, CB_ALLOC, CB_SELL, CB_MAX_CONCURRENT
    ):
        cb = {
            'enabled': True,
            'threshold': thr,
            'alloc': alloc,
            'sell_pct': csell,
            'max_concurrent': maxc,
            'stop_loss_days': 10,
        }
        r = score(base['splits'], base['buy'], base['sell'], crash_buy=cb)
        pro1_cb_results.append({
            'splits': base['splits'],
            'buy': base['buy'],
            'sell': base['sell'],
            'crash_buy': cb,
            **r,
        })
        count += 1
        if count % 100 == 0:
            print(f"    진행: {count}/{total_combos}")

pro1_cb_results.sort(key=lambda x: x['sharpe'], reverse=True)
print("\n[Pro1] crash_buy 탐색 완료. Top5:")
for i, r in enumerate(pro1_cb_results[:5]):
    cb = r['crash_buy']
    splits_name = 'A' if r['splits'] == PRO1_SPLITS_A else 'B'
    print(f"  [{i+1}] splits={splits_name} buy={r['buy']} sell={r['sell']} "
          f"cb_thr={cb['threshold']} cb_alloc={cb['alloc']} cb_sell={cb['sell_pct']} cb_maxc={cb['max_concurrent']}")
    print(f"       → cum={r['cum']}% sharpe={r['sharpe']} mdd={r['mdd']}% worst={r['worst_year']}%")

pro1_best = pro1_cb_results[0]
# 같은 base로 no crash_buy 비교
pro1_base_nocb = score(pro1_best['splits'], pro1_best['buy'], pro1_best['sell'], crash_buy=None)

print("\n[Pro1] crash_buy enabled vs disabled 비교:")
print(f"  with crash_buy : cum={pro1_best['cum']}% sharpe={pro1_best['sharpe']} mdd={pro1_best['mdd']}% worst={pro1_best['worst_year']}%")
print(f"  without crash  : cum={pro1_base_nocb['cum']}% sharpe={pro1_base_nocb['sharpe']} mdd={pro1_base_nocb['mdd']}% worst={pro1_base_nocb['worst_year']}%")


# ─────────────────────────────────────────────
# PRO2 탐색
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRO2 탐색 시작")
print("=" * 60)

PRO2_SPLITS_A = [0.08, 0.12, 0.15, 0.20, 0.21, 0.24]
PRO2_SPLITS_B = [0.07, 0.10, 0.14, 0.19, 0.23, 0.27]
PRO2_BUY = [-0.004, -0.005, -0.006]
PRO2_SELL = [0.020, 0.025, 0.030, 0.035]

# ── Step 1: base params 탐색 ──
print("\n[Pro2] Step1: Base params 탐색 (no crash_buy)...")
pro2_base_results = []
for splits, buy, sell in itertools.product(
    [PRO2_SPLITS_A, PRO2_SPLITS_B], PRO2_BUY, PRO2_SELL
):
    r = score(splits, buy, sell, crash_buy=None)
    pro2_base_results.append({
        'splits': splits,
        'buy': buy,
        'sell': sell,
        **r,
    })

pro2_base_results.sort(key=lambda x: x['sharpe'], reverse=True)
top5_pro2 = pro2_base_results[:5]
print("  Top5 base (sharpe 기준):")
for i, r in enumerate(top5_pro2):
    print(f"    [{i+1}] splits={'A' if r['splits']==PRO2_SPLITS_A else 'B'} "
          f"buy={r['buy']} sell={r['sell']} "
          f"→ cum={r['cum']}% sharpe={r['sharpe']} mdd={r['mdd']}% worst={r['worst_year']}%")

# ── Step 2: crash_buy 세밀 탐색 ──
print("\n[Pro2] Step2: crash_buy 세밀 탐색 (상위 3개 base 기준)...")
pro2_cb_results = []
total_combos2 = 3 * len(CB_THRESHOLD) * len(CB_ALLOC) * len(CB_SELL) * len(CB_MAX_CONCURRENT)
print(f"  총 조합 수: {total_combos2}")

count = 0
for base in top5_pro2[:3]:
    for thr, alloc, csell, maxc in itertools.product(
        CB_THRESHOLD, CB_ALLOC, CB_SELL, CB_MAX_CONCURRENT
    ):
        cb = {
            'enabled': True,
            'threshold': thr,
            'alloc': alloc,
            'sell_pct': csell,
            'max_concurrent': maxc,
            'stop_loss_days': 10,
        }
        r = score(base['splits'], base['buy'], base['sell'], crash_buy=cb)
        pro2_cb_results.append({
            'splits': base['splits'],
            'buy': base['buy'],
            'sell': base['sell'],
            'crash_buy': cb,
            **r,
        })
        count += 1
        if count % 100 == 0:
            print(f"    진행: {count}/{total_combos2}")

pro2_cb_results.sort(key=lambda x: x['sharpe'], reverse=True)
print("\n[Pro2] crash_buy 탐색 완료. Top5:")
for i, r in enumerate(pro2_cb_results[:5]):
    cb = r['crash_buy']
    splits_name = 'A' if r['splits'] == PRO2_SPLITS_A else 'B'
    print(f"  [{i+1}] splits={splits_name} buy={r['buy']} sell={r['sell']} "
          f"cb_thr={cb['threshold']} cb_alloc={cb['alloc']} cb_sell={cb['sell_pct']} cb_maxc={cb['max_concurrent']}")
    print(f"       → cum={r['cum']}% sharpe={r['sharpe']} mdd={r['mdd']}% worst={r['worst_year']}%")

pro2_best = pro2_cb_results[0]
pro2_base_nocb = score(pro2_best['splits'], pro2_best['buy'], pro2_best['sell'], crash_buy=None)

print("\n[Pro2] crash_buy enabled vs disabled 비교:")
print(f"  with crash_buy : cum={pro2_best['cum']}% sharpe={pro2_best['sharpe']} mdd={pro2_best['mdd']}% worst={pro2_best['worst_year']}%")
print(f"  without crash  : cum={pro2_base_nocb['cum']}% sharpe={pro2_base_nocb['sharpe']} mdd={pro2_base_nocb['mdd']}% worst={pro2_base_nocb['worst_year']}%")


# ─────────────────────────────────────────────
# 최종 JSON 추천
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("최종 추천 파라미터 (JSON)")
print("=" * 60)

recommendation = {
    "pro1": {
        "splits": pro1_best['splits'],
        "buy_pct": pro1_best['buy'],
        "sell_pct": pro1_best['sell'],
        "stop_loss_days": 10,
        "buy_on_stop": True,
        "transaction_cost": 0.001,
        "price_stop_loss_pct": None,
        "crash_buy": pro1_best['crash_buy'],
        "performance": {
            "cum_pct": pro1_best['cum'],
            "sharpe": pro1_best['sharpe'],
            "mdd_pct": pro1_best['mdd'],
            "worst_year_pct": pro1_best['worst_year'],
        },
        "vs_no_crash_buy": {
            "cum_pct": pro1_base_nocb['cum'],
            "sharpe": pro1_base_nocb['sharpe'],
            "mdd_pct": pro1_base_nocb['mdd'],
            "worst_year_pct": pro1_base_nocb['worst_year'],
        },
    },
    "pro2": {
        "splits": pro2_best['splits'],
        "buy_pct": pro2_best['buy'],
        "sell_pct": pro2_best['sell'],
        "stop_loss_days": 10,
        "buy_on_stop": True,
        "transaction_cost": 0.001,
        "price_stop_loss_pct": None,
        "crash_buy": pro2_best['crash_buy'],
        "performance": {
            "cum_pct": pro2_best['cum'],
            "sharpe": pro2_best['sharpe'],
            "mdd_pct": pro2_best['mdd'],
            "worst_year_pct": pro2_best['worst_year'],
        },
        "vs_no_crash_buy": {
            "cum_pct": pro2_base_nocb['cum'],
            "sharpe": pro2_base_nocb['sharpe'],
            "mdd_pct": pro2_base_nocb['mdd'],
            "worst_year_pct": pro2_base_nocb['worst_year'],
        },
    },
}

print(json.dumps(recommendation, indent=2, ensure_ascii=False))

# 결과 파일 저장
with open('c:/Users/ze1po/OneDrive/바탕 화면/프로젝트/떨사오팔/pro12_crash_result.json', 'w', encoding='utf-8') as f:
    json.dump(recommendation, f, indent=2, ensure_ascii=False)

print("\n결과 저장 완료: pro12_crash_result.json")
