import numpy as np
import sys
sys.path.insert(0, r'c:\Users\ze1po\OneDrive\바탕 화면\프로젝트\떨사오팔')
from data import get_data
from backtest import run_backtest

df = get_data()

def test_crash(cb_params, tc=0.001):
    base = {
        'splits': [1/6]*6, 'buy_pct': -0.007, 'sell_pct': 0.042,
        'stop_loss_days': 10, 'buy_on_stop': True,
    }
    crash = {'enabled': True, 'stop_loss_days': 10, **cb_params}
    rets, mdds = [], []
    for year in sorted(df.index.year.unique()):
        mask = df.index.year == year
        if mask.sum() < 5:
            continue
        prices = df.loc[mask, 'close']
        ret, mdd = run_backtest(
            prices=prices, splits=base['splits'],
            buy_pct=base['buy_pct'], sell_pct=base['sell_pct'],
            stop_loss_days=base['stop_loss_days'], buy_on_stop=True,
            crash_buy=crash, transaction_cost=tc, price_stop_loss_pct=None,
        )
        rets.append(ret)
        mdds.append(mdd)
    rets = np.array(rets)
    mdds = np.array(mdds)
    std = np.std(rets, ddof=1)
    return {
        'cum': round(float(np.sum(rets)), 1),
        'sharpe': round((np.mean(rets) - 4.5) / std if std > 0 else 0, 2),
        'worst_mdd': round(float(np.min(mdds)), 1),
        'mean': round(float(np.mean(rets)), 1),
    }

# Champion 기준선 확인
champion_cb = {
    'threshold': -0.075, 'alloc': 0.15,
    'sell_pct': 0.025, 'max_concurrent': 3
}
champ = test_crash(champion_cb)
print(f"Champion 기준선: cum={champ['cum']}%, sharpe={champ['sharpe']}, worst_mdd={champ['worst_mdd']}%")
print()

# 그리드 파라미터
thresholds     = [-0.06, -0.075, -0.09, -0.10, -0.12]
allocs         = [0.05, 0.08, 0.10, 0.12, 0.15]
max_concs      = [1, 2, 3]
sell_pcts      = [0.020, 0.025, 0.030, 0.035]

results = []
total = len(thresholds) * len(allocs) * len(max_concs) * len(sell_pcts)
count = 0

for thr in thresholds:
    for alloc in allocs:
        for mc in max_concs:
            for sp in sell_pcts:
                count += 1
                cb_params = {
                    'threshold': thr,
                    'alloc': alloc,
                    'sell_pct': sp,
                    'max_concurrent': mc,
                }
                res = test_crash(cb_params)
                results.append({
                    'threshold': thr,
                    'alloc': alloc,
                    'max_conc': mc,
                    'sell_pct': sp,
                    **res
                })
                if count % 50 == 0:
                    print(f"진행: {count}/{total}", flush=True)

# MDD 오름차순 정렬
results.sort(key=lambda x: x['worst_mdd'])

print()
print("=== crash_buy 파라미터 그리드 결과 (MDD 오름차순) ===")
print(f"{'threshold':>10} | {'alloc':>6} | {'max_conc':>8} | {'sell_pct':>8} | {'cum':>7} | {'sharpe':>6} | {'worst_mdd':>9}")
print("-" * 75)
for r in results[:60]:  # 상위 60개 출력
    print(f"{r['threshold']:>10.3f} | {r['alloc']:>6.2f} | {r['max_conc']:>8} | {r['sell_pct']:>8.3f} | {r['cum']:>7.1f} | {r['sharpe']:>6.2f} | {r['worst_mdd']:>9.1f}")

print()
print("=== 전체 결과 요약 ===")
print(f"총 조합: {len(results)}개")

# MDD 기준 필터링: worst_mdd > -50% 인 조합 중 cum 최대
mdd_filtered = [r for r in results if r['worst_mdd'] > -50.0]
print(f"MDD > -50% 조합: {len(mdd_filtered)}개")

if mdd_filtered:
    mdd_filtered.sort(key=lambda x: -x['cum'])
    best = mdd_filtered[0]
    print()
    print("=== MDD/수익 균형 최적점 (MDD > -50%, 수익 최대) ===")
    print(f"파라미터: threshold={best['threshold']}, alloc={best['alloc']}, max_concurrent={best['max_conc']}, sell_pct={best['sell_pct']}")
    print(f"성능: cum={best['cum']}%, sharpe={best['sharpe']}, worst_mdd={best['worst_mdd']}%")
    print()
    print("=== Champion 대비 ===")
    print(f"cum:    692% → {best['cum']}% ({best['cum']-692:+.1f}%p)")
    print(f"sharpe: 1.56 → {best['sharpe']} ({best['sharpe']-1.56:+.2f})")
    print(f"MDD:   -55.9% → {best['worst_mdd']}% ({best['worst_mdd']-(-55.9):+.1f}%p 개선)")
else:
    print("MDD > -50% 조건을 만족하는 조합 없음. 가장 낮은 MDD 상위 10개:")
    for r in results[:10]:
        print(f"  threshold={r['threshold']}, alloc={r['alloc']}, max_conc={r['max_conc']}, sell_pct={r['sell_pct']} | cum={r['cum']}%, worst_mdd={r['worst_mdd']}%")

# Sharpe 기준 균형점도 출력
print()
print("=== Sharpe 최고 조합 TOP 10 ===")
results_by_sharpe = sorted(results, key=lambda x: -x['sharpe'])
print(f"{'threshold':>10} | {'alloc':>6} | {'max_conc':>8} | {'sell_pct':>8} | {'cum':>7} | {'sharpe':>6} | {'worst_mdd':>9}")
print("-" * 75)
for r in results_by_sharpe[:10]:
    print(f"{r['threshold']:>10.3f} | {r['alloc']:>6.2f} | {r['max_conc']:>8} | {r['sell_pct']:>8.3f} | {r['cum']:>7.1f} | {r['sharpe']:>6.2f} | {r['worst_mdd']:>9.1f}")

# cum/MDD 복합 스코어 (cum - |MDD| * 2)
print()
print("=== 복합 스코어 TOP 10 (cum + MDD*2 최대화) ===")
for r in results:
    r['score'] = r['cum'] + r['worst_mdd'] * 2  # MDD가 음수이므로 패널티
results_by_score = sorted(results, key=lambda x: -x['score'])
print(f"{'threshold':>10} | {'alloc':>6} | {'max_conc':>8} | {'sell_pct':>8} | {'cum':>7} | {'sharpe':>6} | {'worst_mdd':>9} | {'score':>7}")
print("-" * 85)
for r in results_by_score[:10]:
    print(f"{r['threshold']:>10.3f} | {r['alloc']:>6.2f} | {r['max_conc']:>8} | {r['sell_pct']:>8.3f} | {r['cum']:>7.1f} | {r['sharpe']:>6.2f} | {r['worst_mdd']:>9.1f} | {r['score']:>7.1f}")
