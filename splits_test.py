import numpy as np
import sys
sys.path.insert(0, 'c:/Users/ze1po/OneDrive/바탕 화면/프로젝트/떨사오팔')
from data import get_data
from backtest import run_backtest

df = get_data()

base_params = {
    'buy_pct': -0.007,
    'sell_pct': 0.042,
    'stop_loss_days': 10,
    'buy_on_stop': True,
    'crash_buy': {
        'enabled': True, 'threshold': -0.075, 'alloc': 0.15,
        'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3
    }
}

def test_splits(splits, params_base, tc=0.001):
    p = dict(params_base)
    p['splits'] = splits
    crash = p.get('crash_buy', {})
    rets, mdds = [], []
    for year in sorted(df.index.year.unique()):
        mask = df.index.year == year
        if mask.sum() < 5:
            continue
        prices = df.loc[mask, 'close']
        ret, mdd = run_backtest(
            prices=prices, splits=p['splits'],
            buy_pct=p['buy_pct'], sell_pct=p['sell_pct'],
            stop_loss_days=p['stop_loss_days'], buy_on_stop=p['buy_on_stop'],
            crash_buy=crash if crash.get('enabled') else None,
            transaction_cost=tc, price_stop_loss_pct=None,
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
        'win': round(float(np.mean(rets > 0) * 100), 0),
    }

splits_map = {
    'Champion': [1/6]*6,
    'A': [0.05, 0.10, 0.15, 0.20, 0.25, 0.25],
    'B': [0.08, 0.12, 0.15, 0.20, 0.22, 0.23],
    'C': [0.10, 0.13, 0.16, 0.18, 0.20, 0.23],
    'D': [0.05, 0.08, 0.12, 0.18, 0.25, 0.32],
    'E': [0.25, 0.22, 0.20, 0.15, 0.12, 0.06],
    'F': [0.30, 0.25, 0.20, 0.15, 0.07, 0.03],
    'G': [0.10, 0.15, 0.25, 0.25, 0.15, 0.10],
    'H': [0.05, 0.15, 0.25, 0.30, 0.20, 0.05],
    'I': [0.15, 0.20, 0.30, 0.35, 0.0, 0.0],
    'J': [0.10, 0.20, 0.30, 0.40, 0.0, 0.0],
}

results = {}
for name, splits in splits_map.items():
    s = [round(x, 4) for x in splits]
    total = sum(s)
    print(f"Running {name}: splits={s} sum={total:.4f}")
    res = test_splits(splits, base_params)
    results[name] = {'splits': s, **res}
    print(f"  -> cum={res['cum']}%, sharpe={res['sharpe']}, worst_mdd={res['worst_mdd']}%, mean={res['mean']}%, win={res['win']}%")

print()
print("=== splits 구조별 성능 비교 ===")
print(f"{'구조':<10} | {'splits':<45} | {'cum':>6} | {'sharpe':>7} | {'worst_mdd':>10} | {'mean':>6} | {'win':>5}")
print("-" * 105)
for name, r in results.items():
    sp_str = str(r['splits'])
    print(f"{name:<10} | {sp_str:<45} | {r['cum']:>6} | {r['sharpe']:>7} | {r['worst_mdd']:>10} | {r['mean']:>6} | {r['win']:>5}")

champion = results['Champion']
print()
print("=== 최적 splits 탐색 (MDD 개선 우선, 수익 유지) ===")
candidates = []
for name, r in results.items():
    if name == 'Champion':
        continue
    # MDD 개선 = MDD 절댓값 감소 (예: -55.9 → -47.8 이면 8.1%p 개선)
    mdd_improvement = r['worst_mdd'] - champion['worst_mdd']  # 양수 = MDD 악화, 음수 = MDD 개선
    mdd_gain = -mdd_improvement  # 양수 = 개선
    cum_diff = r['cum'] - champion['cum']
    # 목표: MDD 낮추되 수익 최대한 유지. sharpe 저하 패널티 포함
    sharpe_diff = r['sharpe'] - champion['sharpe']
    score = mdd_gain * 3 - max(0, -cum_diff) * 0.05 + sharpe_diff * 10
    candidates.append((name, r, mdd_gain, cum_diff, score))

candidates.sort(key=lambda x: x[4], reverse=True)
best_name, best_r, best_mdd_imp, best_cum_diff, _ = candidates[0]

print(f"\n최적 후보: {best_name}")
print(f"splits: {best_r['splits']}")
print(f"cum={best_r['cum']}%, sharpe={best_r['sharpe']}, worst_mdd={best_r['worst_mdd']}%")

print()
print("=== Champion 대비 ===")
print(f"cum: {champion['cum']}% → {best_r['cum']}% ({best_cum_diff:+.1f}%p)")
print(f"MDD: {champion['worst_mdd']}% → {best_r['worst_mdd']}% ({best_mdd_imp:+.1f}%p 개선)")
print(f"Sharpe: {champion['sharpe']} → {best_r['sharpe']}")

print()
print("=== 전체 후보 순위 (MDD개선*3 - 수익손실*0.05 + Sharpe차이*10 점수) ===")
for i, (name, r, mdd_gain, cum_diff, score) in enumerate(candidates, 1):
    sharpe_diff = r['sharpe'] - champion['sharpe']
    print(f"{i:2}. {name:<10}: MDD {champion['worst_mdd']}%→{r['worst_mdd']}% (개선{mdd_gain:+.1f}%p), cum={r['cum']}% ({cum_diff:+.1f}%p), sharpe={r['sharpe']} ({sharpe_diff:+.2f}), score={score:.1f}")
