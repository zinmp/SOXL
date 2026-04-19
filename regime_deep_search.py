"""
정배열/역배열 국면 세밀 탐색
기준: 정배열 crash OFF + 다양한 파라미터 조합 탐색
"""
import numpy as np
import sys
sys.path.insert(0, r'c:\Users\ze1po\OneDrive\바탕 화면\프로젝트\떨사오팔')

from data import get_data
from backtest import run_backtest
from indicators import calc_indicators

df = get_data(refresh=False)
ind = calc_indicators(df)

TC = 0.001
prices_all = df['close'].values
uptrend_all = ind['uptrend'].reindex(df.index).values
vol_all = ind['volatility'].reindex(df.index).values

# ── 핵심 백테스터: 정배열/역배열 날짜별 동적 파라미터 적용 ──────────────────
def run_dynamic(
    buy_pct_up=-0.007, buy_pct_down=-0.007,
    sell_pct_up=0.042, sell_pct_down=0.042,
    sld_up=10, sld_down=10,
    crash_up_enabled=False,
    crash_down_enabled=True,
    crash_threshold=-0.075,
    crash_alloc=0.15,
    crash_sell_pct=0.025,
    crash_sld=10,
    crash_max=3,
    splits=None,
    tc=TC,
):
    if splits is None:
        splits = [1/6]*6

    slots = [{'active': False, 'entry': 0.0, 'days': 0, 'alloc': s} for s in splits]
    crash_slots = []
    seed = 1.0
    realized_pnl = 0.0
    peak_value = seed
    mdd = 0.0

    def portfolio_value(current_price):
        unrealized = sum(s['alloc'] * (current_price / s['entry'] - 1) for s in slots if s['active'])
        unrealized += sum(s['alloc'] * (current_price / s['entry'] - 1) for s in crash_slots)
        return seed + realized_pnl + unrealized

    for i, price in enumerate(prices_all):
        if i == 0:
            continue
        prev_price = prices_all[i - 1]
        daily_ret = price / prev_price - 1
        effective_exit = price * (1 - tc)

        ut = uptrend_all[i]
        is_up = (ut == 1.0) if not np.isnan(ut) else False

        cur_buy = buy_pct_up if is_up else buy_pct_down
        cur_sell = sell_pct_up if is_up else sell_pct_down
        cur_sld = sld_up if is_up else sld_down
        crash_on = crash_up_enabled if is_up else crash_down_enabled

        # 일반 슬롯 매도
        for s in slots:
            if not s['active']:
                continue
            pr = effective_exit / s['entry'] - 1
            if pr >= cur_sell or s['days'] >= cur_sld:
                realized_pnl += s['alloc'] * pr
                s['active'] = False
                s['days'] = 0

        # 급락 슬롯 매도
        closed = []
        for s in crash_slots:
            pr = effective_exit / s['entry'] - 1
            if pr >= crash_sell_pct or s['days'] >= crash_sld:
                realized_pnl += s['alloc'] * pr
                closed.append(s)
        for s in closed:
            crash_slots.remove(s)

        # 보유 일수 증가
        for s in slots:
            if s['active']:
                s['days'] += 1
        for s in crash_slots:
            s['days'] += 1

        # 일반 매수
        if daily_ret <= cur_buy:
            for s in slots:
                if not s['active']:
                    s['active'] = True
                    s['entry'] = price * (1 + tc)
                    s['days'] = 1
                    break

        # 급락 매수
        if crash_on and daily_ret <= crash_threshold:
            if len(crash_slots) < crash_max:
                crash_slots.append({'entry': price * (1 + tc), 'days': 1, 'alloc': crash_alloc})

        # MDD 추적
        cur_val = portfolio_value(price)
        if cur_val > peak_value:
            peak_value = cur_val
        dd = (cur_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

    # 강제 청산
    fp = prices_all[-1]
    fe = fp * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (fe / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (fe / s['entry'] - 1)

    return round(realized_pnl * 100, 2), round(mdd * 100, 2)


# ── 연도별 uptrend 비율을 이용한 threshold 기반 국면 분류 ─────────────────────
def run_regime_threshold(p_up, p_down, up_threshold=0.5, tc=TC):
    """연도별 uptrend 비율이 up_threshold 이상이면 p_up 적용, 미만이면 p_down"""
    rets, mdds = [], []
    for year in sorted(df.index.year.unique()):
        mask = df.index.year == year
        if mask.sum() < 5:
            continue
        up_ratio = ind.loc[df.index[mask], 'uptrend'].mean()
        p = p_up if up_ratio >= up_threshold else p_down
        cb = p.get('crash_buy', {})
        crash = cb if cb.get('enabled') else None
        ret, mdd = run_backtest(
            prices=df.loc[mask, 'close'],
            splits=p['splits'],
            buy_pct=p['buy_pct'],
            sell_pct=p['sell_pct'],
            stop_loss_days=p['stop_loss_days'],
            buy_on_stop=True,
            crash_buy=crash,
            transaction_cost=tc,
            price_stop_loss_pct=None,
        )
        rets.append(ret)
        mdds.append(mdd)
    a = np.array(rets)
    s = np.std(a, ddof=1)
    return {
        'cum': round(float(np.sum(a)), 1),
        'sharpe': round((np.mean(a) - 4.5) / s if s > 0 else 0, 2),
        'mdd': round(float(np.min(np.array(mdds))), 1)
    }


print("=" * 70)
print("=== 국면 필터 세밀 탐색 결과 ===")
print("=" * 70)

results = []

# ── 기준선: Champion ─────────────────────────────────────────────────────────
r_champ, mdd_champ = run_dynamic(
    crash_up_enabled=True, crash_down_enabled=True,
)
results.append(('Champion 기준선', '-', r_champ, '-', mdd_champ))
print(f"Champion 기준선: cum={r_champ}%, mdd={mdd_champ}%")

# ── 탐색 A: 정배열 crash OFF (기본, up_threshold=0.5 연도별) ─────────────────
print("\n[탐색 A] 정배열 crash OFF, 역배열 crash ON (날짜별 동적)")
r1, m1 = run_dynamic(crash_up_enabled=False, crash_down_enabled=True)
results.append(('A1: 정배열crashOFF(날짜별)', 'daily', r1, '-', m1))
print(f"  A1 날짜별: cum={r1}%, mdd={m1}%")

# ── 탐색 B: 정배열 crash OFF + sell_pct 조합 ─────────────────────────────────
print("\n[탐색 B] 정배열 crash OFF + sell_pct 조합")
sell_up_vals   = [0.030, 0.035, 0.042, 0.050, 0.055]
sell_down_vals = [0.035, 0.042, 0.050, 0.055]

best_b = None
for su in sell_up_vals:
    for sd in sell_down_vals:
        r, m = run_dynamic(
            sell_pct_up=su, sell_pct_down=sd,
            crash_up_enabled=False, crash_down_enabled=True,
        )
        tag = f'B: sell_up={su:.3f} sell_dn={sd:.3f}'
        results.append((tag, 'daily', r, f'su={su},sd={sd}', m))
        if r >= 580 and m >= -53:
            marker = " *** CANDIDATE"
        else:
            marker = ""
        print(f"  sell_up={su:.1%} sell_dn={sd:.1%}: cum={r}%, mdd={m}%{marker}")

# ── 탐색 C: 정배열 crash OFF + stop_loss_days 조합 ───────────────────────────
print("\n[탐색 C] 정배열 crash OFF + stop_loss_days 조합")
sld_up_vals   = [7, 10, 12, 15]
sld_down_vals = [5, 7, 10]

for su in sld_up_vals:
    for sd in sld_down_vals:
        if su < sd:
            continue
        r, m = run_dynamic(
            sld_up=su, sld_down=sd,
            crash_up_enabled=False, crash_down_enabled=True,
        )
        tag = f'C: sld_up={su} sld_dn={sd}'
        results.append((tag, 'daily', r, f'slu={su},sld={sd}', m))
        marker = " *** CANDIDATE" if r >= 580 and m >= -53 else ""
        print(f"  sld_up={su}d sld_dn={sd}d: cum={r}%, mdd={m}%{marker}")

# ── 탐색 D: 정배열 crash OFF + buy_pct 조합 ──────────────────────────────────
print("\n[탐색 D] 정배열 crash OFF + buy_pct 조합")
buy_up_vals   = [-0.005, -0.007, -0.010, -0.012]
buy_down_vals = [-0.005, -0.007, -0.010]

for bu in buy_up_vals:
    for bd in buy_down_vals:
        r, m = run_dynamic(
            buy_pct_up=bu, buy_pct_down=bd,
            crash_up_enabled=False, crash_down_enabled=True,
        )
        tag = f'D: buy_up={bu} buy_dn={bd}'
        results.append((tag, 'daily', r, f'bpu={bu},bpd={bd}', m))
        marker = " *** CANDIDATE" if r >= 580 and m >= -53 else ""
        print(f"  buy_up={bu:.1%} buy_dn={bd:.1%}: cum={r}%, mdd={m}%{marker}")

# ── 탐색 E: crash 임계값 조정 (역배열 시 더 낮게) ────────────────────────────
print("\n[탐색 E] 역배열 시 crash threshold 조정 (-7% vs -10%)")
for thr in [-0.050, -0.060, -0.075, -0.090, -0.100]:
    r, m = run_dynamic(
        crash_up_enabled=False, crash_down_enabled=True,
        crash_threshold=thr,
    )
    tag = f'E: crash_thr={thr}'
    results.append((tag, 'daily', r, f'thr={thr}', m))
    marker = " *** CANDIDATE" if r >= 580 and m >= -53 else ""
    print(f"  crash_threshold={thr:.1%}: cum={r}%, mdd={m}%{marker}")

# ── 탐색 F: 역배열 crash alloc 조정 ─────────────────────────────────────────
print("\n[탐색 F] 역배열 crash alloc 조정 (0.10 ~ 0.20)")
for alloc in [0.10, 0.12, 0.15, 0.18, 0.20]:
    r, m = run_dynamic(
        crash_up_enabled=False, crash_down_enabled=True,
        crash_alloc=alloc,
    )
    tag = f'F: crash_alloc={alloc}'
    results.append((tag, 'daily', r, f'alloc={alloc}', m))
    marker = " *** CANDIDATE" if r >= 580 and m >= -53 else ""
    print(f"  crash_alloc={alloc:.2f}: cum={r}%, mdd={m}%{marker}")

# ── 탐색 G: 연도별 uptrend_threshold 조정 ───────────────────────────────────
print("\n[탐색 G] 연도별 uptrend 비율 임계값 조정")

p_up_base = {
    'splits': [1/6]*6, 'buy_pct': -0.007, 'sell_pct': 0.042,
    'stop_loss_days': 10,
    'crash_buy': {'enabled': False, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3}
}
p_down_base = {
    'splits': [1/6]*6, 'buy_pct': -0.007, 'sell_pct': 0.042,
    'stop_loss_days': 10,
    'crash_buy': {'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3}
}

for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    res = run_regime_threshold(p_up_base, p_down_base, up_threshold=thr)
    tag = f'G: up_threshold={thr}'
    results.append((tag, str(thr), res['cum'], f'thr={thr}', res['mdd']))
    marker = " *** CANDIDATE" if res['cum'] >= 580 and res['mdd'] >= -53 else ""
    print(f"  up_threshold={thr}: cum={res['cum']}%, sharpe={res['sharpe']}, mdd={res['mdd']}%{marker}")

# ── 탐색 H: 정배열 splits 피라미딩 조정 ─────────────────────────────────────
print("\n[탐색 H] 정배열 시 splits 피라미딩 (뒤쪽 슬롯 비중 증가)")
# 정배열: 균등 vs 뒤쪽 가중 (추세 추종적)
splits_flat     = [1/6]*6
splits_ramp_up  = [0.10, 0.13, 0.15, 0.17, 0.20, 0.25]  # 뒤쪽 큰 비중
splits_ramp_dn  = [0.25, 0.20, 0.17, 0.15, 0.13, 0.10]  # 앞쪽 큰 비중 (보수적)
splits_uniform  = [1/6]*6

for name_up, sp_up in [('균등', splits_flat), ('뒤가중', splits_ramp_up), ('앞가중', splits_ramp_dn)]:
    # 역배열은 항상 균등
    r, m = run_dynamic(
        splits=sp_up,
        crash_up_enabled=False, crash_down_enabled=True,
    )
    tag = f'H: 정배열_splits={name_up}'
    results.append((tag, 'daily', r, f'sp={name_up}', m))
    marker = " *** CANDIDATE" if r >= 580 and m >= -53 else ""
    print(f"  정배열splits={name_up}: cum={r}%, mdd={m}%{marker}")

# ── 탐색 I: 복합 최적 조합 ──────────────────────────────────────────────────
print("\n[탐색 I] 복합 조합: crash OFF + sell/sld 동적 최적화")

combos = [
    # (sell_up, sell_dn, sld_up, sld_dn, crash_thr, crash_alloc, crash_sell)
    (0.042, 0.042, 10, 10, -0.075, 0.15, 0.025),   # 기준 (crash OFF only)
    (0.042, 0.050, 10, 10, -0.075, 0.15, 0.025),   # 역배열 sell 더 높게
    (0.042, 0.042, 10,  7, -0.075, 0.15, 0.025),   # 역배열 sld 단축
    (0.042, 0.050, 10,  7, -0.075, 0.15, 0.025),   # sell+sld 복합
    (0.042, 0.042, 10, 10, -0.090, 0.15, 0.025),   # crash thr 낮춤
    (0.042, 0.050, 10,  7, -0.090, 0.15, 0.025),   # sell+sld+thr 복합
    (0.042, 0.042, 10, 10, -0.075, 0.18, 0.025),   # crash alloc 증가
    (0.042, 0.050, 10,  7, -0.090, 0.18, 0.030),   # 강한 역배열 공격적
    (0.035, 0.042, 10, 10, -0.075, 0.15, 0.025),   # 정배열 sell 낮춤
    (0.035, 0.042, 10,  7, -0.075, 0.15, 0.025),   # 정배열 sell + 역배열 sld
    (0.035, 0.050, 10,  7, -0.090, 0.18, 0.030),   # 최적화 강도 높음
    (0.030, 0.042, 10, 10, -0.075, 0.15, 0.025),   # 정배열 sell 더 낮춤
    (0.042, 0.042, 12, 10, -0.075, 0.15, 0.025),   # 정배열 sld 연장
    (0.042, 0.042, 15, 10, -0.075, 0.15, 0.025),   # 정배열 sld 더 연장
    (0.042, 0.042, 12,  7, -0.075, 0.15, 0.025),   # 정배열 연장+역배열 단축
    (0.050, 0.042, 10, 10, -0.075, 0.15, 0.025),   # 정배열 sell 높임
    (0.055, 0.042, 10, 10, -0.075, 0.15, 0.025),   # 정배열 sell 더 높임
]

for su, sd, lu, ld, thr, alloc, csell in combos:
    r, m = run_dynamic(
        sell_pct_up=su, sell_pct_down=sd,
        sld_up=lu, sld_down=ld,
        crash_up_enabled=False, crash_down_enabled=True,
        crash_threshold=thr, crash_alloc=alloc, crash_sell_pct=csell,
    )
    desc = f'sell({su:.3f}/{sd:.3f}) sld({lu}/{ld}) thr={thr} alloc={alloc}'
    tag = f'I: {desc}'
    results.append((tag, 'daily', r, desc, m))
    marker = " *** CANDIDATE" if r >= 600 and m >= -53 else ""
    print(f"  {desc}: cum={r}%, mdd={m}%{marker}")

# ── 탐색 J: 정배열 시 buy_pct 완화 (덜 공격적 매수) ─────────────────────────
print("\n[탐색 J] 정배열 buy_pct 완화 (덜 자주 매수) + crash OFF")
for bpu in [-0.003, -0.005, -0.007, -0.010, -0.015]:
    r, m = run_dynamic(
        buy_pct_up=bpu, buy_pct_down=-0.007,
        sell_pct_up=0.042, sell_pct_down=0.042,
        crash_up_enabled=False, crash_down_enabled=True,
    )
    tag = f'J: buy_up={bpu}'
    results.append((tag, 'daily', r, f'bpu={bpu}', m))
    marker = " *** CANDIDATE" if r >= 580 and m >= -53 else ""
    print(f"  buy_up={bpu:.1%}: cum={r}%, mdd={m}%{marker}")

# ── 최종 결과 정렬 및 출력 ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== 전체 탐색 결과 (cum 내림차순) ===")
print("=" * 70)
print(f"{'설명':<55} {'cum':>7} {'mdd':>7}")
print("-" * 70)

sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
for tag, thr, r, desc, m in sorted_results[:30]:
    print(f"{tag:<55} {r:>6}% {m:>6}%")

# ── 조건 만족 최적 후보 ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== 최적 조합 후보 (cum≥600%, mdd≥-53%) ===")
print("=" * 70)
candidates = [(tag, thr, r, desc, m) for (tag, thr, r, desc, m) in results if r >= 600 and m >= -53]
candidates.sort(key=lambda x: (x[2], -abs(x[4])), reverse=True)
if candidates:
    for tag, thr, r, desc, m in candidates:
        print(f"  {tag}: cum={r}%, mdd={m}%")
else:
    # 조건 완화
    print("(cum>=600, mdd>=-53 조건 없음. 완화 기준으로 출력)")
    candidates2 = [(tag, thr, r, desc, m) for (tag, thr, r, desc, m) in results if r >= 550 and m >= -55]
    candidates2.sort(key=lambda x: x[2], reverse=True)
    for tag, thr, r, desc, m in candidates2[:10]:
        print(f"  {tag}: cum={r}%, mdd={m}%")

# ── Champion 대비 요약 ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("=== Champion 대비 최상위 결과 ===")
print("=" * 70)
CHAMP_CUM = 692.4
CHAMP_MDD = -55.9
best = sorted_results[0]
print(f"Champion: cum={CHAMP_CUM}%, mdd={CHAMP_MDD}%")
print(f"최상위: cum={best[2]}%, mdd={best[4]}%  [{best[0]}]")
print(f"  cum 변화: {CHAMP_CUM}% → {best[2]}% ({best[2]-CHAMP_CUM:+.1f}%p)")
print(f"  MDD 변화: {CHAMP_MDD}% → {best[4]}% ({best[4]-CHAMP_MDD:+.1f}%p 개선)")
