"""
정배열/역배열 국면 세밀 파인튜닝
1차 탐색 결과에서 발견된 최상위 조합들의 세밀 그리드서치
"""
import numpy as np
import sys
sys.path.insert(0, r'c:\Users\ze1po\OneDrive\바탕 화면\프로젝트\떨사오팔')

from data import get_data
from indicators import calc_indicators

df = get_data(refresh=False)
ind = calc_indicators(df)

TC = 0.001
prices_all = df['close'].values
uptrend_all = ind['uptrend'].reindex(df.index).values

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

    def portfolio_value(p):
        u = sum(s['alloc'] * (p / s['entry'] - 1) for s in slots if s['active'])
        u += sum(s['alloc'] * (p / s['entry'] - 1) for s in crash_slots)
        return seed + realized_pnl + u

    for i, price in enumerate(prices_all):
        if i == 0:
            continue
        prev = prices_all[i - 1]
        dr = price / prev - 1
        ee = price * (1 - tc)
        ut = uptrend_all[i]
        is_up = (ut == 1.0) if not np.isnan(ut) else False
        cur_buy = buy_pct_up if is_up else buy_pct_down
        cur_sell = sell_pct_up if is_up else sell_pct_down
        cur_sld = sld_up if is_up else sld_down
        crash_on = crash_up_enabled if is_up else crash_down_enabled

        for s in slots:
            if not s['active']: continue
            pr = ee / s['entry'] - 1
            if pr >= cur_sell or s['days'] >= cur_sld:
                realized_pnl += s['alloc'] * pr
                s['active'] = False; s['days'] = 0

        closed = []
        for s in crash_slots:
            pr = ee / s['entry'] - 1
            if pr >= crash_sell_pct or s['days'] >= crash_sld:
                realized_pnl += s['alloc'] * pr; closed.append(s)
        for s in closed: crash_slots.remove(s)

        for s in slots:
            if s['active']: s['days'] += 1
        for s in crash_slots: s['days'] += 1

        if dr <= cur_buy:
            for s in slots:
                if not s['active']:
                    s['active'] = True; s['entry'] = price * (1 + tc); s['days'] = 1; break

        if crash_on and dr <= crash_threshold:
            if len(crash_slots) < crash_max:
                crash_slots.append({'entry': price * (1 + tc), 'days': 1, 'alloc': crash_alloc})

        cv = portfolio_value(price)
        if cv > peak_value: peak_value = cv
        dd = (cv - peak_value) / peak_value
        if dd < mdd: mdd = dd

    fp = prices_all[-1]; fe = fp * (1 - tc)
    for s in slots:
        if s['active']: realized_pnl += s['alloc'] * (fe / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (fe / s['entry'] - 1)

    return round(realized_pnl * 100, 2), round(mdd * 100, 2)


print("=" * 75)
print("=== 세밀 파인튜닝 결과 ===")
print("=" * 75)

all_results = []

# ── 파인튜닝 1: 정배열 sell_pct 0.042~0.060 세밀 탐색 ──────────────────────
print("\n[파인튜닝 1] 정배열 sell_pct 세밀 탐색 (crash OFF, 역배열 기본)")
for su in [0.042, 0.045, 0.050, 0.053, 0.055, 0.058, 0.060, 0.065]:
    for sd in [0.035, 0.038, 0.042, 0.045, 0.050]:
        r, m = run_dynamic(
            sell_pct_up=su, sell_pct_down=sd,
            crash_up_enabled=False, crash_down_enabled=True,
        )
        all_results.append((f'sell_up={su:.3f} sell_dn={sd:.3f}', r, m))
        if su in [0.050, 0.055, 0.060] and sd in [0.035, 0.042, 0.050]:
            mark = " ***" if r >= 680 and m >= -35 else ""
            print(f"  sell_up={su:.1%} sell_dn={sd:.1%}: cum={r}%, mdd={m}%{mark}")

# ── 파인튜닝 2: 앞가중 splits + 정배열 sell 최적화 ─────────────────────────
print("\n[파인튜닝 2] 앞가중 splits + 정배열 sell 최적화 (최강 조합)")
# 앞가중: 초기 슬롯에 더 많은 자금 배치
splits_front_heavy = [
    [0.25, 0.20, 0.17, 0.15, 0.13, 0.10],  # 앞가중 (1차 탐색에서 731%)
    [0.22, 0.20, 0.18, 0.16, 0.14, 0.10],  # 중간 앞가중
    [0.20, 0.20, 0.17, 0.15, 0.14, 0.14],  # 약한 앞가중
    [0.28, 0.22, 0.18, 0.14, 0.10, 0.08],  # 강한 앞가중
    [0.30, 0.22, 0.17, 0.13, 0.10, 0.08],  # 더 강한 앞가중
]

for sp in splits_front_heavy:
    sp_sum = sum(sp)
    sp_norm = [x / sp_sum for x in sp]  # 합=1 정규화
    for su in [0.042, 0.050, 0.055, 0.060]:
        r, m = run_dynamic(
            splits=sp_norm,
            sell_pct_up=su, sell_pct_down=0.042,
            crash_up_enabled=False, crash_down_enabled=True,
        )
        tag = f'splits={[round(x, 3) for x in sp_norm[:3]]}... sell_up={su:.3f}'
        all_results.append((tag, r, m))
        mark = " ***" if r >= 700 and m >= -36 else ""
        print(f"  {tag}: cum={r}%, mdd={m}%{mark}")

# ── 파인튜닝 3: sell 5.5% + sld 조합 ────────────────────────────────────────
print("\n[파인튜닝 3] sell_up=5.5% 기반 + sld/alloc 조합")
for sld_u in [10, 12, 15]:
    for sld_d in [7, 10]:
        for alloc in [0.15, 0.18, 0.20]:
            r, m = run_dynamic(
                sell_pct_up=0.055, sell_pct_down=0.042,
                sld_up=sld_u, sld_down=sld_d,
                crash_up_enabled=False, crash_down_enabled=True,
                crash_alloc=alloc,
            )
            tag = f'sell5.5% sld({sld_u}/{sld_d}) alloc={alloc}'
            all_results.append((tag, r, m))
            mark = " ***" if r >= 680 and m >= -33 else ""
            print(f"  {tag}: cum={r}%, mdd={m}%{mark}")

# ── 파인튜닝 4: 앞가중 + sell/sld/alloc 종합 조합 ───────────────────────────
print("\n[파인튜닝 4] 앞가중 splits + sell + sld + alloc 종합 최적화")
sp_best = [0.25, 0.20, 0.17, 0.15, 0.13, 0.10]  # 1차에서 731%
sp_sum = sum(sp_best)
sp_norm = [x / sp_sum for x in sp_best]

combos_4 = [
    (0.042, 0.042, 10, 10, 0.15),
    (0.050, 0.042, 10, 10, 0.15),
    (0.055, 0.042, 10, 10, 0.15),
    (0.060, 0.042, 10, 10, 0.15),
    (0.055, 0.035, 10, 10, 0.15),
    (0.055, 0.050, 10, 10, 0.15),
    (0.055, 0.042, 12, 10, 0.15),
    (0.055, 0.042, 15, 10, 0.15),
    (0.055, 0.042, 10, 10, 0.18),
    (0.055, 0.042, 10, 10, 0.20),
    (0.060, 0.042, 15, 10, 0.18),
    (0.055, 0.035, 12, 10, 0.18),
]
for su, sd, lu, ld, alloc in combos_4:
    r, m = run_dynamic(
        splits=sp_norm,
        sell_pct_up=su, sell_pct_down=sd,
        sld_up=lu, sld_down=ld,
        crash_up_enabled=False, crash_down_enabled=True,
        crash_alloc=alloc,
    )
    tag = f'앞가중 sell({su:.3f}/{sd:.3f}) sld({lu}/{ld}) alloc={alloc}'
    all_results.append((tag, r, m))
    mark = " ***" if r >= 720 and m >= -36 else ""
    print(f"  {tag}: cum={r}%, mdd={m}%{mark}")

# ── 파인튜닝 5: crash 파라미터 정밀 탐색 ─────────────────────────────────────
print("\n[파인튜닝 5] crash sell_pct + sld 정밀 탐색 (역배열 최적화)")
for csell in [0.020, 0.025, 0.030, 0.035]:
    for csld in [7, 10, 12]:
        for cthr in [-0.070, -0.075, -0.080, -0.090]:
            r, m = run_dynamic(
                crash_up_enabled=False, crash_down_enabled=True,
                crash_threshold=cthr,
                crash_sell_pct=csell,
                crash_sld=csld,
            )
            if r >= 670 or m >= -30:
                tag = f'crash: sell={csell} sld={csld} thr={cthr}'
                all_results.append((tag, r, m))
                mark = " ***" if r >= 680 and m >= -32 else ""
                print(f"  {tag}: cum={r}%, mdd={m}%{mark}")

# ── 파인튜닝 6: 3중 조합 (앞가중 + 정배열sell + crash 최적) ─────────────────
print("\n[파인튜닝 6] 3중 조합 종합")
sp_norm2 = [0.25/1.0, 0.20/1.0, 0.17/1.0, 0.15/1.0, 0.13/1.0, 0.10/1.0]
sp_sum2 = sum(sp_norm2)
sp_norm2 = [x / sp_sum2 for x in sp_norm2]

for su in [0.050, 0.055, 0.060]:
    for csell in [0.025, 0.030]:
        for cthr in [-0.075, -0.080]:
            for alloc in [0.15, 0.18]:
                r, m = run_dynamic(
                    splits=sp_norm2,
                    sell_pct_up=su, sell_pct_down=0.042,
                    crash_up_enabled=False, crash_down_enabled=True,
                    crash_threshold=cthr,
                    crash_sell_pct=csell,
                    crash_alloc=alloc,
                )
                tag = f'3중: sell_up={su:.3f} csell={csell} cthr={cthr} alloc={alloc}'
                all_results.append((tag, r, m))
                if r >= 720:
                    print(f"  {tag}: cum={r}%, mdd={m}%  ***")

# ── 최종 정렬 출력 ───────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("=== 파인튜닝 종합 결과 Top 20 (cum 내림차순) ===")
print("=" * 75)
sorted_all = sorted(all_results, key=lambda x: x[1], reverse=True)
print(f"{'설명':<65} {'cum':>7} {'mdd':>7}")
print("-" * 80)
for tag, r, m in sorted_all[:20]:
    print(f"{tag:<65} {r:>6}% {m:>6}%")

# ── 최우수 조합 선정 ─────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("=== 최우수 조합 선정 ===")
print("=" * 75)

# cum >= 700, mdd >= -36 조건
top = [(t, r, m) for (t, r, m) in sorted_all if r >= 700 and m >= -36]
if top:
    best_tag, best_r, best_m = top[0]
    print(f"최우수: {best_tag}")
    print(f"성능: cum={best_r}%, mdd={best_m}%")
else:
    best_tag, best_r, best_m = sorted_all[0]
    print(f"최우수 (조건 완화): {best_tag}")
    print(f"성능: cum={best_r}%, mdd={best_m}%")

CHAMP_CUM = 692.4
CHAMP_MDD = -55.9
print(f"\nChampion 대비:")
print(f"  cum: {CHAMP_CUM}% -> {best_r}% ({best_r - CHAMP_CUM:+.1f}%p)")
print(f"  MDD: {CHAMP_MDD}% -> {best_m}% ({best_m - CHAMP_MDD:+.1f}%p 개선)")
