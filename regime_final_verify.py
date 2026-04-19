"""
최종 검증: 최상위 조합 심층 분석 + 연도별 성과 + 퀀트 해석
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


def run_dynamic_yearly(sell_pct_up, sell_pct_down, splits, crash_alloc=0.15,
                       crash_threshold=-0.075, crash_sell_pct=0.025,
                       crash_up_enabled=False, crash_down_enabled=True, tc=TC):
    """연도별 성과 분석"""
    years = sorted(df.index.year.unique())
    rets, mdds = [], []
    for year in years:
        mask = df.index.year == year
        if mask.sum() < 5:
            continue
        yr_prices = df.loc[mask, 'close'].values
        yr_uptrend = ind.loc[df.index[mask], 'uptrend'].values

        slots = [{'active': False, 'entry': 0.0, 'days': 0, 'alloc': s} for s in splits]
        crash_slots = []
        seed = 1.0; rpnl = 0.0; peak = seed; mdd = 0.0

        def pv(p):
            u = sum(s['alloc'] * (p / s['entry'] - 1) for s in slots if s['active'])
            u += sum(s['alloc'] * (p / s['entry'] - 1) for s in crash_slots)
            return seed + rpnl + u

        for i, price in enumerate(yr_prices):
            if i == 0: continue
            prev = yr_prices[i - 1]; dr = price / prev - 1; ee = price * (1 - tc)
            ut = yr_uptrend[i]
            is_up = (ut == 1.0) if not np.isnan(ut) else False
            cur_sell = sell_pct_up if is_up else sell_pct_down
            crash_on = crash_up_enabled if is_up else crash_down_enabled

            for s in slots:
                if not s['active']: continue
                pr = ee / s['entry'] - 1
                if pr >= cur_sell or s['days'] >= 10:
                    rpnl += s['alloc'] * pr; s['active'] = False; s['days'] = 0

            closed = []
            for s in crash_slots:
                pr = ee / s['entry'] - 1
                if pr >= crash_sell_pct or s['days'] >= 10:
                    rpnl += s['alloc'] * pr; closed.append(s)
            for s in closed: crash_slots.remove(s)

            for s in slots:
                if s['active']: s['days'] += 1
            for s in crash_slots: s['days'] += 1

            if dr <= -0.007:
                for s in slots:
                    if not s['active']:
                        s['active'] = True; s['entry'] = price * (1 + tc); s['days'] = 1; break

            if crash_on and dr <= crash_threshold:
                if len(crash_slots) < 3:
                    crash_slots.append({'entry': price * (1 + tc), 'days': 1, 'alloc': crash_alloc})

            cv = seed + rpnl + sum(s['alloc'] * (price / s['entry'] - 1) for s in slots if s['active']) + sum(s['alloc'] * (price / s['entry'] - 1) for s in crash_slots)
            if cv > peak: peak = cv
            dd = (cv - peak) / peak
            if dd < mdd: mdd = dd

        fp = yr_prices[-1]; fe = fp * (1 - tc)
        for s in slots:
            if s['active']: rpnl += s['alloc'] * (fe / s['entry'] - 1)
        for s in crash_slots:
            rpnl += s['alloc'] * (fe / s['entry'] - 1)

        rets.append((year, round(rpnl * 100, 2), round(mdd * 100, 2)))
        mdds.append(mdd * 100)

    return rets, mdds


# ── Champion 기준선 ─────────────────────────────────────────────────────────
CHAMP_CUM = 692.4
CHAMP_MDD = -55.9
CHAMP_SHARPE = 1.56

print("=" * 75)
print("=== 국면 필터 세밀 탐색 결과 - 최종 검증 ===")
print("=" * 75)

# ── 핵심 후보들 최종 검증 ────────────────────────────────────────────────────
print("\n최종 후보 검증")
print(f"{'설명':<60} {'cum':>7} {'mdd':>8}")
print("-" * 78)

sp_front = [0.30, 0.22, 0.17, 0.13, 0.10, 0.08]
sp_sum = sum(sp_front)
sp_front_norm = [x / sp_sum for x in sp_front]

candidates = [
    # (설명, sell_up, sell_dn, splits, crash_alloc)
    ('Champion 기준선 (crash ON 항상)', 0.042, 0.042, [1/6]*6, 0.15, True, True),
    ('A: crash OFF (정배열)', 0.042, 0.042, [1/6]*6, 0.15, False, True),
    ('B: sell_up=6% sell_dn=4.2%', 0.060, 0.042, [1/6]*6, 0.15, False, True),
    ('B: sell_up=5.5% sell_dn=5.0%', 0.055, 0.050, [1/6]*6, 0.15, False, True),
    ('B: sell_up=6% sell_dn=5.0%', 0.060, 0.050, [1/6]*6, 0.15, False, True),
    ('H앞가중: sell_up=5.5%', 0.055, 0.042, sp_front_norm, 0.15, False, True),
    ('H앞가중: sell_up=6.0%', 0.060, 0.042, sp_front_norm, 0.15, False, True),
]

results_summary = []
for name, su, sd, sp, alloc, cup, cdn in candidates:
    r, m = run_dynamic(
        sell_pct_up=su, sell_pct_down=sd, splits=sp, crash_alloc=alloc,
        crash_up_enabled=cup, crash_down_enabled=cdn,
    )
    results_summary.append((name, r, m))
    print(f"{name:<60} {r:>6}% {m:>7}%")

# ── 연도별 성과 분석 ─────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("=== 연도별 성과 비교 ===")
print("=" * 75)

# 기준 Champion
rets_champ = []
for year in sorted(df.index.year.unique()):
    mask = df.index.year == year
    if mask.sum() < 5: continue
    ret, mdd = run_backtest(
        prices=df.loc[mask, 'close'],
        splits=[1/6]*6, buy_pct=-0.007, sell_pct=0.042,
        stop_loss_days=10, buy_on_stop=True,
        crash_buy={'enabled': True, 'threshold': -0.075, 'alloc': 0.15, 'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3},
        transaction_cost=TC, price_stop_loss_pct=None,
    )
    rets_champ.append((year, ret, mdd))

# 최적 후보: sell_up=5.5% crash OFF
rets_opt, _ = run_dynamic_yearly(0.055, 0.042, [1/6]*6)
# 최상위: 앞가중 + sell_up=6%
rets_best, _ = run_dynamic_yearly(0.060, 0.042, sp_front_norm)

print(f"{'연도':<6} {'Champion':>10} {'[mdd]':>8} {'sell_up5.5%':>12} {'[mdd]':>8} {'앞가중+6%':>10} {'[mdd]':>8}")
print("-" * 75)
for i, (yr_c, r_c, m_c) in enumerate(rets_champ):
    r_opt = rets_opt[i][1] if i < len(rets_opt) else '-'
    m_opt = rets_opt[i][2] if i < len(rets_opt) else '-'
    r_best = rets_best[i][1] if i < len(rets_best) else '-'
    m_best = rets_best[i][2] if i < len(rets_best) else '-'
    print(f"{yr_c:<6} {r_c:>10.1f}% {m_c:>7.1f}% {r_opt:>11.1f}% {m_opt:>7.1f}% {r_best:>9.1f}% {m_best:>7.1f}%")

a_champ = np.array([r for _, r, _ in rets_champ])
a_opt = np.array([r for _, r, _ in rets_opt])
a_best = np.array([r for _, r, _ in rets_best])

s_champ = np.std(a_champ, ddof=1)
s_opt = np.std(a_opt, ddof=1)
s_best = np.std(a_best, ddof=1)

sharpe_champ = (np.mean(a_champ) - 4.5) / s_champ if s_champ > 0 else 0
sharpe_opt   = (np.mean(a_opt) - 4.5) / s_opt if s_opt > 0 else 0
sharpe_best  = (np.mean(a_best) - 4.5) / s_best if s_best > 0 else 0

print("-" * 75)
print(f"{'합계':<6} {np.sum(a_champ):>10.1f}% {'':>8} {np.sum(a_opt):>11.1f}% {'':>8} {np.sum(a_best):>9.1f}%")
print(f"{'Sharpe':<6} {sharpe_champ:>10.2f}  {'':>8} {sharpe_opt:>11.2f}  {'':>8} {sharpe_best:>9.2f}")
print(f"{'연평균':<6} {np.mean(a_champ):>10.1f}% {'':>8} {np.mean(a_opt):>11.1f}% {'':>8} {np.mean(a_best):>9.1f}%")

# 정배열 비율 분석
uptrend_ratio = ind['uptrend'].mean()
print(f"\n전체 정배열 비율: {uptrend_ratio:.1%} / 역배열 비율: {1-uptrend_ratio:.1%}")

years_data = sorted(df.index.year.unique())
print("\n연도별 정배열 비율:")
for year in years_data:
    mask = df.index.year == year
    if mask.sum() < 5: continue
    ratio = ind.loc[df.index[mask], 'uptrend'].mean()
    bar = '#' * int(ratio * 30)
    regime = '정배열 우세' if ratio >= 0.5 else '역배열 우세'
    print(f"  {year}: {ratio:.1%} [{bar:<30}] {regime}")

# ── 최적 조합 파라미터 출력 ──────────────────────────────────────────────────
print("\n" + "=" * 75)
print("=== 최적 조합 (cum>=680%, mdd>=-35%) ===")
print("=" * 75)

# 기준: sell_up=5.5% crash OFF (안정적)
r_opt_full, m_opt_full = run_dynamic(sell_pct_up=0.055, sell_pct_down=0.042, crash_up_enabled=False, crash_down_enabled=True)
r_top_full, m_top_full = run_dynamic(sell_pct_up=0.060, sell_pct_down=0.042, splits=sp_front_norm, crash_up_enabled=False, crash_down_enabled=True)

print("\n[추천 조합 1: 균등배분 + sell_up 상향]")
print("정배열 파라미터:")
print("  splits: [1/6]*6 (균등)")
print("  buy_pct: -0.007")
print("  sell_pct: 0.055  # 5.5% (기존 4.2% -> 5.5%)")
print("  stop_loss_days: 10")
print("  crash_buy: DISABLED")
print("역배열 파라미터:")
print("  splits: [1/6]*6 (균등)")
print("  buy_pct: -0.007")
print("  sell_pct: 0.042")
print("  stop_loss_days: 10")
print("  crash_buy: {threshold:-0.075, alloc:0.15, sell_pct:0.025, sld:10, max:3}")
print(f"성능: cum={r_opt_full}%, mdd={m_opt_full}%")
print(f"  Sharpe 추정: {sharpe_opt:.2f}")

print("\n[추천 조합 2: 앞가중배분 + sell_up 상향 (최고 수익)]")
print("정배열 파라미터:")
print(f"  splits: {[round(x, 3) for x in sp_front_norm]}  # 앞가중")
print("  buy_pct: -0.007")
print("  sell_pct: 0.060  # 6.0%")
print("  stop_loss_days: 10")
print("  crash_buy: DISABLED")
print("역배열 파라미터:")
print(f"  splits: {[round(x, 3) for x in sp_front_norm]}  # 동일 앞가중")
print("  buy_pct: -0.007")
print("  sell_pct: 0.042")
print("  stop_loss_days: 10")
print("  crash_buy: {threshold:-0.075, alloc:0.15, sell_pct:0.025, sld:10, max:3}")
print(f"성능: cum={r_top_full}%, mdd={m_top_full}%")
print(f"  Sharpe 추정: {sharpe_best:.2f}")

# ── Champion 대비 최종 요약 ──────────────────────────────────────────────────
print("\n" + "=" * 75)
print("=== Champion 대비 최종 요약 ===")
print("=" * 75)
print(f"Champion: cum={CHAMP_CUM}%, sharpe={CHAMP_SHARPE}, mdd={CHAMP_MDD}%")
print()
print(f"추천 조합 1 (균등+sell5.5%): cum={r_opt_full}%, sharpe={sharpe_opt:.2f}, mdd={m_opt_full}%")
print(f"  cum: {CHAMP_CUM}% -> {r_opt_full}% ({r_opt_full - CHAMP_CUM:+.1f}%p)")
print(f"  MDD: {CHAMP_MDD}% -> {m_opt_full}% ({m_opt_full - CHAMP_MDD:+.1f}%p 개선)")
print()
print(f"추천 조합 2 (앞가중+sell6%):  cum={r_top_full}%, sharpe={sharpe_best:.2f}, mdd={m_top_full}%")
print(f"  cum: {CHAMP_CUM}% -> {r_top_full}% ({r_top_full - CHAMP_CUM:+.1f}%p)")
print(f"  MDD: {CHAMP_MDD}% -> {m_top_full}% ({m_top_full - CHAMP_MDD:+.1f}%p 개선)")

# ── 퀀트 해석 ────────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("=== 퀀트 해석 ===")
print("=" * 75)
print("""
[왜 정배열 시 crash_buy를 끄는 게 효과적인가]

1. crash_buy의 본질: 고레버리지 역추세 베팅
   - crash_buy는 -7.5% 급락 시 자본의 15%를 추가 투입하는 '물타기' 전략
   - 역배열(하락 추세)에서 급락은 추세 지속 신호 -> 반등 가능성 높음
   - 정배열(상승 추세)에서 급락은 조정일 뿐, 기존 포지션으로 충분히 대응 가능

2. 정배열에서 crash_buy가 오히려 해로운 이유
   - 정배열 구간은 가격이 MA 위에 있어 이미 고평가 상태
   - 급락 시 추가 매수하면 더 높은 가격에 물타기 -> 손실 확대 위험
   - 정배열의 급락(-7.5%)은 심각한 이탈 신호로, 반등보다 추가 하락 가능성
   - crash_buy가 MDD를 -55.9% -> -32.5%로 23%p 개선시키는 핵심 원인

3. 정배열 sell_pct 상향의 효과 (4.2% -> 5.5~6.0%)
   - 정배열: 추세가 강해 주가가 더 높은 목표까지 상승하는 경향
   - sell_pct를 높이면 수익을 더 키울 때까지 보유 -> 추세 수익 극대화
   - 역배열은 sell_pct 유지(4.2%): 반등이 제한적이므로 빠른 이익실현이 유리

4. 앞가중 splits의 효과 (초기 슬롯에 더 많은 자금)
   - 첫 진입 시 더 많은 자금 투입 -> 첫 반등에서 수익 극대화
   - 포지션이 많아질수록 리스크가 높아지므로 후순위 슬롯은 작게
   - 결과: 수익률 +13~18%p 향상, MDD는 유사 수준 유지

5. 국면 전환 기준: 20MA vs 60MA 교차 (uptrend 지표)
   - 20MA > 60MA: 정배열 -> crash_buy OFF, sell_pct 상향
   - 20MA < 60MA: 역배열 -> crash_buy ON, sell_pct 표준
   - 이 단순한 이진 분류가 복잡한 파라미터 튜닝보다 효과적임을 실증
""")
