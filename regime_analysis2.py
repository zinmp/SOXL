"""
시장 국면 필터 추가 분석
- 연도별 수익률/MDD 비교
- Sharpe 비교 (연도별 분산 기반)
- 최적 조합 탐색
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, r'c:\Users\ze1po\OneDrive\바탕 화면\프로젝트\떨사오팔')

from data import get_data
from backtest import run_backtest
from indicators import calc_indicators

df = get_data()
ind = calc_indicators(df)
TC = 0.001


def run_regime_backtest_daily(
    sell_uptrend=0.042, sell_downtrend=0.042,
    sld_uptrend=10, sld_downtrend=10,
    crash_uptrend=True, crash_downtrend=True,
    crash_alloc_base=0.15, crash_alloc_high_vol=0.15,
    vol_threshold=999.0,
    tc=TC
):
    """완전 통합 regime 필터 백테스트"""
    prices_all = df['close']
    uptrend_all = ind['uptrend']
    vol_all = ind['volatility']

    splits = [1/6]*6
    buy_pct = -0.007

    cb_params = {
        'threshold': -0.075,
        'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3
    }

    prices = prices_all.values
    uptrend = uptrend_all.reindex(prices_all.index).values
    vol = vol_all.reindex(prices_all.index).values

    slots = [{'active': False, 'entry': 0.0, 'days': 0, 'alloc': s} for s in splits]
    crash_slots = []

    seed = 1.0
    realized_pnl = 0.0
    peak_value = seed
    mdd = 0.0

    # 연도별 추적
    years = prices_all.index.year
    year_start_pnl = {}
    year_peak = {}
    year_mdd = {}
    year_ret = {}

    def portfolio_value(current_price):
        unrealized = sum(
            s['alloc'] * (current_price / s['entry'] - 1)
            for s in slots if s['active']
        )
        unrealized += sum(
            s['alloc'] * (current_price / s['entry'] - 1)
            for s in crash_slots
        )
        return seed + realized_pnl + unrealized

    for i, price in enumerate(prices):
        year = years[i]
        if year not in year_start_pnl:
            year_start_pnl[year] = portfolio_value(price)
            year_peak[year] = year_start_pnl[year]
            year_mdd[year] = 0.0

        if i == 0:
            continue

        prev_price = prices[i - 1]
        daily_ret = price / prev_price - 1
        effective_exit = price * (1 - tc)

        is_uptrend = uptrend[i] == 1.0 if not np.isnan(uptrend[i]) else False
        cur_vol = vol[i]
        is_high_vol = (not np.isnan(cur_vol)) and (cur_vol > vol_threshold)

        # 동적 파라미터
        cur_sell_pct = sell_uptrend if is_uptrend else sell_downtrend
        cur_sld = sld_uptrend if is_uptrend else sld_downtrend
        crash_active = crash_uptrend if is_uptrend else crash_downtrend
        crash_alloc = crash_alloc_high_vol if is_high_vol else crash_alloc_base

        # 1. 일반 슬롯 매도
        for s in slots:
            if not s['active']:
                continue
            profit_ratio = effective_exit / s['entry'] - 1
            if profit_ratio >= cur_sell_pct or s['days'] >= cur_sld:
                realized_pnl += s['alloc'] * profit_ratio
                s['active'] = False
                s['days'] = 0

        # 2. 급락 슬롯 매도
        closed = []
        for s in crash_slots:
            profit_ratio = effective_exit / s['entry'] - 1
            if profit_ratio >= cb_params['sell_pct'] or s['days'] >= cb_params['stop_loss_days']:
                realized_pnl += s['alloc'] * profit_ratio
                closed.append(s)
        for s in closed:
            crash_slots.remove(s)

        # 3. 보유 슬롯 일수 증가
        for s in slots:
            if s['active']:
                s['days'] += 1
        for s in crash_slots:
            s['days'] += 1

        # 4. 일반 매수
        if daily_ret <= buy_pct:
            for s in slots:
                if not s['active']:
                    s['active'] = True
                    s['entry'] = price * (1 + tc)
                    s['days'] = 1
                    break

        # 5. 급락 매수
        if crash_active and crash_alloc > 0 and daily_ret <= cb_params['threshold']:
            if len(crash_slots) < cb_params['max_concurrent']:
                crash_slots.append({
                    'entry': price * (1 + tc),
                    'days': 1,
                    'alloc': crash_alloc,
                })

        # 6. MDD 추적 (전체 및 연도별)
        current_val = portfolio_value(price)
        if current_val > peak_value:
            peak_value = current_val
        dd = (current_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

        if current_val > year_peak[year]:
            year_peak[year] = current_val
        year_dd = (current_val - year_peak[year]) / year_peak[year]
        if year_dd < year_mdd[year]:
            year_mdd[year] = year_dd

    # 연도별 수익률 계산
    year_end_pnl = {}
    for year in year_start_pnl:
        mask_year = (df.index.year == year)
        last_price = df['close'][mask_year].iloc[-1]
        # 해당 연도 마지막 날의 포트폴리오 가치 (미실현 포함)
        # 근사: 강제 청산 없이 마지막 날 기준
        pass

    # 남은 포지션 강제 청산
    final_price = prices[-1]
    final_exit = final_price * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)

    total_return = round(realized_pnl * 100, 2)
    total_mdd = round(mdd * 100, 2)
    return total_return, total_mdd


# 연도별 백테스트로 Sharpe 계산
def run_yearly_with_sharpe(
    sell_uptrend=0.042, sell_downtrend=0.042,
    sld_uptrend=10, sld_downtrend=10,
    crash_uptrend_enabled=True, crash_downtrend_enabled=True,
    crash_alloc_normal=0.15, crash_alloc_high_vol=0.15,
    vol_threshold=999.0,
    tc=TC, name=""):

    rets, mdds = [], []
    for year in sorted(df.index.year.unique()):
        mask = df.index.year == year
        if mask.sum() < 5:
            continue
        prices_yr = df['close'][mask]
        uptrend_yr = ind['uptrend'].reindex(prices_yr.index).values
        vol_yr = ind['volatility'].reindex(prices_yr.index).values

        # 해당 연도 uptrend 비율로 파라미터 선택 (단순화)
        uptrend_ratio = np.nanmean(uptrend_yr)

        # 연도 내 세밀한 구분 대신: 연도 평균 uptrend 비율로 파라미터 혼합
        # sell_pct: uptrend_ratio 가중평균
        eff_sell = sell_uptrend * uptrend_ratio + sell_downtrend * (1 - uptrend_ratio)
        eff_sld = int(round(sld_uptrend * uptrend_ratio + sld_downtrend * (1 - uptrend_ratio)))

        # crash_buy: 역배열 비율이 40% 이상이면 활성
        if uptrend_ratio < 0.6 and crash_downtrend_enabled:
            crash_enabled = True
        elif uptrend_ratio >= 0.6 and crash_uptrend_enabled:
            crash_enabled = True
        else:
            crash_enabled = False

        # 변동성 평균
        vol_mean = np.nanmean(vol_yr)
        if vol_mean > vol_threshold:
            crash_alloc = crash_alloc_high_vol
        else:
            crash_alloc = crash_alloc_normal

        cb = {
            'enabled': crash_enabled,
            'threshold': -0.075,
            'alloc': crash_alloc,
            'sell_pct': 0.025,
            'stop_loss_days': 10,
            'max_concurrent': 3
        } if crash_enabled else None

        ret, mdd_yr = run_backtest(
            prices=prices_yr,
            splits=[1/6]*6,
            buy_pct=-0.007,
            sell_pct=eff_sell,
            stop_loss_days=eff_sld,
            buy_on_stop=True,
            crash_buy=cb,
            transaction_cost=tc,
            price_stop_loss_pct=None,
        )
        rets.append(ret)
        mdds.append(mdd_yr)

    rets = np.array(rets)
    mdds = np.array(mdds)
    std = np.std(rets, ddof=1) if len(rets) > 1 else 1.0
    sharpe = (np.mean(rets) - 4.5) / std if std > 0 else 0

    return {
        'name': name,
        'cum': round(float(np.sum(rets)), 1),
        'sharpe': round(float(sharpe), 2),
        'worst_mdd': round(float(np.min(mdds)), 1),
        'mean_annual': round(float(np.mean(rets)), 1),
    }


# Champion 기준
champ = run_yearly_with_sharpe(
    sell_uptrend=0.042, sell_downtrend=0.042,
    sld_uptrend=10, sld_downtrend=10,
    crash_uptrend_enabled=True, crash_downtrend_enabled=True,
    crash_alloc_normal=0.15, crash_alloc_high_vol=0.15,
    vol_threshold=999.0,
    name="Champion"
)

# 아이디어1: 역배열만 crash_buy
idea1 = run_yearly_with_sharpe(
    sell_uptrend=0.042, sell_downtrend=0.042,
    sld_uptrend=10, sld_downtrend=10,
    crash_uptrend_enabled=False, crash_downtrend_enabled=True,
    crash_alloc_normal=0.15, crash_alloc_high_vol=0.15,
    vol_threshold=999.0,
    name="아이디어1: 역배열 crash_buy"
)

# 아이디어2a: vol>4% alloc 절반
idea2a = run_yearly_with_sharpe(
    crash_alloc_normal=0.15, crash_alloc_high_vol=0.075,
    vol_threshold=0.04,
    name="아이디어2(vol>4% half)"
)

# 아이디어2b: vol>3% alloc 절반
idea2b = run_yearly_with_sharpe(
    crash_alloc_normal=0.15, crash_alloc_high_vol=0.075,
    vol_threshold=0.03,
    name="아이디어2(vol>3% half)"
)

# 아이디어3: sell_pct 동적
idea3 = run_yearly_with_sharpe(
    sell_uptrend=0.025, sell_downtrend=0.042,
    name="아이디어3(sell 2.5/4.2)"
)

# 아이디어4: stop_loss_days 동적
idea4 = run_yearly_with_sharpe(
    sld_uptrend=10, sld_downtrend=7,
    name="아이디어4(sld 10/7)"
)

# 결합 1+2: 역배열 crash_buy + vol 필터
combo12 = run_yearly_with_sharpe(
    crash_uptrend_enabled=False, crash_downtrend_enabled=True,
    crash_alloc_normal=0.15, crash_alloc_high_vol=0.075,
    vol_threshold=0.04,
    name="결합1+2(역배열+vol)"
)

# 결합 1+3: 역배열 crash_buy + sell_pct 동적
combo13 = run_yearly_with_sharpe(
    sell_uptrend=0.025, sell_downtrend=0.042,
    crash_uptrend_enabled=False, crash_downtrend_enabled=True,
    name="결합1+3(역배열+sell)"
)

# 결합 1+3+4
combo134 = run_yearly_with_sharpe(
    sell_uptrend=0.025, sell_downtrend=0.042,
    sld_uptrend=10, sld_downtrend=7,
    crash_uptrend_enabled=False, crash_downtrend_enabled=True,
    name="결합1+3+4"
)

# 전체 결합 1+2+3+4
combo_all = run_yearly_with_sharpe(
    sell_uptrend=0.025, sell_downtrend=0.042,
    sld_uptrend=10, sld_downtrend=7,
    crash_uptrend_enabled=False, crash_downtrend_enabled=True,
    crash_alloc_normal=0.15, crash_alloc_high_vol=0.075,
    vol_threshold=0.04,
    name="전체결합1+2+3+4"
)

results = [champ, idea1, idea2a, idea2b, idea3, idea4, combo12, combo13, combo134, combo_all]

print("\n" + "="*75)
print("=== 시장 국면 필터 아이디어별 백테스트 결과 (연도별 Sharpe) ===")
print("="*75)
print(f"{'전략':<35} {'cum':>8} {'sharpe':>8} {'worst_mdd':>10} {'mean_ann':>10}")
print("-"*75)
for r in results:
    print(f"{r['name']:<35} {r['cum']:>7}% {r['sharpe']:>8.2f} {r['worst_mdd']:>9}% {r['mean_annual']:>9}%")

print("="*75)

# 최적 아이디어 찾기 (MDD 개선 + 수익률 유지)
best = None
best_score = float('-inf')
for r in results[1:]:  # Champion 제외
    mdd_improv = abs(champ['worst_mdd']) - abs(r['worst_mdd'])  # 양수 = 개선
    cum_loss = champ['cum'] - r['cum']  # 양수 = 수익 감소
    # 점수: MDD 개선 가중치 2배, 수익 감소 페널티
    score = mdd_improv * 2 - cum_loss * 0.05
    if score > best_score:
        best_score = score
        best = r

print(f"\n=== 최적 아이디어 ===")
print(f"구현 방법: {best['name']}")
print(f"성능: cum={best['cum']}%, sharpe={best['sharpe']:.2f}, worst_mdd={best['worst_mdd']}%")

print(f"\n=== Champion 대비 ===")
print(f"cum: {champ['cum']}% -> {best['cum']}% ({best['cum']-champ['cum']:+.1f}%p)")
print(f"MDD: {champ['worst_mdd']}% -> {best['worst_mdd']}% ({abs(champ['worst_mdd'])-abs(best['worst_mdd']):+.1f}%p 개선)")
print(f"Sharpe: {champ['sharpe']:.2f} -> {best['sharpe']:.2f} ({best['sharpe']-champ['sharpe']:+.2f})")
