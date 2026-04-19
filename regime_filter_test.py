"""
시장 국면(Market Regime) 필터 아이디어 백테스트
Champion 기준선 대비 각 아이디어별 성능 비교
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

# ── Champion 기준선 ─────────────────────────────────────────────
champion = {
    'splits': [1/6]*6,
    'buy_pct': -0.007,
    'sell_pct': 0.042,
    'stop_loss_days': 10,
    'buy_on_stop': True,
    'crash_buy': {
        'enabled': True, 'threshold': -0.075, 'alloc': 0.15,
        'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3
    }
}


def run_full_backtest(params, tc=TC):
    """전체 기간 단일 백테스트"""
    cb = params.get('crash_buy', {})
    crash = cb if cb.get('enabled') else None
    prices = df['close']
    ret, mdd = run_backtest(
        prices=prices,
        splits=params['splits'],
        buy_pct=params['buy_pct'],
        sell_pct=params['sell_pct'],
        stop_loss_days=params['stop_loss_days'],
        buy_on_stop=True,
        crash_buy=crash,
        transaction_cost=tc,
        price_stop_loss_pct=None,
    )
    return ret, mdd


def run_yearly_backtest(params, tc=TC):
    """연도별 백테스트 후 합산"""
    rets, mdds = [], []
    for year in sorted(df.index.year.unique()):
        mask = df.index.year == year
        if mask.sum() < 5:
            continue
        cb = params.get('crash_buy', {})
        crash = cb if cb.get('enabled') else None
        prices = df.loc[mask, 'close']
        ret, mdd = run_backtest(
            prices=prices,
            splits=params['splits'],
            buy_pct=params['buy_pct'],
            sell_pct=params['sell_pct'],
            stop_loss_days=params['stop_loss_days'],
            buy_on_stop=True,
            crash_buy=crash,
            transaction_cost=tc,
            price_stop_loss_pct=None,
        )
        rets.append(ret)
        mdds.append(mdd)
    return np.array(rets), np.array(mdds)


def summarize(idea_name, rets, mdds):
    std = np.std(rets, ddof=1) if len(rets) > 1 else 1
    mean_ret = np.mean(rets)
    sharpe = (mean_ret - 4.5) / std if std > 0 else 0
    return {
        'idea': idea_name,
        'cum': round(float(np.sum(rets)), 1),
        'sharpe': round(float(sharpe), 2),
        'worst_mdd': round(float(np.min(mdds)), 1),
        'mean_annual': round(float(mean_ret), 1),
    }


# ═══════════════════════════════════════════════════════════════
# Champion 기준선
# ═══════════════════════════════════════════════════════════════
rets_champ, mdds_champ = run_yearly_backtest(champion)
champ_result = summarize('Champion', rets_champ, mdds_champ)
print(f"Champion: cum={champ_result['cum']}%, sharpe={champ_result['sharpe']}, worst_mdd={champ_result['worst_mdd']}%")

# 전체 기간 단일 백테스트도 출력
ret_full, mdd_full = run_full_backtest(champion)
print(f"Champion (전체 단일): cum={ret_full}%, mdd={mdd_full}%")


# ═══════════════════════════════════════════════════════════════
# 아이디어 1: 정배열/역배열에 따른 crash_buy 활성 제어
# 정배열(uptrend=1): crash_buy 비활성
# 역배열(uptrend=0): crash_buy 활성
# ═══════════════════════════════════════════════════════════════
print("\n--- 아이디어 1: 정배열/역배열 crash_buy 필터 ---")

def run_idea1_backtest(tc=TC):
    """날짜별로 uptrend 상태에 따라 crash_buy를 동적 적용"""
    prices_all = df['close']
    uptrend_all = ind['uptrend']

    # 슬롯 상태 초기화 (기본 슬롯 6개)
    splits = [1/6]*6
    buy_pct = -0.007
    sell_pct = 0.042
    stop_loss_days = 10

    cb_params = {
        'threshold': -0.075, 'alloc': 0.15,
        'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3
    }

    prices = prices_all.values
    uptrend = uptrend_all.reindex(prices_all.index).values

    slots = [{'active': False, 'entry': 0.0, 'days': 0, 'alloc': s} for s in splits]
    crash_slots = []

    seed = 1.0
    realized_pnl = 0.0
    peak_value = seed
    mdd = 0.0

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
        if i == 0:
            continue

        prev_price = prices[i - 1]
        daily_ret = price / prev_price - 1
        effective_exit = price * (1 - tc)

        # 오늘 uptrend 상태 (NaN이면 역배열로 처리)
        is_uptrend = uptrend[i] == 1.0 if not np.isnan(uptrend[i]) else False

        # 1. 일반 슬롯 매도
        for s in slots:
            if not s['active']:
                continue
            profit_ratio = effective_exit / s['entry'] - 1
            if profit_ratio >= sell_pct or s['days'] >= stop_loss_days:
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

        # 5. 급락 매수: 역배열 구간에서만 활성
        if not is_uptrend and daily_ret <= cb_params['threshold']:
            if len(crash_slots) < cb_params['max_concurrent']:
                crash_slots.append({
                    'entry': price * (1 + tc),
                    'days': 1,
                    'alloc': cb_params['alloc'],
                })

        # 6. MDD 추적
        current_val = portfolio_value(price)
        if current_val > peak_value:
            peak_value = current_val
        dd = (current_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

    # 남은 포지션 강제 청산
    final_price = prices[-1]
    final_exit = final_price * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)

    return round(realized_pnl * 100, 2), round(mdd * 100, 2)

ret1, mdd1 = run_idea1_backtest()
print(f"아이디어1 (전체 단일): cum={ret1}%, mdd={mdd1}%")


# ═══════════════════════════════════════════════════════════════
# 아이디어 2: 변동성 기반 crash_buy alloc 축소
# 20일 변동성 > 임계값 시 crash_buy alloc 절반으로 축소
# ═══════════════════════════════════════════════════════════════
print("\n--- 아이디어 2: 변동성 기반 crash_buy 축소 ---")

def run_idea2_backtest(vol_threshold=0.04, tc=TC):
    """변동성 높을 때 crash_buy alloc 축소"""
    prices_all = df['close']
    vol_all = ind['volatility']

    splits = [1/6]*6
    buy_pct = -0.007
    sell_pct = 0.042
    stop_loss_days = 10

    cb_params = {
        'threshold': -0.075, 'base_alloc': 0.15, 'low_alloc': 0.075,
        'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3
    }

    prices = prices_all.values
    vol = vol_all.reindex(prices_all.index).values

    slots = [{'active': False, 'entry': 0.0, 'days': 0, 'alloc': s} for s in splits]
    crash_slots = []

    seed = 1.0
    realized_pnl = 0.0
    peak_value = seed
    mdd = 0.0

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
        if i == 0:
            continue

        prev_price = prices[i - 1]
        daily_ret = price / prev_price - 1
        effective_exit = price * (1 - tc)

        # 현재 변동성
        cur_vol = vol[i]
        is_high_vol = (not np.isnan(cur_vol)) and (cur_vol > vol_threshold)
        crash_alloc = cb_params['low_alloc'] if is_high_vol else cb_params['base_alloc']

        # 1. 일반 슬롯 매도
        for s in slots:
            if not s['active']:
                continue
            profit_ratio = effective_exit / s['entry'] - 1
            if profit_ratio >= sell_pct or s['days'] >= stop_loss_days:
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

        # 5. 급락 매수 (변동성 따라 alloc 조정)
        if daily_ret <= cb_params['threshold']:
            if len(crash_slots) < cb_params['max_concurrent']:
                crash_slots.append({
                    'entry': price * (1 + tc),
                    'days': 1,
                    'alloc': crash_alloc,
                })

        # 6. MDD 추적
        current_val = portfolio_value(price)
        if current_val > peak_value:
            peak_value = current_val
        dd = (current_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

    # 남은 포지션 강제 청산
    final_price = prices[-1]
    final_exit = final_price * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)

    return round(realized_pnl * 100, 2), round(mdd * 100, 2)

ret2, mdd2 = run_idea2_backtest(vol_threshold=0.04)
print(f"아이디어2 (vol>4%, 전체 단일): cum={ret2}%, mdd={mdd2}%")

ret2b, mdd2b = run_idea2_backtest(vol_threshold=0.05)
print(f"아이디어2 (vol>5%, 전체 단일): cum={ret2b}%, mdd={mdd2b}%")

ret2c, mdd2c = run_idea2_backtest(vol_threshold=0.03)
print(f"아이디어2 (vol>3%, 전체 단일): cum={ret2c}%, mdd={mdd2c}%")


# ═══════════════════════════════════════════════════════════════
# 아이디어 3: sell_pct 동적 조정
# 정배열: sell_pct=0.025 (빠른 이익실현)
# 역배열: sell_pct=0.042 (충분한 반등 대기)
# ═══════════════════════════════════════════════════════════════
print("\n--- 아이디어 3: sell_pct 동적 조정 ---")

def run_idea3_backtest(sell_uptrend=0.025, sell_downtrend=0.042, tc=TC):
    """정배열/역배열에 따라 sell_pct 동적 조정"""
    prices_all = df['close']
    uptrend_all = ind['uptrend']

    splits = [1/6]*6
    buy_pct = -0.007
    stop_loss_days = 10

    cb_params = {
        'threshold': -0.075, 'alloc': 0.15,
        'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3
    }

    prices = prices_all.values
    uptrend = uptrend_all.reindex(prices_all.index).values

    slots = [{'active': False, 'entry': 0.0, 'days': 0, 'alloc': s} for s in splits]
    crash_slots = []

    seed = 1.0
    realized_pnl = 0.0
    peak_value = seed
    mdd = 0.0

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
        if i == 0:
            continue

        prev_price = prices[i - 1]
        daily_ret = price / prev_price - 1
        effective_exit = price * (1 - tc)

        is_uptrend = uptrend[i] == 1.0 if not np.isnan(uptrend[i]) else False
        cur_sell_pct = sell_uptrend if is_uptrend else sell_downtrend

        # 1. 일반 슬롯 매도 (동적 sell_pct)
        for s in slots:
            if not s['active']:
                continue
            profit_ratio = effective_exit / s['entry'] - 1
            if profit_ratio >= cur_sell_pct or s['days'] >= stop_loss_days:
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
        if daily_ret <= cb_params['threshold']:
            if len(crash_slots) < cb_params['max_concurrent']:
                crash_slots.append({
                    'entry': price * (1 + tc),
                    'days': 1,
                    'alloc': cb_params['alloc'],
                })

        # 6. MDD 추적
        current_val = portfolio_value(price)
        if current_val > peak_value:
            peak_value = current_val
        dd = (current_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

    # 남은 포지션 강제 청산
    final_price = prices[-1]
    final_exit = final_price * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)

    return round(realized_pnl * 100, 2), round(mdd * 100, 2)

ret3, mdd3 = run_idea3_backtest(sell_uptrend=0.025, sell_downtrend=0.042)
print(f"아이디어3 (up=2.5%, down=4.2%, 전체 단일): cum={ret3}%, mdd={mdd3}%")

ret3b, mdd3b = run_idea3_backtest(sell_uptrend=0.030, sell_downtrend=0.050)
print(f"아이디어3 (up=3.0%, down=5.0%, 전체 단일): cum={ret3b}%, mdd={mdd3b}%")


# ═══════════════════════════════════════════════════════════════
# 아이디어 4: stop_loss_days 동적 조정
# 역배열: stop_loss_days=7 (빠른 손절)
# 정배열: stop_loss_days=10 (여유)
# ═══════════════════════════════════════════════════════════════
print("\n--- 아이디어 4: stop_loss_days 동적 조정 ---")

def run_idea4_backtest(sld_uptrend=10, sld_downtrend=7, tc=TC):
    """정배열/역배열에 따라 stop_loss_days 동적 조정"""
    prices_all = df['close']
    uptrend_all = ind['uptrend']

    splits = [1/6]*6
    buy_pct = -0.007
    sell_pct = 0.042

    cb_params = {
        'threshold': -0.075, 'alloc': 0.15,
        'sell_pct': 0.025, 'stop_loss_days': 10, 'max_concurrent': 3
    }

    prices = prices_all.values
    uptrend = uptrend_all.reindex(prices_all.index).values

    slots = [{'active': False, 'entry': 0.0, 'days': 0, 'alloc': s} for s in splits]
    crash_slots = []

    seed = 1.0
    realized_pnl = 0.0
    peak_value = seed
    mdd = 0.0

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
        if i == 0:
            continue

        prev_price = prices[i - 1]
        daily_ret = price / prev_price - 1
        effective_exit = price * (1 - tc)

        is_uptrend = uptrend[i] == 1.0 if not np.isnan(uptrend[i]) else False
        cur_sld = sld_uptrend if is_uptrend else sld_downtrend

        # 1. 일반 슬롯 매도 (동적 stop_loss_days)
        for s in slots:
            if not s['active']:
                continue
            profit_ratio = effective_exit / s['entry'] - 1
            if profit_ratio >= sell_pct or s['days'] >= cur_sld:
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
        if daily_ret <= cb_params['threshold']:
            if len(crash_slots) < cb_params['max_concurrent']:
                crash_slots.append({
                    'entry': price * (1 + tc),
                    'days': 1,
                    'alloc': cb_params['alloc'],
                })

        # 6. MDD 추적
        current_val = portfolio_value(price)
        if current_val > peak_value:
            peak_value = current_val
        dd = (current_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

    # 남은 포지션 강제 청산
    final_price = prices[-1]
    final_exit = final_price * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)

    return round(realized_pnl * 100, 2), round(mdd * 100, 2)

ret4, mdd4 = run_idea4_backtest(sld_uptrend=10, sld_downtrend=7)
print(f"아이디어4 (up=10d, down=7d, 전체 단일): cum={ret4}%, mdd={mdd4}%")

ret4b, mdd4b = run_idea4_backtest(sld_uptrend=10, sld_downtrend=5)
print(f"아이디어4 (up=10d, down=5d, 전체 단일): cum={ret4b}%, mdd={mdd4b}%")


# ═══════════════════════════════════════════════════════════════
# 아이디어 1+2 결합: 정배열 필터 + 변동성 축소
# ═══════════════════════════════════════════════════════════════
print("\n--- 아이디어 1+2 결합: 역배열+저변동성 구간에서만 crash_buy 풀 사이즈 ---")

def run_idea12_combined(vol_threshold=0.04, tc=TC):
    """역배열 + 저변동성 구간에서만 crash_buy alloc 풀 사이즈"""
    prices_all = df['close']
    uptrend_all = ind['uptrend']
    vol_all = ind['volatility']

    splits = [1/6]*6
    buy_pct = -0.007
    sell_pct = 0.042
    stop_loss_days = 10

    cb_params = {
        'threshold': -0.075,
        'full_alloc': 0.15,
        'half_alloc': 0.075,
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
        if i == 0:
            continue

        prev_price = prices[i - 1]
        daily_ret = price / prev_price - 1
        effective_exit = price * (1 - tc)

        is_uptrend = uptrend[i] == 1.0 if not np.isnan(uptrend[i]) else False
        cur_vol = vol[i]
        is_high_vol = (not np.isnan(cur_vol)) and (cur_vol > vol_threshold)

        # crash_buy: 역배열 구간에서만 활성, 고변동성이면 alloc 절반
        crash_enabled = not is_uptrend
        if crash_enabled:
            crash_alloc = cb_params['half_alloc'] if is_high_vol else cb_params['full_alloc']
        else:
            crash_alloc = 0  # 정배열이면 crash_buy 비활성

        # 1. 일반 슬롯 매도
        for s in slots:
            if not s['active']:
                continue
            profit_ratio = effective_exit / s['entry'] - 1
            if profit_ratio >= sell_pct or s['days'] >= stop_loss_days:
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
        if crash_enabled and crash_alloc > 0 and daily_ret <= cb_params['threshold']:
            if len(crash_slots) < cb_params['max_concurrent']:
                crash_slots.append({
                    'entry': price * (1 + tc),
                    'days': 1,
                    'alloc': crash_alloc,
                })

        # 6. MDD 추적
        current_val = portfolio_value(price)
        if current_val > peak_value:
            peak_value = current_val
        dd = (current_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

    # 남은 포지션 강제 청산
    final_price = prices[-1]
    final_exit = final_price * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)

    return round(realized_pnl * 100, 2), round(mdd * 100, 2)

ret12, mdd12 = run_idea12_combined(vol_threshold=0.04)
print(f"아이디어1+2 결합 (vol>4%): cum={ret12}%, mdd={mdd12}%")


# ═══════════════════════════════════════════════════════════════
# 아이디어 1+3+4 결합: 모든 동적 조정 통합
# ═══════════════════════════════════════════════════════════════
print("\n--- 아이디어 1+3+4 결합: 역배열 crash_buy + sell_pct 동적 + stop_loss 동적 ---")

def run_idea134_combined(tc=TC):
    """역배열 crash_buy + sell_pct 동적 + stop_loss_days 동적"""
    prices_all = df['close']
    uptrend_all = ind['uptrend']
    vol_all = ind['volatility']

    splits = [1/6]*6
    buy_pct = -0.007

    cb_params = {
        'threshold': -0.075, 'alloc': 0.15,
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
        if i == 0:
            continue

        prev_price = prices[i - 1]
        daily_ret = price / prev_price - 1
        effective_exit = price * (1 - tc)

        is_uptrend = uptrend[i] == 1.0 if not np.isnan(uptrend[i]) else False

        # 동적 파라미터
        cur_sell_pct = 0.025 if is_uptrend else 0.042   # 아이디어3
        cur_sld = 10 if is_uptrend else 7               # 아이디어4
        crash_enabled = not is_uptrend                   # 아이디어1

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

        # 5. 급락 매수 (역배열만)
        if crash_enabled and daily_ret <= cb_params['threshold']:
            if len(crash_slots) < cb_params['max_concurrent']:
                crash_slots.append({
                    'entry': price * (1 + tc),
                    'days': 1,
                    'alloc': cb_params['alloc'],
                })

        # 6. MDD 추적
        current_val = portfolio_value(price)
        if current_val > peak_value:
            peak_value = current_val
        dd = (current_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

    # 남은 포지션 강제 청산
    final_price = prices[-1]
    final_exit = final_price * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)

    return round(realized_pnl * 100, 2), round(mdd * 100, 2)

ret134, mdd134 = run_idea134_combined()
print(f"아이디어1+3+4 결합 (전체 단일): cum={ret134}%, mdd={mdd134}%")


# ═══════════════════════════════════════════════════════════════
# 전체 결합: 1+2+3+4
# ═══════════════════════════════════════════════════════════════
print("\n--- 전체 결합: 아이디어 1+2+3+4 ---")

def run_all_combined(vol_threshold=0.04, tc=TC):
    """역배열 crash_buy(변동성 축소) + sell_pct 동적 + stop_loss_days 동적"""
    prices_all = df['close']
    uptrend_all = ind['uptrend']
    vol_all = ind['volatility']

    splits = [1/6]*6
    buy_pct = -0.007

    cb_params = {
        'threshold': -0.075,
        'full_alloc': 0.15, 'half_alloc': 0.075,
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
        if i == 0:
            continue

        prev_price = prices[i - 1]
        daily_ret = price / prev_price - 1
        effective_exit = price * (1 - tc)

        is_uptrend = uptrend[i] == 1.0 if not np.isnan(uptrend[i]) else False
        cur_vol = vol[i]
        is_high_vol = (not np.isnan(cur_vol)) and (cur_vol > vol_threshold)

        # 동적 파라미터
        cur_sell_pct = 0.025 if is_uptrend else 0.042
        cur_sld = 10 if is_uptrend else 7
        crash_enabled = not is_uptrend
        if crash_enabled:
            crash_alloc = cb_params['half_alloc'] if is_high_vol else cb_params['full_alloc']
        else:
            crash_alloc = 0

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
        if crash_enabled and crash_alloc > 0 and daily_ret <= cb_params['threshold']:
            if len(crash_slots) < cb_params['max_concurrent']:
                crash_slots.append({
                    'entry': price * (1 + tc),
                    'days': 1,
                    'alloc': crash_alloc,
                })

        # 6. MDD 추적
        current_val = portfolio_value(price)
        if current_val > peak_value:
            peak_value = current_val
        dd = (current_val - peak_value) / peak_value
        if dd < mdd:
            mdd = dd

    # 남은 포지션 강제 청산
    final_price = prices[-1]
    final_exit = final_price * (1 - tc)
    for s in slots:
        if s['active']:
            realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)
    for s in crash_slots:
        realized_pnl += s['alloc'] * (final_exit / s['entry'] - 1)

    return round(realized_pnl * 100, 2), round(mdd * 100, 2)

ret_all, mdd_all = run_all_combined(vol_threshold=0.04)
print(f"전체 결합 (vol>4%): cum={ret_all}%, mdd={mdd_all}%")


# ═══════════════════════════════════════════════════════════════
# 최종 요약 출력
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("=== 시장 국면 필터 아이디어별 백테스트 결과 ===")
print("="*60)
print(f"{'전략':<35} {'cum':>8} {'mdd':>8}")
print("-"*55)
print(f"{'Champion (기준선)':<35} {ret_full:>7}% {mdd_full:>7}%")
print(f"{'아이디어1: 역배열만 crash_buy':<35} {ret1:>7}% {mdd1:>7}%")
print(f"{'아이디어2: 변동성>4% alloc 절반':<35} {ret2:>7}% {mdd2:>7}%")
print(f"{'아이디어2: 변동성>5% alloc 절반':<35} {ret2b:>7}% {mdd2b:>7}%")
print(f"{'아이디어2: 변동성>3% alloc 절반':<35} {ret2c:>7}% {mdd2c:>7}%")
print(f"{'아이디어3: sell_pct 동적(2.5/4.2)':<35} {ret3:>7}% {mdd3:>7}%")
print(f"{'아이디어3b: sell_pct 동적(3.0/5.0)':<35} {ret3b:>7}% {mdd3b:>7}%")
print(f"{'아이디어4: stop_loss 동적(10/7일)':<35} {ret4:>7}% {mdd4:>7}%")
print(f"{'아이디어4b: stop_loss 동적(10/5일)':<35} {ret4b:>7}% {mdd4b:>7}%")
print(f"{'결합1+2: 역배열+변동성 필터':<35} {ret12:>7}% {mdd12:>7}%")
print(f"{'결합1+3+4: 역배열+sell+stop':<35} {ret134:>7}% {mdd134:>7}%")
print(f"{'전체결합1+2+3+4':<35} {ret_all:>7}% {mdd_all:>7}%")
print("="*60)

# uptrend 비율 분석
uptrend_ratio_total = ind['uptrend'].mean()
print(f"\n전체 기간 정배열 비율: {uptrend_ratio_total:.1%}")
print(f"전체 기간 역배열 비율: {1-uptrend_ratio_total:.1%}")

# 변동성 분위수 분석
vol_data = ind['volatility'].dropna()
print(f"\n변동성 분포:")
print(f"  25%: {vol_data.quantile(0.25):.4f}")
print(f"  50%: {vol_data.quantile(0.50):.4f}")
print(f"  75%: {vol_data.quantile(0.75):.4f}")
print(f"  90%: {vol_data.quantile(0.90):.4f}")
print(f"  95%: {vol_data.quantile(0.95):.4f}")
