"""
유사 구간 탐색
- 6개 지표를 Z-score 정규화 후 코사인 유사도로 비교
- 현재 구간과 가장 유사한 과거 Top-K 구간 반환

[퀀트 수정] Z-score 정규화 시 target_date 이전 데이터만 사용.
  기존: 전체 이력(미래 포함) 기준 → 룩어헤드 바이어스 발생
  수정: target_date 이전 이력만 사용 → 실전과 동일한 조건
  최소 60일 이전 데이터 확보 불가 시에만 전체 이력 fallback.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

from config import TOP_K, EVAL_WINDOW
from indicators import FEATURE_COLS

_MIN_NORM_HISTORY = 60  # 정규화에 필요한 최소 과거 데이터 수


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def find_similar_periods(
    ind: pd.DataFrame,
    target_date: pd.Timestamp,
    top_k: int = TOP_K,
    min_gap_days: int = 30,
) -> List[Tuple[pd.Timestamp, float]]:
    """
    target_date 기준 지표값과 유사한 과거 날짜 Top-K 반환.

    Parameters
    ----------
    ind           : calc_indicators() 결과 DataFrame
    target_date   : 기준 날짜
    top_k         : 반환할 유사 구간 수
    min_gap_days  : 결과 간 최소 간격 (중복 구간 방지, 영업일)

    Returns
    -------
    [(past_date, similarity_pct), ...]  — similarity_pct 는 0~100
    """
    valid = ind[FEATURE_COLS].dropna()

    if target_date not in valid.index:
        raise ValueError(f"target_date {target_date.date()} 의 지표값이 없습니다.")

    target_vec = valid.loc[target_date, FEATURE_COLS].values.astype(float)

    # ── [수정] target_date 이전 데이터만으로 Z-score 정규화 ──────
    # 룩어헤드 바이어스 제거: 미래 데이터의 통계가 과거 분석에 개입하지 않도록.
    hist_for_norm = valid.loc[valid.index < target_date, FEATURE_COLS]

    if len(hist_for_norm) >= _MIN_NORM_HISTORY:
        mean = hist_for_norm.mean()
        std  = hist_for_norm.std().replace(0, 1)
    else:
        # 초기 구간 fallback (데이터가 너무 적을 때만)
        mean = valid[FEATURE_COLS].mean()
        std  = valid[FEATURE_COLS].std().replace(0, 1)

    target_z = (target_vec - mean.values) / std.values

    # ── 과거 날짜만 비교 (평가 구간 보장) ──────────────────────────
    eval_dates = valid.index[valid.index < target_date]
    try:
        last_eval_start = ind.index[-EVAL_WINDOW - 1]
    except IndexError:
        last_eval_start = ind.index[0]

    eval_dates = eval_dates[eval_dates <= last_eval_start]

    if len(eval_dates) == 0:
        return []

    hist_matrix = valid.loc[eval_dates, FEATURE_COLS].values
    hist_z = (hist_matrix - mean.values) / std.values

    # ── 코사인 유사도 계산 ──────────────────────────────────────
    norms_hist  = np.linalg.norm(hist_z, axis=1)
    norm_target = np.linalg.norm(target_z)
    denom       = norms_hist * norm_target
    denom       = np.where(denom == 0, 1e-9, denom)
    similarities = hist_z.dot(target_z) / denom
    similarities = np.clip(similarities, -1, 1)

    # ── Top-K (겹치지 않게 min_gap_days 적용) ──────────────────
    sorted_idx = np.argsort(similarities)[::-1]
    results: List[Tuple[pd.Timestamp, float]] = []
    selected_dates: List[pd.Timestamp] = []

    for idx in sorted_idx:
        date = eval_dates[idx]
        sim  = similarities[idx] * 100

        too_close = any(
            abs((date - sd).days) < min_gap_days for sd in selected_dates
        )
        if too_close:
            continue

        results.append((date, round(sim, 2)))
        selected_dates.append(date)

        if len(results) >= top_k:
            break

    return results
