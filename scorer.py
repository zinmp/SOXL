"""
전략 점수 계산

[퀀트 개선] 기존 exp(MDD/100) 페널티 → 제곱 페널티로 교체.

기존 공식: Score = Σ [ (sim_i / Σsim) × return_i(%) × exp(MDD_i/100) ]
  문제: MDD=-80%여도 exp(-0.8)≈0.45 → 55% 감소만. 극단 손실 과소 처벌.

개선 공식: Score = Σ [ (sim_i / Σsim) × return_i(%) × max(0, 1 + MDD_i/100)² ]
  효과:
    MDD= -5% → (0.95)² = 0.90  (10% 페널티)
    MDD=-10% → (0.90)² = 0.81  (19% 페널티)
    MDD=-30% → (0.70)² = 0.49  (51% 페널티, 기존 26%)
    MDD=-50% → (0.50)² = 0.25  (75% 페널티, 기존 39%)
    MDD=-80% → (0.20)² = 0.04  (96% 페널티, 기존 55%)
  → 큰 낙폭을 매우 강하게 페널티화하여 리스크 관리 강화.
"""
from typing import List, Tuple


def calc_score(
    period_results: List[Tuple[float, float, float]]
) -> float:
    """
    period_results : [(similarity, return_pct, mdd_pct), ...]
      similarity   : 유사도 (0~100)
      return_pct   : 수익률 (%, e.g. 9.2)
      mdd_pct      : MDD (%, 음수, e.g. -3.5)

    Returns
    -------
    종합 점수 (float)
    """
    total_sim = sum(r[0] for r in period_results)
    if total_sim == 0:
        return 0.0

    score = 0.0
    for sim, ret, mdd in period_results:
        w = sim / total_sim
        # 제곱 MDD 페널티: max(0, 1 + mdd/100)²
        # mdd는 음수(%) → 1 + mdd/100 ∈ (0, 1]
        mdd_factor = max(0.0, 1.0 + mdd / 100.0) ** 2
        score += w * ret * mdd_factor

    return round(score, 3)
