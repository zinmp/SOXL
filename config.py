"""
떨사오팔 전략 파라미터 설정

런타임 변경 가능한 값(STRATEGIES, EVAL_WINDOW)은 params.json에서 로드.
불변 상수만 이 파일에 유지.
"""
from settings import load_params as _load_params

# ─── 불변 상수 ──────────────────────────────────────────────────
TICKER      = 'SOXL'
DATA_START  = '2010-01-01'
CACHE_FILE  = 'soxl_data.pkl'

# ─── 지표 파라미터 ─────────────────────────────────────────────
MA_SHORT      = 20
MA_LONG       = 60
MA_SLOPE_DAYS = 10
RSI_PERIOD    = 14
ROC_PERIOD    = 12
VOL_PERIOD    = 20

# ─── 유사도 탐색 ────────────────────────────────────────────────
ANALYSIS_WINDOW = 21   # 분석 구간 (영업일)
TOP_K           = 3    # 유사 구간 상위 N개

# ─── 점수 공식 (퀀트 개선: 제곱 MDD 페널티) ───────────────────
# Score = Σ [ (sim_i / Σsim) × return_i(%) × max(0, 1 + MDD_i/100)² ]
# MDD=-80% → (0.2)²=0.04 → 96% 페널티 (기존 exp(-0.8)≈0.45 보다 훨씬 강함)
MDD_WEIGHT = 1.0

# ─── 거래비용 (퀀트 개선: 실거래 비용 반영) ─────────────────────
# 사이트 비교(tab2/tab3/tuner)는 0.0 사용, 실전 추천/통계는 이 값 사용
TRANSACTION_COST    = 0.001   # 편도 0.1% (수수료 + 슬리피지)
PRICE_STOP_LOSS_PCT = None     # 가격 기반 손절 비활성 (수익률 극대화 우선, 리스크는 자산 분산으로 관리)

# ─── 런타임 파라미터 (params.json에서 로드) ──────────────────────
_p = _load_params()

STRATEGIES          = _p.get("STRATEGIES", {})
EVAL_WINDOW         = _p.get("EVAL_WINDOW", 22)
BASELINE_STRATEGIES = _p.get("BASELINE_STRATEGIES", _p.get("STRATEGIES", {}))
