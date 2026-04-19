"""
런타임 파라미터 관리 (params.json 읽기/쓰기)

config.py 소스코드를 직접 수정하는 위험한 방식을 대체.
- atomic write로 파일 손상 방지
- 문법 오류 없이 안전하게 파라미터 저장/로드
"""
import json
from pathlib import Path

_BASE = Path(__file__).parent
_PARAMS_FILE = _BASE / "params.json"


def load_params() -> dict:
    """params.json 로드. 파일 없으면 빈 dict 반환."""
    if not _PARAMS_FILE.exists():
        return {}
    with open(_PARAMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_params(params: dict):
    """params.json에 atomic write (tmp → replace로 중간 손상 방지)."""
    tmp = _PARAMS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    tmp.replace(_PARAMS_FILE)


def get_strategies() -> dict:
    return load_params().get("STRATEGIES", {})


def get_eval_window() -> int:
    return load_params().get("EVAL_WINDOW", 22)


def get_baseline_strategies() -> dict:
    p = load_params()
    return p.get("BASELINE_STRATEGIES", p.get("STRATEGIES", {}))


def update_eval_window(new_window: int):
    """EVAL_WINDOW만 업데이트."""
    params = load_params()
    params["EVAL_WINDOW"] = new_window
    save_params(params)


def update_strategies(new_strategies: dict, save_baseline: bool = True):
    """
    STRATEGIES 업데이트.
    save_baseline=True면 현재 STRATEGIES를 BASELINE_STRATEGIES로 먼저 보존.
    """
    params = load_params()
    if save_baseline:
        params["BASELINE_STRATEGIES"] = {
            name: {k: v for k, v in strat.items()}
            for name, strat in params.get("STRATEGIES", {}).items()
        }
    params["STRATEGIES"] = new_strategies
    save_params(params)
