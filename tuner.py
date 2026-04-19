"""
백테스트 오차 분석 및 파라미터 자동 조정

비교 로그(comparison_log.json)에 쌓인 데이터를 기반으로
우리 백테스트와 사이트 수치 사이의 오차를 최소화하는
EVAL_WINDOW 값을 탐색한 뒤 params.json에 반영한다.

[개발자 수정] config.py를 regex로 직접 패치하는 방식 제거.
  기존: config.py 소스코드를 regex로 수정 → 문법 오류 시 앱 부팅 불가 위험.
  수정: settings.update_eval_window()로 params.json에 안전하게 저장.
"""
import json
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from settings import update_eval_window as _save_eval_window

COMPARISON_FILE = "comparison_log.json"
TUNING_LOG_FILE = "tuning_log.json"


# ── 로그 로드 ──────────────────────────────────────────────────
def load_log() -> List[dict]:
    if not os.path.exists(COMPARISON_FILE):
        return []
    with open(COMPARISON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tuning_log() -> List[dict]:
    if not os.path.exists(TUNING_LOG_FILE):
        return []
    with open(TUNING_LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tuning_log(log: list):
    with open(TUNING_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)


# ── 오차 통계 계산 ─────────────────────────────────────────────
def calc_error_stats(log: List[dict]) -> dict:
    """
    비교 로그에서 전략별, 구간별 오차 통계 계산.
    사이트 비교는 transaction_cost=0 기준 (하위호환).
    """
    if not log:
        return {"total_records": 0, "need_tuning": False}

    errors = {"Pro1": [], "Pro2": [], "Pro3": []}
    for entry in log:
        for period in entry.get("구간별비교", []):
            site = period.get("사이트", {})
            ours = period.get("우리", {})
            for name in ["Pro1", "Pro2", "Pro3"]:
                s_ret = site.get(name, {}).get("ret")
                o_ret = ours.get(name, {}).get("ret")
                if s_ret is not None and o_ret is not None:
                    errors[name].append(o_ret - s_ret)

    by_strategy = {}
    all_errors = []
    for name, errs in errors.items():
        if not errs:
            continue
        arr = np.array(errs)
        by_strategy[name] = {
            "mean_abs": round(float(np.mean(np.abs(arr))), 3),
            "max_abs":  round(float(np.max(np.abs(arr))), 3),
            "rmse":     round(float(np.sqrt(np.mean(arr**2))), 3),
            "bias":     round(float(np.mean(arr)), 3),
            "count":    len(arr),
        }
        all_errors.extend(errs)

    all_arr = np.array(all_errors) if all_errors else np.array([0])
    overall_rmse     = round(float(np.sqrt(np.mean(all_arr**2))), 3)
    overall_mean_abs = round(float(np.mean(np.abs(all_arr))), 3)

    return {
        "total_records":    len(log),
        "by_strategy":      by_strategy,
        "overall_rmse":     overall_rmse,
        "overall_mean_abs": overall_mean_abs,
        "need_tuning":      overall_rmse >= 2.0,
    }


# ── EVAL_WINDOW 그리드 탐색 ────────────────────────────────────
def tune_eval_window(df: pd.DataFrame, log: List[dict]) -> dict:
    """EVAL_WINDOW를 15~28 범위에서 탐색해 RMSE를 최소화하는 값 찾기."""
    from backtest import run_backtest
    from config import STRATEGIES

    best_window = None
    best_rmse   = float("inf")
    results     = {}

    for window in range(15, 29):
        errors = []
        for entry in log:
            for period in entry.get("구간별비교", []):
                eval_range = period.get("구간", "")
                parts = eval_range.split("~")
                if len(parts) != 2:
                    continue
                try:
                    start_ts = pd.Timestamp(parts[0].strip())
                    end_ts   = pd.Timestamp(parts[1].strip())
                except Exception:
                    continue

                mask   = (df.index >= start_ts) & (df.index <= end_ts)
                prices = df.loc[mask, "close"]

                if len(prices) > window + 1:
                    prices = prices.iloc[:window + 1]
                if len(prices) < 5:
                    continue

                site = period.get("사이트", {})
                for name, params in STRATEGIES.items():
                    s_ret = site.get(name, {}).get("ret")
                    if s_ret is None:
                        continue
                    # 사이트 비교: transaction_cost=0 (하위호환)
                    our_ret, _ = run_backtest(
                        prices=prices,
                        splits=params["splits"],
                        buy_pct=params["buy_pct"],
                        sell_pct=params["sell_pct"],
                        stop_loss_days=params["stop_loss_days"],
                        buy_on_stop=params["buy_on_stop"],
                    )
                    errors.append(our_ret - s_ret)

        if errors:
            arr  = np.array(errors)
            rmse = float(np.sqrt(np.mean(arr**2)))
            results[window] = round(rmse, 4)
            if rmse < best_rmse:
                best_rmse   = rmse
                best_window = window

    return {
        "best_window": best_window,
        "best_rmse":   round(best_rmse, 4),
        "all_results": results,
    }


# ── EVAL_WINDOW 저장 (params.json 사용, config.py 패치 제거) ──
def update_eval_window(new_window: int) -> str:
    """
    params.json의 EVAL_WINDOW 값을 안전하게 업데이트.
    [수정] 기존 config.py regex 패치 → settings.update_eval_window() 사용.
    """
    _save_eval_window(new_window)
    return f"EVAL_WINDOW {new_window}으로 업데이트 완료 (params.json)"


# ── 전체 조정 실행 ─────────────────────────────────────────────
def run_tuning(df: pd.DataFrame) -> dict:
    """전체 튜닝 파이프라인 실행."""
    from config import EVAL_WINDOW as current_window

    log   = load_log()
    stats = calc_error_stats(log)

    if stats["total_records"] < 3:
        return {
            "status":  "데이터 부족",
            "message": f"비교 로그 {stats['total_records']}건 (최소 3건 필요)",
            "stats":   stats,
        }

    tune_result = tune_eval_window(df, log)
    best_window = tune_result["best_window"]

    current_rmse = tune_result["all_results"].get(current_window, stats["overall_rmse"])
    improved     = best_window != current_window and tune_result["best_rmse"] < current_rmse - 0.05

    msg = ""
    if improved:
        msg = update_eval_window(best_window)

    tuning_entry = {
        "실행시각":         datetime.now().strftime("%Y-%m-%d %H:%M"),
        "비교데이터수":     stats["total_records"],
        "전체RMSE_전":      current_rmse,
        "전체RMSE_후":      tune_result["best_rmse"],
        "이전_EVAL_WINDOW": current_window,
        "새_EVAL_WINDOW":   best_window if improved else current_window,
        "변경여부":         improved,
        "전략별오차":       stats.get("by_strategy", {}),
        "WINDOW_탐색결과":  tune_result["all_results"],
    }

    tlog = load_tuning_log()
    tlog.append(tuning_entry)
    save_tuning_log(tlog)

    return {
        "status":       "개선됨" if improved else "변경없음",
        "message":      msg if improved else f"현재 EVAL_WINDOW={current_window} 최적 (RMSE={current_rmse:.3f})",
        "stats":        stats,
        "tune_result":  tune_result,
        "improved":     improved,
        "tuning_entry": tuning_entry,
    }
