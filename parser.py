"""
사이트 텍스트 붙여넣기 파서
"""
import re
from datetime import datetime


def parse_site_text(text: str) -> dict:
    """
    사이트에서 복사한 텍스트를 파싱해서 구조화된 dict 반환.

    Returns:
    {
        "ref_date": "2026-04-05",
        "analysis_start": "2026-03-03",
        "analysis_end": "2026-04-02",
        "uptrend": False,
        "slope": -10.68,
        "deviation": 1.86,
        "rsi": 49.11,
        "roc": -4.00,
        "volatility": 0.3748,
        "periods": [
            {
                "analysis_start": "2018-12-07",
                "analysis_end": "2019-01-07",
                "eval_start": "2019-01-07",
                "eval_end": "2019-02-06",
                "similarity": 96.92,
                "uptrend": False,
                "Pro1": {"ret": 3.6, "mdd": -0.2},
                "Pro2": {"ret": 6.2, "mdd": -0.4},
                "Pro3": {"ret": 9.2, "mdd": -0.7},
            },
            ...
        ],
        "scores": {"Pro1": 2.803, "Pro2": 2.391, "Pro3": 4.073},
        "recommended": "Pro3",
    }
    """
    result = {}
    lines = [l.strip() for l in text.strip().splitlines()]
    lines = [l for l in lines if l]  # 빈 줄 제거

    # ── 기준일 ─────────────────────────────────────────────────
    ref_date = None
    for l in lines:
        m = re.search(r'추천 기준일[:\s]*(\d{4}-\d{2}-\d{2})', l)
        if m:
            ref_date = m.group(1)
            break
    result["ref_date"] = ref_date

    # ── 분석 구간 ───────────────────────────────────────────────
    for l in lines:
        m = re.search(r'분석 구간[:\s]*(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})', l)
        if m:
            result["analysis_start"] = m.group(1)
            result["analysis_end"]   = m.group(2)
            break

    # ── 현재 지표 (첫 번째 블록) ───────────────────────────────
    # 정배열 여부: 첫 ✅/❌ 를 정배열로 판단
    uptrend_marks = []
    for l in lines:
        if l in ("✅", "❌"):
            uptrend_marks.append(l == "✅")
    result["uptrend"] = uptrend_marks[0] if uptrend_marks else None

    # 기울기, 이격도, RSI, ROC, 변동성 — 첫 번째 등장값
    def first_match(pattern, lines, cast=float):
        for l in lines:
            m = re.search(pattern, l)
            if m:
                return cast(m.group(1))
        return None

    # 기울기(20ma 10일) 라인 다음 값
    result["slope"]     = _get_value_after(lines, r'기울기', r'([+-]?\d+\.?\d*)%')
    result["deviation"] = _get_value_after(lines, r'이격도', r'([+-]?\d+\.?\d*)%')
    # RSI: "RSI(14)" 라벨 다음 줄의 숫자 (라벨 자체의 14는 무시)
    result["rsi"]       = _get_next_line_number(lines, r'RSI\(\d+\)')
    result["roc"]       = _get_value_after(lines, r'ROC', r'([+-]?\d+\.?\d*)%')
    result["volatility"]= _get_value_after(lines, r'변동성', r'(0\.\d+)')

    # ── 유사 구간 파싱 ─────────────────────────────────────────
    periods = []
    i = 0
    while i < len(lines):
        # 분석 구간 범위 패턴 (과거 구간)
        m = re.match(r'^(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})$', lines[i])
        if m and len(periods) < 3:
            p_start = m.group(1)
            p_end   = m.group(2)

            # 성과 확인 기간
            eval_start, eval_end = None, None
            for j in range(i+1, min(i+4, len(lines))):
                em = re.search(r'(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})', lines[j])
                if em:
                    eval_start = em.group(1)
                    eval_end   = em.group(2)
                    break

            # 유사도
            sim = None
            for j in range(i+1, min(i+6, len(lines))):
                sm = re.search(r'유사도[:\s]*([0-9.]+)%', lines[j])
                if sm:
                    sim = float(sm.group(1))
                    break

            # 정배열 (이 구간의)
            p_uptrend = None
            for j in range(i+1, min(i+10, len(lines))):
                if lines[j] in ("✅", "❌"):
                    p_uptrend = lines[j] == "✅"
                    break

            # Pro1/2/3 수익률 & MDD
            pro_data = {"Pro1": {}, "Pro2": {}, "Pro3": {}}
            cur_pro = None
            for j in range(i+1, min(i+30, len(lines))):
                l = lines[j]
                for pname in ["Pro1", "Pro2", "Pro3"]:
                    if pname in l and "떨사" in l:
                        cur_pro = pname
                        break
                if cur_pro:
                    rm = re.search(r'수익률\s*([+-]?\d+\.?\d*)%', l)
                    mm = re.search(r'MDD\s*([+-]?\d+\.?\d*)%', l)
                    if rm:
                        pro_data[cur_pro]["ret"] = float(rm.group(1))
                    if mm:
                        pro_data[cur_pro]["mdd"] = float(mm.group(1))
                        # MDD는 음수로 통일
                        if pro_data[cur_pro]["mdd"] > 0:
                            pro_data[cur_pro]["mdd"] *= -1

                # 다음 구간이 시작되면 중단
                next_m = re.match(r'^(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})$', l)
                if next_m and j > i + 2:
                    break

            if eval_start and sim is not None:
                periods.append({
                    "analysis_start": p_start,
                    "analysis_end":   p_end,
                    "eval_start":     eval_start,
                    "eval_end":       eval_end,
                    "similarity":     sim,
                    "uptrend":        p_uptrend,
                    "Pro1":           pro_data["Pro1"],
                    "Pro2":           pro_data["Pro2"],
                    "Pro3":           pro_data["Pro3"],
                })

        i += 1

    result["periods"] = periods

    # ── 점수 ───────────────────────────────────────────────────
    scores = {}
    for pname in ["Pro1", "Pro2", "Pro3"]:
        for j, l in enumerate(lines):
            if pname in l and ("점수" in l or "떨사" in l):
                # 다음 줄이 숫자인 경우
                if j + 1 < len(lines):
                    nm = re.match(r'^([+-]?\d+\.?\d+)$', lines[j+1])
                    if nm:
                        scores[pname] = float(nm.group(1))
                        break
                # 같은 줄에 숫자
                nm = re.search(r'([+-]?\d+\.\d+)\s*$', l)
                if nm:
                    scores[pname] = float(nm.group(1))
                    break
    result["scores"] = scores

    # ── 추천 전략 ───────────────────────────────────────────────
    for l in lines:
        m = re.search(r'추천 전략[:\s]*(Pro\d)', l)
        if m:
            result["recommended"] = m.group(1)
            break

    return result


def _get_next_line_number(lines, label_pattern):
    """라벨 줄 바로 다음 줄에서 숫자 추출 (라벨 자체의 숫자는 무시)"""
    for i, l in enumerate(lines):
        if re.search(label_pattern, l, re.IGNORECASE):
            for j in range(i+1, min(i+3, len(lines))):
                m = re.match(r'^([+-]?\d+\.?\d*)$', lines[j])
                if m:
                    return float(m.group(1))
    return None


def _get_value_after(lines, label_pattern, value_pattern):
    """라벨 줄 다음에 오는 값(또는 같은 줄) 추출"""
    for i, l in enumerate(lines):
        if re.search(label_pattern, l, re.IGNORECASE):
            # 같은 줄에 값이 있으면
            m = re.search(value_pattern, l)
            if m:
                return float(m.group(1))
            # 다음 줄에서 찾기
            for j in range(i+1, min(i+3, len(lines))):
                m = re.search(value_pattern, lines[j])
                if m:
                    return float(m.group(1))
    return None
