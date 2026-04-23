"""
SOXL 데이터 다운로드 및 캐싱
"""
import os
import pickle
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from config import TICKER, DATA_START, CACHE_FILE


def _get_latest_available_bday() -> "datetime.date":
    """
    yfinance에서 실제로 조회 가능한 가장 최근 영업일을 반환한다.

    미국 동부시간(ET) 기준 장 마감은 16:00 = UTC 20:00.
    UTC 20:00 이후면 오늘 데이터가 포함되므로 오늘까지,
    UTC 20:00 이전이면 아직 장 마감 전이므로 전일까지만 반환한다.
    """
    now_utc = datetime.now(timezone.utc)
    market_closed_today = now_utc.hour >= 20
    if market_closed_today:
        reference_date = now_utc.date()
    else:
        reference_date = (now_utc - pd.Timedelta(days=1)).date()
    return pd.bdate_range(end=reference_date, periods=1)[0].date()


def get_data(ticker: str = TICKER, start: str = DATA_START, refresh: bool = False) -> pd.DataFrame:
    """
    SOXL 일봉 데이터 반환.
    캐시 파일이 있고 최신이면 캐시 사용, 아니면 yfinance로 재다운로드.
    """
    most_recent_bday = _get_latest_available_bday()

    if not refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            df = pickle.load(f)
        last_date = df.index[-1].date()
        if last_date >= most_recent_bday:
            return df

    print(f"[data] {ticker} 데이터 다운로드 중... (start={start})")

    # yfinance 최신 버전은 curl_cffi로 브라우저 impersonation 내장 — session 파라미터 불필요
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if raw.empty:
        # 캐시가 있으면 오래된 것이라도 반환
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        raise ValueError(f"{ticker} 데이터를 가져올 수 없습니다. 네트워크를 확인하세요.")

    # MultiIndex 컬럼 처리
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(df, f)

    print(f"[data] {len(df)}일 데이터 저장 완료 ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df
