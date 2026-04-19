"""
SOXL 데이터 다운로드 및 캐싱
"""
import os
import pickle
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import TICKER, DATA_START, CACHE_FILE


def get_data(ticker: str = TICKER, start: str = DATA_START, refresh: bool = False) -> pd.DataFrame:
    """
    SOXL 일봉 데이터 반환.
    캐시 파일이 있고 최신이면 캐시 사용, 아니면 yfinance로 재다운로드.
    """
    if not refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            df = pickle.load(f)
        # 가장 최근 영업일보다 캐시가 최신이면 그대로 사용
        last_date = df.index[-1].date()
        most_recent_bday = pd.bdate_range(end=datetime.today().date(), periods=1)[0].date()
        if last_date >= most_recent_bday:
            return df

    print(f"[data] {ticker} 데이터 다운로드 중... (start={start})")
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"{ticker} 데이터를 가져올 수 없습니다.")

    # MultiIndex 컬럼 처리
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(df, f)

    print(f"[data] {len(df)}일 데이터 저장 완료 ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df
