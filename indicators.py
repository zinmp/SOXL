"""
6개 기술 지표 계산
  1. uptrend   : 정배열 (20MA > 60MA) → 0/1
  2. slope     : 20MA 기울기 (10일 변화율)
  3. deviation : 이격도 (주가/20MA - 1)
  4. rsi       : RSI(14)
  5. roc       : ROC(12) (12일 변화율)
  6. volatility: 20일 역사적 변동성 (log return std)
"""
import numpy as np
import pandas as pd

from config import MA_SHORT, MA_LONG, MA_SLOPE_DAYS, RSI_PERIOD, ROC_PERIOD, VOL_PERIOD


def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    df : open/high/low/close/volume 컬럼을 가진 일봉 DataFrame
    반환 : 6개 지표 + 보조 컬럼
    """
    close = df['close']

    ma20 = close.rolling(MA_SHORT).mean()
    ma60 = close.rolling(MA_LONG).mean()

    ind = pd.DataFrame(index=df.index)
    ind['close']      = close
    ind['ma20']       = ma20
    ind['ma60']       = ma60
    ind['uptrend']    = (ma20 > ma60).astype(float)
    ind['slope']      = ma20 / ma20.shift(MA_SLOPE_DAYS) - 1
    ind['deviation']  = close / ma20 - 1
    ind['rsi']        = _calc_rsi(close, RSI_PERIOD)
    ind['roc']        = close / close.shift(ROC_PERIOD) - 1
    log_ret           = np.log(close / close.shift(1))
    ind['volatility'] = log_ret.rolling(VOL_PERIOD).std()

    return ind


# 유사도 계산에 사용할 6개 피처 컬럼명
FEATURE_COLS = ['uptrend', 'slope', 'deviation', 'rsi', 'roc', 'volatility']


def get_feature_vector(ind_row: pd.Series) -> np.ndarray:
    """단일 날짜의 지표값을 numpy array로 반환"""
    return ind_row[FEATURE_COLS].values.astype(float)
