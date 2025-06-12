# 통합-HBAR_최종.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import csv
import os
import numpy as np  # 누락된 numpy import 추가
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input  # Input 추가
from tensorflow.keras.models import load_model
import joblib

# ✅ 반드시 첫 번째 Streamlit 명령어로 설정
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------- AI 예측 모델 ----------------------
def create_prediction_model():
    # 개선된 LSTM 모델 구조
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(64, activation='tanh', return_sequences=True),
        LSTM(32, activation='tanh'),
        Dense(16, activation='relu'),
        Dense(5)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def predict_next_5(df, timeframe):
    """다음 5개 봉 예측 (LSTM 기반)"""
    if len(df) < 60:
        return [], []
    
    # 데이터 정규화
    prices = df['close'].values[-60:]
    min_val, max_val = prices.min(), prices.max()
    norm_prices = (prices - min_val) / (max_val - min_val + 1e-8)
    
    # 모델 생성/예측
    model = create_prediction_model()
    model.fit(norm_prices.reshape(1, 60, 1), 
              norm_prices[-5:].reshape(1, 5), 
              epochs=10, verbose=0)
    
    # 예측 및 역정규화
    prediction = model.predict(norm_prices.reshape(1, 60, 1))
    denorm_pred = prediction[0] * (max_val - min_val) + min_val
    
    # 예측 시간 생성
    last_time = df['datetime'].iloc[-1]
    pred_times = [last_time + timedelta(minutes=timeframe*(i+1)) 
                 for i in range(5)]
    
    return denorm_pred, pred_times

# --- 자동 학습 및 예측 결과 CSV 저장 함수 ---
def auto_train_and_predict(df, selected_tf, model_dir="ai_models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"lstm_{selected_tf}.h5")
    scaler_path = os.path.join(model_dir, f"scaler_{selected_tf}.pkl")
    csv_path = os.path.join(model_dir, f"pred_{selected_tf}.csv")

    # 데이터 준비
    if len(df) < 60:
        return [], []
    prices = df['close'].values[-60:]
    min_val, max_val = prices.min(), prices.max()
    norm_prices = (prices - min_val) / (max_val - min_val + 1e-8)

    # 모델/스케일러 로드 또는 학습
    model = None
    scaler = {'min': min_val, 'max': max_val}
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
        except Exception:
            model = None
    if model is None:
        model = create_prediction_model()
        model.fit(norm_prices.reshape(1, 60, 1), norm_prices[-5:].reshape(1, 5), epochs=10, verbose=0)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

    # 예측 및 역정규화
    prediction = model.predict(norm_prices.reshape(1, 60, 1))
    denorm_pred = prediction[0] * (scaler['max'] - scaler['min']) + scaler['min']

    # 예측 시간 생성
    last_time = df['datetime'].iloc[-1]
    pred_times = [last_time + timedelta(minutes=selected_tf*(i+1)) for i in range(5)]

    # CSV 저장
    pred_df = pd.DataFrame({
        "datetime": pred_times,
        "predicted": denorm_pred
    })
    pred_df.to_csv(csv_path, index=False)

    return denorm_pred, pred_times

# ---------------------- 공통 함수 정의 ----------------------
def send_telegram_alert(message: str):
    bot_token = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
    chat_id = "7890657899"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        res = requests.post(url, data=payload, timeout=5)
        if res.status_code != 200:
            st.error(f"Telegram 전송 실패: {res.status_code} {res.text}")
        else:
            st.session_state.alerts.append(f"텔레그램 전송: {message}")
    except Exception as e:
        st.error(f"Telegram 알림 전송 실패: {str(e)}")

# ---------------------- 실시간 알림 내역(메시지 로그) 표시 ----------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("📢 알림 내역")
    # 최근 10개 알림만 표시
    alert_history = st.session_state.get("alerts", [])
    if alert_history:
        for alert in reversed(alert_history[-10:]):
            st.markdown(
                f"<div style='background:#222;padding:8px 10px;margin:4px 0;border-radius:6px;font-size:14px;color:#39FF14;'>{alert}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("알림 내역이 없습니다.")

def parse_number(x):
    """문자열/숫자 혼합 데이터를 float으로 변환 (단, '-' 문자열은 그대로 반환)"""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            if x.strip() == "-":
                return x
            # 부호는 남기고, 콤마만 제거
            cleaned = x.replace(',', '').strip()
            return float(cleaned) if cleaned else 0.0
        return 0.0
    except Exception:
        return 0.0
    
def wma(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def hma(series, period=16):
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma_half = wma(series, half_length)
    wma_full = wma(series, period)
    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_length)

def hma3(series, period=3):
    return hma(series, period)

def rsi(series, period=7):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def bollinger_bands(series, period=20, std_factor=2):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    return ma, upper, lower

def cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md)

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_chart_patterns(df):
    patterns = []
    if len(df) < 7: return patterns

    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    atr_vals = df['ATR'].values

    # W 패턴 (거래량 + 변동성 필터)
    if len(df) >= 5:
        first_low = lows[-5]
        second_low = lows[-1]
        if (abs(first_low - second_low) < 0.02*(first_low+second_low)/2 and
            volumes[-1] > np.mean(volumes[-5:])*1.2 and
            atr_vals[-1] > np.mean(atr_vals[-5:])*0.7):
            patterns.append("W 패턴 (확률 80%)")
            patterns.append("🚀 5~15분 후 상승 확률 ↑")

    # 역 헤드앤숄더 (다중 조건)
    if len(df) >= 7:
        head_low = lows[-4]
        left_shoulder = lows[-6]
        right_shoulder = lows[-2]
        neckline = (highs[-7] + highs[-5] + highs[-3] + highs[-1])/4
        if (left_shoulder > head_low < right_shoulder and
            abs(left_shoulder - right_shoulder) < 0.03*head_low and
            df['close'].iloc[-1] > neckline and
            volumes[-4] > np.mean(volumes)*1.5):
            patterns.append("역 헤드앤숄더 (확률 85%)")
            patterns.append("🚀 10~30분 후 상승 확률 ↑")

    # 상승 깃발 (추세 지속)
    if len(df) >= 8:
        last8 = df.iloc[-8:]
        range_cond = last8['high'] - last8['low']
        if (np.all(range_cond[:3] > 2*range_cond.mean()) and
            np.all(np.diff(last8['close'][3:]) < 0.02*last8['close'].mean())):
            patterns.append("상승 깃발패턴(추세 지속)")
            patterns.append("🚀 10~20분 후 추가 상승 가능성 ↑")

    # 하락 쐐기 (상승 반전)
    if len(df) >= 10:
        last10 = df.iloc[-10:]
        upper = last10['high'].rolling(3).max().dropna()
        lower = last10['low'].rolling(3).min().dropna()
        if (np.all(np.diff(upper) < 0) and
            np.all(np.diff(lower) < 0)):
            patterns.append("하락 쐐기(상승 반전 예측)")
            patterns.append("🚀 15~30분 후 반등 가능성 ↑")

    # 컵 앤 핸들 (상승 지속)
    if len(df) >= 20:
        last20 = df.iloc[-20:]
        cup = last20[:15]['low'].values
        handle = last20[15:]['low'].values
        if (np.all(cup[0] > cup[1:7]) and
            np.all(cup[-5:] > cup[7]) and
            np.all(handle[-3:] > handle[0])):
            patterns.append("컵 앤 핸들(상승 지속)")
            patterns.append("🚀 30~60분 후 강한 상승 신호")

    # 삼각 수렴 (상승/하락 브레이크아웃)
    if len(df) >= 10:
        last10 = df.iloc[-10:]
        upper = last10['high'].values
        lower = last10['low'].values
        if (np.all(np.diff(upper) < 0) and
            np.all(np.diff(lower) > 0)):
            patterns.append("삼각 수렴(추세 모멘텀)")
            patterns.append("⚡ 10~30분 내 방향성 돌파 가능성 ↑")

    return patterns

# 개선된 신호 점수 계산 함수
def calculate_signal_score(df, latest, pred_prices=None):
    buy_score, sell_score, risk_score = 0, 0, 0
    buy_reasons, sell_reasons, risk_reasons = [], [], []
    latest_rsi = latest.get('RSI', 0) if not df.empty else 0

    # 지표별 가중치
    WEIGHTS = {
        'hma_cross': 1.5,
        'rsi': 1.0,
        'macd': 1.2,
        'volume': 1.0,
        'bb_breakout': 1.3,
        'pattern': 2.0,
        'ai_predict': 2.0,
        'risk': 1.0
    }

    try:
        required_keys = ['HMA3', 'HMA', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
        if not all(key in latest for key in required_keys):
            raise ValueError("필수 기술적 지표 데이터 누락")

        # HMA 교차
        if latest['HMA3'] > latest['HMA']:
            buy_score += WEIGHTS['hma_cross'] * 2
            buy_reasons.append("HMA3 > HMA (가중 2점)")
        if latest['HMA3'] < latest['HMA']:
            sell_score += WEIGHTS['hma_cross'] * 2
            sell_reasons.append("HMA3 < HMA (가중 2점)")

        # RSI
        if latest['RSI'] > 30:
            buy_score += WEIGHTS['rsi'] * 1
            buy_reasons.append("RSI > 30")
        if latest['RSI'] > 70:
            sell_score += WEIGHTS['rsi'] * 2
            sell_reasons.append("RSI > 70")

        # MACD 히스토그램
        if latest['MACD_hist'] > 0:
            buy_score += WEIGHTS['macd'] * 1
            buy_reasons.append("MACD > 0")
        if latest['MACD_hist'] < 0:
            sell_score += WEIGHTS['macd'] * 1
            sell_reasons.append("MACD < 0")

        # 거래량
        if len(df) > 1 and latest['volume'] > df.iloc[-2]['volume']:
            buy_score += WEIGHTS['volume'] * 1
            buy_reasons.append("거래량 증가")
        if len(df) > 1 and latest['volume'] < df.iloc[-2]['volume']:
            sell_score += WEIGHTS['volume'] * 1
            sell_reasons.append("거래량 감소")

        # 볼린저 밴드 돌파
        if latest['close'] < latest['BB_lower']:
            buy_score += WEIGHTS['bb_breakout'] * 1
            buy_reasons.append("BB 하단 이탈")
        if latest['close'] > latest['BB_upper']:
            sell_score += WEIGHTS['bb_breakout'] * 1
            sell_reasons.append("BB 상단 돌파")

        # AI 예측 반영
        if pred_prices is not None and len(pred_prices) > 0:
            predicted = pred_prices[-1]
            if predicted > latest['close']:
                buy_score += WEIGHTS['ai_predict'] * 1
                buy_reasons.append("AI 예측 상승")
            elif predicted < latest['close']:
                sell_score += WEIGHTS['ai_predict'] * 1
                sell_reasons.append("AI 예측 하락")

        # 리스크: 최근 변동성 급등 (가격 변화 or 거래량 폭등)
        if abs(df['close'].pct_change().iloc[-1]) > 0.03:
            risk_score += 1
            risk_reasons.append("단기 급등/급락 경고")
        if latest['volume'] > df['volume'].mean() * 4:
            risk_score += 1
            risk_reasons.append("비정상 거래량")

    except Exception as e:
        buy_reasons.append(f"오류 발생: {str(e)}")
        sell_reasons.append(f"오류 발생: {str(e)}")

    return buy_score, sell_score, risk_score, buy_reasons, sell_reasons, risk_reasons, latest_rsi

def calculate_signal_score_old(df, latest):
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []
    latest_rsi = latest.get('RSI', 0) if not df.empty else 0

    # --- 이동평균선 계산 (5, 10, 20, 60, 120, 200) ---
    ma_periods = [5, 10, 20, 60, 120, 200]
    for period in ma_periods:
        col = f"SMA{period}"
        if col not in df.columns:
            df[col] = df['close'].rolling(period).mean()

    # 골든크로스/데드크로스 감지 및 점수 반영
    cross_msgs = []
    # (5,20), (20,60), (60,120), (120,200) 등 주요 구간만 체크
    cross_pairs = [(5, 20), (20, 60), (60, 120), (120, 200)]
    for short, long in cross_pairs:
        s_col = f"SMA{short}"
        l_col = f"SMA{long}"
        if s_col in df.columns and l_col in df.columns and len(df) > long:
            prev_short = df[s_col].iloc[-2]
            prev_long = df[l_col].iloc[-2]
            curr_short = df[s_col].iloc[-1]
            curr_long = df[l_col].iloc[-1]
            # 골든크로스: 단기선이 장기선을 아래에서 위로 돌파
            if prev_short < prev_long and curr_short > curr_long:
                buy_score += 3
                msg = f"{short}/{long} 골든크로스 발생 (3점)"
                buy_reasons.append(msg)
                cross_msgs.append(f"⚡ {short}/{long} 골든크로스 발생!")
            # 데드크로스: 단기선이 장기선을 위에서 아래로 돌파
            if prev_short > prev_long and curr_short < curr_long:
                sell_score += 3
                msg = f"{short}/{long} 데드크로스 발생 (3점)"
                sell_reasons.append(msg)
                cross_msgs.append(f"⚡ {short}/{long} 데드크로스 발생!")
            # 임박(±1% 이내)
            if abs(curr_short - curr_long) / curr_long < 0.01:
                if curr_short > curr_long:
                    cross_msgs.append(f"⏳ {short}/{long} 골든크로스 임박")
                else:
                    cross_msgs.append(f"⏳ {short}/{long} 데드크로스 임박")

    if df.empty or len(df) < 2:
        return buy_score, sell_score, buy_reasons, sell_reasons, latest_rsi

    try:
        required_keys = ['HMA3', 'HMA', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
        if not all(key in latest for key in required_keys):
            raise ValueError("필수 기술적 지표 데이터 누락")

        # 1. 기존 신호
        if latest['HMA3'] > latest['HMA']:
            buy_score += 3
            buy_reasons.append("HMA3 > HMA (3점)")
        if latest['RSI'] > 30:
            buy_score += 2
            buy_reasons.append(f"RSI({latest['RSI']:.1f}) > 40 (2점)")
        if latest['MACD_hist'] > 0 and latest['MACD_hist'] > df.iloc[-2]['MACD_hist']:
            buy_score += 1
            buy_reasons.append("MACD 히스토그램 > 0 (1점)")
        if len(df) > 1 and latest['volume'] > df.iloc[-2]['volume']:
            buy_score += 2
            buy_reasons.append("거래량 증가 (2점)")
        if latest['close'] < latest['BB_lower']:
            buy_score += 2
            buy_reasons.append("가격 < BB 하한선 (2점)")

        if latest['HMA3'] < latest['HMA']:
            sell_score += 3
            sell_reasons.append("HMA3 < HMA (3점)")
        if latest['RSI'] > 70:
            sell_score += 2
            sell_reasons.append(f"RSI({latest['RSI']:.1f}) > 70 (2점)")
        if latest['MACD_hist'] < 0:
            sell_score += 1
            sell_reasons.append("MACD 히스토그램 < 0 (1점)")
        if len(df) > 1 and latest['volume'] < df.iloc[-2]['volume']:
            sell_score += 2
            sell_reasons.append("거래량 감소 (2점)")
        if latest['close'] > latest['BB_upper']:
            sell_score += 2
            sell_reasons.append("가격 > BB 상한선 (2점)")

        # 2. 볼린저 밴드 압축 점수
        if 'BB_width' in df.columns and len(df) > 20:
            bb_width_ratio = latest['BB_width'] / df['BB_width'].mean()
            if bb_width_ratio < 0.7:
                buy_score += 1
                sell_score += 1
                buy_reasons.append("볼린저 밴드 압축 (1점)")
                sell_reasons.append("볼린저 밴드 압축 (1점)")

        # 3. 볼린저 밴드 위치 + 거래량 연동
        if latest['close'] < latest['BB_lower']:
            if latest['volume'] > df['volume'].rolling(10).mean().iloc[-1] * 1.5:
                buy_score += 3
                buy_reasons.append("하한선 돌파 + 거래량 폭등 (3점)")
        elif latest['close'] > latest['BB_upper']:
            if latest['volume'] > df['volume'].rolling(10).mean().iloc[-1] * 1.5:
                sell_score += 3
                sell_reasons.append("상한선 돌파 + 거래량 폭등 (3점)")

        # 4. 볼린저 밴드 상대 위치 점수
        try:
            bb_position = (latest['close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
            if bb_position < 0.2:
                add = max(0, int(2 * (1 - bb_position/0.2)))
                buy_score += add
                buy_reasons.append(f"볼린저 하한선 근접 ({bb_position:.2f})")
            elif bb_position > 0.8:
                add = max(0, int(2 * ((bb_position-0.8)/0.2)))
                sell_score += add
                sell_reasons.append(f"볼린저 상한선 근접 ({bb_position:.2f})")
        except Exception:
            pass

        # 5. 거래량 모멘텀 점수
        if 'volume_momentum' in df.columns and len(df) > 3:
            vol_momentum = df['volume_momentum'].iloc[-3:].mean()
            if vol_momentum > 1.8 and latest['close'] > df['close'].iloc[-2]:
                buy_score += 2
                buy_reasons.append("거래량 가속 상승 (2점)")
            if vol_momentum > 1.8 and latest['close'] < df['close'].iloc[-2]:
                sell_score += 2
                sell_reasons.append("거래량 가속 하락 (2점)")

        # 6. 거래량 클러스터링(상위 거래량) 점수
        if len(df) > 20:
            vol_percentile = df['volume'].rank(pct=True).iloc[-1]
            if vol_percentile > 0.9:
                if latest['close'] > df['open'].iloc[-1]:
                    buy_score += 2
                    buy_reasons.append("고거래량 양봉 (2점)")
                else:
                    sell_score += 2
                    sell_reasons.append("고거래량 음봉 (2점)")

        # 7. 패턴 기반 점수 보정
        patterns = detect_chart_patterns(df)
        for pattern in patterns:
            if "상승" in pattern or "W 패턴" in pattern:
                buy_score += 3
                buy_reasons.append(f"패턴 보정: {pattern} (3점)")
            elif "하락" in pattern or "M 패턴" in pattern:
                sell_score += 3
                sell_reasons.append(f"패턴 보정: {pattern} (3점)")

    except Exception as e:
        st.error(f"신호 계산 오류: {str(e)}")
        buy_reasons = [f"오류 발생: {str(e)}"]
        sell_reasons = [f"오류 발생: {str(e)}"]

    # --- 골든/데드크로스 임박/발생 알림 ---
    if cross_msgs:
        now = datetime.now()
        for msg in cross_msgs:
            alert_key = f"ma_cross_{msg}"
            last_alert_time = st.session_state.last_alert_time.get(alert_key, datetime.min)
            # 10분 이내 중복 알림 방지
            if (now - last_alert_time) > timedelta(minutes=10):
                st.session_state.alerts.append(f"🚨 {msg} ({now.strftime('%H:%M:%S')})")
                if 'telegram_enabled' in globals() and telegram_enabled:
                    send_telegram_alert(f"🚨 {msg} ({now.strftime('%H:%M:%S')})")
                st.session_state.last_alert_time[alert_key] = now

    return buy_score, sell_score, buy_reasons, sell_reasons, latest_rsi

# ---------------------- 초기 설정 ----------------------
# 형광 녹색 스타일(검정배경+녹색글씨) 적용
DOS_GREEN = "#39FF14"
DOS_BG = "#000000"
st.markdown(
    f"""
    <style>
    body, .stApp {{
        background-color: {DOS_BG} !important;
    }}
    .dos-green, .stMarkdown, .stDataFrame, .stTable, .stText, .stSubheader, .stHeader, .stTitle, .stMetric, .stSidebar, .stSidebarContent, .stSidebar .stTextInput, .stSidebar .stNumberInput, .stSidebar .stCheckbox, .stSidebar .stSelectbox, .stSidebar .stMetric {{
        color: {DOS_GREEN} !important;
        background-color: {DOS_BG} !important;
    }}
    .stMetric label, .stMetric div, .stMetric span {{
        color: {DOS_GREEN} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 메인 타이틀/서브타이틀 형광녹색, 폰트 두단계 작게(h3/h5)
st.markdown(f"<h3 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 If you're not desperate, don't even think about it!</h3>", unsafe_allow_html=True)
st.markdown(f"<h5 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 Live Cryptocurrency Analytics Dashboard</h5>", unsafe_allow_html=True)
st_autorefresh(interval=5000, key="auto_refresh")

# ---------------------- 전역 변수 ----------------------
default_holdings = {
    'KRW-STX': 13702.725635935, #13580.63263846, #13700,  #13092.79231352, 15743.86
    'KRW-HBAR': 62216.22494886, #62248.73694539, #1595.26778585,  61507.68894681 57267.71799311 57703.21327885
    'KRW-DOGE': 61194.37067502,
    }
markets = list(default_holdings.keys())
timeframes = {1: '1분', 3: '3분', 5: '5분', 15: '15분', 60: '60분', 240: '240분'}
TOTAL_INVESTMENT = 58500000

# ---------------------- 데이터 함수 ----------------------
@st.cache_data(ttl=30)
def fetch_ohlcv(market, timeframe, count=300):
    try:
        url = f"https://api.upbit.com/v1/candles/minutes/{timeframe}"
        params = {'market': market, 'count': count}
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        df = pd.DataFrame(res.json())[::-1]

        if df.empty:
            st.warning(f"{market} 데이터가 비어 있습니다.")
            return pd.DataFrame()

        df = df[['candle_date_time_kst','opening_price','high_price','low_price','trade_price','candle_acc_trade_volume']]
        df.columns = ['datetime','open','high','low','close','volume']
        df['datetime'] = pd.to_datetime(df['datetime'])

        df['HL2'] = (df['high'] + df['low']) / 2
        df['HMA'] = hma(df['HL2'])
        df['HMA3'] = hma3(df['HL2'])
        df['Signal'] = np.where(df['HMA3'] > df['HMA'], '매수', '매도')
        df['RSI'] = rsi(df['close'])
        df['MACD_line'], df['Signal_line'], df['MACD_hist'] = macd(df['close'])
        df['BB_ma'], df['BB_upper'], df['BB_lower'] = bollinger_bands(df['close'])
        df['CCI'] = cci(df)
        df['ATR'] = atr(df)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_ma']
        df['volume_momentum'] = df['volume'] / df['volume'].shift(1)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"데이터 요청 실패: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"데이터 처리 오류: {str(e)}")
        return pd.DataFrame()

# 예측에 필요한 최소 데이터 수 계산 함수
def get_required_candle_count(selected_tf, prediction_steps, window=60):
    """
    학습에 필요한 최소 캔들 수 계산
    - window: 예측에 필요한 과거 데이터 길이 (예: 60봉)
    - prediction_steps: 예측할 봉 개수
    """
    return window + prediction_steps + 10  # 여유분 포함

# 업비트 캔들 데이터 다운로드 함수
def fetch_candles(market="KRW-BTC", interval="minute1", count=200, to=None):
    """
    업비트에서 분봉 데이터 수집
    - interval 예: minute1, minute3, minute5, minute15, minute30, minute60, minute240
    """
    url = f"https://api.upbit.com/v1/candles/{interval}"
    params = {"market": market, "count": count}
    if to:
        params["to"] = to

    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        df = pd.DataFrame(response.json())
        df = df.rename(columns={
            "candle_date_time_kst": "time",
            "opening_price": "open",
            "high_price": "high",
            "low_price": "low",
            "trade_price": "close",
            "candle_acc_trade_volume": "volume"
        })
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        return df
    else:
        print("Error fetching data:", response.status_code)
        return pd.DataFrame()

# 예측할 봉 개수 자동 계산 함수 (예시)
def get_prediction_steps(selected_tf):
    # 예: 1분봉~15분봉은 5, 60분봉 이상은 3
    if selected_tf < 60:
        return 5
    else:
        return 3

# ---------------------- 모바일 대응 및 사용자 입력 세션 상태 초기화 ----------------------
if 'holdings' not in st.session_state:
    st.session_state.holdings = {
        'KRW-STX': 13702.725635935,
        'KRW-HBAR': 62216.22494886,
        'KRW-DOGE': 61194.37067502,
    }
if 'total_investment' not in st.session_state:
    st.session_state.total_investment = 58500000
if 'telegram_bot_token' not in st.session_state:
    st.session_state.telegram_bot_token = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
if 'telegram_chat_id' not in st.session_state:
    st.session_state.telegram_chat_id = "7890657899"

# ---------------------- 모바일 레이아웃 및 CSS ----------------------
st.markdown("""
<style>
@media (max-width: 768px) {
    .table-wrapper {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    body, .stSelectbox, .stNumberInput, .stTextInput, .stButton > button {
        font-size: 14px !important;
    }
    .stPlotlyChart {
        height: 300px !important;
    }
    .stSidebar {
        padding: 5px !important;
    }
    .stTabs [role="tablist"] button {
        padding: 8px 12px !important;
        font-size: 12px !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------------- 사이드바 사용자 입력 ----------------------
with st.sidebar:
    st.header("🔑 사용자 설정")
    st.subheader("💰 보유 코인 설정")
    for market in ['KRW-STX', 'KRW-HBAR', 'KRW-DOGE']:
        coin = market.split('-')[1]
        st.session_state.holdings[market] = st.number_input(
            f"{coin} 보유량",
            value=st.session_state.holdings.get(market, 0.0),
            step=0.0001,
            format="%.4f",
            key=f"holding_{market}"
        )
    st.session_state.total_investment = st.number_input(
        "총 투자금액 (원)",
        value=st.session_state.total_investment,
        step=100000,
        format="%d"
    )
    st.subheader("✉️ 텔레그램 설정")
    st.session_state.telegram_bot_token = st.text_input(
        "텔레그램 봇 토큰",
        value=st.session_state.telegram_bot_token
    )
    st.session_state.telegram_chat_id = st.text_input(
        "텔레그램 채팅 ID",
        value=st.session_state.telegram_chat_id
    )
    st.subheader("AI 예측 설정")
    pred_horizon = st.slider("예측 기간 (봉)", 1, 10, 5)
    ai_confidence = st.slider("신뢰도 임계값", 50, 100, 80)
    st.subheader("차트 주기(분) 선택")
    selected_tf = st.selectbox("차트 주기", list(timeframes.keys()), format_func=lambda x: timeframes[x], key="selected_tf", index=2)

# ---------------------- send_telegram_alert 함수 수정 ----------------------
def send_telegram_alert(message: str):
    bot_token = st.session_state.telegram_bot_token
    chat_id = st.session_state.telegram_chat_id
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        res = requests.post(url, data=payload, timeout=5)
        if res.status_code != 200:
            st.error(f"Telegram 전송 실패: {res.status_code} {res.text}")
        else:
            st.session_state.alerts.append(f"텔레그램 전송: {message}")
    except Exception as e:
        st.error(f"Telegram 알림 전송 실패: {str(e)}")

# ---------------------- generate_coin_table 함수에서 holdings/total_investment 사용 ----------------------
def generate_coin_table(selected_tf):
    signal_scores = {} 
    base_market = 'KRW-STX'
    base_qty = st.session_state.holdings[base_market]
    base_price_data = prices.get(base_market, {'trade_price': 0, 'signed_change_rate': 0})
    base_price = base_price_data['trade_price']
    base_krw = base_qty * base_price * 0.9995

    compare_data = []
    telegram_msgs = []  # 텔레그램 전송용 메시지 리스트
    for market in markets:
        coin = market.split('-')[1]
        price_data = prices.get(market, {'trade_price': 0, 'signed_change_rate': 0})
        price = price_data['trade_price']
        change_rate = price_data['signed_change_rate'] * 100
        qty = st.session_state.holdings[market]
        value = price * qty
        
        df = fetch_ohlcv(market, selected_tf)
        buy_score, sell_score = 0, 0
        latest_rsi = 0
        pattern_msgs = []

        if not df.empty and len(df) >= 2:
            try:
                latest = df.iloc[-1]
                required_cols = ['HMA', 'HMA3', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
                if all(col in df.columns for col in required_cols):
                    buy_score, sell_score, _, _, _, _, latest_rsi = calculate_signal_score(df, latest, pred_prices=None)
                    # 패턴 감지 및 텔레그램 메시지 준비
                    patterns = detect_chart_patterns(df)
                    if patterns:
                        pattern_msgs = [f"{coin} {p}" for p in patterns]
                        for msg in pattern_msgs:
                            telegram_msgs.append(msg)
                else:
                    st.error(f"{coin} 데이터 컬럼 누락")
            except Exception as e:
                st.error(f"{coin} 신호 계산 오류: {str(e)}")
        else:
            st.warning(f"{coin} 데이터 부족으로 분석 생략")
        
        if market != base_market:
            replace_qty = (base_krw * 0.9995) / price if price else 0
            diff_qty = replace_qty - qty
            replace_value = replace_qty * price * 0.9995
        else:
            replace_qty = "-"
            diff_qty = "-"
            replace_value = "-"
        
        change_color = "red" if change_rate < 0 else "green"
        change_emoji = "🔻" if change_rate < 0 else "🟢"
        buy_color = "green" if buy_score >= 7 else "gray"
        sell_color = "red" if sell_score >= 7 else "gray"
        
        compare_data.append({
            '코인명': coin,
            '시세': f"{price:,.1f} 원",
            'RSI': f"{latest_rsi:.1f}",
            '매수신호': f"<span style='color:{buy_color}'>매수({buy_score}/10)</span>",
            '매도신호': f"<span style='color:{sell_color}'>매도({sell_score}/10)</span>",
            '등락률': f"<span style='color:{change_color}'>{change_emoji} {change_rate:+.2f}%</span>",
            '보유수량': qty,
            '대체가능수량': replace_qty if market != base_market else "-",
            '차이수량': diff_qty if market != base_market else "-",
            '평가금액': value,
            '대체평가액': replace_value if market != base_market else "-"
        })

    # 비교테이블 요약 텔레그램 메시지 전송 (최초 탭 진입 시 1회만)
    if telegram_msgs:
        now = datetime.now().strftime("%H:%M:%S")
        for msg in telegram_msgs:
            send_telegram_alert(f"[{now}] {msg}")

    df_compare = pd.DataFrame(compare_data)
    df_compare['차이수량'] = df_compare['차이수량'].apply(lambda x: float(x) if x != "-" else x)
    df_compare['대체가능수량'] = df_compare['대체가능수량'].apply(lambda x: float(x) if x != "-" else x)
    return df_compare

def format_number(x):
    if isinstance(x, (int, float)):
        if x == "-": return x
        return f"{x:,.2f}" if abs(x) < 10000 else f"{x:,.0f}"
    return x

# ---------------------- 탭 구성 ----------------------
prices = get_current_prices()

tab1, tab2 = st.tabs(["📊 코인 비교 테이블", "📈 코인 분석"])

with tab1:
    st.subheader("📊 코인 비교 테이블 (RSI 포함)")
    df_compare = generate_coin_table(selected_tf)
    
    def diff_qty_color(val):
        try:
            v = float(val)
            if v > 0: return "color:#FF2222"
            elif v < 0: return "color:#00BFFF"
            else: return f"color:{DOS_GREEN}"
        except: return f"color:{DOS_GREEN}"

    def change_rate_color(val):
        try:
            import re
            match = re.search(r'([+\-]?\d+(\.\d+)?)', str(val))
            num = float(match.group(1)) if match else 0
            if '-' in str(val): return "color:#00BFFF"
            elif '+' in str(val): return "color:#FF2222"
            elif num > 0: return "color:#FF2222"
            elif num < 0: return "color:#00BFFF"
            else: return f"color:{DOS_GREEN}"
        except: return f"color:{DOS_GREEN}"

    styled = (
        df_compare.style.format({
            '보유수량': format_number,
            '대체가능수량': format_number,
            '차이수량': lambda x: f"{x:+.0f}" if isinstance(x, (int, float)) else x,
            '평가금액': format_number,
            '대체평가액': format_number,
            'RSI': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
        })
        .map(diff_qty_color, subset=['차이수량'])
        .map(change_rate_color, subset=['등락률'])
        .map(lambda _: 'text-align: center')
        # .hide_index()  # hide_index() 미지원 환경에서는 아래처럼 직접 index 숨김
    )
    # 인덱스 숨기기: to_html(escape=False, index=False) 사용
    st.markdown('<div class="table-wrapper">' + styled.to_html(escape=False, index=False) + '</div>', unsafe_allow_html=True)

with tab2:
    # 2코인 분석 탭: RSI 비교 차트 
    st.subheader("📈 RSI 비교 차트")
    fig_rsi = make_subplots(rows=1, cols=1)
    # 2주(14일) 데이터만 표시
    rsi_time_window = timedelta(days=14)
    now_time = datetime.now()
    for market in markets:
        coin = market.split('-')[1]
        df = fetch_ohlcv(market, selected_tf)
        if not df.empty:
            df_recent = df[df['datetime'] >= now_time - rsi_time_window]
            fig_rsi.add_trace(go.Scatter(
                x=df_recent['datetime'],
                y=df_recent['RSI'],
                name=f'{coin} RSI',
                line=dict(width=2)
            ))
    fig_rsi.update_layout(
        height=400,
        title=f"RSI 비교 ({timeframes[selected_tf]} 차트)",
        yaxis_title="RSI",
        hovermode="x unified"
    )
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    st.plotly_chart(fig_rsi, use_container_width=True)


    # --- 이평선 + AI 예측 + 실제가 비교 차트 ---
    st.subheader("📈 이평선 & AI 예측 vs 실제가")
    selected_coin = st.selectbox("코인 선택 (이평선/AI예측)", [m.split('-')[1] for m in markets], key="ma_ai_coin")
    market_for_coin = [m for m in markets if m.split('-')[1] == selected_coin][0]
    df_ma = fetch_ohlcv(market_for_coin, selected_tf)
    ai_preds, ai_pred_times = [], []
    if not df_ma.empty:
        # 이동평균선 추가
        df_ma['SMA5'] = df_ma['close'].rolling(5).mean()
        df_ma['SMA10'] = df_ma['close'].rolling(10).mean()
        df_ma['SMA20'] = df_ma['close'].rolling(20).mean()
        df_ma['SMA60'] = df_ma['close'].rolling(60).mean()
        df_ma['SMA120'] = df_ma['close'].rolling(120).mean()
        df_ma['SMA200'] = df_ma['close'].rolling(200).mean()

        # --- AI 예측값 준비 (자동 학습/예측 및 CSV 저장) ---
        try:
            ai_preds, ai_pred_times = auto_train_and_predict(df_ma, selected_tf)
        except Exception as e:
            ai_preds, ai_pred_times = [], []
            st.warning(f"AI 예측 오류: {e}")

        fig_ma = go.Figure()
        # 실제 가격
        fig_ma.add_trace(go.Scatter(
            x=df_ma['datetime'], y=df_ma['close'],
            name='실제 가격', line=dict(color='blue', width=2)
        ))
        # 이동평균선
        for period, color in zip([5, 10, 20, 60, 120, 200], ['orange', 'green', 'purple', 'gray', 'brown', 'black']):
            col = f"SMA{period}"
            if col in df_ma.columns:
                fig_ma.add_trace(go.Scatter(
                    x=df_ma['datetime'], y=df_ma[col],
                    name=f'SMA{period}', line=dict(width=1, dash='dot', color=color)
                ))
        # AI 예측값 (명확하게 강조, 예측가 점/선/텍스트 모두 표시)
        if ai_preds is not None and ai_pred_times and len(ai_preds) > 0:
            # --- 예측 차트 표기 단위 결정 ---
            # 1분봉: 4시간(240분) = 240개, 3분봉: 80개, 5분봉: 48개, 15분봉: 16개, 60분봉: 4개, 240분봉: 1개
            tf_minutes = int(selected_tf)
            if tf_minutes == 1:
                pred_range = 240  # 4시간
                pred_label = "4시간(240개봉)"
            elif tf_minutes == 3:
                pred_range = 80
                pred_label = "4시간(80개봉)"
            elif tf_minutes == 5:
                pred_range = 48
                pred_label = "4시간(48개봉)"
            elif tf_minutes == 15:
                pred_range = 16
                pred_label = "4시간(16개봉)"
            elif tf_minutes == 60:
                pred_range = 4
                pred_label = "4시간(4개봉)"
            elif tf_minutes == 240:
                pred_range = 1
                pred_label = "4시간(1개봉)"
            else:
                pred_range = len(ai_preds)
                pred_label = f"{len(ai_preds)}개봉"

            # 실제 예측 표시 범위 제한
            show_preds = ai_preds[:pred_range]
            show_times = ai_pred_times[:pred_range]

            # 예측가 점
            fig_ma.add_trace(go.Scatter(
                x=show_times, y=show_preds,
                name=f'AI 예측가 ({pred_label})',
                mode='markers+text',
                marker=dict(size=14, color='red', symbol='star'),
                text=[f"{y:,.0f}" for y in show_preds],
                textposition="top center",
                hovertemplate='AI 예측가: %{y:,.0f}원<br>시간: %{x}<extra></extra>'
            ))
            # 예측가 선
            fig_ma.add_trace(go.Scatter(
                x=show_times, y=show_preds,
                name=f'AI 예측선 ({pred_label})',
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                showlegend=True
            ))
            # 예측가와 실제가 연결선(점선)
            for t, pred in zip(show_times, show_preds):
                actual_row = df_ma[df_ma['datetime'] == t]
                if not actual_row.empty:
                    actual = actual_row['close'].values[0]
                    fig_ma.add_trace(go.Scatter(
                        x=[t, t], y=[actual, pred],
                        mode='lines',
                        line=dict(color='gray', dash='dot'),
                        showlegend=False
                    ))
            # 예측 범위 안내
            st.info(f"예측 표기 범위: {pred_label} (차트 주기: {tf_minutes}분)")
        else:
            st.info("AI 예측값이 없습니다. 데이터가 부족하거나 예측에 실패했습니다.")

        fig_ma.update_layout(
            title=f"{selected_coin} 이평선 & AI 예측 vs 실제가",
            yaxis_title="가격 (원)",
            xaxis_title="시간",
            height=600,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ma, use_container_width=True)
    else:
        st.warning(f"{selected_coin} 데이터를 불러오지 못했습니다.")

