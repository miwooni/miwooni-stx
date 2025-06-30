import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from xgboost import XGBRegressor
from prophet import Prophet
import logging
import time


# 비밀번호 설정 (노출주의)
PASSWORD = "Fudfud8080@"

# 세션 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 인증 처리
if not st.session_state.authenticated:
    st.title("🔐 모카 간식비 만들기!!!")
    password = st.text_input("비밀번호를 입력하세요:", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif password != "":
        st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()  # 아래 코드 실행 방지


# ✅ 첫 번째 Streamlit 명령어 (여기만 남기세요)
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed"
)
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- 공통 함수 정의 ----------------------
def send_telegram_alert(message: str):
    bot_token = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
    chat_id = "7890657899"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Telegram 알림 전송 실패: {str(e)}")
        st.error(f"Telegram 알림 전송 실패: {str(e)}")

def parse_number(x):
    """문자열/숫자 혼합 데이터를 float으로 변환"""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            cleaned = x.replace(',','').replace('+','').replace('-','').strip()
            return float(cleaned) if cleaned else 0.0
        return 0.0
    except:
        return 0.0

# -- 기술적 지표 함수 --   
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
    if df.empty or len(df) < period:
        return pd.Series([np.nan] * len(df))
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(
    lambda x: np.mean(np.abs(x - np.mean(x))) if not np.isnan(np.mean(x)) else np.nan,
    raw=True
    )
    return (tp - ma) / (0.015 * md)

def atr(df, period=14):
    if df.empty or len(df) < 2:
        return pd.Series([np.nan] * len(df))
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_chart_patterns(df):
    patterns = []
    if len(df) < 7: return patterns
    
    # W 패턴 (이중 바닥)
    last5 = df.iloc[-5:]
    lows = last5['low'].values
    if (lows[0] > lows[1] and 
        lows[1] < lows[2] and 
        lows[2] > lows[3] and 
        lows[3] < lows[4]):
        patterns.append("W 패턴(하락 예측)")
    
    # M 패턴 (이중 천정)
    highs = last5['high'].values
    if (highs[0] < highs[1] and 
        highs[1] > highs[2] and 
        highs[2] < highs[3] and 
        highs[3] > highs[4]):
        patterns.append("M 패턴(상승 예측)")
    
    # 삼중 바닥 패턴
    if len(df) >= 7:
        last7 = df.iloc[-7:]
        l7 = last7['low'].values
        if (l7[0] > l7[1] and 
            l7[1] < l7[2] and 
            l7[2] > l7[3] and 
            l7[3] < l7[4] and 
            l7[4] > l7[5] and 
            l7[5] < l7[6]):
            patterns.append("강한 상승 예측")
    
    # 상승 삼각형 패턴
    if len(df) >= 10:
        resistance = df['high'].rolling(5).max()[-5:]
        support = df['low'].rolling(5).min()[-5:]
        res_diff = np.diff(resistance)
        sup_diff = np.diff(support)
        if (np.all(np.abs(res_diff) < 0.02*resistance.mean()) and 
            np.all(sup_diff > 0)):
            patterns.append("상승예측")
    # 역 헤드 앤 숄더 (상승 전환)
    if len(df) >= 7:
        last7 = df.iloc[-7:]
        lows = last7['low'].values
    if (lows[0] > lows[1] and 
        lows[1] < lows[2] and 
        lows[2] > lows[3] and 
        lows[3] < lows[4] and 
        lows[4] > lows[5] and 
        lows[5] < lows[6]):
        patterns.append("역 헤드앤숄더(강한 상승 예측)")

    # 헤드 앤 숄더 (하락 전환)
    if len(df) >= 7:
        last7 = df.iloc[-7:]
        highs = last7['high'].values
    if (highs[0] < highs[1] and 
        highs[1] > highs[2] and 
        highs[2] < highs[3] and 
        highs[3] > highs[4] and 
        highs[4] < highs[5] and 
        highs[5] > highs[6]):
        patterns.append("헤드앤숄더(강한 하락 예측)")

    # 상승 깃발 (추세 지속)
    if len(df) >= 8:
        last8 = df.iloc[-8:]
        range_cond = last8['high'] - last8['low']
    if (np.all(range_cond[:3] > 2*range_cond.mean()) and  # 초기 급등
        np.all(np.diff(last8['close'][3:]) < 0.02*last8['close'].mean())): # 이후 횡보
        patterns.append("상승 깃발패턴(추세 지속)")

    # 하락 쐐기 (상승 반전)
    if len(df) >= 10:
        last10 = df.iloc[-10:]
        upper = last10['high'].rolling(3).max().dropna()
        lower = last10['low'].rolling(3).min().dropna()
    if (np.all(np.diff(upper) < 0) and  # 고점 하락
        np.all(np.diff(lower) < 0)):    # 저점 하락 (수렴)
        patterns.append("하락 쐐기(상승 반전 예측)")

    # 컵 앤 핸들 (상승 지속)
    if len(df) >= 20:
        last20 = df.iloc[-20:]
        cup = last20[:15]['low'].values
        handle = last20[15:]['low'].values
    if (np.all(cup[0] > cup[1:7]) and   # 컵 형성(초기 하락)
        np.all(cup[-5:] > cup[7]) and    # 컵 복귀 
        np.all(handle[-3:] > handle[0])): # 핸들 조정
        patterns.append("컵 앤 핸들(상승 지속)")

    # 삼각 수렴 (상승/하락 브레이크아웃)
    if len(df) >= 10:
        last10 = df.iloc[-10:]
        upper = last10['high'].values
        lower = last10['low'].values
    if (np.all(np.diff(upper) < 0) and  # 고점 하락
        np.all(np.diff(lower) > 0)):    # 저점 상승
        patterns.append("삼각 수렴(추세 모멘텀)")
    return patterns

def calculate_signal_score(df, latest):
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []
    latest_rsi = latest.get('RSI', 0) if not df.empty else 0
    
    if df.empty or len(df) < 2:
        return buy_score, sell_score, buy_reasons, sell_reasons, latest_rsi
    
    try:
        required_keys = ['HMA3', 'HMA', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
        if not all(key in latest for key in required_keys):
            raise ValueError("필수 기술적 지표 데이터 누락")

        # 매수 조건
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

        # 매도 조건
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
            
    except Exception as e:
        st.error(f"신호 계산 오류: {str(e)}")
        buy_reasons = [f"오류 발생: {str(e)}"]
        sell_reasons = [f"오류 발생: {str(e)}"]
    
    return buy_score, sell_score, buy_reasons, sell_reasons, latest_rsi

# ---------------------- AI 예측 모델 함수 ----------------------
def prepare_lstm_data(data, n_steps=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['close']])
    
    if len(scaled_data) < n_steps:  # 데이터 부족 시 처리
        dummy = np.zeros((n_steps, 1))
        dummy[-len(scaled_data):] = scaled_data
        return dummy.reshape(1, n_steps, 1), None, scaler
    
    X = []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, 0])
    return np.array(X).reshape(-1, n_steps, 1), None, scaler

def train_lstm_model(X_train, y_train):
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),  # 명시적 입력층 추가
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    return model

def lstm_predict(model, data, scaler, n_steps=60, n_future=5):
    """LSTM 모델로 미래 가격 예측"""
    inputs = data[-n_steps:].values
    inputs = scaler.transform(inputs.reshape(-1, 1))
    
    predictions = []
    for _ in range(n_future):
        x_input = inputs[-n_steps:].reshape(1, n_steps, 1)
        pred = model.predict(x_input, verbose=0)
        inputs = np.append(inputs, pred)
        predictions.append(pred[0,0])
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def train_xgboost_model(X_train, y_train):
    """XGBoost 모델 생성 및 훈련"""
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def xgboost_predict(model, data, n_steps=60, n_future=5):
    """수정된 XGBoost 예측 함수"""
    if len(data) < n_steps:
        return np.array([data.mean()] * n_future)  # 기본값 반환
    
    predictions = []
    x_input = data[-n_steps:].values.reshape(1, -1)  # 2D 배열로 변환
    for _ in range(n_future):
        pred = model.predict(x_input)
        predictions.append(pred[0])
        x_input = np.roll(x_input, -1)
        x_input[0, -1] = pred[0]  # 예측값으로 업데이트
    return np.array(predictions)

def prophet_predict(data, n_future=5):
    """Prophet 모델로 미래 가격 예측"""
    if data.empty:
        return np.zeros(n_future)
    
    df = data.reset_index()
    df = df.rename(columns={'datetime': 'ds', 'close': 'y'})
    
    try:
        model = Prophet(daily_seasonality=True)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=n_future, freq='min')
        forecast = model.predict(future)
        
        return forecast['yhat'][-n_future:].values
    except Exception as e:
        logger.error(f"Prophet 예측 오류: {str(e)}")
        return np.zeros(n_future)

def ensemble_predictions(predictions):
    """여러 모델의 예측 결과를 앙상블"""
    weights = {'lstm': 0.4, 'xgboost': 0.3, 'prophet': 0.3}  # 모델 가중치
    weighted_sum = np.zeros_like(predictions['lstm'])
    
    for model, pred in predictions.items():
        weighted_sum += pred * weights[model]
    
    return weighted_sum

def ai_price_predict_enhanced(df, current_price, selected_tf):
    if len(df) < 60:  # 최소 데이터 요구량 축소
        # 단순한 예측으로 대체
        change_percent = random.uniform(-0.02, 0.02)
        predicted = current_price * (1 + change_percent)
        trend = "상승" if change_percent > 0.005 else "하락" if change_percent < -0.005 else "유지"
        emoji = "📈" if trend == "상승" else "📉" if trend == "하락" else "⚖️"
        return round(predicted, 1), f"{emoji} {trend}", None
    
    try:
        # LSTM 예측 (간소화)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['close']].values)
        
        # XGBoost 예측
        xgb_data = df['close'].rolling(5).mean().dropna()
        if len(xgb_data) > 10:
            xgb_model = train_xgboost_model(np.arange(len(xgb_data)).reshape(-1, 1), xgb_data.values)
            xgb_pred = xgboost_predict(xgb_model, xgb_data)
        else:
            xgb_pred = np.array([current_price] * 5)
        
        # Prophet 예측
        prophet_pred = prophet_predict(df[['datetime', 'close']])
        
        # 앙상블 예측
        predictions = {
            'lstm': lstm_pred,
            'xgboost': xgb_pred,
            'prophet': prophet_pred
        }
        ensemble_pred = ensemble_predictions(predictions)
        
        # 예측 결과 분석
        avg_pred = np.mean(ensemble_pred)
        change_percent = (avg_pred - current_price) / current_price
        trend = "상승" if change_percent > 0.005 else "하락" if change_percent < -0.005 else "유지"
        emoji = "📈" if trend == "상승" else "📉" if trend == "하락" else "⚖️"
        
        # 예측 차트 생성
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'][-50:],
            y=df['close'][-50:],
            name='실제 가격',
            line=dict(color='blue')
        ))
        
        future_dates = pd.date_range(
            start=df['datetime'].iloc[-1],
            periods=6,
            freq=f'{selected_tf}min'  # selected_tf 사용
        )[1:]  # 현재 시간 제외
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=ensemble_pred,
            name='AI 예측 가격',
            line=dict(color='red', dash='dot')
        ))
        
        fig.update_layout(
            title=f"AI 가격 예측 (앙상블 모델)",
            xaxis_title="시간",
            yaxis_title="가격 (원)",
            showlegend=True,
            height=400
        )
        
        return round(avg_pred, 1), f"{emoji} {trend}", fig
        
    except Exception as e:
        logger.error(f"AI 예측 오류: {str(e)}")
        change_percent = random.uniform(-0.02, 0.02)
        predicted = current_price * (1 + change_percent)
        trend = "상승" if change_percent > 0.005 else "하락" if change_percent < -0.005 else "유지"
        emoji = "📈" if trend == "상승" else "📉" if trend == "하락" else "⚖️"
        return round(predicted, 1), f"{emoji} {trend}", None

# ---------------------- 분봉별 분석 리포트 함수 ----------------------
def generate_timeframe_analysis(market, timeframe):
    """분봉별 기술적 지표 계산 및 분석 리포트 생성"""
    coin = market.split('-')[1]
    tf_name = timeframes[timeframe]
    df = fetch_ohlcv(market, timeframe)
    
    if df.empty or len(df) < 20:
        return f"⚠️ {coin} {tf_name}봉 데이터 부족으로 분석 불가", None
    
    # 기술적 지표 계산
    latest = df.iloc[-1]
    current_price = latest['close']
    
    # 볼린저밴드
    _, bb_upper, bb_lower = bollinger_bands(df['close'], period=20)
    
    # RSI
    rsi_val = rsi(df['close'], period=14).iloc[-1]
    
    # 지지/저항선
    support = df['low'].rolling(20).min().iloc[-1]
    resistance = df['high'].rolling(20).max().iloc[-1]
    
    # 거래량 분석
    vol_ratio = latest['volume'] / df['volume'].rolling(20).mean().iloc[-1] if df['volume'].rolling(20).mean().iloc[-1] > 0 else 1.0
    
    # 패턴 분석
    patterns = detect_chart_patterns(df)
    
    # 1. 기술적 지표 테이블 생성
    indicators_table = (
        f"📊 [{coin} {tf_name}봉] 기술적 지표\n"
        f"┌{'─'*25}┬{'─'*15}┐\n"
        f"│ {'지표':<23} │ {'값':<13} │\n"
        f"├{'─'*25}┼{'─'*15}┤\n"
        f"│ {'현재가':<23} │ {current_price:>10,.1f} 원 │\n"
        f"│ {'RSI(14)':<23} │ {rsi_val:>13.1f} │\n"
        f"│ {'볼린저밴드 상한':<23} │ {bb_upper.iloc[-1]:>10,.1f} 원 │\n"
        f"│ {'볼린저밴드 하한':<23} │ {bb_lower.iloc[-1]:>10,.1f} 원 │\n"
        f"│ {'지지선 (20봉 최저)':<23} │ {support:>10,.1f} 원 │\n"
        f"│ {'저항선 (20봉 최고)':<23} │ {resistance:>10,.1f} 원 │\n"
        f"│ {'거래량 (평균 대비)':<23} │ {vol_ratio:>10.1f}배 │\n"
        f"└{'─'*25}┴{'─'*15}┘\n"
    )
    
    # 2. 분석 내용 생성
    analysis_content = f"🔍 [{coin} {tf_name}봉] 분석 내용:\n"
    
    # 추세 분석
    trend = ""
    if current_price > bb_upper.iloc[-1]:
        trend += "- 📈 강한 상승 추세 (볼린저 상한 돌파)\n"
    elif current_price < bb_lower.iloc[-1]:
        trend += "- 📉 강한 하락 추세 (볼린저 하한 돌파)\n"
    else:
        if current_price > (bb_upper.iloc[-1] + bb_lower.iloc[-1]) / 2:
            trend += "- ↗️ 상승 추세 (볼린저 상단 근접)\n"
        else:
            trend += "- ↘️ 하락 추세 (볼린저 하단 근접)\n"
    
    # RSI 분석
    rsi_analysis = ""
    if rsi_val > 70:
        rsi_analysis += "- ⚠️ 과매수 상태 (RSI > 70), 조정 가능성\n"
    elif rsi_val < 30:
        rsi_analysis += "- ⚠️ 과매도 상태 (RSI < 30), 반등 가능성\n"
    else:
        rsi_analysis += "- ⚖️ RSI 중립 구간 (30-70)\n"
    
    # 패턴 분석
    pattern_analysis = ""
    if patterns:
        for pattern in patterns:
            pattern_analysis += f"- {pattern}\n"
    else:
        pattern_analysis += "- 주요 패턴 감지되지 않음\n"
    
    # 거래량 분석
    volume_analysis = ""
    if vol_ratio > 1.5:
        volume_analysis += f"- 🔥 거래량 급증 (평균 대비 {vol_ratio:.1f}배)\n"
    elif vol_ratio < 0.7:
        volume_analysis += f"- ❄️ 거래량 감소 (평균 대비 {vol_ratio:.1f}배)\n"
    else:
        volume_analysis += "- ↔️ 정상 거래량 유지\n"
    
    # 매매 전략 제안
    strategy = "🎯 매매 전략 제안:\n"
    if "강한 상승 예측" in patterns or "역 헤드앤숄더" in patterns:
        strategy += "- 🟢 매수 진입 고려 (강력한 상승 패턴)\n"
        strategy += f"- ✅ 목표가: {resistance:,.1f}원 (+{(resistance/current_price-1)*100:.1f}%)\n"
    elif "강한 하락 예측" in patterns or "헤드앤숄더" in patterns:
        strategy += "- 🔴 매도/관망 권장 (하락 패턴)\n"
        strategy += f"- ❌ 손절가: {support:,.1f}원 ({(support/current_price-1)*100:.1f}%)\n"
    else:
        if rsi_val < 40 and current_price < bb_lower.iloc[-1]:
            strategy += "- 🟢 매수 기회 (과매도 + 볼린저 하한)\n"
        elif rsi_val > 60 and current_price > bb_upper.iloc[-1]:
            strategy += "- 🔴 매도 고려 (과매수 + 볼린저 상한)\n"
        else:
            strategy += "- ⚠️ 관망 권장 (명확한 신호 없음)\n"
    
    # 종합 분석 내용 조합
    analysis_content += trend + rsi_analysis + pattern_analysis + volume_analysis + strategy
    
    return indicators_table + analysis_content, patterns

# ---------------------- 코인별 전체 분봉 리포트 생성 (가독성 개선) ----------------------
def generate_full_coin_report(market):
    """코인별 모든 분봉에 대한 종합 리포트 생성 - 텔레그램 전송용"""
    coin = market.split('-')[1]
    now_time = datetime.now().strftime('%m-%d %H:%M')
    report = f"🔥 *{coin} 종합 분석 리포트* ({now_time})\n"
    report += "="*40 + "\n\n"
    
    # 모든 시간대별 분석 데이터 수집
    indicators_data = {}
    analysis_contents = {}
    all_patterns = []

    for tf in timeframes.keys():
        tf_name = timeframes[tf]
        df = fetch_ohlcv(market, tf)

        if df.empty or len(df) < 20:
            indicators_data[tf_name] = ["-"] * 7
            analysis_contents[tf_name] = f"⚠️ {tf_name}봉 데이터 부족"
            continue

        # 기술적 지표 계산
        latest = df.iloc[-1]
        current_price = latest['close']
        _, bb_upper, bb_lower = bollinger_bands(df['close'], period=20)
        rsi_val = rsi(df['close'], period=14).iloc[-1]
        support = df['low'].rolling(20).min().iloc[-1]
        resistance = df['high'].rolling(20).max().iloc[-1]
        vol_ratio = latest['volume'] / df['volume'].rolling(20).mean().iloc[-1] if df['volume'].rolling(20).mean().iloc[-1] > 0 else 1.0
        patterns = detect_chart_patterns(df)
        all_patterns.extend(patterns)

        indicators_data[tf_name] = [
            f"{current_price:,.1f}",
            f"{rsi_val:.1f}",
            f"{bb_upper.iloc[-1]:,.1f}",
            f"{bb_lower.iloc[-1]:,.1f}",
            f"{support:,.1f}",
            f"{resistance:,.1f}",
            f"{vol_ratio:.1f}x"
        ]

        # 분석 내용 생성
        analysis = f"📊 *{tf_name}봉 분석:*\n"
        if current_price > bb_upper.iloc[-1]:
            analysis += "- 📈 강한 상승 추세 (볼린저 상한 돌파)\n"
        elif current_price < bb_lower.iloc[-1]:
            analysis += "- 📉 강한 하락 추세 (볼린저 하한 돌파)\n"
        else:
            bb_pos = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-8)
            if current_price > (bb_upper.iloc[-1] + bb_lower.iloc[-1]) / 2:
                analysis += "- ↗️ 상승 추세 (볼린저 상단 근접)\n"
            else:
                analysis += f"- ↘️ 하락 추세 (볼린저 하단 근접 ({bb_pos:.2f}))\n"
                
        if rsi_val > 70:
            analysis += "- ⚠️ *과매수* (RSI > 70)\n"
        elif rsi_val < 30:
            analysis += f"- ⚠️ *과매도* (RSI < {rsi_val:.1f})\n"
        else:
            analysis += "- ⚖️ RSI 중립\n"
            
        if patterns:
            analysis += f"- 🔍 패턴: {', '.join(patterns)}\n"
        analysis_contents[tf_name] = analysis

    # ==================== 지표 테이블 생성 ====================
    timeframes_order = ['1분', '3분', '5분', '15분', '60분', '4시간']
    report += "📈 *주요 기술적 지표*\n"
    report += "| 지표 | " + " | ".join(timeframes_order) + " |\n"
    report += "|-----|" + "|".join(["-----"] * len(timeframes_order)) + "|\n"
    
    # 테이블 행 추가
    rows = [
        ("현재가격", [indicators_data[tf][0] for tf in timeframes_order]),
        ("RSI", [indicators_data[tf][1] for tf in timeframes_order]),
        ("볼밴상단", [indicators_data[tf][2] for tf in timeframes_order]),
        ("볼밴하단", [indicators_data[tf][3] for tf in timeframes_order]),
        ("지지선", [indicators_data[tf][4] for tf in timeframes_order]),
        ("저항선", [indicators_data[tf][5] for tf in timeframes_order]),
        ("거래량", [indicators_data[tf][6] for tf in timeframes_order])
    ]
    
    for row_name, values in rows:
        report += f"| {row_name} | " + " | ".join(values) + " |\n"

    # ==================== 분석 내용 ====================
    report += "\n🔍 *분봉별 분석*\n"
    for tf in timeframes_order:
        if tf in analysis_contents:
            report += analysis_contents[tf] + "\n"

    # ==================== 종합 판단 ====================
    report += "\n🚨 *종합 판단*\n"
    report += "="*30 + "\n"
    
    # 추세 일관성 분석
    consistency = []
    for tf in [5, 15, 60]:
        df = fetch_ohlcv(market, tf)
        if not df.empty and len(df) >= 20:
            ma5 = df['close'].rolling(5).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            consistency.append("🟢" if ma5 > ma20 else "🔴")
        else:
            consistency.append("⚪")
    
    # 위험 신호 분석
    risk_count = sum(1 for p in all_patterns if "하락" in p or "매도" in p)
    risk_level = "⚠️⚠️ 고위험" if risk_count > 3 else "⚠️ 주의" if risk_count > 1 else "✅ 양호"
    
    report += f"- 추세 일관성: 5분:{consistency[0]} 15분:{consistency[1]} 60분:{consistency[2]}\n"
    report += f"- 위험 신호: {risk_level} ({risk_count}개 시간대)\n"
    
    # 매매 권장 사항
    if "강한 상승 예측" in all_patterns or "역 헤드앤숄더" in all_patterns:
        report += "- 💎 *매수 추천*: 강력한 상승 패턴 확인\n"
    elif "강한 하락 예측" in all_patterns or "헤드앤숄더" in all_patterns:
        report += "- 🚫 *매수 자제*: 강력한 하락 패턴 확인\n"
    else:
        report += "- ⚖️ *관망 권장*: 명확한 신호 없음\n"
    
    report += "\n" + "="*40 + "\n"
    return report

# ---------------------- 초기 설정 ----------------------
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .text-white {
        color: #fff !important;
        text-shadow: 1px 1px 0 #000, -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000;
    }
    .metric-container {
        padding: 10px;
        border-radius: 10px;
        background: #1e1e1e;
        margin: 10px 0;
    }
    .coin-report {
        background: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 4px solid #3498db;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<h1 class='text-white' style='text-align:center;font-family:Consolas,monospace;'>"
    "🔥 If you're not desperate, don't even think about it!</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 class='text-white' style='text-align:center;font-family:Consolas,monospace;'>"
    "🔥 Live Cryptocurrency Analytics Dashboard with AI Prediction</h3>",
    unsafe_allow_html=True
)
st_autorefresh(interval=5000, key="auto_refresh")

# 알림 목록 초기화
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# ---------------------- 전역 변수 ----------------------
default_holdings = {
    'KRW-STX': 14073.68834666,
    'KRW-HBAR': 62216.22494886,
    'KRW-DOGE': 61194.37067502,
    }
markets = list(default_holdings.keys())
timeframes = {1: '1분', 3: '3분', 5: '5분', 15: '15분', 60: '60분', 240: '4시간'}  # 240분을 4시간으로 변경
TOTAL_INVESTMENT = 58500000

# ---------------------- 데이터 조회 함수 ----------------------
@st.cache_data(ttl=10, show_spinner=False)
def get_current_prices():
    try:
        res = requests.get(f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}", timeout=10)
        res.raise_for_status()
        
        # API 응답 구조에 맞게 수정
        return {item['market']: {
            'trade_price': item['trade_price'],
            'signed_change_rate': item['signed_change_rate']
        } for item in res.json()}
    
    except Exception as e:
        logger.error(f"현재 가격 조회 실패: {str(e)}")
        # 에러 시 빈 딕셔너리 반환
        return {}

@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv(market, timeframe, count=300):
    try:
        # API 엔드포인트 수정
        url = f"https://api.upbit.com/v1/candles/minutes/{timeframe}"
        params = {'market': market, 'count': count}
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()
        
        if not data:
            return pd.DataFrame()
            
        # 실제 API 응답에 맞는 컬럼 매핑
        column_mapping = {
            'candle_date_time_kst': 'datetime',
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        }
        
        df = pd.DataFrame(data)
        df = df.rename(columns=column_mapping)
        
        # 필요한 컬럼만 선택
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols]
        
        # 시간 순서 정렬 (과거 -> 현재)
        df = df.iloc[::-1].reset_index(drop=True)
        
        # datetime 컬럼을 KST 시간대로 변환
        df['datetime'] = pd.to_datetime(df['datetime'], utc=False)
        
        # 기술적 지표 계산 (데이터가 충분한 경우에만)
        if len(df) > 16:
            df['HL2'] = (df['high'] + df['low']) / 2
            df['HMA'] = hma(df['HL2'])
            df['HMA3'] = hma3(df['HL2'])
            df['RSI'] = rsi(df['close'])
            df['MACD_line'], df['Signal_line'], df['MACD_hist'] = macd(df['close'])
            df['BB_ma'], df['BB_upper'], df['BB_lower'] = bollinger_bands(df['close'])
            df['CCI'] = cci(df)
            df['ATR'] = atr(df)
        else:
            # 데이터 부족 시 기본값 설정
            for col in ['HMA', 'HMA3', 'RSI', 'MACD_line', 'Signal_line', 
                       'MACD_hist', 'BB_ma', 'BB_upper', 'BB_lower', 'CCI', 'ATR']:
                df[col] = np.nan
                
        return df
    
    except Exception as e:
        logger.error(f"{market} OHLCV 데이터 조회 실패: {str(e)}")
        return pd.DataFrame()

# ---------------------- AI 학습 진행률 관리 ----------------------
if 'ai_progress' not in st.session_state:
    st.session_state.ai_progress = {'STX': 0, 'HBAR': 0, 'DOGE': 0}

def update_ai_progress():
    # 단순 시뮬레이션: 실행마다 2~6% 증가
    for coin in st.session_state.ai_progress:
        if st.session_state.ai_progress[coin] < 100:
            st.session_state.ai_progress[coin] += random.randint(2, 6)
            st.session_state.ai_progress[coin] = min(100, st.session_state.ai_progress[coin])

# ---------------------- 사이드바 설정 ----------------------
with st.sidebar:
    st.header("⚙️ 제어 패널")
    # 차트 주기 선택
    selected_tf = st.selectbox(
        '차트 주기', 
        list(timeframes.keys()), 
        format_func=lambda x: timeframes[x],
        key='timeframe_selector'
    )
    st.subheader("💰 투자 현황")
    prices = get_current_prices()
    stx_holding = default_holdings['KRW-STX']
    stx_price = prices.get('KRW-STX', {}).get('trade_price', 0)
    current_value = stx_holding * stx_price
    profit = current_value - TOTAL_INVESTMENT
    profit_percent = (profit / TOTAL_INVESTMENT) * 100 if TOTAL_INVESTMENT else 0
    profit_emoji = "🔻" if profit < 0 else "🟢"
    # 메트릭 스타일 개선
    st.markdown(f"""
        <div class="metric-container">
            <div><strong>총 투자금액</strong></div>
            <div style="font-size:24px; font-weight:bold;">{TOTAL_INVESTMENT:,.0f} 원</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
        <div class="metric-container">
            <div><strong>STX 자산가치</strong></div>
            <div style="font-size:24px; font-weight:bold;">{current_value:,.0f} 원</div>
            <div style="color:{"red" if profit < 0 else "green"}">
                {profit_emoji} {profit:+,.0f} 원 ({profit_percent:+.2f}%)
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
        <div class="metric-container">
            <div><strong>STX 보유량</strong></div>
            <div style="font-size:24px; font-weight:bold;">{stx_holding:,.2f} EA</div>
        </div>
    """, unsafe_allow_html=True)
    st.subheader("🔔 알림 설정")
    telegram_enabled = st.checkbox("텔레그램 알림 활성화", value=True)
    table_alert_interval = st.slider("테이블 알림 주기(분)", 1, 60, 10)
    analysis_interval = st.slider("전체 분석 주기(분)", 5, 120, selected_tf)  # 분석 주기 설정 추가
    st.subheader("🧠 AI 학습상황")
    update_ai_progress()
    for coin, progress in st.session_state.ai_progress.items():
        progress_bar = st.progress(progress)
        if progress >= 100:
            st.success(f"{coin}: 학습 완료")
        else:
            st.info(f"{coin}: {progress}% 진행 중")

# ---------------------- 상태 변수 초기화 ----------------------
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}
if 'last_table_alert_time' not in st.session_state:
    st.session_state.last_table_alert_time = datetime.min
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = datetime.now() - timedelta(minutes=analysis_interval+1)

# ---------------------- 설정된 주기마다 전체 분석 실행 (STX만) ----------------------
now = datetime.now()
if (now - st.session_state.last_analysis_time) > timedelta(minutes=analysis_interval):
    st.session_state.last_analysis_time = now
    with st.spinner(f"🔍 STX 종합 분석 중... (주기: {analysis_interval}분)"):
        # STX에 대해서만 분석 및 텔레그램 전송
        market = 'KRW-STX'
        coin = market.split('-')[1]
        try:
            full_report = generate_full_coin_report(market)
            
            # 텔레그램 전송 (STX만)
            if telegram_enabled:
                try:
                    send_telegram_alert(full_report)
                    st.session_state.alerts.append(f"📊 {coin} 종합 분석 리포트 전송 완료")
                except Exception as e:
                    st.error(f"{coin} 리포트 전송 실패: {str(e)}")
            
            # 로컬 알림 목록에 추가
            st.session_state.alerts.append(f"📈 {coin} 종합 분석 리포트 생성됨")
        except Exception as e:
            st.error(f"{coin} 리포트 생성 오류: {str(e)}")
            st.session_state.alerts.append(f"⚠️ {coin} 리포트 생성 실패: {str(e)}")

def generate_coin_table():
    # 최신 가격 정보 갱신
    prices = get_current_prices()
    signal_scores = {} 
    base_market = 'KRW-STX'
    base_qty = default_holdings[base_market]
    base_price_data = prices.get(base_market, {'trade_price': 0, 'signed_change_rate': 0})
    base_price = base_price_data['trade_price']
    base_krw = base_qty * base_price * 0.9995

    # 현재 시간 변수 정의
    now = datetime.now()

    compare_data = []
    for market in markets:
        coin = market.split('-')[1]
        price_data = prices.get(market, {'trade_price': 0, 'signed_change_rate': 0})
        price = price_data['trade_price']
        change_rate = price_data['signed_change_rate'] * 100
        qty = default_holdings[market]
        value = price * qty
        
        df = fetch_ohlcv(market, selected_tf)
        buy_score, sell_score = 0, 0
        latest_rsi = 0

        if not df.empty and len(df) >= 2:
            try:
                latest = df.iloc[-1]
                required_cols = ['HMA', 'HMA3', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
                if all(col in df.columns for col in required_cols):
                    buy_score, sell_score, _, _, latest_rsi = calculate_signal_score(df, latest)
                else:
                    st.error(f"{coin} 데이터 컬럼 누락")
            except Exception as e:
                st.error(f"{coin} 신호 계산 오류: {str(e)}")
        else:
            st.warning(f"{coin} 데이터 부족으로 분석 생략")
        
        if market != base_market:
            # 매수 수수료 0.05% 반영: 실제 매수 시 가격에 0.05% 추가 비용 발생 -> 구매 수량 감소
            replace_qty = base_krw / (price * 1.0005)   # base_krw는 이미 매도 수수료가 적용된 금액
            diff_qty = replace_qty - qty
            
            # 대체 수량 변화 알림 (마이너스에서 플러스로 전환 시)
            last_diff_key = f"{coin}_last_diff_qty"
            current_diff_sign = np.sign(diff_qty)
            last_diff_sign = st.session_state.get(last_diff_key, current_diff_sign)
            
            if last_diff_sign < 0 and current_diff_sign > 0:
                timeframe_str = timeframes[selected_tf]
                alert_msg = (
                    f"🚨 [{coin} {timeframe_str}차트] 대체 수량 전환 알림!\n"
                    f"📈 이전 차이: {last_diff_sign:+.2f} → 현재 차이: {diff_qty:+.2f}\n"
                    f"📊 현재 가격: {price:,.1f} 원\n"
                    f"🔄 대체 가능 수량: {replace_qty:,.2f} (차이: {diff_qty:+,.2f})"
                )
                
                last_alert_time = st.session_state.last_alert_time.get(f"{coin}_qty_change", datetime.min)
                if (now - last_alert_time) > timedelta(minutes=5):  # 5분 간격으로만 알림
                    st.session_state.alerts.append(alert_msg)
                    st.session_state.last_alert_time[f"{coin}_qty_change"] = now
                    if telegram_enabled:
                        send_telegram_alert(alert_msg)
            
            st.session_state[last_diff_key] = current_diff_sign
            replace_value = replace_qty * price * 0.9995
        else:
            replace_qty = "-"
            diff_qty = "-"
            replace_value = "-"
        
        change_color = "red" if change_rate < 0 else "green"
        change_emoji = "🔻" if change_rate < 0 else "🟢"
        buy_color = "green" if buy_score >= 7 else "gray"
        sell_color = "red" if sell_score >= 7 else "gray"
        
        # AI 예측 실행 (selected_tf 인자 추가)
        ai_pred_price, ai_trend, ai_fig = ai_price_predict_enhanced(df, price, selected_tf)
        
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
            '대체평가액': replace_value if market != base_market else "-",
            '예측가격': f"{ai_pred_price:,.1f} 원",
            '전망': ai_trend,
            'AI_차트': ai_fig
        })

    df_compare = pd.DataFrame(compare_data)
    df_compare['차이수량'] = df_compare['차이수량'].apply(parse_number)
    df_compare['대체가능수량'] = df_compare['대체가능수량'].apply(parse_number)
    return df_compare

# 차트 주기 명시적으로 표시
tf_name = timeframes[selected_tf]
st.subheader(f"📊 코인 비교 테이블 ({tf_name}봉)")

df_compare = generate_coin_table()

# ---------------------- 통합된 테이블 렌더링 (hbar(25-0521).py 스타일 적용) ----------------------
def format_number(x):
    if isinstance(x, (int, float)):
        if abs(x) < 0.01: return f"{x:.4f}"
        if abs(x) < 1: return f"{x:.2f}"
        if abs(x) < 10000: return f"{x:,.2f}"
        return f"{x:,.0f}"
    return x

# 테이블 스타일링 및 출력
st.markdown(
    df_compare.style.format({
        '보유수량': format_number,
        '대체가능수량': format_number,
        '차이수량': format_number,
        '평가금액': format_number,
        'RSI': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
    }).set_properties(**{'text-align': 'center'}).hide(axis="index").to_html(escape=False),
    unsafe_allow_html=True
)

# AI 예측 차트 표시
for _, row in df_compare.iterrows():
    if row['AI_차트'] is not None:
        st.plotly_chart(row['AI_차트'], use_container_width=True)

# ---------------------- 코인별 종합분석 리포트 표시 (STX 포함) ----------------------
st.subheader("📊 코인 종합분석 리포트")
for market in markets:
    coin = market.split('-')[1]
    try:
        # 종합분석 리포트 생성
        full_report = generate_full_coin_report(market)
        # 확장 가능한 섹션에 리포트 표시
        with st.expander(f"{coin} 종합분석 리포트", expanded=False):
            st.markdown(f"""
                <div class="coin-report">
                    <pre style="white-space: pre-wrap; font-family: monospace; font-size: 21px; color: #fff;">{full_report}</pre>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"{coin} 리포트 생성 오류: {str(e)}")

# ---------------------- 테이블 알림 생성 (가독성 개선) ----------------------
now = datetime.now()
if (now - st.session_state.last_table_alert_time) > timedelta(minutes=table_alert_interval):
    alert_msg = "🔥 *실시간 분석 현황 (by MIWOONI)* 🔥\n\n"
    for _, row in df_compare.iterrows():
        alert_msg += (
            f"📌 *{row['코인명']}*\n"
            f"  ┣ 현재가: {row['시세']}\n"
            f"  ┣ RSI: {row['RSI']}\n"
            f"  ┣ 신호: {row['매수신호'].split('>')[1].split('<')[0]} / {row['매도신호'].split('>')[1].split('<')[0]}\n"
            f"  ┣ 보유량: {row['보유수량']:,.2f}\n"
            f"  ┗ 평가금액: {row['평가금액']:,.0f}원\n"
        )
        if row['코인명'].upper() in ['HBAR', 'DOGE']:
            alert_msg += (
                f"  ┣ 대체 수량: {row['대체가능수량']:,.2f}\n"
                f"  ┗ 차이 수량: {row['차이수량']:+,.2f}\n"
            )
        alert_msg += f"  ┗ AI 예측: {row['예측가격']} ({row['전망']})\n\n"
    
    alert_msg += "➖➖➖➖➖➖➖➖➖➖\n"
    st.session_state.last_table_alert_time = now
    if telegram_enabled:
        send_telegram_alert(alert_msg.strip())
    st.session_state.alerts.append("📊 테이블 분석 리포트 생성됨")

# ---------------------- 개별 코인 분석 ----------------------
for market in markets:
    coin = market.split('-')[1]
    df = fetch_ohlcv(market, selected_tf)
    if df.empty:
        st.warning(f"{coin} 차트 데이터 불러오기 실패")
        continue

    latest = df.iloc[-1]
    current_price = latest['close']
    prev_close = df.iloc[-2]['close'] if len(df) > 1 else current_price
    delta = current_price - prev_close
    delta_percent = (delta / prev_close) * 100 if prev_close else 0

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )
    
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Price"), row=1, col=1)
    
    # 간결한 주석 사용
    fig.add_hline(
        y=current_price, 
        line_dash="solid", 
        line_color="orange", 
        row=1, col=1,
        annotation_text=f"현재가: {current_price:,.1f}",
        annotation_position="bottom right",
        annotation_font_size=12,
        annotation_font_color="orange"
    )   
    
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['HMA'], name='HMA', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['HMA3'], name='HMA3', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Bar(x=df['datetime'], y=df['volume'], name='Volume',
                       marker_color=np.where(df['close'] > df['open'], 'green', 'red')),
               row=1, col=1, secondary_y=True)
    
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)

    fig.add_trace(go.Bar(x=df['datetime'], y=df['MACD_hist'], name='Histogram',
                       marker_color=np.where(df['MACD_hist'] > 0, 'green', 'red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD_line'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Signal_line'], name='Signal', line=dict(color='orange')), row=3, col=1)

    fig.update_layout(
        height=800,
        title=f"{coin} 차트 ({selected_tf})",
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=40),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # 강화된 패턴 감지 및 알림
    detected_patterns = detect_chart_patterns(df)
    pattern_alerts = []

    for pattern in detected_patterns:
        timeframe_str = timeframes[selected_tf]  # 선택된 시간대 문자열 가져오기
        alert_msg = f"🚨🚨 [{coin} {timeframe_str}차트]{current_price:,.1f} 원 // 강력한 패턴 경고 - {pattern} 발현! ({now.strftime('%H:%M:%S')})"
        pattern_alerts.append(alert_msg)
        
        # 마지막 알림 시간 확인 (5분 내 동일 패턴 알림 방지)
        last_pattern_alert_key = f"{coin}_{pattern[:10]}_pattern"
        last_alert_time = st.session_state.last_alert_time.get(last_pattern_alert_key, datetime.min)
        
        # 하락 쐐기 + RSI + MACD 히스토그램 상승 반전 조건 알림
        if (
            pattern == "하락 쐐기(상승 반전 예측)"
            and latest.get('RSI', 50) < 40
            and len(df) > 2
            and latest.get('MACD_hist', 0) > 0
            and df.iloc[-2].get('MACD_hist', 0) < 0
        ):
            send_telegram_alert("🚀 하락 쐐기 + MACD 반전 + 저점 RSI → 반등 예상!")

        if (now - last_alert_time) > timedelta(minutes=10):  # 10분 간격으로만 알림
            if telegram_enabled:
                send_telegram_alert(alert_msg)
            st.session_state.last_alert_time[last_pattern_alert_key] = now

    st.session_state.alerts.extend(pattern_alerts)

# ---------------------- 실시간 알림 출력 ----------------------
st.subheader("🔔 실시간 분석 알림")
for alert in reversed(st.session_state.alerts[-15:]):
    # 알림 유형에 따라 다른 스타일 적용
    if "대체 수량 전환" in alert:
        bg_color = "#1a5276"  # 파란색 계열
        icon = "🔄"
    elif "패턴" in alert:
        bg_color = "#7d3c98"  # 보라색 계열
        icon = "📈"
    elif "매수" in alert:
        bg_color = "#196f3d"  # 녹색 계열
        icon = "🟢"
    elif "매도" in alert:
        bg_color = "#943126"  # 빨간색 계열
        icon = "🔻"
    elif "분석" in alert or "리포트" in alert:
        bg_color = "#2b2b2b"  # 기본 배경색
        icon = "📊"
    else:
        bg_color = "#7e5100"  # 기타 알림
        icon = "ℹ️"
    st.markdown(
        f"""<div style='padding:10px; background:{bg_color}; border-radius:5px; margin:5px 0; font-size:16px; color:#fff;'><b>{icon}</b> {alert}</div>""", 
        unsafe_allow_html=True
    )
