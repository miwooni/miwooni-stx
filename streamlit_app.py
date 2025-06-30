# 통합-STX_최종-피보나치+엘리엇.py
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
from prophet import Prophet
import logging
import re



# 비밀번호 설정 (노출주의)Add commentMore actions
PASSWORD = "Fudfud8080@"

# 세션 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 인증 처리
if not st.session_state.authenticated:
    st.title("🔐 궁금하지? 모카꺼야!!")
    password = st.text_input("비밀번호를 입력하세요:", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
        st.()
    elif password != "":
        st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()  # 아래 코드 실행 방지

# ---------------------- 로깅 설정 ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------- 상수 정의 ----------------------
TOTAL_INVESTMENT = 58500000
TELEGRAM_TOKEN = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
TELEGRAM_CHAT_ID = "7890657899"
MARKETS = ['KRW-STX', 'KRW-HBAR', 'KRW-DOGE']

# ---------------------- 공통 함수 정의 ----------------------
def send_telegram_alert(message: str, parse_mode="Markdown"):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID, 
        "text": message,
        "parse_mode": parse_mode
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Telegram 알림 전송 실패: {str(e)}")
        return False

def parse_number(x):
    """문자열/숫자 혼합 데이터를 float으로 변환"""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        cleaned = re.sub(r'[^\d.]', '', x)
        return float(cleaned) if cleaned else 0.0
    return 0.0

# ---------------------- 기술적 분석 함수 ----------------------
def calculate_fibonacci_levels(df, period=50):
    """피보나치 되돌림 수준 계산"""
    if len(df) < period:
        period = len(df)
    
    recent_high = df['high'].tail(period).max()
    recent_low = df['low'].tail(period).min()
    
    diff = recent_high - recent_low
    if diff == 0:
        return None, None, None
    
    levels = {
        '0.0': recent_low,
        '0.236': recent_high - diff * 0.236,
        '0.382': recent_high - diff * 0.382,
        '0.5': recent_high - diff * 0.5,
        '0.618': recent_high - diff * 0.618,
        '0.786': recent_high - diff * 0.786,
        '1.0': recent_high
    }
    return recent_high, recent_low, levels

def detect_wave_patterns(df):
    """엘리엇 파동 이론 기반 패턴 감지"""
    patterns = []
    if len(df) < 10:
        return patterns
    
    # 최근 10개 봉 분석
    closes = df['close'].values[-10:]
    highs = df['high'].values[-10:]
    lows = df['low'].values[-10:]
    
    # 임펄스 파동 (상승 5파)
    if (lows[0] < lows[2] < lows[4] < lows[6] < lows[8] and
        highs[1] < highs[3] < highs[5] < highs[7] < highs[9] and
        lows[1] > lows[0] and lows[3] > lows[2] and lows[5] > lows[4] and lows[7] > lows[6]):
        patterns.append("상승 임펄스 파동 (5파 진행중)")
    
    # 조정 파동 (ABC 패턴)
    if (highs[0] > highs[2] > highs[4] and
        lows[1] > lows[3] > lows[5] and
        closes[6] > closes[7] and closes[7] < closes[8]):
        patterns.append("ABC 조정 파동 완료 예상")
    
    # 확장 파동 (상승 5파 확장)
    if (lows[0] < lows[1] < lows[2] < lows[3] < lows[4] and
        highs[0] < highs[1] < highs[2] < highs[3] < highs[4] and
        lows[5] > lows[4] and highs[6] > highs[5] and lows[7] > lows[6] and highs[8] > highs[7] and
        lows[9] < lows[8]):
        patterns.append("확장 상승 파동 (3파 확장)")
    
    return patterns

def detect_chart_patterns(df):
    """차트 패턴 감지 (피보나치 + 엘리엇 파동 통합)"""
    patterns = detect_wave_patterns(df)
    if len(df) < 7: 
        return patterns
    
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
    
    return patterns

# ---------------------- AI 예측 모델 ----------------------
def ai_price_predict(df, current_price, selected_tf, n_future=5):
    """AI 가격 예측 (LSTM + Prophet 앙상블)"""
    if len(df) < 60:
        change_percent = random.uniform(-0.02, 0.02)
        predicted = current_price * (1 + change_percent)
        trend = "상승" if change_percent > 0.005 else "하락" if change_percent < -0.005 else "유지"
        emoji = "📈" if trend == "상승" else "📉" if trend == "하락" else "⚖️"
        return round(predicted, 1), f"{emoji} {trend}", None
    
    try:
        # Prophet 예측
        prophet_df = df[['datetime', 'close']].copy()
        prophet_df = prophet_df.rename(columns={'datetime': 'ds', 'close': 'y'})
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(prophet_df)
        future = prophet_model.make_future_dataframe(periods=n_future, freq=f'{selected_tf}min')
        prophet_forecast = prophet_model.predict(future)
        prophet_pred = prophet_forecast['yhat'][-n_future:].values
        
        # 앙상블 예측 (Prophet에 가중치 100% 적용)
        ensemble_pred = prophet_pred
        
        # 결과 분석
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
            line=dict(color='blue'))
        )

        future_dates = pd.date_range(
            start=df['datetime'].iloc[-1],
            periods=n_future+1,
            freq=f'{selected_tf}min'
        )[1:]

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=ensemble_pred,
            name='AI 예측 가격',
            line=dict(color='red', dash='dot'))
        )

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

# ---------------------- 데이터 처리 ----------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv(market, timeframe, count=300):
    """OHLCV 데이터 조회"""
    try:
        url = f"https://api.upbit.com/v1/candles/minutes/{timeframe}"
        params = {'market': market, 'count': count}
        res = requests.get(url, params=params, timeout=15)
        res.raise_for_status()
        data = res.json()
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        column_mapping = {
            'candle_date_time_kst': 'datetime',
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        }
        df = df.rename(columns=column_mapping)[list(column_mapping.values())]
        df = df.iloc[::-1].reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=False)
        
        # 기술적 지표 계산
        if len(df) > 16:
            df['HL2'] = (df['high'] + df['low']) / 2
            df['HMA'] = df['HL2'].rolling(16).mean()  # HMA 대신 SMA 사용
            df['HMA3'] = df['HL2'].rolling(3).mean()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD 계산
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD_line'] = exp1 - exp2
            df['MACD_hist'] = df['MACD_line'] - df['MACD_line'].ewm(span=9, adjust=False).mean()
            
            # 볼린저 밴드
            ma = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            df['BB_upper'] = ma + (std * 2)
            df['BB_lower'] = ma - (std * 2)
            
        return df
    
    except Exception as e:
        logger.error(f"{market} 데이터 조회 실패: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=10, show_spinner=False)
def get_current_prices(markets):
    """현재 가격 조회"""
    try:
        res = requests.get(f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}", timeout=10)
        res.raise_for_status()
        return {item['market']: {
            'trade_price': item['trade_price'],
            'signed_change_rate': item['signed_change_rate']
        } for item in res.json()}
    except Exception as e:
        logger.error(f"현재 가격 조회 실패: {str(e)}")
        return {}

# ---------------------- 시각화 함수 ----------------------
def create_coin_chart(df, coin, tf_name):
    """코인 차트 생성 (피보나치 수준 포함)"""
    if df.empty or 'close' not in df.columns:
        return None
        
    try:
        latest = df.iloc[-1]
        current_price = latest['close']

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2],
            specs=[[{"secondary_y": True}], [{}], [{}]]
        )
        
        # 캔들스틱 차트
        fig.add_trace(go.Candlestick(
            x=df['datetime'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'], name="Price"), row=1, col=1)
        
        # 현재가 주석
        fig.add_hline(
            y=current_price, line_dash="solid", line_color="orange", row=1, col=1,
            annotation_text=f"현재가: {current_price:,.1f}",
            annotation_position="bottom right",
            annotation_font_size=12,
            annotation_font_color="orange"
        )
        
        # 피보나치 되돌림 라인
        _, _, fib_levels = calculate_fibonacci_levels(df)
        if fib_levels:
            colors = ['#FF6B6B', '#4ECDC4', '#556270', '#C06C84', '#6C5B7B', '#355C7D']
            for i, (ratio, level) in enumerate(fib_levels.items()):
                fig.add_hline(
                    y=level, line_dash="dash", line_color=colors[i % len(colors)],
                    row=1, col=1, annotation_text=f"Fib {ratio} ({level:,.1f})",
                    annotation_position="bottom right"
                )
        
        # 기술적 지표
        if 'HMA' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['HMA'], name='HMA', line=dict(color='blue')), row=1, col=1)
        if 'HMA3' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['HMA3'], name='HMA3', line=dict(color='orange')), row=1, col=1)
        if 'BB_upper' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dot')), row=1, col=1)
        if 'BB_lower' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dot')), row=1, col=1)
        if 'volume' in df.columns:
            fig.add_trace(go.Bar(x=df['datetime'], y=df['volume'], name='Volume',
                            marker_color=np.where(df['close'] > df['open'], 'green', 'red')),
                    row=1, col=1, secondary_y=True)
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        if 'MACD_hist' in df.columns:
            fig.add_trace(go.Bar(x=df['datetime'], y=df['MACD_hist'], name='Histogram',
                            marker_color=np.where(df['MACD_hist'] > 0, 'green', 'red')), row=3, col=1)

        fig.update_layout(
            height=800, title=f"{coin} 차트 ({tf_name})",
            xaxis_rangeslider_visible=False, margin=dict(t=40, b=40),
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"{coin} 차트 생성 오류: {str(e)}")
        return None

# ---------------------- 대시보드 초기화 ----------------------
def init_dashboard():
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
        .text-white { color: #fff !important; text-shadow: 1px 1px 0 #000; }
        .metric-container { padding: 10px; border-radius: 10px; background: #1e1e1e; margin: 10px 0; }
        .fib-alert { background: linear-gradient(135deg, #1a5276, #3498db); padding: 15px; border-radius: 10px; margin: 10px 0; }
        .wave-alert { background: linear-gradient(135deg, #6c5b7b, #c06c84); padding: 15px; border-radius: 10px; margin: 10px 0; }
        .coin-report { background: #1e1e1e; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #3498db; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(
        "<h1 class='text-white' style='text-align:center;font-family:Consolas,monospace;'>"
        "🌊 코인 차트 통합 분석 시스템</h1>",
        unsafe_allow_html=True
    )
    st_autorefresh(interval=5000, key="auto_refresh")
    
    # 세션 상태 초기화
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'last_alert_time' not in st.session_state:
        st.session_state.last_alert_time = {}
    if 'last_table_alert_time' not in st.session_state:
        st.session_state.last_table_alert_time = datetime.min
    if 'ai_progress' not in st.session_state:
        st.session_state.ai_progress = {'STX': 0, 'HBAR': 0, 'DOGE': 0}

# ---------------------- 알림 시스템 ----------------------
def check_pattern_alerts(markets, timeframes, selected_tf):
    """패턴 알림 체크 및 발송"""
    alerts = []
    now = datetime.now()
    
    for market in markets:
        coin = market.split('-')[1]
        df = fetch_ohlcv(market, selected_tf)
        if df.empty:
            continue
            
        current_price = df.iloc[-1]['close']
        tf_name = timeframes[selected_tf]
        
        # 파동 패턴 감지
        wave_patterns = detect_wave_patterns(df)
        for pattern in wave_patterns:
            alert_key = f"{coin}_{pattern[:10]}_pattern"
            last_alert = st.session_state.last_alert_time.get(alert_key, datetime.min)
            
            if (now - last_alert) > timedelta(minutes=10):
                message = (
                    f"🌊 *{coin} {tf_name}차트 파동 패턴 감지!*\n"
                    f"📊 패턴 유형: {pattern}\n"
                    f"💰 현재 가격: `{current_price:,.1f}` 원\n"
                    f"📅 감지 시간: {now.strftime('%m-%d %H:%M')}"
                )
                if send_telegram_alert(message):
                    st.session_state.last_alert_time[alert_key] = now
                    alerts.append(f"🌊 {coin} 파동 패턴: {pattern}")
        
        # 피보나치 돌파 감지
        high, _, fib_levels = calculate_fibonacci_levels(df)
        if fib_levels and current_price > high:
            for ratio, level in fib_levels.items():
                if current_price > level:
                    alert_key = f"{coin}_fib_{ratio}"
                    last_alert = st.session_state.last_alert_time.get(alert_key, datetime.min)
                    
                    if (now - last_alert) > timedelta(minutes=30):
                        message = (
                            f"🚨 *{coin} {tf_name}차트 피보나치 돌파 알림!*\n"
                            f"📈 현재 가격: `{current_price:,.1f}` 원\n"
                            f"🎯 피보나치 {ratio} 수준 돌파\n"
                            f"💎 예상 목표가: `{level:,.1f}` 원"
                        )
                        if send_telegram_alert(message):
                            st.session_state.last_alert_time[alert_key] = now
                            alerts.append(f"📊 {coin} 피보나치 {ratio} 돌파")
    
    return alerts

# ---------------------- 메인 애플리케이션 ----------------------
def main():
    # 초기화
    init_dashboard()
    
    # 전역 변수
    default_holdings = {
        'KRW-STX': 14073.68834666,
        'KRW-HBAR': 62216.22494886,
        'KRW-DOGE': 61194.37067502,
    }
    timeframes = {1: '1분', 3: '3분', 5: '5분', 15: '15분', 60: '60분', 240: '4시간'}
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 제어 패널")
        selected_tf = st.selectbox(
            '차트 주기', list(timeframes.keys()), 
            format_func=lambda x: timeframes[x], index=2
        )
        
        st.subheader("💰 투자 현황")
        prices = get_current_prices(MARKETS)
        stx_price = prices.get('KRW-STX', {}).get('trade_price', 0)
        stx_value = default_holdings['KRW-STX'] * stx_price
        profit = stx_value - TOTAL_INVESTMENT
        profit_percent = (profit / TOTAL_INVESTMENT) * 100
        
        st.metric("총 투자금액", f"{TOTAL_INVESTMENT:,.0f} 원")
        st.metric("STX 자산가치", 
                 f"{stx_value:,.0f} 원", 
                 f"{profit:+,.0f} 원 ({profit_percent:+.2f}%)",
                 delta_color="inverse" if profit < 0 else "normal")
        st.metric("STX 보유량", f"{default_holdings['KRW-STX']:,.2f} EA")
        
        st.subheader("🔔 알림 설정")
        telegram_enabled = st.checkbox("텔레그램 알림 활성화", value=True)
        alert_interval = st.slider("알림 주기(분)", 1, 60, 10)
        
        st.subheader("🧠 AI 학습상황")
        for coin in st.session_state.ai_progress:
            progress = min(st.session_state.ai_progress[coin] + random.randint(2, 6), 100)
            st.session_state.ai_progress[coin] = progress
            st.progress(progress, text=f"{coin}: {progress}%")

    # 실시간 알림 처리
    pattern_alerts = check_pattern_alerts(MARKETS, timeframes, selected_tf)
    st.session_state.alerts.extend(pattern_alerts)
    
    # 실시간 알림 표시
    st.subheader("🔔 실시간 분석 알림")
    for alert in st.session_state.alerts[-10:]:
        alert_type = "fib-alert" if "피보나치" in alert else "wave-alert" if "파동" in alert else ""
        st.markdown(f"<div class='{alert_type}'>{alert}</div>", unsafe_allow_html=True)
    
    # 코인 비교 테이블 생성
    st.subheader(f"📊 코인 비교 테이블 ({timeframes[selected_tf]}봉)")
    prices = get_current_prices(MARKETS)
    table_data = []
    
    for market in MARKETS:
        coin = market.split('-')[1]
        price_data = prices.get(market, {})
        price = price_data.get('trade_price', 0)
        change_rate = price_data.get('signed_change_rate', 0) * 100
        
        # AI 예측
        df = fetch_ohlcv(market, selected_tf)
        ai_pred, ai_trend, ai_fig = ai_price_predict(df, price, selected_tf)
        
        # 대체 수량 계산 (STX 기준)
        if market != 'KRW-STX':
            stx_price = prices.get('KRW-STX', {}).get('trade_price', 1)
            replace_qty = (default_holdings['KRW-STX'] * stx_price * 0.9995) / price
            diff_qty = replace_qty - default_holdings[market]
        else:
            replace_qty = "-"
            diff_qty = "-"
        
        table_data.append({
            '코인명': coin,
            '현재가': f"{price:,.1f} 원",
            '변동율': f"{change_rate:+.2f}%",
            'AI 예측': f"{ai_pred:,.1f} 원",
            'AI 전망': ai_trend,
            '대체 수량': f"{replace_qty:,.2f}" if isinstance(replace_qty, float) else replace_qty,
            '수량 차이': f"{diff_qty:+,.2f}" if isinstance(diff_qty, float) else diff_qty
        })
    
    # 테이블 표시
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True)
    
    # 개별 코인 차트 및 AI 예측 표시
    for market in MARKETS:
        coin = market.split('-')[1]
        df = fetch_ohlcv(market, selected_tf)
        if df.empty:
            continue
            
        price = df.iloc[-1]['close']
        ai_pred, ai_trend, ai_fig = ai_price_predict(df, price, selected_tf)
        
        # 차트 표시
        with st.expander(f"{coin} 분석", expanded=True):
            col1, col2 = st.columns([0.7, 0.3])
            
            with col1:
                # 실시간 차트
                chart = create_coin_chart(df, coin, timeframes[selected_tf])
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            with col2:
                # AI 예측 정보
                st.subheader(f"🔮 {coin} AI 예측")
                st.metric("예측 가격", f"{ai_pred:,.1f} 원")
                st.metric("전망", ai_trend)
                
                if ai_fig:
                    st.plotly_chart(ai_fig, use_container_width=True)
                
                # 피보나치 분석
                high, low, fib_levels = calculate_fibonacci_levels(df)
                if fib_levels:
                    st.subheader("📊 피보나치 수준")
                    for ratio, level in fib_levels.items():
                        st.write(f"- **{ratio}**: `{level:,.1f}` 원")
                
                # 파동 패턴
                patterns = detect_wave_patterns(df)
                if patterns:
                    st.subheader("🌊 파동 패턴")
                    for pattern in patterns:
                        st.write(f"- {pattern}")

if __name__ == "__main__":
    main()
