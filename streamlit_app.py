import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import load_model
import joblib

# ✅ 반드시 첫 번째 Streamlit 명령어로 설정
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed"  # 사이드바 초기 숨김
)

# ---------------------- 세션 상태 초기화 ----------------------
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}
if 'ma_cross_alerts' not in st.session_state:
    st.session_state.ma_cross_alerts = {}
if 'total_investment' not in st.session_state:
    st.session_state.total_investment = 58500000
if 'telegram_bot_token' not in st.session_state:
    st.session_state.telegram_bot_token = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
if 'telegram_chat_id' not in st.session_state:
    st.session_state.telegram_chat_id = "7890657899"

# 자동 갱신 설정 (5초 간격)
st_autorefresh(interval=5000, key="auto_refresh")

# ---------------------- 형광 녹색 스타일 적용 ----------------------
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
    @media (max-width: 600px) {{
        .stSelectbox, .stNumberInput, .stTextInput, .stButton > button {{
            font-size: 16px !important;
        }}
        .stAlert {{
            padding: 5px !important;
        }}
        .table-wrapper {{
            overflow-x: auto;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- 메인 타이틀 ----------------------
st.markdown(f"<h3 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 If you're not desperate, don't even think about it!</h3>", unsafe_allow_html=True)
st.markdown(f"<h5 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 Live Cryptocurrency Analytics Dashboard</h5>", unsafe_allow_html=True)

# ---------------------- 전역 변수 ----------------------
default_holdings = {
    'KRW-STX': 13702.725635935,
    'KRW-HBAR': 62216.22494886,
    'KRW-DOGE': 61194.37067502,
}
markets = list(default_holdings.keys())
timeframes = {1: '1분', 3: '3분', 5: '5분', 15: '15분', 60: '60분', 240: '240분'}

# ---------------------- AI 예측 모델 ----------------------
def create_prediction_model():
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, activation='relu', return_sequences=False),
        Dense(25, activation='relu'),
        Dense(5)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def apply_prediction_correction(pred_prices, actual_prices):
    """실제 가격과 예측값 차이 기반 보정"""
    if len(actual_prices) < 5 or len(pred_prices) < 5:
        return pred_prices
    errors = []
    for i in range(1, min(6, len(actual_prices))):
        if len(pred_prices) >= i:
            errors.append(actual_prices[-i] - pred_prices[-i])
    if not errors:
        return pred_prices
    avg_error = sum(errors) / len(errors)
    return [p + avg_error * 0.7 for p in pred_prices]

# --- 자동 학습 및 예측 결과 CSV 저장 함수 ---
def auto_train_and_predict(df, selected_tf, model_dir="ai_models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"lstm_{selected_tf}.h5")
    scaler_path = os.path.join(model_dir, f"scaler_{selected_tf}.pkl")
    csv_path = os.path.join(model_dir, f"pred_{selected_tf}.csv")
    correction_path = os.path.join(model_dir, f"correction_{selected_tf}.csv")

    if len(df) < 65:  # 60(과거) + 5(예측)
        return [], []

    prices = df['close'].values
    min_val, max_val = prices.min(), prices.max()
    scaler = {'min': min_val, 'max': max_val}

    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
        except:
            model = create_prediction_model()
    else:
        model = create_prediction_model()

    norm_prices = (prices - min_val) / (max_val - min_val + 1e-8)
    X = norm_prices[-65:-5].reshape(1, 60, 1)
    y = norm_prices[-5:].reshape(1, 5)

    model.fit(X, y, epochs=3, verbose=0)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    prediction = model.predict(X)
    denorm_pred = prediction[0] * (max_val - min_val) + min_val

    corrected_pred = denorm_pred
    if os.path.exists(correction_path):
        correction_df = pd.read_csv(correction_path)
        actuals = correction_df['actual'].values
        preds = correction_df['predicted'].values
        corrected_pred = apply_prediction_correction(denorm_pred, actuals)

    last_time = df['datetime'].iloc[-1]
    pred_times = [last_time + timedelta(minutes=selected_tf*(i+1)) for i in range(5)]

    # 예측 결과 DataFrame에 verified/accuracy 컬럼 추가
    pred_df = pd.DataFrame({
        "datetime": pred_times,
        "predicted": denorm_pred,
        "corrected": corrected_pred,
        "verified": [False] * 5,
        "accuracy": [None] * 5
    })
    pred_df.to_csv(csv_path, index=False)

    new_correction = pd.DataFrame({
        "datetime": [datetime.now()],
        "actual": [df['close'].iloc[-1]],
        "predicted": [denorm_pred[0]]
    })
    if os.path.exists(correction_path):
        correction_df = pd.read_csv(correction_path)
        correction_df = pd.concat([correction_df, new_correction])
    else:
        correction_df = new_correction
    correction_df.to_csv(correction_path, index=False)

    return corrected_pred, pred_times

# ---------------------- 기술적 지표 계산 함수 ----------------------
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

# ---------------------- 신호 점수 계산 함수 ----------------------
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

# ---------------------- 실시간 알림 내역 표시 ----------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("📢 알림 내역")
    alert_history = st.session_state.get("alerts", [])
    if alert_history:
        for alert in reversed(alert_history[-10:]):
            st.markdown(
                f"<div style='background:#222;padding:8px 10px;margin:4px 0;border-radius:6px;font-size:14px;color:#39FF14;'>{alert}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("알림 내역이 없습니다.")

# ---------------------- 데이터 함수 ----------------------
@st.cache_data(ttl=10)
def get_current_prices():
    try:
        res = requests.get(f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}")
        return {x['market']:x for x in res.json()}
    except:
        return {market: {'trade_price': 0, 'signed_change_rate': 0} for market in markets}

@st.cache_data(ttl=30)
def fetch_ohlcv(market, timeframe, count=300):
    try:
        url = f"https://api.upbit.com/v1/candles/minutes/{timeframe}"
        params = {'market': market, 'count': count}
        res = requests.get(url, params=params, timeout=10)
        df = pd.DataFrame(res.json())[::-1]
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
        # --- 강화 신호용 추가 지표 ---
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_ma']
        df['volume_momentum'] = df['volume'] / df['volume'].shift(1)
        return df
    except:
        return pd.DataFrame()

# ---------------------- 코인 테이블 생성 함수 ----------------------
def generate_coin_table(selected_tf):
    signal_scores = {} 
    base_market = 'KRW-STX'
    base_qty = default_holdings[base_market]
    base_price_data = prices.get(base_market, {'trade_price': 0, 'signed_change_rate': 0})
    base_price = base_price_data['trade_price']
    base_krw = base_qty * base_price * 0.9995

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
                    buy_score, sell_score, _, _, _, _, latest_rsi = calculate_signal_score(df, latest, pred_prices=None)
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

    df_compare = pd.DataFrame(compare_data)
    df_compare['차이수량'] = df_compare['차이수량'].apply(lambda x: float(x) if x != "-" else x)
    df_compare['대체가능수량'] = df_compare['대체가능수량'].apply(lambda x: float(x) if x != "-" else x)
    return df_compare

def format_number(x):
    if isinstance(x, (int, float)):
        if x == "-": return x
        return f"{x:,.2f}" if abs(x) < 10000 else f"{x:,.0f}"
    return x

# ---------------------- 사용자 입력 ----------------------
with st.sidebar:
    st.subheader("📈 차트 주기(분) 선택")
    selected_tf = st.selectbox("차트 주기", list(timeframes.keys()), 
                              format_func=lambda x: timeframes[x], key="selected_tf", index=2)
    st.header("🔑 사용자 설정")
    telegram_enabled = st.checkbox("✉️ 텔레그램 알림 활성화", value=False)
    st.subheader("✉️ 텔레그램 설정")
    st.session_state.telegram_bot_token = st.text_input(
        "텔레그램 봇 토큰",
        value=st.session_state.telegram_bot_token
    )
    st.session_state.telegram_chat_id = st.text_input(
        "텔레그램 채팅 ID",
        value=st.session_state.telegram_chat_id
    )
    st.subheader("💰 보유 코인 설정")
    for market in ['KRW-STX', 'KRW-HBAR', 'KRW-DOGE']:
        coin = market.split('-')[1]
        default_holdings[market] = st.number_input(
            f"{coin} 보유량",
            value=default_holdings.get(market, 0.0),
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
    st.subheader("🤖 AI 예측 설정")
    pred_horizon = st.slider("예측 기간 (봉)", 1, 10, 5)
    ai_confidence = st.slider("신뢰도 임계값", 50, 100, 80)

# ---------------------- 텔레그램 알림 함수 ----------------------
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

# ---------------------- 메인 콘텐츠 ----------------------
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
    )
    st.markdown('<div class="table-wrapper">' + styled.to_html(escape=False, index=False) + '</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("📈 RSI 비교 차트")
    fig_rsi = make_subplots(rows=1, cols=1)
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

    st.subheader("📈 이평선 & AI 예측 vs 실제가")
    selected_coin = st.selectbox("코인 선택 (이평선/AI예측)", [m.split('-')[1] for m in markets], key="ma_ai_coin")
    market_for_coin = [m for m in markets if m.split('-')[1] == selected_coin][0]
    df_ma = fetch_ohlcv(market_for_coin, selected_tf)
    ai_preds, ai_pred_times = [], []
    if not df_ma.empty:
        df_ma['SMA5'] = df_ma['close'].rolling(5).mean()
        df_ma['SMA10'] = df_ma['close'].rolling(10).mean()
        df_ma['SMA20'] = df_ma['close'].rolling(20).mean()
        df_ma['SMA60'] = df_ma['close'].rolling(60).mean()
        df_ma['SMA120'] = df_ma['close'].rolling(120).mean()
        df_ma['SMA200'] = df_ma['close'].rolling(200).mean()

        try:
            ai_preds, ai_pred_times = auto_train_and_predict(df_ma, selected_tf)
            actuals = []
            for pred_time in ai_pred_times:
                actual_row = df_ma[df_ma['datetime'] == pred_time]
                if not actual_row.empty:
                    actuals.append(actual_row['close'].values[0])
            if ai_preds and actuals:
                ai_preds = apply_prediction_correction(ai_preds, actuals)
        except Exception as e:
            ai_preds, ai_pred_times = [], []
            st.warning(f"AI 예측 오류: {e}")

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=df_ma['datetime'], y=df_ma['close'],
            name='실제 가격', line=dict(color='blue', width=2)
        ))
        for period, color in zip([5, 10, 20, 60, 120, 200], ['orange', 'green', 'purple', 'gray', 'brown', 'black']):
            col = f"SMA{period}"
            if col in df_ma.columns:
                fig_ma.add_trace(go.Scatter(
                    x=df_ma['datetime'], y=df_ma[col],
                    name=f'SMA{period}', line=dict(width=1, dash='dot', color=color)
                ))
        if ai_preds is not None and ai_pred_times and len(ai_preds) > 0:
            tf_minutes = int(selected_tf)
            if tf_minutes == 1:
                pred_range = 240
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

            show_preds = ai_preds[:pred_range]
            show_times = ai_pred_times[:pred_range]

            fig_ma.add_trace(go.Scatter(
                x=show_times, y=show_preds,
                name=f'AI 예측가 ({pred_label})',
                mode='markers+text',
                marker=dict(size=14, color='red', symbol='star'),
                text=[f"{y:,.0f}" for y in show_preds],
                textposition="top center",
                hovertemplate='AI 예측가: %{y:,.0f}원<br>시간: %{x}<extra></extra>'
            ))
            fig_ma.add_trace(go.Scatter(
                x=show_times, y=show_preds,
                name=f'AI 예측선 ({pred_label})',
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                showlegend=True
            ))
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

# ---------------------- 이동평균선 교차 알림 로직 ----------------------
for market in markets:
    coin = market.split('-')[1]
    df = fetch_ohlcv(market, selected_tf)
    if len(df) > 200:
        df['SMA5'] = df['close'].rolling(5).mean()
        df['SMA20'] = df['close'].rolling(20).mean()
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        golden_cross = (prev['SMA5'] < prev['SMA20']) and (latest['SMA5'] > latest['SMA20'])
        dead_cross = (prev['SMA5'] > prev['SMA20']) and (latest['SMA5'] < latest['SMA20'])
        now = datetime.now()
        alert_key = f"{market}_cross"
        last_alert = st.session_state.ma_cross_alerts.get(alert_key, datetime.min)
        if (golden_cross or dead_cross) and (now - last_alert) > timedelta(minutes=10):
            msg_type = "골든크로스" if golden_cross else "데드크로스"
            message = f"🚨 {coin} {msg_type} 발생! ({now.strftime('%H:%M:%S')})"
            st.session_state.alerts.append(message)
            st.session_state.ma_cross_alerts[alert_key] = now
            if telegram_enabled:
                send_telegram_alert(message)
