import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from streamlit_autorefresh import st_autorefresh

# 기술 지표 계산 함수
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

def rsi(series, period=14):
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

st.set_page_config(layout="wide")
st.title("🔥 실시간 분석 (정교한 조건 기반)")

init_investment = 58721620
default_holdings = {'KRW-STX': 15694.14065172}
markets = list(default_holdings.keys())
timeframes = {1: '1분', 3: '3분', 5: '5분', 15: '15분', 60: '60분', 240: '240분'}
price_init = {market: 0 for market in markets}

@st.cache_data(ttl=30)
def get_current_prices():
    try:
        url = f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}"
        res = requests.get(url, timeout=5)
        return {x['market']: x['trade_price'] for x in res.json()}
    except:
        return price_init

@st.cache_data(ttl=30)
def fetch_ohlcv(market, timeframe, count=200):
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
        df['RSI'] = rsi(df['close'])
        df['MACD_line'], df['Signal_line'], df['MACD_hist'] = macd(df['close'])
        df['BB_ma'], df['BB_upper'], df['BB_lower'] = bollinger_bands(df['close'])

        # 정교한 예측 신호 생성
        df['predicted_signal'] = np.where(
            (df['HMA3'] > df['HMA']) &
            (df['MACD_hist'] > 0) &
            (df['MACD_line'] > df['Signal_line']) &
            (df['RSI'].between(50, 65)),
            'up',
            np.where(
                (df['HMA3'] < df['HMA']) &
                (df['MACD_hist'] < 0) &
                (df['MACD_line'] < df['Signal_line']) &
                (df['RSI'].between(35, 50)),
                'down',
                'neutral'
            )
        )
        return df
    except:
        return pd.DataFrame()

# 사이드바
with st.sidebar:
    st.header("⚙️ 제어 패널")
    selected_tf = st.selectbox('차트 주기', list(timeframes.keys()), format_func=lambda x: timeframes[x])

    prices = get_current_prices()
    st.header("💰 자산 현황")

    stx_qty = default_holdings['KRW-STX']
    stx_price = prices.get('KRW-STX', 0)
    current_value = stx_qty * stx_price
    profit_loss = current_value - init_investment
    profit_percent = profit_loss / init_investment * 100 if init_investment else 0

    st.metric(label="총 자산 평가액", value=f"{current_value:,.0f} KRW",
              delta=f"{profit_loss:,.0f} KRW ({profit_percent:.2f}%)")

# 메인화면 - 차트
try:
    df = fetch_ohlcv('KRW-STX', selected_tf)
    if df.empty:
        st.warning("차트 데이터를 불러올 수 없습니다.")
    else:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03,
                            row_heights=[0.6, 0.2, 0.2],
                            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]])

        fig.add_trace(go.Candlestick(x=df['datetime'], open=df['open'], 
                                     high=df['high'], low=df['low'], close=df['close'], 
                                     name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['HMA'], name='HMA'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['HMA3'], name='HMA3'), row=1, col=1)

        fig.add_trace(go.Bar(x=df['datetime'], y=df['volume'], name='Volume',
                             marker_color=np.where(df['close'] > df['open'], 'green', 'red')),
                     row=1, col=1, secondary_y=True)

        # 정교한 예측 이모지 표시
        interval = 5
        for i in range(interval, len(df), interval):
            signal = df['predicted_signal'].iloc[i]
            if signal == 'up':
                emoji = "🚀"
            elif signal == 'down':
                emoji = "⚡"
            else:
                continue
            fig.add_annotation(
                x=df['datetime'].iloc[i],
                y=df['high'].iloc[i] * 1.01,
                text=emoji,
                showarrow=False,
                font=dict(size=20),
                xanchor='center'
            )

        fig.add_trace(go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI',
                                 line=dict(color='purple')), row=2, col=1)

        fig.add_trace(go.Bar(x=df['datetime'], y=df['MACD_hist'], name='Histogram',
                             marker_color=np.where(df['MACD_hist'] > 0, 'green', 'red')), 
                     row=3, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD_line'], name='MACD',
                                 line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['Signal_line'], name='Signal',
                                 line=dict(color='orange')), row=3, col=1)

        fig.update_layout(height=1000, template='plotly_dark', margin=dict(t=100))
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"⚠️ 차트 생성 오류: {str(e)}")

# 자동 새로고침
st_autorefresh(interval=30000, key="auto_refresh")
