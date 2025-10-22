# 통합-HBAR_최종.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# ---------------------- 로그인 설정 ----------------------
def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.title("🔐 로그인 필요")
        
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            with st.form("login_form"):
                st.subheader("모카 대시보드 로그인")
                input_id = st.text_input("아이디")
                input_pw = st.text_input("비밀번호", type="password")
                login_btn = st.form_submit_button("로그인")
                
                if login_btn:
                    if input_id == "miwooni" and input_pw == "Fudfud8080@":
                        st.session_state.logged_in = True
                        st.success("로그인 성공! 대시보드로 이동합니다.")
                        st.rerun()
                    else:
                        st.error("아이디 또는 비밀번호가 incorrect.")
            return False
        return False
    return True

# 로그인 체크 - 로그인 안되었으면 여기서 실행 중단
if not check_login():
    st.stop()

# ---------------------- 공통 함수 정의 ----------------------
def send_telegram_alert(message: str):
    bot_token = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
    chat_id = "7890657899"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    
    try:
        now = datetime.now()
        # 텔레그램 전송 로그 초기화
        if 'telegram_sent_log' not in st.session_state:
            st.session_state.telegram_sent_log = []
            
        # 최근 5분 내 전송된 메시지 필터링
        recent_messages = [
            msg for msg, sent_time in st.session_state.telegram_sent_log
            if (now - sent_time).total_seconds() < 300  # 5분 = 300초
        ]
        
        # 동일 메시지가 있는지 확인
        is_duplicate = message in recent_messages
        
        # 10% 이상 등락률은 중복 체크 무시
        is_significant = False
        if "등락률" in message:
            try:
                pct_str = message.split("등락률")[1].split("%")[0]
                pct_val = float(pct_str.replace(":", "").replace(" ", "").replace("+", ""))
                is_significant = abs(pct_val) >= 10
            except:
                pass
                
        # 중복이 아니거나 중요 알림인 경우만 전송
        if not is_duplicate or is_significant:
            requests.post(url, data=payload, timeout=5)
            
            # 알림 로그 추가
            if 'alerts' not in st.session_state:
                st.session_state.alerts = []
            st.session_state.alerts.append(f"텔레그램 전송: {message}")
            
            # 전송 로그에 기록
            st.session_state.telegram_sent_log.append((message, now))
            
            # 오래된 로그 정리 (5분 이상 지난 로그 제거)
            st.session_state.telegram_sent_log = [
                (msg, time) for msg, time in st.session_state.telegram_sent_log
                if (now - time).total_seconds() < 300
            ]
            
    except Exception as e:
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
            patterns.append("강한 상승 예측측")
    
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

# ---------------------- 초기 설정 ----------------------
st.set_page_config(layout="wide")
st.title("🔥 If you're not desperate, don't even think about it!")
st.markdown("🔥 Live Cryptocurrency Analytics Dashboard", unsafe_allow_html=True)
st_autorefresh(interval=5000, key="auto_refresh")

# ---------------------- 전역 변수 ----------------------
default_holdings = {
    'KRW-STX': 15000,
    'KRW-ENA': 13000,
    'KRW-HBAR': 62216,
    'KRW-DOGE': 61194.37067502,
}
markets = list(default_holdings.keys())
timeframes = {1: '1분', 3: '3분', 5: '5분', 15: '15분', 60: '60분', 240: '240분', 360: '360분'}
TOTAL_INVESTMENT = 58500000

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
        return df
    except:
        return pd.DataFrame()

# ---------------------- 사이드바 설정 ----------------------
with st.sidebar:
    st.header("⚙️ 제어 패널")
    selected_tf = st.selectbox('차트 주기', list(timeframes.keys()), format_func=lambda x: timeframes[x])
    
    # 로그아웃 버튼
    if st.button("🚪 로그아웃"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.subheader("💰 투자 현황")
    prices = get_current_prices()
    stx_holding = default_holdings['KRW-STX']
    stx_price = prices['KRW-STX']['trade_price']
    current_value = stx_holding * stx_price
    profit = current_value - TOTAL_INVESTMENT
    profit_percent = (profit / TOTAL_INVESTMENT) * 100
    profit_emoji = "🔻" if profit < 0 else "🟢"
    
    st.metric("총 투자금액", f"{TOTAL_INVESTMENT:,.0f} 원")
    st.metric("STX 자산가치", 
            f"{current_value:,.0f} 원", 
            f"{profit_emoji} {profit:+,.0f} 원 ({profit_percent:+.2f}%)")
    st.metric("STX 보유량", f"{stx_holding:,.2f} EA")
    
    st.subheader("🔔 텔레그램 알림")
    telegram_enabled = st.checkbox("텔레그램 알림 활성화")
    table_alert_interval = st.number_input("테이블 알림 주기(분)", min_value=1, value=10)

# ---------------------- 메인 화면 ----------------------
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}
if 'last_table_alert_time' not in st.session_state:
    st.session_state.last_table_alert_time = datetime.min

def generate_coin_table():
    signal_scores = {} 
    base_market = 'KRW-STX'
    base_qty = default_holdings[base_market]
    base_price_data = prices.get(base_market, {'trade_price': 0, 'signed_change_rate': 0})
    base_price = base_price_data['trade_price']
    base_krw = base_qty * base_price * 0.9995

    compare_data = []
    now = datetime.now()  # <-- 추가: now를 함수 내부에서 정의
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
            replace_qty = (base_krw * 0.9995) / price if price else 0
            diff_qty = replace_qty - qty
            replace_value = replace_qty * price * 0.9995
            
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
    df_compare['차이수량'] = df_compare['차이수량'].apply(parse_number)
    df_compare['대체가능수량'] = df_compare['대체가능수량'].apply(parse_number)
    return df_compare

st.subheader("📊 코인 비교 테이블 (RSI 포함)")
df_compare = generate_coin_table()

def format_number(x):
    if isinstance(x, (int, float)):
        if x == "-": return x
        return f"{x:,.2f}" if abs(x) < 10000 else f"{x:,.0f}"
    return x

st.markdown(
    df_compare.style.format({
        '보유수량': format_number,
        '대체가능수량': format_number,
        '차이수량': format_number,
        '평가금액': format_number,
        '대체평가액': format_number,
        'RSI': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
    }).set_properties(**{'text-align': 'center'}).hide(axis="index").to_html(escape=False),
    unsafe_allow_html=True
)

# ---------------------- 테이블 알림 생성 ----------------------
now = datetime.now()
if (now - st.session_state.last_table_alert_time) > timedelta(minutes=table_alert_interval):
    alert_msg = "📊 분석현황(by MIWOONI)\n\n"
    for _, row in df_compare.iterrows():
        alert_msg += (
            f"[{row['코인명']}]\n"
            f"시세: {row['시세']}\n"
            f"RSI: {row['RSI']}\n"
            f"매수신호: {row['매수신호'].split('>')[1].split('<')[0]}\n"
            f"매도신호: {row['매도신호'].split('>')[1].split('<')[0]}\n"
            f"보유량: {row['보유수량']:,.2f}\n"
            f"평가금액: {row['평가금액']:,.0f}원\n"
        )
        if row['코인명'].upper() in ['HBAR', 'DOGE', 'ENA']:
            alert_msg += (
                f"대체 가능 수량: {row['대체가능수량']:,.2f}\n"
                f"차이 수량: {row['차이수량']:+,.2f}\n"
            )
        alert_msg += "\n"
    
    st.session_state.last_table_alert_time = now
    if telegram_enabled:
        send_telegram_alert(alert_msg.strip())
    st.session_state.alerts.append(alert_msg)

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
    
    fig.add_hline(y=current_price, line_dash="solid", line_color="orange", row=1, col=1,
                 annotation_text=f"      _____________________________________________________________________________________________________________________________ 현재가: {current_price:,.1f}______________________________________", 
                 annotation_position="top left",
                 annotation_font_color="white",
                 annotation_font_size=14,
                 annotation_xanchor='center')
    
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
    
    # 패턴 감지 및 알림
    detected_patterns = detect_chart_patterns(df)
    pattern_alerts = []
    for pattern in detected_patterns:
        timeframe_str = timeframes[selected_tf]  # 선택된 시간대 문자열 가져오기
        alert_msg = f"🚨🚨 [{coin} {timeframe_str}차트]{current_price:,.1f} 원 // 강력한 패턴 경고 - {pattern} 발현! ({now.strftime('%H:%M:%S')})"
        pattern_alerts.append(alert_msg)
        
        # 마지막 알림 시간 확인 (10분 내 동일 패턴 알림 방지)
        last_pattern_alert_key = f"{coin}_{pattern[:10]}_pattern"
        last_alert_time = st.session_state.last_alert_time.get(last_pattern_alert_key, datetime.min)
        
        if (now - last_alert_time) > timedelta(minutes=10):  # 10분 간격으로만 알림
            if telegram_enabled:
                send_telegram_alert(alert_msg)
            st.session_state.last_alert_time[last_pattern_alert_key] = now

    st.session_state.alerts.extend(pattern_alerts)

# ---------------------- 실시간 알림 출력 ----------------------
st.subheader("🔔 분석알림 (RSI, 패턴, 수량 변화 포함)")
for alert in reversed(st.session_state.alerts[-10:]):
    # 알림 유형에 따라 다른 스타일 적용
    if "대체 수량 전환" in alert:
        bg_color = "#1a5276"  # 파란색 계열 (수량 변화 알림)
    elif "패턴 경고" in alert:
        bg_color = "#7d3c98"  # 보라색 계열 (패턴 알림)
    elif "매수 신호" in alert:
        bg_color = "#196f3d"  # 녹색 계열 (매수 알림)
    elif "매도 신호" in alert:
        bg_color = "#943126"  # 빨간색 계열 (매도 알림)
    else:
        bg_color = "#2b2b2b"  # 기본 배경색
        
    st.markdown(
        f"""<div style='padding:10px; background:{bg_color}; border-radius:5px; margin:5px 0;'>{alert}</div>""", 
        unsafe_allow_html=True
    )
