import streamlit as st
import sqlite3
from datetime import datetime, timedelta
import os
import glob
import base64
import fitz  # PyMuPDF
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pytube import YouTube
import requests
from bs4 import BeautifulSoup
import webbrowser
from PIL import Image
import io

# --- 페이지 설정 ---
st.set_page_config(
    page_title="성진아 도전!!!",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 스타일 적용 (검정 바탕에 녹색 글씨, 사이드바 폰트 2배 확대, 커서 색상 변경) ---
st.markdown("""
<style>
    * {
        background-color: #000000 !important;
        color: #00FF00 !important;
    }
    .stApp, .stSidebar {
        background-color: #000000 !important;
    }
    .stButton>button {
        background-color: #006600;
        color: #00FF00;
        border: 1px solid #00FF00;
    }
    .stButton>button:hover {
        background-color: #004400;
        color: #00FF00;
        border: 1px solid #00FF00;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #001100;
        color: #00FF00;
    }
    .stSelectbox>div>div>select {
        background-color: #001100;
        color: #00FF00;
    }
    .stRadio>div {
        background-color: #001100;
        color: #00FF00;
    }
    .stSlider>div>div>div>div {
        background-color: #006600;
    }
    .stProgress>div>div>div {
        background-color: #006600;
    }
    .stExpander>div>div>div {
        background-color: #001100;
        color: #00FF00;
    }
    .stMarkdown {
        color: #00FF00;
    }
    .stAlert {
        background-color: #001100;
        color: #00FF00;
    }
    .comcbt-iframe {
        width: 100%;
        height: 800px;
        border: 2px solid #00FF00;
        border-radius: 10px;
        overflow: hidden;
    }
    /* 사이드바 메뉴 글자 크기 2.6배 확대 (30% 증가) */
    .stSidebar .stRadio > label > div {
        font-size: 31.2px !important;  /* 24px * 1.3 = 31.2px */
        padding: 15px 0 !important;
    }
    /* 페이징 컨트롤 스타일 */
    .pagination-control {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .pagination-control button {
        margin: 0 10px;
        padding: 5px 15px;
    }
    /* 입력 커서(caret) 형광색 설정 */
    input, textarea, [contenteditable] {
        caret-color: #00FF00 !important; /* 형광 초록 */
        color: #00FF00 !important;       /* 입력 글자도 형광 */
        background-color: #000000 !important; /* 배경을 어두운 색으로 대비 ↑ */
        border: 1px solid #00FF00 !important; /* 테두리도 형광색으로 */
    }
    /* 고정 영역 스타일 */
    .fixed-section {
        position: sticky;
        top: 0;
        z-index: 100;
        background-color: #000000;
        padding: 15px;
        border-bottom: 2px solid #00FF00;
        margin-bottom: 20px;
    }
    /* 스크롤 영역 스타일 */
    .scrollable-section {
        max-height: calc(100vh - 300px);
        overflow-y: auto;
        padding: 10px;
    }
    /* 검색 기준 선택시 강조 */
    .search-highlight {
        background-color: #004400;
        padding: 5px;
        border-radius: 5px;
        border: 1px solid #00FF00;
    }
    /* 회로이론 기호 테이블 스타일 */
    .symbol-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 0.95em;
        background-color: #002200;
        border: 1px solid #00FF00;
        border-radius: 10px;
        overflow: hidden;
    }
    .symbol-table th {
        background-color: #004400;
        color: #00FF00;
        text-align: center;
        padding: 12px 15px;
        font-weight: bold;
        border-bottom: 2px solid #00FF00;
    }
    .symbol-table td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid #00AA00;
    }
    .symbol-table tr:nth-of-type(even) {
        background-color: #001100;
    }
    .symbol-table tr:last-of-type {
        border-bottom: 2px solid #00FF00;
    }
    .symbol-table tr:hover {
        background-color: #003300;
    }
    .symbol-table td:first-child {
        text-align: center;
        font-weight: bold;
    }
    .symbol-table td:nth-child(2) {
        text-align: center;
        font-family: monospace;
    }
    .highlight-row {
        background-color: #003300 !important;
    }
    /* 3분할 레이아웃 설정 */
    .main-layout {
        display: flex;
        width: 100%;
        gap: 20px;
    }
    .left-sidebar {
        width: 20%;
        min-width: 250px;
    }
    .main-content {
        flex: 1;
        width: 60%;
    }
    .right-sidebar {
        width: 20%;
        min-width: 300px;
        background-color: #001100;
        padding: 15px;
        border-left: 2px solid #00FF00;
        border-radius: 10px;
        max-height: calc(100vh - 100px);
        overflow-y: auto;
    }
    .right-sidebar-tabs .stTabs [role="tablist"] {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    .right-sidebar-tabs .stTabs [role="tab"] {
        flex: 1;
        text-align: center;
        padding: 10px 0;
        background-color: #002200;
        border: 1px solid #00FF00;
        border-radius: 5px;
        cursor: pointer;
    }
    .right-sidebar-tabs .stTabs [role="tab"][aria-selected="true"] {
        background-color: #004400;
        font-weight: bold;
    }
    /* 반응형 디자인 */
    @media (max-width: 1200px) {
        .main-layout {
            flex-direction: column;
        }
        .left-sidebar, .main-content, .right-sidebar {
            width: 100%;
        }
        .right-sidebar {
            margin-top: 30px;
            border-left: none;
            border-top: 2px solid #00FF00;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- 개발자 크레딧 ---
def show_developer_credit():
    st.sidebar.divider()
    # 사이드바에 이미지 추가 (크기 40% 확대: 150 -> 210)
    try:
        st.sidebar.image("moca.jpg", width=210)
        st.sidebar.markdown("<center>나는 할 수 밖에 없다.!!!<br>⚡ Made by Sung Jin ⚡</center>", unsafe_allow_html=True)
    except:
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 10px; background-color: #002200; border-radius: 10px; margin-top: 20px;">
            <p style="color: #00FF00; margin-bottom: 5px;">나는 할 수 밖에 없다.!!!</p>
            <h4 style="color: #00FF00; margin-top: 0;">⚡ Made by Sung Jin ⚡</h4>
        </div>
        """, unsafe_allow_html=True)

# --- 데이터베이스 초기화 ---
def init_databases():
    # 동영상 데이터베이스
    with sqlite3.connect("videos.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                subject TEXT,
                title TEXT,
                watch_count INTEGER DEFAULT 0,
                last_watched TEXT,
                url TEXT
            )
        """)
    
    # 학습 자료 저장 테이블 (기존 메모장 대체)
    with sqlite3.connect("study_materials.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS study_materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                subject TEXT,
                title TEXT,
                content TEXT,
                timestamp TEXT
            )
        """)
    
    # 용어집 저장 테이블 (이미지 경로 저장 필드 추가)
    with sqlite3.connect("glossary.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT UNIQUE,
                definition TEXT,
                subject TEXT,
                timestamp TEXT,
                image_path TEXT
            )
        """)
        
        # 기존 테이블에 image_path 컬럼이 없는 경우 추가
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(glossary)")
        columns = [col[1] for col in cursor.fetchall()]
        if "image_path" not in columns:
            cursor.execute("ALTER TABLE glossary ADD COLUMN image_path TEXT")
        conn.commit()

# --- 데이터베이스 유틸리티 함수 ---
def db_query(db_name, query, params=(), fetch=False, fetch_one=False):
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        if isinstance(params, list) and params and isinstance(params[0], (tuple, list)):
            cursor.executemany(query, params)
        else:
            cursor.execute(query, params)
        if fetch_one:
            return cursor.fetchone()
        if fetch:
            return cursor.fetchall()
        conn.commit()

# --- 이미지 저장 함수 ---
def save_uploaded_image(uploaded_file):
    # 이미지 저장 디렉토리 생성
    os.makedirs("glossary_images", exist_ok=True)
    
    # 고유한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_ext = uploaded_file.name.split('.')[-1]
    filename = f"term_{timestamp}.{file_ext}"
    filepath = os.path.join("glossary_images", filename)
    
    # 이미지 저장
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return filepath

# --- COMCBT.COM 문제 통합 ---
def integrate_comcbt_exam():
    st.title("🧠 CBT 모의고사")
    
    # COMCBT 전기산업기사 카테고리 ID
    ELECTRIC_CATEGORY_ID = "29"
    
    # COMCBT 문제 프레임 URL
    comcbt_url = f"https://www.comcbt.com/cbt/index2.php?hack_number={ELECTRIC_CATEGORY_ID}"
    
    st.markdown(f"""
    <div class="comcbt-iframe">
        <iframe src="{comcbt_url}" width="100%" height="800px" frameborder="0"></iframe>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #002200; padding: 15px; border-radius: 10px; margin-top: 20px;">
        <h4>📌 COMCBT.COM 사용 안내</h4>
        <ul>
            <li>위 프레임은 COMCBT.COM의 전기산업기사 문제를 직접 표시합니다</li>
            <li>문제 풀이, 채점 등 모든 기능을 이 창에서 바로 사용 가능합니다</li>
            <li>문제가 표시되지 않으면 <a href="{comcbt_url}" target="_blank">여기를 클릭</a>하여 새 창에서 열어주세요</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- 회로이론 기호 설명 표 ---
def circuit_symbols_table():
    st.title("⚡ 회로이론 필수 기호 사전")
    st.markdown("### 전기기사 시험을 위한 핵심 회로이론 기호 총정리")
    
    # 표 데이터
    table_html = """
    <table class="symbol-table">
        <thead>
            <tr>
                <th>No.</th>
                <th>기호</th>
                <th>한글 읽는법</th>
                <th>의미</th>
                <th>단위</th>
                <th>공식 및 설명</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td>V</td>
                <td>-</td>
                <td>전압</td>
                <td>[V]</td>
                <td>V = I·R, V = W/Q</td>
            </tr>
            <tr>
                <td>2</td>
                <td>I</td>
                <td>-</td>
                <td>전류</td>
                <td>[A]</td>
                <td>I = V/R, I = Q/t</td>
            </tr>
            <tr>
                <td>3</td>
                <td>R</td>
                <td>-</td>
                <td>저항</td>
                <td>[Ω]</td>
                <td>R = V/I, R = ρ·l/A</td>
            </tr>
            <tr>
                <td>4</td>
                <td>C</td>
                <td>-</td>
                <td>정전용량</td>
                <td>[F]</td>
                <td>C = Q/V, C합 직렬/병렬 공식</td>
            </tr>
            <tr>
                <td>5</td>
                <td>L</td>
                <td>-</td>
                <td>인덕턴스</td>
                <td>[H]</td>
                <td>V = L·(di/dt)</td>
            </tr>
            <tr>
                <td>6</td>
                <td>Z</td>
                <td>-</td>
                <td>임피던스</td>
                <td>[Ω]</td>
                <td>Z = √(R² + (XL − XC)²)</td>
            </tr>
            <tr>
                <td>7</td>
                <td>XL</td>
                <td>-</td>
                <td>인덕턴스 리액턴스</td>
                <td>[Ω]</td>
                <td>XL = 2πfL</td>
            </tr>
            <tr>
                <td>8</td>
                <td>XC</td>
                <td>-</td>
                <td>커패시턴스 리액턴스</td>
                <td>[Ω]</td>
                <td>XC = 1 / (2πfC)</td>
            </tr>
            <tr>
                <td>9</td>
                <td>S</td>
                <td>-</td>
                <td>피상전력</td>
                <td>[VA]</td>
                <td>S = VI</td>
            </tr>
            <tr>
                <td>10</td>
                <td>P</td>
                <td>-</td>
                <td>유효전력</td>
                <td>[W]</td>
                <td>P = VIcosθ, P = I²R, P = V²/R</td>
            </tr>
            <tr>
                <td>11</td>
                <td>Q</td>
                <td>-</td>
                <td>무효전력</td>
                <td>[Var]</td>
                <td>Q = VIsinθ</td>
            </tr>
            <tr>
                <td>12</td>
                <td>PF</td>
                <td>-</td>
                <td>역률</td>
                <td>-</td>
                <td>PF = cosθ</td>
            </tr>
            <tr>
                <td>13</td>
                <td>f</td>
                <td>-</td>
                <td>주파수</td>
                <td>[Hz]</td>
                <td>f = 1/T</td>
            </tr>
            <tr>
                <td>14</td>
                <td>T</td>
                <td>-</td>
                <td>주기</td>
                <td>[s]</td>
                <td>T = 1/f</td>
            </tr>
            <tr>
                <td>15</td>
                <td>ω</td>
                <td>오메가</td>
                <td>각속도</td>
                <td>[rad/s]</td>
                <td>ω = 2πf</td>
            </tr>
            <tr>
                <td>16</td>
                <td>t</td>
                <td>-</td>
                <td>시간</td>
                <td>[s]</td>
                <td>전류, 전하량 계산에 사용</td>
            </tr>
            <tr>
                <td>17</td>
                <td>Q</td>
                <td>-</td>
                <td>전하</td>
                <td>[C]</td>
                <td>Q = I·t</td>
            </tr>
            <tr>
                <td>18</td>
                <td>ρ</td>
                <td>로</td>
                <td>고유저항</td>
                <td>[Ω·m]</td>
                <td>R = ρ·l/A</td>
            </tr>
            <tr>
                <td>19</td>
                <td>l</td>
                <td>-</td>
                <td>도선 길이</td>
                <td>[m]</td>
                <td>길이 증가 시 저항 증가</td>
            </tr>
            <tr>
                <td>20</td>
                <td>A</td>
                <td>-</td>
                <td>단면적</td>
                <td>[m²]</td>
                <td>A 증가 시 저항 감소</td>
            </tr>
            <tr>
                <td>21</td>
                <td>j</td>
                <td>제이</td>
                <td>허수</td>
                <td>-</td>
                <td>j² = -1, 교류 해석에 사용</td>
            </tr>
            <tr>
                <td>22</td>
                <td>jθ</td>
                <td>제이 세타</td>
                <td>복소수 위상 표현</td>
                <td>-</td>
                <td>e^{jθ} = cosθ + jsinθ (오일러 공식)</td>
            </tr>
            <tr>
                <td>23</td>
                <td>τ</td>
                <td>타우</td>
                <td>시정수</td>
                <td>[s]</td>
                <td>RC, L/R</td>
            </tr>
            <tr>
                <td>24</td>
                <td>φ</td>
                <td>파이</td>
                <td>위상각</td>
                <td>[°], [rad]</td>
                <td>tanφ = (XL − XC)/R</td>
            </tr>
            <tr>
                <td>25</td>
                <td>π</td>
                <td>파이</td>
                <td>원주율</td>
                <td>-</td>
                <td>180° = π rad</td>
            </tr>
            <tr>
                <td>26</td>
                <td>rad</td>
                <td>라디안</td>
                <td>라디안</td>
                <td>-</td>
                <td>1° = π/180 rad</td>
            </tr>
            <tr>
                <td>27</td>
                <td>Δ / Y</td>
                <td>델타 / 와이</td>
                <td>결선 방식</td>
                <td>-</td>
                <td>3상 결선: Δ = 삼각형, Y = 별</td>
            </tr>
            <tr>
                <td>28</td>
                <td>E</td>
                <td>-</td>
                <td>기전력</td>
                <td>[V]</td>
                <td>E = V + Ir (내부 저항 포함)</td>
            </tr>
        </tbody>
    </table>
    """
    
    # 표 표시
    st.markdown(table_html, unsafe_allow_html=True)
    
    # 추가 설명
    st.markdown("""
    <div style="background-color: #002200; padding: 15px; border-radius: 10px; margin-top: 20px; border: 1px solid #00FF00;">
        <h4>📌 회로이론 기호 학습 가이드</h4>
        <ul>
            <li>기호와 단위를 매칭하는 연습을 통해 회로 해석 능력 향상</li>
            <li>공식 열은 해당 기호가 사용되는 대표적인 공식을 표시</li>
            <li>빨간색으로 표시된 읽는법은 주의가 필요한 발음</li>
            <li>표를 클릭하면 해당 행이 강조되어 가시성 향상</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #002200; border-radius: 10px; margin-top: 30px; border: 1px solid #00FF00;">
        <h4>⚡ 학습 팁</h4>
        <p>이 표를 매일 5분씩 복습하면 회로이론 문제 해결 속도가 크게 향상됩니다!</p>
        <p>특히 <strong style="color:#00FF99;">임피던스(Z), 리액턴스(X), 위상각(φ)</strong> 관련 기호는 반드시 숙지해야 합니다.</p>
    </div>
    """, unsafe_allow_html=True)

# --- 홈 화면 ---
def home():
    circuit_symbols_table()

# --- 사이드바 메뉴 ---
def sidebar_menu():
    st.sidebar.title("📚 학습 메뉴")
    menu = st.sidebar.radio(
        "메인 메뉴",
        ["🏠 회로이론 기호", "🧠 CBT 모의고사", "🎥 동영상 학습", "📚 학습 자료", "📖 용어집", "📊 학습 통계"],
        key="main_menu"
    )
    
    # 개발자 크레딧
    show_developer_credit()
    
    return menu

# --- 오른쪽 사이드바 (학습 자료 & 용어집) ---
def right_sidebar():
    st.markdown('<div class="right-sidebar">', unsafe_allow_html=True)
    st.markdown('<div class="right-sidebar-tabs">', unsafe_allow_html=True)
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📝 학습 자료", "📖 용어집"])
    
    with tab1:
        # 학습 자료 입력
        st.subheader("📝 학습 자료 입력")
        user_id = "miwooni"
        subjects = ["회로이론", "전기이론", "전기기기", "전력공학", "전기설비"]
        
        with st.form("material_form"):
            material_subject = st.selectbox("과목 선택", subjects, key="material_subject")
            material_title = st.text_input("제목", key="material_title")
            material_content = st.text_area("내용", height=200, key="material_content")
            submitted = st.form_submit_button("저장")
            
            if submitted:
                if material_title and material_content:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    db_query(
                        "study_materials.db",
                        "INSERT INTO study_materials (user_id, subject, title, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                        (user_id, material_subject, material_title, material_content, timestamp)
                    )
                    st.success("학습 자료가 저장되었습니다!")
                else:
                    st.warning("제목과 내용을 입력해주세요")
        
        # 저장된 학습 자료 검색
        st.subheader("🔍 학습 자료 검색")
        search_subject = st.selectbox("검색 과목", subjects, key="search_subject")
        search_keyword = st.text_input("검색어", key="material_search", placeholder="제목 또는 내용 검색")
        
        # 검색 실행
        materials = db_query(
            "study_materials.db",
            "SELECT id, subject, title, content, timestamp FROM study_materials WHERE user_id=? AND subject=? AND (title LIKE ? OR content LIKE ?) ORDER BY timestamp DESC LIMIT 5",
            (user_id, search_subject, f'%{search_keyword}%', f'%{search_keyword}%'),
            fetch=True
        )
        
        if materials:
            for material in materials:
                mat_id, subject, title, content, timestamp = material
                with st.expander(f"{subject} - {title} ({timestamp[:10]})", expanded=False):
                    st.write(content)
                    if st.button("삭제", key=f"delete_mat_{mat_id}"):
                        db_query("study_materials.db", "DELETE FROM study_materials WHERE id=?", (mat_id,))
                        st.experimental_rerun()
        else:
            st.info("검색 결과가 없습니다.")
    
    with tab2:
        # 용어 추가 폼
        st.subheader("📝 용어 추가")
        with st.form("term_form"):
            term = st.text_input("용어", key="term")
            definition = st.text_area("정의", height=150, key="definition")
            subject = st.selectbox(
                "과목", 
                ["공통", "전기이론", "전기기기", "전력공학", "회로이론", "전기설비"],
                key="term_subject"
            )
            
            # 이미지 업로드
            uploaded_image = st.file_uploader(
                "이미지 업로드 (선택사항)", 
                type=['jpg', 'jpeg', 'png'], 
                key="term_image"
            )
            
            submitted = st.form_submit_button("추가")
            
            if submitted:
                if term and definition:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    image_path = None
                    
                    if uploaded_image is not None:
                        image_path = save_uploaded_image(uploaded_image)
                    
                    try:
                        db_query(
                            "glossary.db",
                            "INSERT INTO glossary (term, definition, subject, timestamp, image_path) VALUES (?, ?, ?, ?, ?)",
                            (term, definition, subject, timestamp, image_path)
                        )
                        st.success("용어가 추가되었습니다!")
                    except sqlite3.IntegrityError:
                        st.error("이미 존재하는 용어입니다.")
                else:
                    st.warning("용어와 정의를 모두 입력해주세요")
        
        # 용어 검색
        st.subheader("🔍 용어 검색")
        search_term = st.text_input("검색어", key="search_term", placeholder="용어 또는 정의 검색")
        
        # 검색 실행
        if search_term:
            terms = db_query(
                "glossary.db",
                "SELECT id, term, definition, subject, image_path FROM glossary WHERE term LIKE ? OR definition LIKE ? ORDER BY term LIMIT 10",
                (f"%{search_term}%", f"%{search_term}%"),
                fetch=True
            )
        else:
            terms = db_query(
                "glossary.db",
                "SELECT id, term, definition, subject, image_path FROM glossary ORDER BY term LIMIT 10",
                fetch=True
            )
        
        if terms:
            for term_id, term, definition, subject, image_path in terms:
                with st.expander(f"{subject} - {term}", expanded=False):
                    st.write(definition)
                    if image_path and os.path.exists(image_path):
                        st.image(image_path, caption=f"{term} 이미지", use_container_width=True)
                    if st.button("삭제", key=f"delete_{term_id}"):
                        db_query("glossary.db", "DELETE FROM glossary WHERE id=?", (term_id,))
                        st.experimental_rerun()
        else:
            st.info("검색 결과가 없습니다.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 동영상 학습 화면 ---
def video_learning():
    st.title("🎥 동영상 학습")
    
    # 페이징 상태 관리
    if 'video_page' not in st.session_state:
        st.session_state.video_page = 1
    page_size = 5
    
    # 과목 선택
    subjects = ["회로이론", "전기이론", "전기기기", "전력공학", "전기설비"]
    selected_subject = st.selectbox("과목 선택", subjects, key="video_subject")
    
    # 정렬 기준 선택
    sort_options = ["검색 기준", "제목순", "인기순", "최신순"]
    sort_by = st.selectbox("정렬 기준", sort_options, key="video_sort")
    
    # 검색 기준이 선택된 경우 검색어 입력 필드 표시
    search_keyword = ""
    if sort_by == "검색 기준":
        search_keyword = st.text_input("검색어 입력", key="video_search", placeholder="동영상 제목을 입력하세요")
    
    # 동영상 추가 폼
    with st.expander("새 동영상 추가", expanded=False):
        with st.form("video_form"):
            video_url = st.text_input("유튜브 URL", key="video_url")
            video_title = st.text_input("동영상 제목", key="video_title")
            submitted = st.form_submit_button("추가")
            
            if submitted and video_url and video_title:
                video_id = None
                if "youtube.com" in video_url or "youtu.be" in video_url:
                    try:
                        yt = YouTube(video_url)
                        video_id = yt.video_id
                    except:
                        if "v=" in video_url:
                            video_id = video_url.split("v=")[1].split("&")[0]
                        elif "youtu.be/" in video_url:
                            video_id = video_url.split("youtu.be/")[1].split("?")[0]
                
                if video_id:
                    with sqlite3.connect("videos.db") as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO videos (video_id, subject, title, watch_count, last_watched, url)
                            VALUES (?, ?, ?, 1, ?, ?)
                            ON CONFLICT(video_id) DO UPDATE SET
                                title = excluded.title,
                                subject = excluded.subject
                        """, (video_id, selected_subject, video_title, 
                              datetime.now().strftime("%Y-%m-%d %H:%M:%S"), video_url))
                        conn.commit()
                    st.success("동영상이 추가되었습니다!")
                    st.experimental_rerun()
                else:
                    st.error("유효한 YouTube URL을 입력해주세요.")
    
    # 동영상 목록
    st.subheader(f"📺 {selected_subject} 동영상 목록")
    
    # 정렬 기준에 따른 쿼리
    if sort_by == "최신순":
        order_clause = "ORDER BY last_watched DESC"
    elif sort_by == "인기순":
        order_clause = "ORDER BY watch_count DESC"
    elif sort_by == "제목순":
        order_clause = "ORDER BY title ASC"
    else:  # 검색 기준
        order_clause = "ORDER BY title ASC"
    
    # 검색 조건 처리
    where_clause = "subject=?"
    params = (selected_subject,)
    
    if sort_by == "검색 기준" and search_keyword:
        where_clause = "subject=? AND title LIKE ?"
        params = (selected_subject, f'%{search_keyword}%')
    
    # 전체 동영상 수 조회
    total_videos = db_query(
        "videos.db",
        f"SELECT COUNT(*) FROM videos WHERE {where_clause}",
        params,
        fetch_one=True
    )[0]
    
    # 페이징 계산
    total_pages = max(1, (total_videos + page_size - 1) // page_size)
    offset = (st.session_state.video_page - 1) * page_size
    
    # 현재 페이지 동영상 조회
    videos = db_query(
        "videos.db",
        f"SELECT video_id, title, watch_count, url FROM videos WHERE {where_clause} {order_clause} LIMIT ? OFFSET ?",
        params + (page_size, offset),
        fetch=True
    )
    
    if not videos:
        st.info("등록된 동영상이 없습니다. 위에서 동영상을 추가하세요.")
        return
    
    # 검색 결과 강조 표시
    if sort_by == "검색 기준" and search_keyword:
        st.markdown(f"<div class='search-highlight'>검색 결과: '{search_keyword}' (총 {len(videos)}개)</div>", unsafe_allow_html=True)
    
    # 동영상 목록 표시
    for i, (video_id, title, count, url) in enumerate(videos):
        # 검색어 강조 표시
        display_title = title
        if sort_by == "검색 기준" and search_keyword:
            display_title = title.replace(search_keyword, f"<mark style='background-color:#004400;'>{search_keyword}</mark>")
        
        with st.expander(f"{display_title} (시청 {count}회)", expanded=False):
            st.markdown(f"""
            <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%;">
                <iframe src="https://www.youtube.com/embed/{video_id}?rel=0" 
                        style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;" 
                        allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("시청 기록 추가", key=f"watch_{video_id}"):
                    with sqlite3.connect("videos.db") as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE videos 
                            SET watch_count = watch_count + 1, 
                                last_watched = ?
                            WHERE video_id = ?
                        """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), video_id))
                        conn.commit()
                    st.experimental_rerun()
            with col2:
                if st.button("삭제", key=f"delete_{video_id}"):
                    db_query(
                        "videos.db", 
                        "DELETE FROM videos WHERE video_id=?", 
                        (video_id,)
                    )
                    st.experimental_rerun()
            
            st.markdown(f"[원본 보기]({url})", unsafe_allow_html=True)
    
    # 페이징 컨트롤
    if total_pages > 1:
        st.divider()
        col_prev, col_page, col_next = st.columns([1, 2, 1])
        
        with col_prev:
            if st.button("◀ 이전", disabled=st.session_state.video_page <= 1):
                st.session_state.video_page -= 1
                st.experimental_rerun()
        
        with col_page:
            st.markdown(f"**페이지 {st.session_state.video_page}/{total_pages}**")
        
        with col_next:
            if st.button("다음 ▶", disabled=st.session_state.video_page >= total_pages):
                st.session_state.video_page += 1
                st.experimental_rerun()

# --- 학습 통계 화면 ---
def learning_stats():
    st.title("📊 학습 통계")
    
    # 동영상 시청 통계
    st.subheader("동영상 시청 통계")
    try:
        video_df = pd.read_sql("""
            SELECT subject, title, watch_count, last_watched
            FROM videos
            ORDER BY last_watched DESC
        """, sqlite3.connect("videos.db"))
        
        if not video_df.empty:
            # 과목별 시청 횟수
            st.write("### 과목별 시청 횟수")
            subject_views = video_df.groupby('subject')['watch_count'].sum().reset_index()
            st.bar_chart(subject_views.set_index('subject'))
            
            # 인기 동영상
            st.write("### 인기 동영상 TOP 5")
            top_videos = video_df.sort_values('watch_count', ascending=False).head(5)
            st.dataframe(top_videos[['title', 'subject', 'watch_count']], hide_index=True)
        else:
            st.info("동영상 시청 기록이 없습니다.")
    except:
        st.info("동영상 시청 기록이 없습니다.")
    
    # 학습 자료 통계
    st.subheader("학습 자료 통계")
    try:
        material_df = pd.read_sql("""
            SELECT subject, COUNT(*) as count
            FROM study_materials
            GROUP BY subject
        """, sqlite3.connect("study_materials.db"))
        
        if not material_df.empty:
            # 과목별 학습 자료 수
            st.write("### 과목별 학습 자료 수")
            st.bar_chart(material_df.set_index('subject'))
        else:
            st.info("학습 자료 기록이 없습니다.")
    except:
        st.info("학습 자료 기록이 없습니다.")

# --- 학습 자료 메인 페이지 ---
def study_materials():
    st.title("📚 학습 자료 관리")
    
    # 검색 필터
    col1, col2 = st.columns([1, 3])
    with col1:
        subjects = ["회로이론", "전기이론", "전기기기", "전력공학", "전기설비"]
        subject_filter = st.selectbox("과목 선택", ["전체"] + subjects)
    with col2:
        search_query = st.text_input("검색어", placeholder="제목 또는 내용 검색")
    
    # 검색 쿼리 생성
    query = "SELECT * FROM study_materials WHERE 1=1"
    params = []
    
    if subject_filter != "전체":
        query += " AND subject=?"
        params.append(subject_filter)
    
    if search_query:
        query += " AND (title LIKE ? OR content LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
    
    query += " ORDER BY timestamp DESC"
    
    # 자료 조회
    materials = db_query("study_materials.db", query, params, fetch=True)
    
    if materials:
        st.subheader(f"총 {len(materials)}개의 학습 자료")
        
        for mat in materials:
            id, user_id, subject, title, content, timestamp = mat
            with st.expander(f"{subject} - {title} ({timestamp[:10]})", expanded=False):
                st.write(content)
                if st.button("삭제", key=f"delete_{id}"):
                    db_query("study_materials.db", "DELETE FROM study_materials WHERE id=?", (id,))
                    st.experimental_rerun()
    else:
        st.info("검색 결과가 없습니다. 오른쪽 사이드바에서 새 학습 자료를 추가하세요.")

# --- 용어집 메인 페이지 ---
def glossary():
    st.title("📖 용어집 관리")
    
    # 검색 필터
    col1, col2 = st.columns([1, 3])
    with col1:
        subjects = ["공통", "전기이론", "전기기기", "전력공학", "회로이론", "전기설비"]
        subject_filter = st.selectbox("과목 선택", ["전체"] + subjects)
    with col2:
        search_query = st.text_input("검색어", placeholder="용어 또는 정의 검색")
    
    # 검색 쿼리 생성
    query = "SELECT * FROM glossary WHERE 1=1"
    params = []
    
    if subject_filter != "전체":
        query += " AND subject=?"
        params.append(subject_filter)
    
    if search_query:
        query += " AND (term LIKE ? OR definition LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
    
    query += " ORDER BY term ASC"
    
    # 용어 조회
    terms = db_query("glossary.db", query, params, fetch=True)
    
    if terms:
        st.subheader(f"총 {len(terms)}개의 용어")
        
        for term in terms:
            id, term_text, definition, subject, timestamp, image_path = term
            with st.expander(f"{subject} - {term_text}", expanded=False):
                st.write(definition)
                if image_path and os.path.exists(image_path):
                    st.image(image_path, caption=f"{term_text} 이미지", use_container_width=True)
                if st.button("삭제", key=f"delete_{id}"):
                    db_query("glossary.db", "DELETE FROM glossary WHERE id=?", (id,))
                    st.experimental_rerun()
    else:
        st.info("검색 결과가 없습니다. 오른쪽 사이드바에서 새 용어를 추가하세요.")

# --- 메인 앱 ---
def main():
    # 초기화
    if "init" not in st.session_state:
        init_databases()
        st.session_state.init = True
    
    # 메뉴 라우팅
    menu_functions = {
        "🏠 회로이론 기호": home,
        "🧠 CBT 모의고사": integrate_comcbt_exam,
        "🎥 동영상 학습": video_learning,
        "📚 학습 자료": study_materials,
        "📖 용어집": glossary,
        "📊 학습 통계": learning_stats
    }
    
    menu = sidebar_menu()
    
    # 3분할 레이아웃 컨테이너
    st.markdown('<div class="main-layout">', unsafe_allow_html=True)
    
    # 좌측 사이드바 (고정)
    with st.container():
        st.markdown('<div class="left-sidebar"></div>', unsafe_allow_html=True)
    
    # 중앙 메인 콘텐츠
    with st.container():
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        
        if menu in menu_functions:
            menu_functions[menu]()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 우측 사이드바 (조건부 표시)
    if menu in ["🎥 동영상 학습", "📚 학습 자료", "📖 용어집"]:
        with st.container():
            right_sidebar()
    
    st.markdown('</div>', unsafe_allow_html=True)  # main-layout 종료
    
    # 푸터
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #002200; border-radius: 10px; margin-top: 30px;">
        <h3 style="color: #00FF00;">⚡ Made by Sung Jin ⚡</h3>
        <p style="color: #00FF00;">성진아 너두 ? </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
