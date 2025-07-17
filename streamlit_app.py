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

# --- 페이지 설정 (갤럭시 탭 최적화) ---
st.set_page_config(
    page_title="성진아 도전!!!",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 스타일 적용 (갤럭시 탭 최적화) ---
st.markdown("""
<style>
    * {
        background-color: #000000 !important;
        color: #00FF00 !important;
        font-size: 1.1rem !important;  /* 기본 폰트 크기 확대 */
    }
    .stApp, .stSidebar {
        background-color: #000000 !important;
    }
    .stButton>button {
        background-color: #006600;
        color: #00FF00;
        border: 1px solid #00FF00;
        font-size: 1.2rem !important;  /* 버튼 폰트 크기 확대 */
        padding: 12px 24px !important; /* 버튼 패딩 확대 (터치 최적화) */
        min-height: 50px !important;   /* 버튼 최소 높이 확대 */
    }
    .stButton>button:hover {
        background-color: #004400;
        color: #00FF00;
        border: 1px solid #00FF00;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #001100;
        color: #00FF00;
        font-size: 1.2rem !important;  /* 입력 필드 폰트 크기 확대 */
        min-height: 50px !important;   /* 입력 필드 높이 확대 */
    }
    .stSelectbox>div>div>select {
        background-color: #001100;
        color: #00FF00;
        font-size: 1.2rem !important;  /* 셀렉트박스 폰트 크기 확대 */
        min-height: 50px !important;   /* 셀렉트박스 높이 확대 */
    }
    .stRadio>div {
        background-color: #001100;
        color: #00FF00;
        font-size: 1.2rem !important;  /* 라디오 버튼 폰트 크기 확대 */
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
        font-size: 1.2rem !important;  /* 확장기 폰트 크기 확대 */
    }
    .stMarkdown {
        color: #00FF00;
        font-size: 1.2rem !important;  /* 마크다운 폰트 크기 확대 */
    }
    .stAlert {
        background-color: #001100;
        color: #00FF00;
        font-size: 1.2rem !important;  /* 알림 폰트 크기 확대 */
    }
    .comcbt-iframe {
        width: 100%;
        height: 70vh;  /* 화면 높이에 비례하는 높이로 변경 */
        border: 2px solid #00FF00;
        border-radius: 10px;
        overflow: hidden;
    }
    /* 사이드바 메뉴 글자 크기 확대 (갤럭시 탭 최적화) */
    .stSidebar .stRadio > label > div {
        font-size: 1.8rem !important;  /* 더 큰 폰트 크기 */
        padding: 20px 0 !important;    /* 더 큰 패딩 */
        margin: 10px 0 !important;     /* 마진 추가 */
    }
    /* 이미지 표시 스타일 */
    .term-image {
        max-width: 100%;
        border: 1px solid #00FF00;
        border-radius: 5px;
        margin-top: 10px;
    }
    /* 페이징 컨트롤 스타일 (터치 최적화) */
    .pagination-control {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .pagination-control button {
        margin: 0 15px;  /* 버튼 사이 간격 확대 */
        padding: 12px 24px !important; /* 버튼 크기 확대 */
        font-size: 1.3rem !important;  /* 버튼 폰트 크기 확대 */
        min-height: 60px !important;   /* 버튼 최소 높이 확대 */
    }
    /* 입력 커서(caret) 형광색 설정 */
    input, textarea, [contenteditable] {
        caret-color: #00FF00 !important;
        color: #00FF00 !important;
        background-color: #000000 !important;
        border: 1px solid #00FF00 !important;
    }
    /* 탭 레이아웃 최적화 */
    .stTabs [role="tab"] {
        font-size: 1.4rem !important;  /* 탭 폰트 크기 확대 */
        padding: 15px 25px !important; /* 탭 패딩 확대 */
    }
    /* 갤럭시 탭 화면 크기에 따른 반응형 조정 */
    @media (max-width: 1024px) {
        /* 갤럭시 탭 가로 모드 */
        .stButton>button {
            padding: 10px 20px !important;
            min-height: 45px !important;
        }
        .stSidebar .stRadio > label > div {
            font-size: 1.6rem !important;
        }
    }
    @media (max-width: 768px) {
        /* 갤럭시 탭 세로 모드 */
        .stButton>button {
            padding: 8px 16px !important;
            min-height: 40px !important;
            font-size: 1.1rem !important;
        }
        .stSidebar .stRadio > label > div {
            font-size: 1.4rem !important;
            padding: 15px 0 !important;
        }
        .stTabs [role="tab"] {
            font-size: 1.2rem !important;
            padding: 10px 15px !important;
        }
        .comcbt-iframe {
            height: 50vh; /* 세로 모드에서 높이 축소 */
        }
    }
</style>
""", unsafe_allow_html=True)

# --- 개발자 크레딧 ---
def show_developer_credit():
    st.sidebar.divider()
    # 사이드바에 이미지 추가 (반응형 크기 조정)
    try:
        st.sidebar.image("화면 캡처 2025-07-15 094924.jpg", use_column_width=True)
        st.sidebar.markdown("<center style='font-size:1.4rem;'>나는 할 수 밖에 없다.!!!<br>⚡ Made by Sung Jin ⚡</center>", unsafe_allow_html=True)
    except:
        st.sidebar.markdown("""
        <div style="text-align: center; padding: 10px; background-color: #002200; border-radius: 10px; margin-top: 20px;">
            <p style="color: #00FF00; margin-bottom: 5px; font-size:1.4rem;">나는 할 수 밖에 없다.!!!</p>
            <h4 style="color: #00FF00; margin-top: 0; font-size:1.6rem;">⚡ Made by Sung Jin ⚡</h4>
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
    
    # 학습 자료 저장 테이블
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
    
    # 용어집 저장 테이블
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
        <iframe src="{comcbt_url}" width="100%" height="100%" frameborder="0"></iframe>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #002200; padding: 15px; border-radius: 10px; margin-top: 20px;">
        <h4 style="font-size:1.4rem;">📌 COMCBT.COM 사용 안내</h4>
        <ul style="font-size:1.2rem;">
            <li>위 프레임은 COMCBT.COM의 전기산업기사 문제를 직접 표시합니다</li>
            <li>문제 풀이, 채점 등 모든 기능을 이 창에서 바로 사용 가능합니다</li>
            <li>문제가 표시되지 않으면 <a href="{comcbt_url}" target="_blank">여기를 클릭</a>하여 새 창에서 열어주세요</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- 홈 화면 ---
def home():
    st.title("불가능은 있다!! 하지만 난 불가능에 도전한다!!")
    st.markdown("""
    <div style="font-size:1.3rem;">
    ### 🚀 간절하지 않으면 생각도 말라 !!!. 🚀
    - **🧠 CBT 모의고사**: COMCBT 통합 모의고사
    - **🎥 동영상 학습**: 필요한 강의만 집중해서 시청
    - **📚 학습 자료**: 과목별 학습 메모 관리
    - **📖 용어집**: 전기기사 필수 용어 사전
    - **📊 학습 통계**: 나의 학습 패턴 분석
    </div>
    """, unsafe_allow_html=True)
    
    # 홈 화면 이미지
    try:
        st.image("화면 캡처 2025-07-15 094924.jpg", 
                 use_column_width=True, caption="모카 멋진척 하기!!!")
    except:
        pass

# --- 사이드바 메뉴 (갤럭시 탭 최적화) ---
def sidebar_menu():
    st.sidebar.title("📚 학습 메뉴")
    menu = st.sidebar.radio(
        "메인 메뉴",
        ["🏠 홈", "🧠 CBT 모의고사", "🎥 동영상 학습", "📚 학습 자료", "📖 용어집", "📊 학습 통계"],
        key="main_menu"
    )
    
    # 개발자 크레딧
    show_developer_credit()
    
    return menu

# --- 동영상 학습 화면 (갤럭시 탭 최적화) ---
def video_learning():
    st.title("🎥 동영상 학습")
    
    # 페이징 상태 관리
    if 'video_page' not in st.session_state:
        st.session_state.video_page = 1
    page_size = 3 if st.session_state.get('is_mobile', False) else 5  # 모바일에서는 3개, PC에서는 5개
    
    # 과목 선택
    subjects = ["회로이론", "전기이론", "전기기기", "전력공학", "전기설비"]
    selected_subject = st.selectbox("과목 선택", subjects, key="video_subject")
    
    # 정렬 기준 선택
    sort_options = ["제목순", "인기순", "최신순"]
    sort_by = st.selectbox("정렬 기준", sort_options, key="video_sort")
    
    # 동영상 추가 폼
    with st.expander("새 동영상 추가", expanded=False):
        with st.form("video_form", clear_on_submit=True):
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
                    st.rerun()
                else:
                    st.error("유효한 YouTube URL을 입력해주세요.")
    
    # 학습 자료를 위한 레이아웃 (용어집 제거)
    col_video, col_memo = st.columns([3, 2])
    
    with col_video:
        # 동영상 목록
        st.subheader(f"{selected_subject} 동영상 목록")
        
        # 정렬 기준에 따른 쿼리
        if sort_by == "최신순":
            order_clause = "ORDER BY last_watched DESC"
        elif sort_by == "인기순":
            order_clause = "ORDER BY watch_count DESC"
        else:  # 제목순
            order_clause = "ORDER BY title ASC"
        
        # 전체 동영상 수 조회
        total_videos = db_query(
            "videos.db",
            f"SELECT COUNT(*) FROM videos WHERE subject=?",
            (selected_subject,),
            fetch_one=True
        )[0]
        
        # 페이징 계산
        total_pages = max(1, (total_videos + page_size - 1) // page_size)
        offset = (st.session_state.video_page - 1) * page_size
        
        # 현재 페이지 동영상 조회
        videos = db_query(
            "videos.db",
            f"SELECT video_id, title, watch_count, url FROM videos WHERE subject=? {order_clause} LIMIT ? OFFSET ?",
            (selected_subject, page_size, offset),
            fetch=True
        )
        
        if not videos:
            st.info("등록된 동영상이 없습니다. 위에서 동영상을 추가하세요.")
            return
        
        # 동영상 목록 표시 (항상 접힌 상태로)
        for i, (video_id, title, count, url) in enumerate(videos):
            with st.expander(f"{title} (시청 {count}회)", expanded=False):
                # 동영상 플레이어 크기 조정 (갤럭시 탭 최적화)
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
                        st.rerun()
                with col2:
                    if st.button("삭제", key=f"delete_{video_id}"):
                        db_query(
                            "videos.db", 
                            "DELETE FROM videos WHERE video_id=?", 
                            (video_id,)
                        )
                        st.rerun()
                
                st.markdown(f"[원본 보기]({url})", unsafe_allow_html=True)
        
        # 페이징 컨트롤 (터치 최적화)
        if total_pages > 1:
            st.divider()
            col_prev, col_page, col_next = st.columns([1, 2, 1])
            
            with col_prev:
                if st.button("◀ 이전", disabled=st.session_state.video_page <= 1):
                    st.session_state.video_page -= 1
                    st.rerun()
            
            with col_page:
                st.markdown(f"<div style='text-align:center; font-size:1.3rem;'>페이지 {st.session_state.video_page}/{total_pages}</div>", unsafe_allow_html=True)
            
            with col_next:
                if st.button("다음 ▶", disabled=st.session_state.video_page >= total_pages):
                    st.session_state.video_page += 1
                    st.rerun()
    
    with col_memo:
        # 학습 자료 입력
        st.subheader("학습 자료")
        
        # 사용자 ID 고정값 사용
        user_id = "miwooni"
        
        # 과목 선택
        material_subject = st.selectbox(
            "과목 선택", 
            subjects,
            key="material_subject"
        )
        
        # 제목 입력
        material_title = st.text_input("제목", key="material_title")
        
        # 내용 입력 (갤럭시 탭 최적화)
        material_content = st.text_area("내용", height=300, key="material_content")
        
        # 저장 버튼
        if st.button("학습 자료 저장", key="save_material"):
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
        
        # 저장된 학습 자료 보기
        st.subheader("저장된 학습 자료")
        materials = db_query(
            "study_materials.db",
            "SELECT id, subject, title, content, timestamp FROM study_materials WHERE user_id=? ORDER BY timestamp DESC",
            (user_id,),
            fetch=True
        )
        
        if materials:
            for material in materials:
                mat_id, subject, title, content, timestamp = material
                with st.expander(f"{subject} - {title} ({timestamp[:10]})", expanded=False):
                    st.write(content)
                    if st.button("삭제", key=f"delete_mat_{mat_id}"):
                        db_query("study_materials.db", "DELETE FROM study_materials WHERE id=?", (mat_id,))
                        st.rerun()
        else:
            st.info("저장된 학습 자료가 없습니다.")

# --- 학습 자료 화면 (갤럭시 탭 최적화) ---
def study_materials():
    st.title("📚 학습 자료")
    
    # 화면을 두 개의 열로 분할 (갤럭시 탭 최적화)
    col_list, col_glossary = st.columns([1, 1], gap="large")
    
    with col_list:
        # 학습 자료 목록
        st.subheader("📋 학습 자료 목록")
        
        # 사용자 ID 고정값 사용
        user_id = "miwooni"
        
        # 페이징 상태 관리
        if 'material_page' not in st.session_state:
            st.session_state.material_page = 1
        page_size = 3 if st.session_state.get('is_mobile', False) else 5  # 모바일에서는 3개, PC에서는 5개
        
        # 과목 선택
        subjects = ["회로이론", "전기이론", "전기기기", "전력공학", "전기설비"]
        selected_subject = st.selectbox("과목 선택", subjects, key="list_subject")
        
        # 전체 학습 자료 수 조회
        total_materials = db_query(
            "study_materials.db",
            "SELECT COUNT(*) FROM study_materials WHERE user_id=? AND subject=?",
            (user_id, selected_subject),
            fetch_one=True
        )[0]
        
        # 페이징 계산
        total_pages = max(1, (total_materials + page_size - 1) // page_size)
        offset = (st.session_state.material_page - 1) * page_size
        
        # 현재 페이지 학습 자료 조회
        materials = db_query(
            "study_materials.db",
            "SELECT id, title, content, timestamp FROM study_materials WHERE user_id=? AND subject=? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (user_id, selected_subject, page_size, offset),
            fetch=True
        )
        
        if materials:
            for material in materials:
                mat_id, title, content, timestamp = material
                with st.expander(f"{title} ({timestamp[:10]})", expanded=False):
                    st.write(content)
                    if st.button("삭제", key=f"delete_list_{mat_id}"):
                        db_query("study_materials.db", "DELETE FROM study_materials WHERE id=?", (mat_id,))
                        st.rerun()
        else:
            st.info("해당 과목의 학습 자료가 없습니다.")
        
        # 페이징 컨트롤 (터치 최적화)
        if total_pages > 1:
            st.divider()
            col_prev, col_page, col_next = st.columns([1, 2, 1])
            
            with col_prev:
                if st.button("◀ 이전", key="prev_mat", disabled=st.session_state.material_page <= 1):
                    st.session_state.material_page -= 1
                    st.rerun()
            
            with col_page:
                st.markdown(f"<div style='text-align:center; font-size:1.3rem;'>페이지 {st.session_state.material_page}/{total_pages}</div>", unsafe_allow_html=True)
            
            with col_next:
                if st.button("다음 ▶", key="next_mat", disabled=st.session_state.material_page >= total_pages):
                    st.session_state.material_page += 1
                    st.rerun()
    
    with col_glossary:
        # 용어집
        st.subheader("📖 용어집")
        
        # 용어 추가 폼 (이미지 업로드 추가)
        with st.expander("새 용어 추가", expanded=False):
            with st.form("term_form", clear_on_submit=True):
                term = st.text_input("용어", key="term")
                definition = st.text_area("정의", height=150, key="definition")
                subject = st.selectbox(
                    "과목", 
                    ["공통", "전기이론", "전기기기", "전력공학", "회로이론", "전기설비"],
                    key="term_subject"
                )
                
                # 이미지 업로드 추가
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
                        
                        # 이미지 업로드 처리
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
        st.subheader("용어 검색")
        search_term = st.text_input("용어 검색", key="search_term")
        
        # 검색 실행
        if search_term:
            terms = db_query(
                "glossary.db",
                "SELECT id, term, definition, subject, image_path FROM glossary WHERE term LIKE ? OR definition LIKE ?",
                (f"%{search_term}%", f"%{search_term}%"),
                fetch=True
            )
        else:
            terms = db_query(
                "glossary.db",
                "SELECT id, term, definition, subject, image_path FROM glossary ORDER BY term",
                fetch=True
            )
        
        if terms:
            # 과목별 탭 생성 (용어 개수 포함)
            subjects = sorted(set([term[3] for term in terms]))
            subject_counts = {}
            for term in terms:
                subj = term[3]
                subject_counts[subj] = subject_counts.get(subj, 0) + 1
            
            # 탭 생성 (과목명 + 용어 개수)
            tab_labels = [f"📚 {sub} ({subject_counts[sub]})" for sub in subjects]
            tabs = st.tabs(tab_labels)
            
            for i, subject in enumerate(subjects):
                with tabs[i]:
                    subject_terms = [t for t in terms if t[3] == subject]
                    for term_id, term, definition, _, image_path in subject_terms:
                        with st.expander(f"**{term}**", expanded=False):
                            st.write(definition)
                            
                            # 이미지 표시
                            if image_path and os.path.exists(image_path):
                                st.image(image_path, caption=f"{term} 이미지", use_column_width=True)
                            
                            if st.button("삭제", key=f"delete_{term_id}"):
                                db_query("glossary.db", "DELETE FROM glossary WHERE id=?", (term_id,))
                                st.rerun()
        else:
            st.info("용어가 없습니다. 위에서 새로운 용어를 추가해주세요.")

# --- 용어집 화면 (갤럭시 탭 최적화) ---
def glossary():
    st.title("📖 용어집")
    
    # 용어 추가 폼 (이미지 업로드 추가)
    with st.expander("새 용어 추가", expanded=False):
        with st.form("term_form", clear_on_submit=True):
            term = st.text_input("용어", key="term")
            definition = st.text_area("정의", height=150, key="definition")
            subject = st.selectbox(
                "과목", 
                ["공통", "전기이론", "전기기기", "전력공학", "회로이론", "전기설비"],
                key="term_subject"
            )
            
            # 이미지 업로드 추가
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
                    
                    # 이미지 업로드 처리
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
    
    st.divider()
    
    # 용어 검색
    st.subheader("용어 검색")
    search_term = st.text_input("용어 검색", key="search_term")
    
    # 검색 실행
    if search_term:
        terms = db_query(
            "glossary.db",
            "SELECT id, term, definition, subject, image_path FROM glossary WHERE term LIKE ? OR definition LIKE ?",
            (f"%{search_term}%", f"%{search_term}%"),
            fetch=True
        )
    else:
        terms = db_query(
            "glossary.db",
            "SELECT id, term, definition, subject, image_path FROM glossary ORDER BY term",
            fetch=True
        )
    
    if terms:
        # 과목별 탭 생성 (용어 개수 포함)
        subjects = sorted(set([term[3] for term in terms]))
        subject_counts = {}
        for term in terms:
            subj = term[3]
            subject_counts[subj] = subject_counts.get(subj, 0) + 1
        
        # 탭 생성 (과목명 + 용어 개수)
        tab_labels = [f"📚 {sub} ({subject_counts[sub]})" for sub in subjects]
        tabs = st.tabs(tab_labels)
        
        for i, subject in enumerate(subjects):
            with tabs[i]:
                subject_terms = [t for t in terms if t[3] == subject]
                for term_id, term, definition, _, image_path in subject_terms:
                    with st.expander(f"**{term}**", expanded=False):
                        st.write(definition)
                        
                        # 이미지 표시
                        if image_path and os.path.exists(image_path):
                            st.image(image_path, caption=f"{term} 이미지", use_column_width=True)
                        
                        if st.button("삭제", key=f"delete_{term_id}"):
                            db_query("glossary.db", "DELETE FROM glossary WHERE id=?", (term_id,))
                            st.rerun()
    else:
        st.info("용어가 없습니다. 위에서 새로운 용어를 추가해주세요.")

# --- 학습 통계 화면 (갤럭시 탭 최적화) ---
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
            
            # 차트 크기 조정
            fig, ax = plt.subplots(figsize=(10, 6) if not st.session_state.get('is_mobile', False) else plt.subplots(figsize=(6, 4)))
            ax.bar(subject_views['subject'], subject_views['watch_count'], color='#00FF00')
            ax.set_title('과목별 시청 횟수', fontsize=14)
            ax.set_ylabel('시청 횟수', fontsize=12)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
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
            
            # 차트 크기 조정
            fig, ax = plt.subplots(figsize=(10, 6) if not st.session_state.get('is_mobile', False) else plt.subplots(figsize=(6, 4))
            ax.bar(material_df['subject'], material_df['count'], color='#00FF00')
            ax.set_title('과목별 학습 자료 수', fontsize=14)
            ax.set_ylabel('자료 수', fontsize=12)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("학습 자료 기록이 없습니다.")
    except:
        st.info("학습 자료 기록이 없습니다.")

# --- 메인 앱 (갤럭시 탭 감지) ---
def main():
    # 화면 크기 감지 (갤럭시 탭 여부)
    if "is_mobile" not in st.session_state:
        try:
            # Streamlit의 내장 기능으로 모바일 감지
            from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
            ctx = get_script_run_ctx()
            if ctx and hasattr(ctx, 'request') and hasattr(ctx.request, 'headers'):
                user_agent = ctx.request.headers.get("User-Agent", "").lower()
                st.session_state.is_mobile = any(m in user_agent for m in ["mobile", "android", "iphone"])
            else:
                st.session_state.is_mobile = False
        except:
            st.session_state.is_mobile = False
    
    # 초기화
    if "init" not in st.session_state:
        init_databases()
        st.session_state.init = True
    
    # 메뉴 라우팅
    menu_functions = {
        "🏠 홈": home,
        "🧠 CBT 모의고사": integrate_comcbt_exam,
        "🎥 동영상 학습": video_learning,
        "📚 학습 자료": study_materials,
        "📖 용어집": glossary,
        "📊 학습 통계": learning_stats
    }
    
    menu = sidebar_menu()
    menu_functions[menu]()
    
    # 푸터
    st.divider()
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #002200; border-radius: 10px; margin-top: 30px;">
        <h3 style="color: #00FF00; font-size:1.6rem;">⚡ Made by Sung Jin ⚡</h3>
        <p style="color: #00FF00; font-size:1.3rem;">성진아 너두 ? </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
