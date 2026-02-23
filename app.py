"""
Emotion Diary App for middle school EFL students.

Quick start:
    pip install streamlit pandas google-genai
    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import os
import re
from typing import Dict

import pandas as pd
import streamlit as st
from google import genai

CSV_PATH = "diary_logs.csv"
MODEL_NAME = "gemini-3-flash-preview"


def init_gemini_client(api_key: str):
    """Initialize Gemini client with a user-provided API key."""
    return genai.Client(api_key=api_key)


def parse_feedback_response(text: str) -> Dict[str, str]:
    """Parse model output into structured sections with safe fallbacks."""
    patterns = {
        "corrected_diary": r"Corrected\s*Diary\s*:\s*(.*?)(?=Feedback\s*:|Korean\s*Summary\s*:|$)",
        "feedback_en": r"Feedback\s*:\s*(.*?)(?=Korean\s*Summary\s*:|$)",
        "summary_ko": r"Korean\s*Summary\s*:\s*(.*)$",
    }

    parsed: Dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        parsed[key] = match.group(1).strip() if match else ""

    # 파싱 실패 시 전체 텍스트라도 보여주기
    if not parsed["corrected_diary"] and not parsed["feedback_en"] and not parsed["summary_ko"]:
        parsed["corrected_diary"] = text.strip()

    return parsed


def generate_feedback(client, diary_text: str) -> Dict[str, str]:
    """Call Gemini and return corrected diary, simple feedback, and Korean summary."""
    prompt = f'''
You are an English teacher for middle school EFL students.
The student wrote an English emotion diary.

Please respond with these exact section headers:
Corrected Diary:
Feedback:
Korean Summary:

Rules:
1) Rewrite the diary in correct and natural English, but do not change the meaning or level too much.
2) Give friendly feedback in simple English (2-3 sentences, CEFR A2-B1 level).
   - Show empathy for the student's feelings.
   - Praise 1-2 good things about the writing.
   - Suggest one small thing to improve next time.
3) Write a one-sentence summary in Korean for the teacher.

Student diary: """{diary_text}"""
'''

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw_text = (response.text or "").strip()

    parsed = parse_feedback_response(raw_text)
    parsed["raw_response"] = raw_text
    return parsed


def save_diary_to_csv(
    timestamp: str,
    selected_date: str,
    student_id: str,
    mood: str,
    original_diary: str,
    corrected_diary: str,
    feedback_en: str,
    summary_ko: str,
    csv_path: str = CSV_PATH,
) -> None:
    """Append one diary record to CSV; create file with headers if missing."""
    row_df = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "date": selected_date,
                "student_id": student_id,
                "mood": mood,
                "original_diary": original_diary,
                "corrected_diary": corrected_diary,
                "feedback_en": feedback_en,
                "summary_ko": summary_ko,
            }
        ]
    )

    file_exists = os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", index=False, header=not file_exists, encoding="utf-8-sig")


def load_diary_csv(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load diary CSV or return an empty DataFrame with expected columns."""
    expected_cols = [
        "timestamp",
        "date",
        "student_id",
        "mood",
        "original_diary",
        "corrected_diary",
        "feedback_en",
        "summary_ko",
    ]

    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=expected_cols)

    df = pd.read_csv(csv_path)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    return df[expected_cols]


def main() -> None:
    """Render Streamlit app UI and handle interactions."""
    st.set_page_config(page_title="EFL Emotion Diary", page_icon="📝", layout="wide")

    st.title("📝 Daily Emotion Diary (EFL)")
    st.caption("학생들이 매일 영어 감정일기를 쓰고, AI 피드백을 받는 앱")

    # 1) API 키 입력
    st.sidebar.header("🔐 Gemini API Settings")
    api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")
    if not api_key:
        st.sidebar.info("Please enter your Gemini API key to use the app. / API 키를 입력해주세요.")

    # 탭 구성: 작성 / 기록 조회
    diary_tab, history_tab = st.tabs(["✍️ Write Diary", "📚 History"])

    with diary_tab:
        st.subheader("Today's Emotion Diary")

        # 2) 메인 입력 UI
        selected_date = st.date_input("Date", value=dt.date.today())
        student_input = st.text_input("Name or Student ID", placeholder="e.g., Mina-03")
        student_id = student_input.strip() or "Anonymous"

        mood = st.selectbox(
            "Today's Mood",
            ["😊 Happy", "😢 Sad", "😡 Angry", "😴 Tired", "😌 Calm", "🤩 Excited", "😐 So-so"],
        )

        diary_text = st.text_area(
            "Write about your day and how you felt in English.",
            height=180,
            placeholder="Try writing 3-5 sentences in English...",
        )

        if st.button("Save & Get Feedback", type="primary", use_container_width=True):
            # 버튼 클릭 시점에만 호출/저장
            if not api_key:
                st.error("Please enter your Gemini API key first. / 먼저 Gemini API 키를 입력해주세요.")
                st.stop()

            if not diary_text.strip():
                st.error("Please write your diary before submitting. / 일기를 먼저 작성해주세요.")
                st.stop()

            try:
                with st.spinner("Generating feedback... 잠시만 기다려주세요."):
                    client = init_gemini_client(api_key)
                    result = generate_feedback(client, diary_text.strip())

                    corrected_diary = result.get("corrected_diary", "")
                    feedback_en = result.get("feedback_en", "")
                    summary_ko = result.get("summary_ko", "")

                    # 3) CSV 저장
                    save_diary_to_csv(
                        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                        selected_date=selected_date.isoformat(),
                        student_id=student_id,
                        mood=mood,
                        original_diary=diary_text.strip(),
                        corrected_diary=corrected_diary,
                        feedback_en=feedback_en,
                        summary_ko=summary_ko,
                    )

                st.success("Saved successfully! 저장이 완료되었습니다.")

                # 4) 결과 표시
                st.subheader("Original Diary")
                st.code(diary_text.strip(), language="markdown")

                st.subheader("Corrected Diary")
                st.markdown(corrected_diary or "(No parsed corrected diary)")

                st.subheader("Feedback (Easy English)")
                st.markdown(feedback_en or "(No parsed feedback)")

                st.subheader("Korean Summary for Teacher")
                st.markdown(summary_ko or "(No parsed Korean summary)")

            except Exception as exc:
                st.error(
                    f"Could not generate feedback. Please check your API key/network. "
                    f"/ 피드백 생성에 실패했습니다. API 키와 네트워크를 확인해주세요.\n\nError: {exc}"
                )

    with history_tab:
        st.subheader("Past Diary Records")
        logs_df = load_diary_csv()

        if logs_df.empty:
            st.info("No records yet. / 아직 저장된 기록이 없습니다.")
        else:
            # 날짜 형식 변환
            logs_df["date"] = pd.to_datetime(logs_df["date"], errors="coerce").dt.date

            # 5) 필터 UI
            col1, col2 = st.columns(2)
            with col1:
                filter_student = st.text_input("Filter by Name/ID", key="filter_student")
            with col2:
                min_date = logs_df["date"].min()
                max_date = logs_df["date"].max()
                date_range = st.date_input(
                    "Filter by Date Range",
                    value=(min_date, max_date) if pd.notna(min_date) and pd.notna(max_date) else (dt.date.today(), dt.date.today()),
                )

            filtered_df = logs_df.copy()
            if filter_student.strip():
                filtered_df = filtered_df[
                    filtered_df["student_id"].astype(str).str.contains(filter_student.strip(), case=False, na=False)
                ]

            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
                filtered_df = filtered_df[
                    (filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)
                ]

            st.dataframe(filtered_df, use_container_width=True)


if __name__ == "__main__":
    main()
