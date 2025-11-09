import streamlit as st
import json
import os
import re
import random
import datetime
import time
from dotenv import load_dotenv
from card_rag import search_card
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
import gspread
from google.oauth2.service_account import Credentials


# ------------------------------- 초기 설정 -------------------------------
load_dotenv()
SHEET_ID = st.secrets["SHEET_ID"]

# Google 인증
raw_json = st.secrets["GOOGLE_SERVICE_ACCOUNT"]
parsed = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
creds = Credentials.from_service_account_info(
    parsed,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SHEET_ID).sheet1


# ------------------------------- 로그 저장 -------------------------------
def append_log_to_sheet(log_entry):
    try:
        ab_value = log_entry.get("ab_version", "")
        if log_entry.get("report_flag"):
            ab_value = f"{ab_value} (신고)"
        row = [
            log_entry.get("timestamp", ""),
            log_entry.get("user_info", {}).get("name", ""),
            log_entry.get("user_info", {}).get("age_group", ""),
            log_entry.get("user_info", {}).get("occupation", ""),
            log_entry.get("query", ""),
            ", ".join(log_entry.get("card_ids", [])),
            ", ".join(log_entry.get("clicked_cards", [])),
            log_entry.get("session_duration_sec", 0),
            ab_value,
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.error(f"로그 저장 실패: {e}")


# ------------------------------- 세션 초기화 -------------------------------
if "pre_memory" not in st.session_state:
    st.session_state["pre_memory"] = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요. AI 카드 추천 전문가입니다."}
    ]
if "clicked_cards" not in st.session_state:
    st.session_state["clicked_cards"] = []
if "reported_cards" not in st.session_state:
    st.session_state["reported_cards"] = []

AB_VERSION = random.choice(["A", "B"])
SESSION_START = datetime.datetime.now()


# ------------------------------- 카드 로드 -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LINK_IMAGE_PATH = os.path.join(BASE_DIR, "cards_link_image.json")
with open(LINK_IMAGE_PATH, "r", encoding="utf-8") as f:
    link_data = json.load(f)
LINK_DB = {str(item["card_id"]): item for item in link_data}


# ------------------------------- 카드 표시 -------------------------------
def show_card_details(card_ids, user_info, question):
    for cid in card_ids:
        data = LINK_DB.get(str(cid))
        if not data:
            continue

        img_path = data.get("image")
        if img_path:
            abs_img_path = os.path.normpath(
                os.path.join(BASE_DIR, "..", img_path.replace("./", ""))
            )
            if os.path.exists(abs_img_path):
                st.image(abs_img_path, width=250)

        pc_link = data.get("request_pc")
        m_link = data.get("request_m")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if pc_link:
                st.markdown(
                    f'<a href="{pc_link}" target="_blank"><button style="background-color:#0072C6;'
                    f'color:white;border:none;padding:8px 16px;border-radius:6px;">PC 신청 ({cid})</button></a>',
                    unsafe_allow_html=True,
                )
                if f"{cid}_pc" not in st.session_state["clicked_cards"]:
                    st.session_state["clicked_cards"].append(f"{cid}_pc")

        with col2:
            if m_link:
                st.markdown(
                    f'<a href="{m_link}" target="_blank"><button style="background-color:#28a745;'
                    f'color:white;border:none;padding:8px 16px;border-radius:6px;">모바일 ({cid})</button></a>',
                    unsafe_allow_html=True,
                )
                if f"{cid}_m" not in st.session_state["clicked_cards"]:
                    st.session_state["clicked_cards"].append(f"{cid}_m")

        with col3:
            if f"{cid}_report" in st.session_state["reported_cards"]:
                st.info(f"카드ID {cid} 신고 완료됨")
            else:
                if st.button(f"신고 ({cid})", key=f"report_{cid}"):
                    st.session_state["reported_cards"].append(f"{cid}_report")
                    log_entry = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "user_info": user_info,
                        "query": f"불일치 신고 (카드ID: {cid})",
                        "response": "",
                        "card_ids": [str(cid)],
                        "clicked_cards": st.session_state["clicked_cards"],
                        "session_duration_sec": 0,
                        "ab_version": AB_VERSION,
                        "report_flag": True,
                    }
                    append_log_to_sheet(log_entry)
                    st.success(f"카드ID {cid} 신고가 기록되었습니다.")
        st.write("---")


# ------------------------------- LangChain 모델 -------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=512)
system_prompt = """
너는 카드사 직원이야. 고객의 질문에 따라 context에 있는 카드 중에서 혜택이 가장 많은 카드 2개를 추천해줘.
각 카드 설명의 마지막 줄에는 반드시 '카드ID: {{card_id}}'를 포함해줘.
"""
user_prompt = """\
아래의 사용자 question을 읽고 context를 참고하여 카드를 추천하세요.
{chat_history}
{question}
{context}
"""
final_prompt = ChatPromptTemplate([("system", system_prompt), ("user", user_prompt)])


def get_user_input(question):
    return {
        "chat_history": st.session_state["pre_memory"].chat_memory.messages,
        "question": question,
        "context": search_card(question),
    }


chain = RunnableLambda(get_user_input) | final_prompt | model | StrOutputParser()


# ------------------------------- 대화 함수 -------------------------------
def conversation_with_memory(question, user_info):
    full_response = ""
    retries = 3
    for attempt in range(retries):
        try:
            for chunk in chain.stream(question):
                full_response += chunk
                st.write(full_response)
            break
        except Exception as e:
            if attempt < retries - 1:
                st.warning(f"서버 오류 발생, 재시도 중... ({attempt + 1}/{retries})")
                time.sleep(2)
            else:
                st.error("서버 오류가 계속 발생했습니다. 나중에 다시 시도해주세요.")
                return ""

    card_ids = re.findall(r"카드ID\s*:\s*(\d+)", full_response)
    show_card_details(card_ids, user_info, question)

    duration = (datetime.datetime.now() - SESSION_START).total_seconds()

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_info": user_info,
        "query": question,
        "response": full_response,
        "card_ids": card_ids,
        "clicked_cards": st.session_state["clicked_cards"],
        "session_duration_sec": duration,
        "ab_version": AB_VERSION,
        "report_flag": False,
    }
    append_log_to_sheet(log_entry)
    return full_response


# ------------------------------- UI -------------------------------
st.title("AI 맞춤 카드 추천 챗봇")

col1, col2 = st.columns(2)
with col1:
    age_group = st.radio("연령대", ["10대", "20대", "30대", "40대", "50대 이상"])
with col2:
    occupation = st.radio("직업", ["학생", "직장인", "취준생", "기타"])

user_name = st.text_input("닉네임을 입력하세요:", "")
user_info = {
    "name": user_name or "익명",
    "age_group": age_group,
    "occupation": occupation,
}

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("카드 관련 질문을 입력하세요.")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            answer = conversation_with_memory(question, user_info)
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )
        except Exception as e:
            st.error(f"오류 발생: {e}")
