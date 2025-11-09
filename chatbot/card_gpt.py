import streamlit as st
import json
import os
import re
import random
import datetime
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

# Streamlit Secrets에서 환경 변수 불러오기
SHEET_ID = st.secrets["SHEET_ID"]

# Google Service Account JSON 파싱
raw_json = st.secrets["GOOGLE_SERVICE_ACCOUNT"]
try:
    parsed = json.loads(raw_json)
    if isinstance(parsed, str):
        service_account_info = json.loads(parsed)
    else:
        service_account_info = parsed
except json.JSONDecodeError as e:
    st.error(f"❌ JSON 파싱 오류: {e}")
    st.stop()

# Google 인증 객체 생성
creds = Credentials.from_service_account_info(
    service_account_info,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SHEET_ID).sheet1


# ------------------------------- 로그 저장 함수 -------------------------------
def append_log_to_sheet(log_entry):
    """Google Sheets에 로그 추가"""
    try:
        row = [
            log_entry.get("timestamp"),
            log_entry.get("user_info", {}).get("name", ""),
            log_entry.get("user_info", {}).get("age_group", ""),
            log_entry.get("user_info", {}).get("occupation", ""),
            log_entry.get("query", ""),
            ", ".join(log_entry.get("card_ids", [])),
            ", ".join(log_entry.get("clicked_cards", [])),
            log_entry.get("session_duration_sec", 0),
            log_entry.get("ab_version", ""),
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        print(f"[로그 저장 실패] Google Sheets → {e}")


# ------------------------------- 세션 및 A/B 설정 -------------------------------
AB_VERSION = random.choice(["A", "B"])
SESSION_START = datetime.datetime.now()


# ------------------------------- 카드 링크·이미지 로드 -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LINK_IMAGE_PATH = os.path.join(BASE_DIR, "cards_link_image.json")

with open(LINK_IMAGE_PATH, "r", encoding="utf-8") as f:
    link_data = json.load(f)

LINK_DB = {str(item["card_id"]): item for item in link_data}


# ------------------------------- 함수 정의 -------------------------------
def extract_card_ids(text):
    """AI 응답에서 카드ID 추출"""
    return re.findall(r"카드ID\s*:\s*(\d+)", text)


def show_card_details(card_ids):
    """카드ID 기반으로 이미지·링크 표시 + 클릭 추적 + 오류 신고 기능"""

    # 신고 상태 저장용 세션 추가
    if "reported_cards" not in st.session_state:
        st.session_state["reported_cards"] = []

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
            else:
                st.warning(f"이미지 파일을 찾을 수 없습니다: {abs_img_path}")

        pc_link = data.get("request_pc")
        m_link = data.get("request_m")

        # 링크 표시
        if pc_link:
            st.markdown(
                f"[🖥️ **PC 신청 링크 열기 ({cid})**]({pc_link})",
                unsafe_allow_html=True,
            )
            if f"{cid}_pc" not in st.session_state["clicked_cards"]:
                st.session_state["clicked_cards"].append(f"{cid}_pc")

        if m_link:
            st.markdown(
                f"[📱 **모바일 신청 링크 열기 ({cid})**]({m_link})",
                unsafe_allow_html=True,
            )
            if f"{cid}_m" not in st.session_state["clicked_cards"]:
                st.session_state["clicked_cards"].append(f"{cid}_m")

        if not pc_link and not m_link:
            st.write("신청 링크 없음")

        if cid not in st.session_state["reported_cards"]:
            if st.button(f"⚠️ 이미지·링크 불일치 신고 ({cid})", key=f"report_{cid}"):
                report_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "report_type": "불일치 신고",
                    "card_id": cid,
                    "user_name": st.session_state.get("user_name", "익명"),
                    "ab_version": AB_VERSION,
                }
                try:
                    sheet.append_row(
                        [
                            report_entry["timestamp"],
                            report_entry["user_name"],
                            "",
                            "",
                            f"불일치 신고 (카드ID: {cid})",
                            cid,
                            "",
                            "",
                            report_entry["ab_version"],
                            "신고됨",
                        ],
                        value_input_option="USER_ENTERED",
                    )
                    st.session_state["reported_cards"].append(cid)
                    st.success(f"카드ID {cid} 신고가 접수되었습니다.")
                except Exception as e:
                    st.error(f"신고 저장 실패: {e}")
        else:
            st.info(f"이미 카드ID {cid}는 신고가 접수되었습니다.")

        st.write("---")


# ------------------------------- 세션 초기화 -------------------------------
if "pre_memory" not in st.session_state:
    st.session_state["pre_memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "안녕하세요. 저는 AI 카드 추천 전문가입니다. 당신에게 맞는 카드를 추천해드릴게요.",
        }
    ]

if "clicked_cards" not in st.session_state:
    st.session_state["clicked_cards"] = []


# ------------------------------- 모델 설정 -------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = """
너는 카드사 직원이야. 고객의 질의가 들어오면 context에 따라 가장 혜택이 2개 추천해줘.
신용카드, 체크카드에 대한 명시가 없을 경우 신용카드, 체크카드 각각 1개씩 추천하고,
명시할 경우 해당 카드로 2개 추천해줘.
context 내용에 한해서만 추천해주되, context에 없는 내용은 발설하지 말아줘.
각 카드의 마지막 줄에는 반드시 '카드ID: {{card_id}}'를 포함시켜줘.

--출력 포맷--
📌 해당란에 먼저 사용자가 어떤 카드를 원하는지 파악해서 요약본을 한 줄로 작성해줘.
💳 추천카드명 
- 추천 이유 
- 해당 카드의 혜택
💳 추천카드명 
- 추천 이유 
- 해당 카드의 혜택
"""

user_prompt = """
아래의 사용자 question을 읽고 context를 참고하여
가장 적합한 카드(사용자가 혜택을 최대로 받을 수 있는 카드)를 추천해주세요.

--chat_history--
{chat_history}

--question--
{question}

--context--
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
    stream_placeholder = st.empty()
    image_placeholder = st.empty()
    full_response = ""

    for chunk in chain.stream(question):
        full_response += chunk
        stream_placeholder.write(full_response)

    card_ids = extract_card_ids(full_response)

    with image_placeholder.container():
        show_card_details(card_ids)

    session_duration = (datetime.datetime.now() - SESSION_START).total_seconds()

    st.session_state["pre_memory"].save_context(
        {"input": question}, {"output": full_response}
    )

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_info": user_info,
        "query": question,
        "response": full_response,
        "card_ids": card_ids,
        "clicked_cards": st.session_state.get("clicked_cards", []),
        "session_duration_sec": session_duration,
        "ab_version": AB_VERSION,
    }

    append_log_to_sheet(log_entry)

    return full_response


# ------------------------------- 메인 화면 -------------------------------
st.title("AI의 맞춤 카드 추천 챗봇🥰")

col1, col2 = st.columns(2)
with col1:
    age_group = st.radio(
        "연령대",
        ["10대", "20대", "30대", "40대", "50대 이상"],
        index=0,
    )

with col2:
    occupation = st.radio(
        "직업",
        ["학생", "직장인", "취업 준비생", "기타"],
        index=0,
    )

user_name = st.text_input("닉네임을 입력하세요:", "")

user_info = {
    "name": user_name or "익명",
    "age_group": age_group,
    "occupation": occupation,
}

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("메시지를 입력하세요.")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    if st.session_state["messages"][-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            try:
                ai_response = conversation_with_memory(question, user_info)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": ai_response}
                )
            except Exception as e:
                st.error(f"오류 발생: {e}")
