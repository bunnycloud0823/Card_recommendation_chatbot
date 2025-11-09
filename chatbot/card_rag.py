# from langchain_community.document_loaders import JSONLoader
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()


# vectorstore 객체 생성 함수 정의
def get_or_create_vectorstore(
    persist_directory="./Chroma", collection_name="Card_recommendation_chatbot"
):
    """
    vectorstore가 존재하면 로드하고, 없으면 새로 생성합니다.
    """
    # 임베딩 객체 정의
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    # vectorstore가 이미 존재하는지 확인
    # Chroma는 persist_directory/collection_name 형태로 저장됨
    vectorstore_exists = os.path.exists(persist_directory) and os.path.isdir(
        persist_directory
    )

    # 존재하면 기존의 벡터스토어를 로드하고 반환
    if vectorstore_exists:
        try:
            # 기존 vectorstore 로드 시도
            vectorstore = Chroma(
                embedding_function=embedding,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
            # 만약에 벡터스토어는 있는데 문서 개수가 없을경우
            # collection에 데이터가 있는지 확인
            if vectorstore._collection.count() > 0:
                print(
                    f"기존 vectorstore를 로드했습니다. (문서 수: {vectorstore._collection.count()})"
                )
                return vectorstore  # 반환
            else:
                print("Vectorstore는 존재하지만 비어있습니다. 새로 생성합니다.")
        except Exception as e:
            print(f"Vectorstore 로드 중 오류 발생: {e}")
            print("새로운 vectorstore를 생성합니다.")
    else:
        print("Vectorstore가 존재하지 않습니다. 새로 생성합니다.")

    # ⚠️ 존재하지 않을 경우 ⚠️
    # 새로운 vectorstore 생성
    # 1. 데이터 로드
    with open("cards_info.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    # 2. 문서 분할 (Chunking)
    splitter = RecursiveCharacterTextSplitter()
    split_docs = splitter.create_documents([str(dict_) for dict_ in docs])

    # 3. vectorstore 생성 및 청크 저장
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"새로운 vectorstore를 생성했습니다. (문서 수: {len(split_docs)})")
    return vectorstore


# 가장 유사한 청크 검색하는 함수 정의
def search_card(
    question,
    persist_directory="./Chroma",
    collection_name="Card_recommendation_chatbot",
):
    """
    카드 정보를 검색합니다.
    """
    # 벡터스토어 재정의
    # vectorstore 가져오기 (없으면 자동 생성)
    vectorstore = get_or_create_vectorstore(persist_directory, collection_name)

    # retriever 실행
    retriever = vectorstore.as_retriever()
    result = retriever.invoke(question)

    card_context = []
    for page in result:
        card_context.append(page.page_content)  # 텍스트만 추출하여 저장한다.

    return card_context  # 리스트 형태로 반환
