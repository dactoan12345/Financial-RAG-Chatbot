# app_v2_improved.py
import os
import re 
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from qa_system import FinancialQASystem

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Pro Financial Q&A", page_icon="📈", layout="wide")

# --- CÁC THAM SỐ VÀ BIẾN TOÀN CỤC ---
MODEL_NAME = 'all-MiniLM-L6-v2'
PINECONE_INDEX_NAME = 'financial-qa' 

# --- DANH SÁCH TICKERS ---
AVAILABLE_TICKERS = sorted([
    "NVDA", "LULU", "BRK-A", "AEN", "ICE", "AAPL", "PG", "META", "VSC", "INTU", 
    "TSLA", "COST", "AXP", "PHE", "IRM", "ABNB", "PTON", "DAL", "JNJ", "JPM", 
    "MSFT", "SBUX", "TRDL", "KR", "LVS", "AMZN", "NKE", "EBAY", "HD", "WMT", 
    "NFLX", "PLTR", "AMD", "CVX", "GOOGL", "ABBV", "BAC", "KO", "V", "GME", 
    "EFX", "T", "AZO", "AMC", "CRM", "ETSY", "CAT", "SCHW", "LLY", "AVGO", 
    "FDX", "CMG", "CB", "UNH", "F", "GRMN", "GIS", "GM", "GILD", "GS", "HAS", 
    "HSY", "HPE", "LTH", "HPQ", "HUM", "IBM"
])

# --- TẢI API KEYS ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# --- KHỞI TẠO DỊCH VỤ (CACHE LẠI) ---
@st.cache_resource
def initialize_services():
    if not gemini_api_key or not pinecone_api_key:
        st.error("Vui lòng cung cấp GEMINI_API_KEY và PINECONE_API_KEY trong file .env")
        st.stop()
        
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    pc = Pinecone(api_key=pinecone_api_key)

    if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        st.error(f"Index '{PINECONE_INDEX_NAME}' không tồn tại. Vui lòng chạy 'preprocess_data_optimized.py' trước.")
        st.stop()
        
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    embedding_model = SentenceTransformer(MODEL_NAME)
    return FinancialQASystem(gemini_model, pinecone_index, embedding_model)

qa_system = initialize_services()

# --- KHỞI TẠO SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "NVDA"

# --- GIAO DIỆN STREAMLIT ---
st.title("📈 Pro Financial Q&A with RAG")
st.markdown("Hệ thống có thể tự động phát hiện và trả lời các câu hỏi so sánh giữa nhiều công ty.")

with st.sidebar:
    st.header("Cài đặt Truy vấn")
    st.session_state.selected_ticker = st.selectbox(
        "Chọn Ticker mặc định:", 
        options=AVAILABLE_TICKERS,
        key="ticker_selector"
    )
    st.info(f"Ticker mặc định là **{st.session_state.selected_ticker}**.")
    if st.button("Bắt đầu cuộc trò chuyện mới"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "contexts" in message and message["contexts"]:
             with st.expander("Xem các nguồn đã sử dụng"):
                for i, context in enumerate(message["contexts"], 1):
                    ticker_badge = f"`{context['ticker']}`"
                    st.info(f'**Nguồn {i} ({ticker_badge}):** "{context["text"]}"')

# --- LOGIC CHÍNH ---
if prompt := st.chat_input("Hỏi về một hoặc nhiều công ty..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        prompt_upper = prompt.upper()
        tickers_in_prompt = sorted(list(set([
            ticker for ticker in AVAILABLE_TICKERS 
            if re.search(r'\b' + re.escape(ticker) + r'\b', prompt_upper)
        ])))

        if not tickers_in_prompt:
            tickers_to_query = [st.session_state.selected_ticker]
            st.info(f"🔍 Không tìm thấy ticker trong câu hỏi. Sử dụng ticker mặc định: **{tickers_to_query[0]}**")
        else:
            tickers_to_query = tickers_in_prompt
            st.info(f"🔍 Đã phát hiện và sẽ phân tích các ticker: **{', '.join(tickers_to_query)}**")

        all_contexts = []
        with st.spinner(f"Đang tìm kiếm thông tin cho {', '.join(tickers_to_query)}..."):
            for ticker in tickers_to_query:
                contexts = qa_system.find_top_contexts(
                    ticker=ticker,
                    question=prompt,
                    top_k=5 
                )
                all_contexts.extend(contexts)
        
        full_response = ""
        response_generator = qa_system.get_answer_stream(prompt, all_contexts)
        
        for chunk in response_generator:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "contexts": all_contexts
        })