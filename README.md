# 📈 Financial Q&A Chatbot with RAG

Dự án này xây dựng một chatbot Hỏi-Đáp (Q&A) chuyên về lĩnh vực tài chính, sử dụng kiến trúc Retrieval-Augmented Generation (RAG). Hệ thống cho phép người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên về các báo cáo tài chính của nhiều công ty và nhận được câu trả lời được tổng hợp bởi mô hình ngôn ngữ lớn, dựa trên các ngữ cảnh chính xác được truy xuất từ cơ sở dữ liệu véc-tơ.

## ✨ Các tính năng chính

* **Giao diện Chat trực quan:** Xây dựng bằng Streamlit, cung cấp trải nghiệm hỏi-đáp thân thiện.
* **Truy vấn nhiều công ty:** Tự động phát hiện và so sánh thông tin từ nhiều mã chứng khoán (ticker) trong cùng một câu hỏi.
* **Tìm kiếm ngữ nghĩa tốc độ cao:** Sử dụng Pinecone để lưu trữ và truy xuất nhanh các đoạn văn bản liên quan nhất.
* **Tổng hợp câu trả lời thông minh:** Dùng Google Gemini 1.5 Flash để đọc hiểu ngữ cảnh và tạo ra câu trả lời mạch lạc.
* **Phản hồi Streaming:** Hiển thị câu trả lời ngay lập tức, tương tự như các chatbot hiện đại.
* **Hiển thị nguồn:** Cho phép người dùng xem các đoạn văn bản gốc đã được sử dụng để tạo ra câu trả lời, tăng tính minh bạch.

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 3
* **Giao diện Web:** Streamlit
* **Cơ sở dữ liệu Vector:** Pinecone
* **Mô hình ngôn ngữ (LLM):** Google Gemini 1.5 Flash
* **Mô hình Embedding:** `all-MiniLM-L6-v2` (từ Sentence-Transformers)
* **Xử lý dữ liệu:** Pandas, LangChain

## 🚀 Cài đặt và Chạy dự án

### Yêu cầu
* Python 3.8+
* Tài khoản Pinecone
* API Key từ Google AI Studio (cho Gemini)

### Các bước cài đặt

1.  **Clone repository về máy:**
    ```bash
    git clone [https://github.com/TEN_CUA_BAN/TEN_REPOSITORY.git](https://github.com/TEN_CUA_BAN/TEN_REPOSITORY.git)
    cd TEN_REPOSITORY
    ```

2.  **Tạo file `.env`:**
    Tạo một file tên là `.env` trong thư mục gốc và điền API key của bạn vào:
    ```
    PINECONE_API_KEY="YOUR_PINECONE_KEY_HERE"
    GEMINI_API_KEY="YOUR_GEMINI_KEY_HERE"
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Chuẩn bị dữ liệu và nạp vào Pinecone:**
    Đặt file dữ liệu `Financial-QA-10k.csv` của bạn vào thư mục gốc và chạy script sau (chỉ cần chạy 1 lần):
    ```bash
    python preprocess_data_optimized.py
    ```

5.  **Khởi động ứng dụng:**
    ```bash
    streamlit run app_v2_improved.py
    ```
    Mở trình duyệt và truy cập vào `http://localhost:8501`.

## 使い方 (Cách sử dụng)

1.  Truy cập ứng dụng qua trình duyệt.
2.  Sử dụng thanh bên để chọn một ticker mặc định.
3.  Nhập câu hỏi vào ô chat ở dưới cùng. Bạn có thể hỏi về một công ty hoặc so sánh nhiều công ty.
    * **Ví dụ 1 (Một công ty):** `What were the total revenues for AAPL last year?`
    * **Ví dụ 2 (Nhiều công ty):** `Compare the main business risks of GOOGL and META.`
4.  Nhấn Enter và xem câu trả lời được tạo ra. Bạn có thể nhấp vào expander "Xem các nguồn đã sử dụng" để kiểm chứng thông tin.

---
*Dự án được phát triển nhằm mục đích học tập và trình diễn khả năng của kiến trúc RAG.*