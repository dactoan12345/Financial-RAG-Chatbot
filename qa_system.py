# qa_system.py
import textwrap

class FinancialQASystem:
    def __init__(self, gemini_model, pinecone_index, embedding_model):
        self.gemini = gemini_model
        self.pinecone_index = pinecone_index
        self.sbert = embedding_model

    def find_top_contexts(self, ticker: str, question: str, top_k: int = 5, threshold: float = 0.3) -> list:
        """
        Tìm kiếm ngữ cảnh liên quan trong Pinecone và lọc theo ngưỡng.
        Lưu ý: Ngưỡng có thể cần điều chỉnh tùy vào dữ liệu.
        """
        question_embedding = self.sbert.encode(question).tolist()
        
        query_result = self.pinecone_index.query(
            vector=question_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={'ticker': {'$eq': ticker.upper()}}
        )
        
        top_contexts = [
            match['metadata'] for match in query_result['matches'] 
            if match['score'] >= threshold
        ]
        
        return top_contexts

    def get_answer_stream(self, question: str, optimized_contexts: list):
        """
        Tạo prompt và gọi Gemini để lấy câu trả lời DẠNG STREAM.
        """
        if not optimized_contexts:
            yield "Dựa trên các tài liệu liên quan nhất, thông tin bạn hỏi không có sẵn."
            return

        context_texts = [ctx['text'] for ctx in optimized_contexts]
        full_context = "\n\n---\n\n".join(context_texts)
        
        prompt = f"""
        Task: You are a professional financial analyst assistant. Based ONLY on the highly relevant context snippets provided below, answer the user's question clearly and concisely.
        If the information is still not sufficient, state clearly: "Based on the most relevant documents, this information is not available." Do not make up information.

        RELEVANT CONTEXT:
        {full_context}

        ---
        USER'S QUESTION:
        {question}

        ANSWER (based on the provided context):
        """
        try:
            response_stream = self.gemini.generate_content(prompt, stream=True)
            for chunk in response_stream:
                yield chunk.text
        except Exception as e:
            yield f"Lỗi khi gọi Gemini API: {e}"