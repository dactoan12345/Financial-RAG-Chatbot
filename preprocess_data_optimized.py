# preprocess_data_optimized.py
import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import time

# --- CÁC THAM SỐ ---
CSV_PATH = 'Financial-QA-10k.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'
PINECONE_INDEX_NAME = 'financial-qa'
PINECONE_UPSERT_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 32

# --- TẢI API KEYS ---
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# --- LOGIC XỬ LÝ ---
def preprocess_and_upload_to_pinecone():
    """
    Hàm đọc dữ liệu, xử lý chunking, tạo embeddings THEO BATCH và tải lên Pinecone.
    """
    print("Bắt đầu quá trình xử lý và tải dữ liệu lên Pinecone...")

    # 1. Khởi tạo kết nối
    print(f"Đang tải model SentenceTransformer: '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()

    print("Đang khởi tạo Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)

    # 2. Tạo hoặc kết nối tới Pinecone Index
    if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes()]:
        print(f"Index '{PINECONE_INDEX_NAME}' không tồn tại. Đang tạo index mới...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=embedding_dim,
            metric='cosine',
            spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            time.sleep(1)
        print("Đã tạo index thành công.")
    else:
        print(f"Đã kết nối tới index '{PINECONE_INDEX_NAME}'.")

    index = pc.Index(PINECONE_INDEX_NAME)

    # 3. Đọc và xử lý dữ liệu
    print(f"Đang đọc dữ liệu từ '{CSV_PATH}'...")
    df = pd.read_csv(CSV_PATH, encoding='utf-8').dropna(subset=['ticker', 'context'])

    # 4. Cải thiện chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    print("Bắt đầu quá trình tạo chunk và metadata...")
    all_chunks = []
    
    for doc_id, row in tqdm(df.iterrows(), total=df.shape[0], desc="Chunking documents"):
        ticker = row['ticker']
        context = row['context']
        
        if not context or not isinstance(context, str):
            continue

        chunks_text = text_splitter.split_text(context)
        
        for i, chunk_text in enumerate(chunks_text):
            metadata = {
                'ticker': ticker,
                'text': chunk_text,
                'source_document': CSV_PATH,
                'document_id': f"doc_{doc_id}",
                'chunk_index': i
            }
            unique_id = f"doc_{doc_id}_chunk_{i}"
            all_chunks.append({'id': unique_id, 'text': chunk_text, 'metadata': metadata})

    print(f"✅ Đã tạo tổng cộng {len(all_chunks)} chunks.")

    # 5. TẠO EMBEDDING THEO BATCH và UPSERT LÊN PINECONE
    print(f"Bắt đầu tạo embedding và upsert lên Pinecone theo batch (size={PINECONE_UPSERT_BATCH_SIZE})...")
    
    for i in tqdm(range(0, len(all_chunks), PINECONE_UPSERT_BATCH_SIZE), desc="Upserting to Pinecone"):
        batch_chunks = all_chunks[i:i + PINECONE_UPSERT_BATCH_SIZE]
        texts_to_embed = [chunk['text'] for chunk in batch_chunks]
        
        embeddings = model.encode(texts_to_embed, batch_size=EMBEDDING_BATCH_SIZE).tolist()
        
        vectors_to_upsert = []
        for j, chunk in enumerate(batch_chunks):
            vectors_to_upsert.append({
                'id': chunk['id'],
                'values': embeddings[j],
                'metadata': chunk['metadata']
            })
            
        index.upsert(vectors=vectors_to_upsert)

    print("\nQuá trình xử lý và tải lên Pinecone hoàn tất!")
    print(f"Tổng quan Index: {index.describe_index_stats()}")


if __name__ == '__main__':
    preprocess_and_upload_to_pinecone()