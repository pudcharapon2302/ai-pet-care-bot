import os
import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import GoogleGenerativeAiEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

print("===== Script is starting... =====")

# โหลด API Key จากไฟล์ .env
load_dotenv()

# ตั้งค่าตัวแปร
KNOWLEDGE_BASE_DIR = "knowledge_base"
CHROMA_PATH = "chroma_db"


# 1. เชื่อมต่อไปยัง ChromaDB ที่รันบน Docker ผ่าน HTTP
client = chromadb.HttpClient(host='localhost', port=8000)

# 2. สร้างหรือโหลด Collection
collection = client.get_or_create_collection(name="my_docker_collection")

# 3. เพิ่มและค้นหาข้อมูล (เหมือนวิธีแรกทุกประการ)
collection.add(
    documents=["นี่คือข้อมูลบน Docker"],
    ids=["docker_id1"]
)

results = collection.query(
    query_texts=["ข้อมูล"],
    n_results=1
)

print(results)

def create_or_update_vector_store():
    """
    สร้างหรืออัปเดต Vector Store 
    โดยจะเพิ่มเฉพาะไฟล์ใหม่ที่ยังไม่มีในฐานข้อมูล
    """
    print("กำลังตรวจสอบและอัปเดต Vector Store (ChromaDB)...")
    
    # 1. เชื่อมต่อ DB และ Collection ที่มีอยู่
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="pet_care_knowledge") # ใช้ชื่อเฉพาะสำหรับ Collection ของคุณ

    # 2. ดึงรายชื่อไฟล์ source ที่มีอยู่แล้วใน DB
    existing_files = set([metadata['source'] for metadata in collection.get()['metadatas']])
    print(f"ไฟล์ที่มีอยู่แล้วใน DB: {len(existing_files)} ไฟล์")

    # 3. หาสิ่งที่ต้องทำ Embedding เพิ่ม
    all_files = [os.path.join(KNOWLEDGE_BASE_DIR, f) for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".txt")]
    new_files = [f for f in all_files if f not in existing_files]

    if not new_files:
        print("ไม่มีไฟล์ใหม่ ไม่ต้องอัปเดต")
        return

    print(f"พบไฟล์ใหม่ {len(new_files)} ไฟล์ กำลังทำ Embedding...")
    
    # 4. โหลดและประมวลผลเฉพาะไฟล์ใหม่
    docs_to_add = []
    for file_path in new_files:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        docs_to_add.extend(documents)

    if docs_to_add:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(docs_to_add)
        
        # 5. เพิ่มข้อมูลใหม่ลงใน Collection โดยตรง
        # แปลง docs ของ Langchain เป็นสิ่งที่ collection.add() ต้องการ
        contents = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        ids = [f"{meta['source']}_{i}" for i, meta in enumerate(metadatas)] # สร้าง ID ที่คาดเดาได้

        collection.add(documents=contents, metadatas=metadatas, ids=ids)
        print(f"เพิ่มข้อมูลจากไฟล์ใหม่ {len(new_files)} ไฟล์เรียบร้อย")

def get_rag_chain():
    """โหลด Vector Store และสร้าง RAG Chain"""
    # --- ส่วนที่เปลี่ยนแปลง ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # -------------------------
    
    # โหลด Vector Store ที่มีอยู่
    if not os.path.exists(CHROMA_PATH):
        create_or_update_vector_store()
        
    db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings 
    )
    retriever = db.as_retriever()
    

    # สร้าง Prompt Template (สามารถใช้ Prompt เดิมได้)
    prompt_template = """
    คุณคือ "ผู้ช่วยรักสัตว์เลี้ยง" เป็น AI ที่ใจดีและเป็นมิตรมากๆ ให้คำปรึกษาเหมือนเพื่อนที่รักสัตว์เหมือนกัน
    - เริ่มต้นทักทายอย่างอบอุ่นเสมอ
    - ใช้ภาษาที่เข้าใจง่าย พูดคุยเหมือนคนจริงๆ ไม่ใช้ศัพท์เทคนิคเยอะเกินไป
    - ลงท้ายประโยคด้วย "นะคะ", "นะครับ" หรือคำที่น่ารักๆ เพื่อให้ดูเข้าถึงง่าย
    - ตอบคำถามโดยอ้างอิงจากข้อมูลใน "บริบท" ที่ให้มาเป็นหลัก
    - หากข้อมูลไม่เพียงพอ ให้ตอบอย่างนุ่มนวลว่า "เรื่องนี้อาจจะต้องปรึกษาสัตวแพทย์โดยตรงเพื่อความแม่นยำนะคะ" หรือ "ข้อมูลส่วนนี้ยังไม่มีในระบบเลยค่ะ"
    - เน้นย้ำเสมอว่าคำแนะนำเป็นเพียงข้อมูลเบื้องต้น และการพาสัตว์เลี้ยงไปพบสัตวแพทย์คือสิ่งที่ดีที่สุด

    บริบท:
    {context}

    คำถาม: {input}

    คำตอบจากเพื่อนรักสัตว์เลี้ยง:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # --- ส่วนที่เปลี่ยนแปลง ---
    # สร้าง LLM Chain โดยใช้ ChatGoogleGenerativeAI (Gemini Pro)
    # ของใหม่
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    # -------------------------
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # สร้าง Retrieval Chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# สำหรับการรันครั้งแรกเพื่อสร้าง index
if __name__ == '__main__':
    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)
    create_or_update_vector_store()