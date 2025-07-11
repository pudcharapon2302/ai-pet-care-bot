import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- ส่วนที่เปลี่ยนแปลง ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# -------------------------
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

print("===== Script is starting... =====")

# โหลด API Key จากไฟล์ .env
load_dotenv()

# ตั้งค่าตัวแปร
KNOWLEDGE_BASE_DIR = "knowledge_base"
FAISS_PATH = "faiss_index"

def create_vector_store():
    """สร้างและบันทึก Vector Store จาก Knowledge Base"""
    print("กำลังสร้าง Vector Store...")
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # --- ส่วนที่เปลี่ยนแปลง ---
    # ใช้ GoogleGenerativeAIEmbeddings แทน OpenAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # -------------------------

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_PATH)
    print("Vector Store ถูกสร้างและบันทึกเรียบร้อย")

def get_rag_chain():
    """โหลด Vector Store และสร้าง RAG Chain"""
    # --- ส่วนที่เปลี่ยนแปลง ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # -------------------------
    
    # โหลด Vector Store ที่มีอยู่
    if not os.path.exists(FAISS_PATH):
        create_vector_store()
        
    db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
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
    # เนื่องจากเราเปลี่ยนโมเดล Embeddings เราต้องสร้าง Index ใหม่
    print("กำลังลบ Index เก่า...")
    if os.path.exists(FAISS_PATH):
        import shutil
        shutil.rmtree(FAISS_PATH)
    create_vector_store()