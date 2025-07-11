# main.py
# ไฟล์หลักสำหรับรันเซิร์ฟเวอร์ FastAPI และทำหน้าที่เป็น Webhook ให้กับ LINE

import os
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# --- ขั้นตอนที่ 1: นำเข้า RAG Chain ---
# ตรวจสอบให้แน่ใจว่าคุณมีไฟล์ rag_handler.py ที่สร้างฟังก์ชัน get_rag_chain()
# ซึ่งทำหน้าที่สร้างและคืนค่า RAG chain ที่พร้อมใช้งาน
try:
    from rag_handler import get_rag_chain
except ImportError:
    print("="*50)
    print("ข้อผิดพลาด: ไม่พบไฟล์ rag_handler.py หรือฟังก์ชัน get_rag_chain")
    print("โปรดตรวจสอบว่าคุณได้สร้างไฟล์ตามขั้นตอนก่อนหน้านี้แล้ว")
    print("="*50)
    exit()


# --- ขั้นตอนที่ 2: โหลด Environment Variables ---
# โหลดค่าที่ตั้งไว้ในไฟล์ .env เช่น CHANNEL_ACCESS_TOKEN และ CHANNEL_SECRET
load_dotenv()

# --- ขั้นตอนที่ 3: ตั้งค่า FastAPI และ LINE Bot ---
# สร้าง Instance ของ FastAPI application
app = FastAPI(title="AI Pet Care Bot Service")

# ดึงค่า Access Token และ Channel Secret จาก Environment variables
channel_access_token = os.getenv('CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('CHANNEL_SECRET')

# ตรวจสอบว่าค่าที่จำเป็นถูกตั้งค่าไว้ครบถ้วนหรือไม่
if not channel_access_token or not channel_secret:
    raise RuntimeError("กรุณาตั้งค่า CHANNEL_ACCESS_TOKEN และ CHANNEL_SECRET ในไฟล์ .env")

# สร้าง Instance ของ LineBotApi และ WebhookHandler
line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# --- ขั้นตอนที่ 4: โหลด RAG Chain ---
# เรียกใช้ฟังก์ชันเพื่อสร้าง Chain ที่จะใช้ประมวลผลคำถาม
print("กำลังเริ่มต้น RAG chain...")
try:
    rag_chain = get_rag_chain()
    print("RAG chain เริ่มต้นสำเร็จ!")
except Exception as e:
    print(f"เกิดข้อผิดพลาดขณะเริ่มต้น RAG chain: {e}")
    rag_chain = None

# --- ขั้นตอนที่ 5: สร้าง API Endpoints ---
@app.get("/")
def read_root():
    """Endpoint หลักสำหรับตรวจสอบว่าเซิร์ฟเวอร์ทำงานอยู่หรือไม่"""
    return {"message": "AI Pet Care Bot is running! สามารถเชื่อมต่อได้"}

@app.post("/webhook")
async def webhook(request: Request):
    """
    Endpoint ที่รับ Webhook events จาก LINE Platform
    LINE จะส่งข้อมูลมาที่นี่ทุกครั้งที่มีคนส่งข้อความถึงบอท
    """
    # ดึงค่า Signature จาก Header เพื่อยืนยันว่า Request มาจาก LINE จริงๆ
    signature = request.headers.get('X-Line-Signature')
    if not signature:
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    # ดึงข้อมูล (body) จาก Request
    body = await request.body()

    try:
        # ตรวจสอบ Signature และประมวลผล Event ต่างๆ
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        print("ลายเซ็นไม่ถูกต้อง! โปรดตรวจสอบ Channel Secret ของคุณ")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการจัดการ Webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    return 'OK'

# --- ขั้นตอนที่ 6: สร้างฟังก์ชันสำหรับจัดการข้อความ ---
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """
    ฟังก์ชันนี้จะถูกเรียกใช้เมื่อ Event ที่ได้รับเป็นข้อความ (TextMessage)
    """
    user_message = event.message.text
    reply_token = event.reply_token
    print(f"ได้รับข้อความจากผู้ใช้: '{user_message}'")

    if not rag_chain:
        # กรณีที่ RAG Chain ไม่สามารถเริ่มต้นได้
        ai_answer = "ขออภัยค่ะ ระบบประมวลผลกำลังมีปัญหา โปรดลองอีกครั้งในภายหลัง"
    else:
        try:
            # ส่งข้อความของผู้ใช้ไปยัง RAG chain เพื่อหาคำตอบ
            print("กำลังส่งคำถามไปยัง RAG chain...")
            response = rag_chain.invoke({"input": user_message})
            
            # ดึงคำตอบจาก response object (ใช้ .get เพื่อความปลอดภัย)
            ai_answer = response.get('answer', "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในขณะนี้")
            print(f"คำตอบจาก AI: '{ai_answer}'")

        except Exception as e:
            print(f"เกิดข้อผิดพลาดระหว่างการประมวลผล RAG chain: {e}")
            ai_answer = "ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผล"

    # ส่งคำตอบกลับไปหาผู้ใช้ผ่าน LINE
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text=ai_answer)
    )

# --- ขั้นตอนที่ 7: รันเซิร์ฟเวอร์ ---
if __name__ == "__main__":
    # ใช้ uvicorn เพื่อรัน FastAPI application
    # host="0.0.0.0" ทำให้สามารถเข้าถึงได้จากภายนอกเครื่อง
    port = int(os.getenv("PORT", 8000))
    print(f"กำลังจะเริ่มเซิร์ฟเวอร์ที่ http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
