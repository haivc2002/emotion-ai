import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
from mtcnn import MTCNN
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

app = FastAPI(title="Emotion Recognition AI")

# Thiết lập thư mục tĩnh và templates
base_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(base_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# ============================================================
# Load HSEmotion pre-trained model (EfficientNet-B0 + VGGFace2)
# Đạt ~66% accuracy trên AffectNet 8-class (state-of-the-art)
# ============================================================
print("Loading HSEmotion pre-trained model...")
emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf')
print("HSEmotion model loaded successfully!")

# Load bộ phát hiện khuôn mặt MTCNN
detector = MTCNN()

# Nhãn cảm xúc tiếng Việt (8 lớp)
# Thứ tự khớp với HSEmotion idx_to_class:
# 0: Anger, 1: Contempt, 2: Disgust, 3: Fear, 4: Happiness, 5: Neutral, 6: Sadness, 7: Surprise
EMOTIONS_VI = {
    'Anger': 'Tức giận',
    'Contempt': 'Khinh bỉ',
    'Disgust': 'Ghê tởm',
    'Fear': 'Sợ hãi',
    'Happiness': 'Vui vẻ',
    'Neutral': 'Bình thường',
    'Sadness': 'Buồn bã',
    'Surprise': 'Ngạc nhiên'
}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def process_image(img_array):
    # Chuyển ảnh sang RGB cho MTCNN (OpenCV mặc định dùng BGR)
    rgb_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Phát hiện khuôn mặt bằng MTCNN
    detections = detector.detect_faces(rgb_img)
    
    results = []
    
    for face in detections:
        if face['confidence'] < 0.9:
            continue
            
        x, y, w, h = face['box']
        
        # --- Xử lý Bounding Box ---
        # Mở rộng khung cắt thành hình vuông + padding 25%
        # để khớp với cách HSEmotion xử lý ảnh khuôn mặt
        center_x = x + w // 2
        center_y = y + h // 2
        
        side = max(w, h)
        side = int(side * 1.25)
        
        new_x = max(0, center_x - side // 2)
        new_y = max(0, center_y - side // 2)
        new_w = min(rgb_img.shape[1] - new_x, side)
        new_h = min(rgb_img.shape[0] - new_y, side)
        
        # Cắt khuôn mặt từ ảnh RGB
        face_img = rgb_img[new_y:new_y+new_h, new_x:new_x+new_w]
        
        if face_img.size == 0:
            continue
        
        # --- Dự đoán bằng HSEmotion ---
        # HSEmotion tự xử lý resize + preprocessing bên trong
        emotion_en, scores = emotion_recognizer.predict_emotions(face_img, logits=False)
        
        # Chuyển sang tiếng Việt
        emotion_vi = EMOTIONS_VI.get(emotion_en, emotion_en)
        confidence = float(scores.max())
        
        # In kết quả raw ra Terminal
        print("\n--- HSEmotion PREDICTION ---")
        for i, (en_name, vi_name) in enumerate(EMOTIONS_VI.items()):
            print(f"  {vi_name}: {scores[i]*100:.2f}%")
        print(f"  → Kết quả: {emotion_vi} ({confidence*100:.1f}%)")
        print("----------------------------")
        
        results.append({
            "box": [int(new_x), int(new_y), int(new_w), int(new_h)],
            "emotion": emotion_vi,
            "confidence": confidence
        })
        
    return {"faces": results}

import urllib.request

@app.post("/predict")
async def predict(request: Request):
    content_type = request.headers.get("content-type", "")
    img = None
    
    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            if "file" in form:
                contents = await form["file"].read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif "image" in form:
                image_data = form["image"].split(',')[1] if ',' in form["image"] else form["image"]
                nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif "url" in form:
                req = urllib.request.Request(form["url"], headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    contents = response.read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
        elif "application/json" in content_type:
            data = await request.json()
            if "image" in data:
                image_data = data["image"].split(',')[1] if ',' in data["image"] else data["image"]
                nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif "url" in data:
                req = urllib.request.Request(data["url"], headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    contents = response.read()
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid request. Please provide 'file' (upload), 'image' (base64), or 'url'."})
            
        results = process_image(img)
        return JSONResponse(content=results)
        
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Error processing request: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
