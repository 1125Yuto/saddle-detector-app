#app.py 2回目(画像のインポートをOpencvにやらせてみた)

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

# ======================
# モデルの読み込み
# ======================
model = YOLO("best.pt")

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="サドル検出アプリ", layout="centered")
st.title("🚲 サドル検出・カウントアプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 一時ファイルに保存（OpenCVで読み込むため）
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    image_bgr = cv2.imread(tmp_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="アップロードされた画像", use_column_width=True)

    with st.spinner("🔍 サドルを検出中..."):
        results = model.predict(
            source=tmp_path,
            save=False,
            save_txt=False,
            save_crop=False,
            conf=0.2,
            imgsz=1024,
            device='cpu'
        )

        result = results[0]
        boxes = result.boxes
        count = boxes.shape[0]

        # バウンディングボックスの描画
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(image_rgb, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    st.image(image_rgb, caption=f"🔍 検出結果：{count} 個のサドルを検出", use_column_width=True)
    st.success(f"✅ サドルの検出数：{count} 個")



