import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# โหลดโมเดลและ tokenizer
model = joblib.load("model.pkl")
tokenizer = joblib.load("tokenizer.pkl")
maxlen = 100  # ต้องใช้ค่า maxlen เดียวกับตอนเทรน

# emoji map ตาม label
emoji_map = {
    'emoticons': {
        0: "😜",
        1: "📸",
        2: "😍",
        3: "😂",
        4: "😉",
        5: "🎄",
        6: "📷",
        7: "🔥",
        8: "😘",
        9: "❤",
        10: "😁",
        11: "🇺🇸",
        12: "☀",
        13: "✨",
        14: "💙",
        15: "💕",
        16: "😎",
        17: "😊",
        18: "💜",
        19: "💯"
    }
}

st.title("🔍 Text Classification Web App with Emoji")
st.write("กรอกข้อความหลายบรรทัด แล้วระบบจะทำนายและแสดง emoji")

user_input = st.text_area("✏️ ป้อนข้อความ (หนึ่งบรรทัดต่อหนึ่งตัวอย่าง):", height=200)

if st.button("วิเคราะห์"):
    if user_input.strip() == "":
        st.warning("กรุณาใส่ข้อความก่อน")
    else:
        test = user_input.strip().split("\n")
        test_seq = tokenizer.texts_to_sequences(test)
        Xtest = pad_sequences(test_seq, maxlen=maxlen, padding='post', truncating='post')

        y_pred = model.predict(Xtest)
        y_pred = np.argmax(y_pred, axis=1)  # ปรับตาม output ของโมเดล

        st.subheader("📊 ผลลัพธ์")
        for i in range(len(test)):
            emoji_icon = emoji_map['emoticons'].get(y_pred[i], "❓")
            st.write(f"➡️ {test[i]} → {emoji_icon}")