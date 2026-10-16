from faster_whisper import WhisperModel
from pathlib import Path
from scipy.signal import resample_poly
from Artificial_mind.STT_and_TTS.Database_manager import save_person_name

import threading
import numpy as np
import re

# Khởi tạo biến model ban đầu là None để tránh tải nặng lúc khởi động ứng dụng
_model = None
_model_lock = threading.Lock()

def get_whisper_model():
    """Hàm tải và trả về model theo cơ chế Lazy Loading an toàn cho luồng (thread-safe)"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("Đang tải mô hình Whisper lần đầu tiên...")
                _model = WhisperModel("base", device="cpu", compute_type="int8")
    return _model

def process_audio(audio_data):
    print("Hàm process_audio đã được kích hoạt")
    audio_16k = resample_poly(audio_data, up=1, down=3).astype(np.float32)

    puntuation_tokens = [0, 2467]

    # Lấy model thông qua hàm lazy loading khi thực sự cần xử lý âm thanh
    model = get_whisper_model()
    segments, _ = model.transcribe(audio_16k, task="translate", beam_size=5, suppress_tokens=puntuation_tokens)

    person_name = None 

    for segment in segments:
        # Debug
        # print(f"Phát hiện câu: {segment}")

        text = segment.text.strip()
       
        match = re.search(r"\bmy\s+name\s+is\s+(.+)", text, re.IGNORECASE)

        if match:
            # Tách toàn bộ các từ nằm sau cụm "my name is"
            words = match.group(1).split()
            name_words = []
            
            # Danh sách các từ nối hoặc từ khóa bắt đầu thông tin phụ phía sau tên
            stop_words = {"and", "i", "im", "i'm", "who", "am", "my", "is", "the", "a", "an", "years", "old"}
            
            for word in words:
                # Làm sạch từ khỏi các dấu câu (dấu phẩy, chấm,...)
                clean_word = re.sub(r'[^\w]', '', word)
                
                # Nếu gặp từ dừng, từ trống, hoặc chứa chữ số (như số tuổi 18) thì dừng lại ngay
                if not clean_word or clean_word.lower() in stop_words or any(char.isdigit() for char in clean_word):
                    break
                    
                name_words.append(clean_word)
                
                # Giới hạn tối đa 3 từ cho tên người (phòng trường hợp câu nói không chứa stop word)
                if len(name_words) >= 3:
                    break

            if name_words:
                person_name = " ".join(name_words)
                save_person_name(person_name)
                return person_name