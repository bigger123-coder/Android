import webrtcvad
import sounddevice as sd
import numpy as np
import time
import queue
import collections
import threading

class VADlistener:
    def __init__(self, stop_event):
        # Initiate necessary variables
        self.rate = 48000
        self.frame_duration = 20
        self.chunk_size = int(self.rate * (self.frame_duration / 1000))
        self.num_voices = 0

        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)

        self.audio_queue = queue.Queue()
        self.utterance_queue = queue.Queue()

        self.stop_event = stop_event

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Cảnh báo lỗi: {status}")

        # Converse float32 to int16 --> int16 to bytes
        audio_int16 = (indata * 32767).astype(np.int16)
        self.audio_queue.put(audio_int16.tobytes())

    def process_vad_loop(self):
        # Saving 10 audio frames to avoid missing first word
        ring_buffer = collections.deque(maxlen=10)
        audio_frames = []
        trigger = False

        silience_limit_frame = 15
        silience_count = 0

        while True:
            if self.stop_event and self.stop_event.is_set():
                break

            try:
                frame_bytes = self.audio_queue.get(timeout=1.0)

                is_speech = self.vad.is_speech(frame_bytes, self.rate)

                if not trigger:
                    ring_buffer.append((frame_bytes, is_speech))

                    self.num_voices = 0

                    # Count the number of voices
                    for _, speech in ring_buffer:
                        if speech:
                            self.num_voices += 1

                    if self.num_voices > 0.6 * ring_buffer.maxlen:
                        trigger = True
                        self.num_voices = 0
                        print ("Người dùng đang nói, trigger=True")

                        for f, _ in ring_buffer:
                            audio_frames.append(f)
                        ring_buffer.clear()

                else:
                    audio_frames.append(frame_bytes)

                    if not is_speech:
                        silience_count += 1
                    else:
                        # Reset silience_count to 0 when ppl continue speaking
                        silience_count = 0 

                    if silience_count >= silience_limit_frame:
                        print("Người dùng đã dừng nói")

                        # Gom các mảng byte lại, đổi về mảng NumPy kiểu float32 chuẩn để Whisper đọc
                        raw_bytes = b"".join(audio_frames)
                        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32767

                        self.utterance_queue.put(audio_np)

                        audio_frames.clear()
                        del raw_bytes
                        del audio_np

                        silience_count = 0 
                        trigger = False
                        break
            except queue.Empty: 
                continue

    def listen_until_utterance(self):
        # Clean RAM
        while not self.utterance_queue.empty():
            self.utterance_queue.get()

        threading.Thread(target=self.process_vad_loop, daemon=True).start()

        with sd.InputStream(samplerate=self.rate, blocksize=self.chunk_size, device=14, channels=1, dtype="float32",callback=self.audio_callback):
            while self.utterance_queue.empty():
                if self.stop_event and self.stop_event.is_set():
                    break
                time.sleep(0.1)