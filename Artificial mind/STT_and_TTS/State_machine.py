from statemachine import StateChart, State
from .VAD import VADlistener
from .STT import process_audio

import threading


class STTStateMachine(StateChart):
    # Initiate events
    IDLE = State(initial=True)
    LISTENING = State()
    PROCESSING = State()

    trigger_listening = IDLE.to(LISTENING)
    trigger_process = LISTENING.to(PROCESSING)
    trigger_idle = PROCESSING.to(IDLE)
    
    def __init__(self, stop_event=None):
        super().__init__()
        self.listener = VADlistener(stop_event=stop_event)
        self.recognized_name = None
        self._known_faces_lock = threading.Lock()
        self.known_faces = []
    
    def start_listening(self):
        self.trigger_listening()

        self.listener.listen_until_utterance()

        self.trigger_process()

    def on_enter_processing(self):
        user_name = None

        # Lấy mảng âm thanh
        if not self.listener.utterance_queue.empty():
            audio_data = self.listener.utterance_queue.get()

            user_name = process_audio(audio_data)

            # Delete audio_data after processing
            del audio_data

        self.recognized_name = user_name

        self.trigger_idle()