from statemachine import StateChart, State
from VAD import VADlistener
from STT import process_audio


class STTStateMachine(StateChart):
    # Initiate events
    IDLE = State(initial=True)
    LISTENING = State()
    PROCESSING = State(final=True)

    STT_cycle = (IDLE.to(LISTENING) | LISTENING.to(PROCESSING) | PROCESSING.to(IDLE))

    def __init__(self):
        super().__init__()
        self.listener = VADlistener()
    
    def on_listen(self):
        self.listener.listen_until_utterance()

        self.PROCESSING()

    def after_listen(self):
        # Lấy mảng âm thanh
        if not self.listener.utterance_queue.empty():
            audio_data = self.listener.utterance_queue.get()

            process_audio(audio_data)

            # Delete audio_data after processing
            del audio_data

        self.IDLE()