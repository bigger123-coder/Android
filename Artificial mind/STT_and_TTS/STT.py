from faster_whisper import WhisperModel

# Download model
model = WhisperModel("base", device="cpu", compute_type="int8")

def process_audio(audio_data):
    segments, _ = model.transcribe(audio_data, beam_size=5)

    for segment in segments:
        if "my name is" in segment.text.lower():
            parts = segment.text.split("is", 1)

            if len(parts) > 1:
                person_name = parts[1].split()

        return person_name