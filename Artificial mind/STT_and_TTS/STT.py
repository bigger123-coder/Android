from faster_whisper import WhisperModel
from pathlib import Path
import json
import re

# Directory of person_ID json file
base_dir = Path(__file__).resolve().parent
json_path = base_dir / "memory" / "Person_ID.json"

# Ensure memory directory exists
json_path.parent.mkdir(parents=True, exist_ok=True)

# Download model
model = WhisperModel("base", device="cpu", compute_type="int8")

def process_audio(audio_data):
    """Transcribe audio, with the current per-face known flags available.

    Args:
        audio_data: Audio data captured by the VAD listener.
        known_faces: Boolean values corresponding to detected faces.
    """
    puntuation_tokens = [0, 2467]

    segments, _ = model.transcribe(audio_data, language="en", beam_size=5, suppress_tokens=puntuation_tokens)

    person_name = None 

    for segment in segments:
        text = segment.text.strip()
       
        match = re.search(r"\bmy\s+name\s+is\s+(.+)", text,  re.IGNORECASE)

        if match:
            person_name = " ".join(match.group(1).split())
            return person_name

        #else: --> Khi làm TTS thì viết thêm dòng này để android báo người dùng trả lời lỗi

    return None
    

def load_person_database():
    if not json_path.exists():
        return {}

    with open(json_path, "r") as file:
        return json.load(file)

def get_next_id(database):
    if not database:
        return "00001"

    ids = []

    for person_id in database.keys():
        try:
            ids.append(int(person_id))
        except ValueError:
            continue

    if not ids:
        return "00001"

    next_id = max(ids) + 1
    return f"{next_id:05d}"
        
# Lưu tên người dùng vào file json kèm theo ID của người đó
def save_person_name(person_name):
    if not person_name:
        return None

    # Load database
    database = load_person_database()

    # Checking is current person existed
    for person_id, person_data in database.items():
        saved_name = person_data.get("name")

        if saved_name and saved_name.lower() == person_name.lower():
            print(f"Person already exists: {saved_name}({person_id})")
            return person_id

    # Create new ID for unvalid person
    person_id = get_next_id(database)

    # Save new person
    database[person_id] = {"name":person_name}

    # Save json
    with open(json_path, "w") as file:
        json.dump(database, file, indent=4)1