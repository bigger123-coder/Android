import json
from pathlib import Path

# Directory of person_ID json file
base_dir = Path(__file__).resolve().parent
json_path = base_dir / "memory" / "Identify_face_name"/ "Person_ID.json"

# Ensure memory directory exists
json_path.parent.mkdir(parents=True, exist_ok=True)

def load_person_database():
    if not json_path.exists():
        return {}

    # If json file was empty, return an empty dict
    if json_path.stat().st_size == 0:
        return {}

    try:
        with open(json_path, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}

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

def update_person_embed_dir(person_id, embed_dir):
    database = load_person_database()
    if person_id in database:
        database[person_id]["Embed_dir"] = str(embed_dir)
        with open(json_path, "w") as file:
            json.dump(database, file, indent=4)
        
# Lưu tên người dùng vào file json kèm theo ID của người đó
def save_person_name(person_name=None, embed_dir=None):
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
    database[person_id] = {"name": person_name, "embed_dir": embed_dir}

    # Save json
    with open(json_path, "w") as file:
        json.dump(database, file, indent=4)

    # Trả về ID mới để hệ thống nhận diện
    return person_id