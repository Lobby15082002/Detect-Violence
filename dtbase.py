# database.py
from pymongo import MongoClient
import datetime

# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["violence_detection"]
collection = db["events"]

def save_event(video_path, event_type="Violence"):
    """
    Lưu thông tin video vào database
    """
    data = {
        "event_type": event_type,
        "video_path": video_path,
        "timestamp": datetime.datetime.now().isoformat()
    }
    collection.insert_one(data)
    print(f"✅ Đã lưu vào database: {video_path}")
