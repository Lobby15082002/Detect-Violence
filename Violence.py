import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import time
import threading
import platform
import os
from dtbase import save_event  # lưu clip vào DB

# --- tắt log OpenCV/FFmpeg để tránh overread ---

# --- beep setup ---
if platform.system() == "Windows":
    import winsound

def alert_sound():
    if platform.system() == "Windows":
        winsound.Beep(1000, 400)
        time.sleep(0.05)
        winsound.Beep(1200, 300)
    else:
        print("\a")

# --- alert ---
ALERT_COOLDOWN = 3.0  # giây giữa 2 cảnh báo
last_alert_time = 0.0
def trigger_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        return
    last_alert_time = now
    threading.Thread(target=alert_sound, daemon=True).start()

# --- CameraStream ---
class CameraStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.cap.release()

# ================= CONFIG =================
rtsp_url2 = "http://192.168.0.20:8080/video"  # MJPEG, khuyến nghị H.264 nếu camera hỗ trợ
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["NonViolence", "Violence"]
PROB_THRESHOLD = 0.5
SAVE_DIR = "record_clip"
os.makedirs(SAVE_DIR, exist_ok=True)

# delay & merge clip
NON_VIOLENCE_DELAY = 2  # giây
MERGE_DELAY = 5          # giây

# --- load model ---
MODEL_PATH = "MoBiLSTM_model.h5"
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded.")
except Exception as e:
    print("⚠️ Load lỗi:", e)
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded (compile=False).")

# --- preprocess ---
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype("float32") / 255.0
    return frame_norm

# --- save clip ---
def save_clip(frames, fps):
    if not frames:
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_DIR}/clip_{timestamp}.mp4"
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    save_event(filename)
    print(f"💾 Clip đã lưu: {filename}")

# --- MAIN ---
def main():
    cam = CameraStream(rtsp_url2)
    time.sleep(1.0)  # đợi camera ổn định

    fps = 25
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_label_name = "None"
    prob_violate = 0.0

    # --- biến toàn cục ---
    global recording, recorded_frames, non_violence_count, last_violence_time
    recording = False
    recorded_frames = []
    non_violence_count = 0
    last_violence_time = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("⚠️ Không đọc được frame.")
            time.sleep(0.01)  # tránh CPU spin
            continue

        proc = preprocess_frame(frame)
        frames_queue.append(proc)

        if len(frames_queue) == SEQUENCE_LENGTH:
            input_arr = np.expand_dims(np.array(frames_queue), axis=0)
            preds = model.predict(input_arr, verbose=0)[0]
            predicted_idx = np.argmax(preds)
            predicted_label_name = CLASSES_LIST[predicted_idx]
            prob_violate = float(preds[1]) if len(preds) > 1 else 0.0

            # --- logic ghi clip thông minh ---
            if predicted_label_name == "Violence" and prob_violate >= PROB_THRESHOLD:
                trigger_alert()
                last_violence_time = time.time()
                non_violence_count = 0
                if not recording:
                    print("🎬 Bắt đầu ghi clip (Violence phát hiện)...")
                    recording = True
                    recorded_frames = []

            else:  # NonViolence
                if recording:
                    non_violence_count += 1 / fps
                    if non_violence_count >= NON_VIOLENCE_DELAY:
                        time_since_last_violence = time.time() - last_violence_time
                        if time_since_last_violence < MERGE_DELAY:
                            non_violence_count = 0
                        else:
                            print("🛑 Dừng ghi và lưu clip sau delay...")
                            save_clip(recorded_frames, fps)
                            recorded_frames = []
                            recording = False
                            non_violence_count = 0

        # --- thêm frame nếu đang ghi ---
        if recording:
            recorded_frames.append(frame.copy())

        # --- vẽ thông tin ---
        label_text = f"{predicted_label_name} ({prob_violate:.2f})"
        color = (0, 255, 0) if predicted_label_name == "NonViolence" else (0, 0, 255)
        cv2.rectangle(frame, (0, 0), (460, 60), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        if recording:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 0, 255), 6)

        cv2.imshow("Violence Detection (Realtime)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
