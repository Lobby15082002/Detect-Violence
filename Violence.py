import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import time
import threading
import platform

# --- beep setup ---
if platform.system() == "Windows":
    import winsound

# xử lý video
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
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        self.cap.release()
# ================= CONFIG =================
rtsp_url2 = "http://192.168.1.105:8080/video"
VIDEO_PATH = CameraStream(rtsp_url2)       
MODEL_PATH = "MoBiLSTM_model.h5"     
SEQUENCE_LENGTH = 16                 
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64   
CLASSES_LIST = ["NonViolence", "Violence"]
PROB_THRESHOLD = 0.5
ALERT_COOLDOWN = 3.0  # giây giữa 2 cảnh báo


# --- load model ---
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded.")
except Exception as e:
    print("⚠️ Load lỗi:", e)
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded (compile=False).")

# --- alert ---
last_alert_time = 0.0
def alert_sound():
    if platform.system() == "Windows":
        winsound.Beep(1000, 400)
        time.sleep(0.05)
        winsound.Beep(1200, 300)
    else:
        print("\a")

def trigger_alert():
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_COOLDOWN:
        return
    last_alert_time = now
    threading.Thread(target=alert_sound, daemon=True).start()

# --- preprocess ---
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype("float32") / 255.0
    return frame_norm

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Không mở được video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Video FPS: {fps:.2f}")

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_label_name = "None"
    prob_violate = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video đã phát hết.")
            break
        frame_count += 1

        # Preprocess frame
        proc = preprocess_frame(frame)
        frames_queue.append(proc)

        # Predict nếu đủ sequence
        if len(frames_queue) == SEQUENCE_LENGTH:
            input_arr = np.expand_dims(np.array(frames_queue), axis=0)
            preds = model.predict(input_arr, verbose=0)[0]
            predicted_idx = np.argmax(preds)
            predicted_label_name = CLASSES_LIST[predicted_idx]
            prob_violate = float(preds[1]) if len(preds) > 1 else 0.0

            # Nếu là bạo lực → cảnh báo ngay
            if predicted_label_name == "Violence" and prob_violate >= PROB_THRESHOLD:
                trigger_alert()

        # --- vẽ thông tin lên video ---
        label_text = f"{predicted_label_name} ({prob_violate:.2f})"
        color = (0, 255, 0) if predicted_label_name == "NonViolence" else (0, 0, 255)
        cv2.rectangle(frame, (0,0), (460,60), (0,0,0), -1)
        cv2.putText(frame, label_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # Nếu là violence → khung đỏ
        if predicted_label_name == "Violence" and prob_violate >= PROB_THRESHOLD:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (5,5), (w-5,h-5), (0,0,255), 6)

        cv2.imshow("Violence Detection (Video Test) - press q to quit", frame)

        # delay theo FPS gốc để giả lập realtime
        key = cv2.waitKey(int(1000 / max(fps, 1))) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
