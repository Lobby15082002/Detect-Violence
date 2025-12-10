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

# ================= CONFIG =================
VIDEO_PATH = "D:\\Capstone project\\Data test predict\\Test\\14_13_1.avi"       
MODEL_PATH = "D:\\Capstone project\\Model\\model_2.h5"       


SEQUENCE_LENGTH = 16                 
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64   
CLASSES_LIST = ["NonViolence", "Violence"]

PROB_THRESHOLD = 0.5          # ngưỡng để báo Violence
ALERT_COOLDOWN = 3.0          # giây giữa hai cảnh báo


# =====================================================
#                  LOAD MODEL
# =====================================================
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded.")
except Exception as e:
    print("⚠️ Lỗi load model:", e)
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded (compile=False).")


# =====================================================
#                  ALERT SYSTEM
# =====================================================
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


# =====================================================
#                PREPROCESS FRAME
# =====================================================
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_norm = frame_rgb.astype("float32") / 255.0
    return frame_norm


# =====================================================
#         FINAL VIDEO RESULT (SUMMARY REPORT)
# =====================================================
def final_video_prediction(pred_list):
    if len(pred_list) == 0:
        print("Không đủ dữ liệu để kết luận video.")
        return "Unknown"

    total = len(pred_list)
    violence_count = pred_list.count("Violence")
    nonviolence_count = pred_list.count("NonViolence")

    violence_ratio = violence_count / total

    print("\n========== VIDEO RESULT ==========")
    print(f"Total predicted frames: {total}")
    print(f"Violence frames: {violence_count}")
    print(f"NonViolence frames: {nonviolence_count}")
    print(f"Violence ratio: {violence_ratio:.2f}")

    if violence_ratio >= 0.5:
        print("➡️ FINAL VIDEO PREDICTION: **VIOLENCE**")
        return "Violence"
    else:
        print("➡️ FINAL VIDEO PREDICTION: **NON-VIOLENCE**")
        return "NonViolence"


# =====================================================
#                  MAIN LOOP
# =====================================================
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 Video FPS: {fps:.2f}")

    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    pred_history = []      # lưu lại toàn bộ dự đoán
    predicted_label_name = "None"
    prob_violate = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video đã phát hết.")
            break

        # ----------------------------------------
        #           PREPROCESS FRAME
        # ----------------------------------------
        proc = preprocess_frame(frame)
        frames_queue.append(proc)

        # ----------------------------------------
        #            PREDICT SEQUENCE
        # ----------------------------------------
        if len(frames_queue) == SEQUENCE_LENGTH:
            input_arr = np.expand_dims(np.array(frames_queue), axis=0)
            preds = model.predict(input_arr, verbose=0)[0]

            predicted_idx = np.argmax(preds)
            predicted_label_name = CLASSES_LIST[predicted_idx]
            prob_violate = float(preds[1]) if len(preds) > 1 else 0.0

            pred_history.append(predicted_label_name)

            if predicted_label_name == "Violence" and prob_violate >= PROB_THRESHOLD:
                trigger_alert()

        # ----------------------------------------
        #           DRAW RESULT ON VIDEO
        # ----------------------------------------
        label_text = f"{predicted_label_name} ({prob_violate:.2f})"
        color = (0, 255, 0) if predicted_label_name == "NonViolence" else (0, 0, 255)

        cv2.rectangle(frame, (0,0), (460,60), (0,0,0), -1)
        cv2.putText(frame, label_text, (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        # khung đỏ nếu violence
        if predicted_label_name == "Violence" and prob_violate >= PROB_THRESHOLD:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (5,5), (w-5,h-5), (0,0,255), 6)

        cv2.imshow("Violence Detection (Video Test) - press Q to quit", frame)

        # giữ FPS tự nhiên
        key = cv2.waitKey(int(1000 / max(fps, 1))) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ----------------------------------------
    #     PRINT FINAL RESULT AFTER VIDEO
    # ----------------------------------------
    final_video_prediction(pred_history)


# =====================================================
#                 RUN PROGRAM
# =====================================================
if __name__ == "__main__":
    main()
