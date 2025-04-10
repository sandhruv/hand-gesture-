import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False
    print("Note: scikit-learn not found. Using basic gesture detection.")

# ================== Base Class ==========================
class GestureRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Gesture Recognition System")
        self.root.configure(bg="#1c1c1c")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils

        # Camera Setup
        self.cap = None
        self.running = False

        # UI
        self.setup_ui()

        if not ML_ENABLED:
            self.status_var.set("⚠️ ML Mode: Disabled (Install scikit-learn)")

    def setup_ui(self):
        # Fonts and Styles
        font_heading = ('Helvetica', 22, 'bold')
        font_button = ('Helvetica', 12, 'bold')
        font_label = ('Helvetica', 14, 'bold')

        # Video Display Frame
        self.video_label = tk.Label(self.root, bg="black", bd=3, relief="ridge")
        self.video_label.place(x=150, y=60, width=600, height=400)

        # Control Buttons
        self.start_btn = tk.Button(self.root, text="🚀 Start", font=font_button, bg="#00FFAB", fg="black",
                                   activebackground="#32e0c4", command=self.start_camera, relief='raised', bd=2)
        self.start_btn.place(x=250, y=480, width=100, height=40)

        self.stop_btn = tk.Button(self.root, text="🛑 Stop", font=font_button, bg="#FF6464", fg="white",
                                  activebackground="#ff4d6d", command=self.stop_camera, relief='raised', bd=2,
                                  state=tk.DISABLED)
        self.stop_btn.place(x=550, y=480, width=100, height=40)

        # Gesture Display
        self.gesture_var = tk.StringVar(value="Gesture: None")
        self.gesture_label = tk.Label(self.root, textvariable=self.gesture_var, font=font_heading,
                                      fg="#00FFD1", bg="#1c1c1c")
        self.gesture_label.place(x=330, y=530)

        # Status Bar
        self.status_var = tk.StringVar(value="🔵 System Ready")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=font_label,
                                     bg="#121212", fg="#00FFAB", relief="sunken", anchor="w")
        self.status_label.place(x=0, y=670, relwidth=1, height=30)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not access the webcam.")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("🟢 Camera Started")
        self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.video_label.config(image='')
        self.status_var.set("🔴 Camera Stopped")

    def count_fingers(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_dips = [6, 10, 14, 18]
        count = 0
        for tip, dip in zip(finger_tips, finger_dips):
            if landmarks[tip].y < landmarks[dip].y:
                count += 1

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        wrist = landmarks[0]

        if thumb_tip.x < thumb_ip.x:
            if thumb_tip.x < wrist.x:
                count += 1
        else:
            if thumb_tip.x > wrist.x:
                count += 1

        return count

    def detect_gesture(self, finger_count):
        gestures = {
            0: "Fist 👊",
            1: "Point 👆",
            2: "Victory ✌️",
            3: "Three 🤟",
            4: "Four 🖖",
            5: "Open 🖐️"
        }
        return gestures.get(finger_count, "🤖 Unknown")

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                count = self.count_fingers(hand_landmarks.landmark)
                gesture = self.detect_gesture(count)
                self.gesture_var.set(f"Gesture: {gesture}")
        else:
            self.gesture_var.set("Gesture: No Hand ✋")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(15, self.update_frame)

# ================== ML Version ==========================
class MLGestureRecognizer(GestureRecognizer):
    def __init__(self, root):
        super().__init__(root)
        if ML_ENABLED:
            self.train_model()
        else:
            self.status_var.set("⚠️ ML Disabled. Using basic gesture logic.")

    def train_model(self):
        self.status_var.set("⚙️ Training ML Model...")
        self.model = RandomForestClassifier()
        X = np.random.rand(20, 21 * 3)  # Dummy data
        y = np.random.randint(0, 6, size=20)
        self.model.fit(X, y)
        self.status_var.set("✅ ML Model Ready")

# ================== Main Entry ==========================
if __name__ == "__main__":
    root = tk.Tk()
    app = MLGestureRecognizer(root) if ML_ENABLED else GestureRecognizer(root)
    root.mainloop()
