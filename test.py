#!/usr/bin/env python3
"""
Integrated Face Recognition and Hand Landmark Tracking System (Final Stable Version 4.0)
Re-architected to use a fully Tkinter-based GUI, removing all cv2.imshow calls
to ensure stability and resolve threading/state conflicts.
"""

# --- Standard Library Imports ---
import os
import pickle
import sys
import time
import threading
import queue
import tkinter as tk
from tkinter import simpledialog

# --- GUI / Input Libraries ---
from pynput.mouse import Button, Controller
from PIL import Image, ImageTk

# --- Computer Vision / AI Libraries ---
import cv2
import numpy as np
import mediapipe as mp
import tflite_runtime.interpreter as tflite
try:
    import tensorflow as tf
except ImportError:
    print("Warning: TensorFlow not found. OCR functionality will be limited.")
    tf = None

# --- Utility and Camera Imports ---
try:
    sys.path.append('handMini2')
    from utils import calc_landmark
    from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
except ImportError as e:
    print(f"✗ Critical Error: Failed to import from 'handMini2'. Details: {e}")
    sys.exit(1)

from picamera2 import Picamera2
import pyautogui

# --- Configuration ---
FACE_RECOGNITION_DURATION = 2.0
SIMILARITY_THRESHOLD = 0.75
CONFIDENCE_THRESHOLD = 0.15
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "face/Lightweight-Face-Detection.tflite")
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "face/MobileFaceNet_9925_9680.tflite")
FACE_DATABASE_FILENAME = os.path.join(BASE_DIR, "face/pi_face_database_multi.pkl")
FRAME_WIDTH, FRAME_HEIGHT = 480, 360  # Performance optimized
EAST_MODEL_PATH = os.path.join(BASE_DIR, "frozen_east_text_detection.pb")
RECOGNIZER_MODEL_PATH = os.path.join(BASE_DIR, "recognizer_model.tflite")


class SharedCamera:
    def __init__(self):
        self.picam2 = None
        self.is_running = False
        self.camera_thread = None
        self.frame_lock = threading.Lock()
        self.current_frame = None

    def initialize(self):
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
                controls={"FrameDurationLimits": (66666, 66666)}  # ~15 FPS
            )
            self.picam2.configure(config)
            self.picam2.start()
            print("✓ Shared camera initialized.")
            return True
        except Exception as e:
            print(f"✗ Camera initialization error: {e}")
            return False

    def start_camera_stream(self):
        if self.is_running: return
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _camera_loop(self):
        while self.is_running:
            try:
                frame = self.picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_bgr = cv2.flip(frame_bgr, 1)
                with self.frame_lock:
                    self.current_frame = frame_bgr
            except Exception as e:
                print(f"✗ Frame capture error: {e}")
                time.sleep(0.1)

    def get_current_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1)
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            self.picam2 = None
            print("✓ Shared camera stopped and closed.")


# FaceRecognitionManager와 HandTrackingManager는 로직 변경 없이 그대로 사용 가능

class FaceRecognitionManager:
    def __init__(self):
        self.models = {}
        self.is_loaded = False
        self.face_database = {}
        self.recognition_start_time = None
        self.current_recognized_face = None
        self.detection_input = np.zeros((1, FRAME_HEIGHT, FRAME_WIDTH, 1), dtype=np.float32)

    def load_models(self):
        if self.is_loaded: return True
        try:
            det_interp = tflite.Interpreter(model_path=DETECTION_MODEL_PATH)
            det_interp.allocate_tensors()
            self.models['detection'] = {'interpreter': det_interp, 'input': det_interp.get_input_details()}

            emb_interp = tflite.Interpreter(model_path=EMBEDDING_MODEL_PATH)
            emb_interp.allocate_tensors()
            self.models['embedding'] = {'interpreter': emb_interp, 'input': emb_interp.get_input_details()}
            
            if os.path.exists(FACE_DATABASE_FILENAME):
                with open(FACE_DATABASE_FILENAME, 'rb') as f: self.face_database = pickle.load(f)
                print(f"✓ Face DB loaded: {len(self.face_database)} people")
            
            self.is_loaded = True
            print("✓ Face models loaded.")
            return True
        except Exception as e:
            print(f"✗ Face model loading error: {e}")
            return False
            
    # ... FaceRecognitionManager의 나머지 함수들은 이전과 동일 ...
    def save_database(self):
        with open(FACE_DATABASE_FILENAME, 'wb') as f: pickle.dump(self.face_database, f)
    def detect_face(self, image):
        H0, W0 = image.shape[:2]
        resized = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self.detection_input[0, :, :, 0] = gray / 255.0
        interpreter = self.models['detection']['interpreter']
        interpreter.set_tensor(self.models['detection']['input'][0]['index'], self.detection_input)
        interpreter.invoke()
        out = interpreter.get_output_details()
        heatmap = interpreter.get_tensor(out[0]['index'])[0, :, :, 0]
        bbox_data = interpreter.get_tensor(out[1]['index'])[0]
        ys, xs = np.where(heatmap > CONFIDENCE_THRESHOLD)
        if ys.size == 0: return None
        scores = heatmap[ys, xs]; STRIDE = 8
        cx, cy = (xs + 0.5) * STRIDE, (ys + 0.5) * STRIDE
        dx1, dy1 = bbox_data[ys, xs, 0] * STRIDE, bbox_data[ys, xs, 1] * STRIDE
        dx2, dy2 = bbox_data[ys, xs, 2] * STRIDE, bbox_data[ys, xs, 3] * STRIDE
        x1, y1, x2, y2 = cx - dx1, cy - dy1, cx + dx2, cy + dy2
        boxes_pix = np.column_stack([x1, y1, x2, y2])
        boxes_pix[:, [0, 2]] *= W0 / FRAME_WIDTH; boxes_pix[:, [1, 3]] *= H0 / FRAME_HEIGHT
        bboxes_for_nms = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_pix]
        idxs = cv2.dnn.NMSBoxes(bboxes_for_nms, scores.tolist(), CONFIDENCE_THRESHOLD, 0.3)
        if idxs is not None and len(idxs) > 0:
            best_idx = idxs.flatten()[0]; x1_b, y1_b, x2_b, y2_b = boxes_pix[best_idx]
            return (int(x1_b), int(y1_b), int(x2_b - x1_b), int(y2_b - y1_b))
        return None
    def get_face_embedding(self, face_img):
        face_resized = cv2.resize(face_img, (112, 112))
        embedding_input = (cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 128.0
        interpreter = self.models['embedding']['interpreter']
        interpreter.set_tensor(self.models['embedding']['input'][0]['index'], embedding_input.reshape(1, 112, 112, 3))
        interpreter.invoke()
        return interpreter.get_tensor(interpreter.get_output_details()[0]['index']).flatten()
    def find_best_match(self, emb_check):
        best_name, max_sim = None, 0.0
        for name, emb_list in self.face_database.items():
            sims = [np.dot(emb_check, reg_emb) / (np.linalg.norm(emb_check) * np.linalg.norm(reg_emb)) for reg_emb in emb_list]
            if not sims: continue
            current_max_sim = max(sims)
            if current_max_sim > max_sim: max_sim, best_name = current_max_sim, name
        return best_name, max_sim
    def register_face(self, name, embedding):
        if name in self.face_database: self.face_database[name].append(embedding)
        else: self.face_database[name] = [embedding]
        print(f"✓ Embedding registered for '{name}'.")
    # FaceRecognitionManager 클래스 내부에 이 함수를 붙여넣으세요.

# FaceRecognitionManager 클래스 내부에 이 함수를 붙여넣으세요.

def process_frame(self, frame):
    # 1. 얼굴 탐지
    face_box = self.detect_face(frame)

    # 2. 얼굴이 감지되지 않으면, 항상 일관되게 3개의 값을 반환하고 종료합니다.
    if not face_box:
        self.recognition_start_time = None
        self.current_recognized_face = None
        return frame, None, None  # (처리된 프레임, 얼굴 없음, 결과 없음)

    # 3. 얼굴이 감지된 경우의 로직
    # 이 아래의 모든 return 구문도 3개의 값을 반환하도록 보장됩니다.
    x, y, w, h = face_box
    face_roi = frame[y:y+h, x:x+w]

    try:
        current_embedding = self.get_face_embedding(face_roi)
        if current_embedding is None:
            return frame, face_box, None # 결과 없음

        best_match, best_sim = self.find_best_match(current_embedding)

        if best_match and best_sim > SIMILARITY_THRESHOLD:
            # 아는 얼굴을 연속으로 인식하는 경우
            if self.current_recognized_face == best_match:
                if self.recognition_start_time is None:
                    self.recognition_start_time = time.time()
                
                # 인식 유지 시간이 충족되면 로그인 성공
                if time.time() - self.recognition_start_time >= FACE_RECOGNITION_DURATION:
                    return frame, face_box, (best_match, best_sim, True)
                # 시간이 아직 부족한 경우
                else:
                    rem_time = FACE_RECOGNITION_DURATION - (time.time() - self.recognition_start_time)
                    return frame, face_box, (best_match, best_sim, False, rem_time)
            # 아는 얼굴을 처음 인식한 경우
            else:
                self.current_recognized_face = best_match
                self.recognition_start_time = time.time()
                return frame, face_box, (best_match, best_sim, False, FACE_RECOGNITION_DURATION)
        # 모르는 얼굴인 경우
        else:
            self.current_recognized_face = None
            self.recognition_start_time = None
            return frame, face_box, (None, best_sim, False)

    except Exception as e:
        print(f"✗ Face recognition processing error: {e}")
        return frame, face_box, None # 에러 발생 시 결과 없음


class HandTrackingManager:
    def __init__(self, screen_size):
        self.screen_width, self.screen_height = screen_size
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = None
        self.keypoint_classifier = None
        self.is_initialized = False
        self.mouse_controller = Controller()
        self.current_mode = "Mouse Control"
        self.mode_toggle_cooldown = 0
        self.screen_capture_points = []
        self.last_finger_pos = None
        self.finger_stable_start_time = None
        self.finger_stable_threshold = 20
        self.dwell_click_duration = 1.5

    def initialize(self):
        if self.is_initialized: return True
        try:
            self.hands = self.mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.8)
            model_path = os.path.join(BASE_DIR, 'handMini2/model/keypoint_classifier/keypoint_classifier.tflite')
            self.keypoint_classifier = KeyPointClassifier(model_path=model_path)
            self.is_initialized = True
            print("✓ Hand tracking models initialized.")
            return True
        except Exception as e:
            print(f"✗ Hand tracking initialization error: {e}")
            return False
            
    # ... HandTrackingManager의 나머지 함수들은 이전과 동일 ...
    def process_frame(self, frame, app_queue):
        if not self.is_initialized: return
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmark_list = calc_landmark.calc_landmark(frame, hand_landmarks)
                pre_processed = calc_landmark.pre_process_landmark(landmark_list)
                gesture_id = self.keypoint_classifier(pre_processed)
                gesture = self.keypoint_classifier.labels[gesture_id]
                self.handle_gestures(gesture, hand_landmarks, frame, app_queue)
        self.draw_ui(frame)
        self.mode_toggle_cooldown = max(0, self.mode_toggle_cooldown - 1)
    def handle_gestures(self, gesture, hand_landmarks, frame, app_queue):
        if gesture == "Open" and self.mode_toggle_cooldown == 0:
            self.current_mode = "Screen Capture & OCR" if self.current_mode == "Mouse Control" else "Mouse Control"
            print(f"Mode changed to: {self.current_mode}")
            self.reset_mode_state(app_queue)
            self.mode_toggle_cooldown = 30
        tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        screen_pos = self.map_finger_to_screen(tip.x, tip.y, frame.shape[1], frame.shape[0])
        self.mouse_controller.position = screen_pos
        if self.current_mode == "Mouse Control":
            self.handle_dwell_click(screen_pos)
        elif self.current_mode == "Screen Capture & OCR":
            if len(self.screen_capture_points) == 0: self.handle_screen_capture_start_point(screen_pos, app_queue)
            elif len(self.screen_capture_points) == 1: app_queue.put(('update_box_drawing', self.screen_capture_points[0], screen_pos))
            if gesture == "Close" and len(self.screen_capture_points) == 1:
                self.screen_capture_points.append(screen_pos)
                app_queue.put(('perform_ocr', self.screen_capture_points))
                self.reset_mode_state(app_queue)
    def handle_dwell_click(self, pos):
        dist = np.linalg.norm(np.array(pos) - np.array(self.last_finger_pos)) if self.last_finger_pos else float('inf')
        if dist <= self.finger_stable_threshold:
            if self.finger_stable_start_time is None: self.finger_stable_start_time = time.time()
            elif time.time() - self.finger_stable_start_time >= self.dwell_click_duration:
                self.mouse_controller.click(Button.left); print(f"✓ Dwell click at {pos}")
                self.finger_stable_start_time = None
        else: self.finger_stable_start_time = None
        self.last_finger_pos = pos
    def handle_screen_capture_start_point(self, pos, app_queue):
        dist = np.linalg.norm(np.array(pos) - np.array(self.last_finger_pos)) if self.last_finger_pos else float('inf')
        if dist <= self.finger_stable_threshold:
            if self.finger_stable_start_time is None: self.finger_stable_start_time = time.time()
            elif time.time() - self.finger_stable_start_time >= self.dwell_click_duration:
                if not self.screen_capture_points:
                    self.screen_capture_points.append(pos); print(f"Capture start set: {pos}")
                    app_queue.put(('start_box_drawing',))
                self.finger_stable_start_time = None
        else: self.finger_stable_start_time = None
        self.last_finger_pos = pos
    def map_finger_to_screen(self, x, y, fw, fh):
        sx = int((x / fw) * self.screen_width); sy = int((y / fh) * self.screen_height)
        return max(0, min(sx, self.screen_width - 1)), max(0, min(sy, self.screen_height - 1))
    def draw_ui(self, frame):
        cv2.putText(frame, f"Mode: {self.current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        instruction = ""
        if self.current_mode == "Screen Capture & OCR":
            if not self.screen_capture_points: instruction = "Dwell with 'Pointer' to set START point"
            else: instruction = "Move 'Pointer' to draw box | 'Close' to CAPTURE"
        elif self.current_mode == "Mouse Control": instruction = "Dwell for 1.5s to CLICK"
        if instruction: cv2.putText(frame, instruction, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    def reset_mode_state(self, app_queue):
        self.screen_capture_points = []; self.last_finger_pos = None; self.finger_stable_start_time = None
        app_queue.put(('stop_box_drawing',))


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated Control System")
        self.root.configure(bg='#2c3e50')

        self.screen_width, self.screen_height = 1920, 1080

        self.camera = SharedCamera()
        self.face_manager = FaceRecognitionManager()
        self.hand_manager = HandTrackingManager(screen_size=(self.screen_width, self.screen_height))
        
        self.app_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False

        self.state = "idle" # idle, face_recognition, registration, hand_tracking
        self.current_user = None
        self.registration_name = None
        self.registration_angles = []
        self.registration_angle_idx = 0

        self.setup_gui()
        self.process_queue()

    def setup_gui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50'); main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Video display
        self.video_label = tk.Label(main_frame, bg="black"); self.video_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Control frame
        control_frame = tk.Frame(main_frame, bg='#34495e'); control_frame.pack(fill=tk.X, padx=10, pady=(0,10))

        self.status_label = tk.Label(control_frame, text="Status: Ready", font=('Arial', 12), fg='white', bg='#34495e'); self.status_label.pack(side=tk.LEFT, padx=10)
        self.user_label = tk.Label(control_frame, text="User: None", font=('Arial', 12), fg='white', bg='#34495e'); self.user_label.pack(side=tk.LEFT, padx=10)
        
        self.start_button = tk.Button(control_frame, text="Start System", command=self.start_system); self.start_button.pack(side=tk.RIGHT, padx=10, pady=5)
        self.logout_button = tk.Button(control_frame, text="Logout", command=self.logout); # Packed later

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_system(self):
        if self.is_running: return
        self.update_status("Initializing...")
        if not self.camera.initialize() or not self.face_manager.load_models():
            self.update_status("Initialization Failed!"); return
        
        self.is_running = True
        self.camera.start_camera_stream()
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.state = "face_recognition"
        self.update_status("Face Recognition Active")
        self.start_button.config(state=tk.DISABLED)
        self.logout_button.pack(side=tk.RIGHT, padx=10, pady=5)

    def logout(self):
        if not self.is_running: return
        self.update_status("Logging out...")
        self.state = "idle"
        self.current_user = None
        self.update_user_label()
        
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        
        self.camera.stop()
        
        self.start_button.config(state=tk.NORMAL)
        self.logout_button.pack_forget()
        self.video_label.config(image=''); self.video_label.image = None
        self.update_status("Ready")

    def processing_loop(self):
        while self.is_running:
            frame = self.camera.get_current_frame()
            if frame is None:
                time.sleep(0.01); continue

            if self.state == "face_recognition":
                frame, face_box, rec_res = self.face_manager.process_frame(frame)
                if face_box:
                    x, y, w, h = face_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if rec_res:
                        name, sim, ready, *opt = rec_res
                        if ready:
                            self.current_user = name; self.state = "hand_tracking"
                            self.root.after(0, self.on_login_success)
                        else:
                            text = f"{name}: {sim:.2f} ({opt[0]:.1f}s)" if name and opt else (f"Unknown: {sim:.2f}" if not name else f"{name}: {sim:.2f}")
                            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            elif self.state == "registration":
                self.process_registration_frame(frame)

            elif self.state == "hand_tracking":
                self.hand_manager.process_frame(frame, self.app_queue)
            
            self.update_video_feed(frame)
            time.sleep(0.01) # Give GUI thread a chance to breathe
        print("Processing loop finished.")

    def on_login_success(self):
        self.update_status(f"Hand Tracking Active")
        self.update_user_label()
        if not self.hand_manager.is_initialized:
            self.hand_manager.initialize()

    def process_registration_frame(self, frame):
        if self.registration_angle_idx >= len(self.registration_angles):
            self.face_manager.save_database()
            self.update_status(f"Registration complete for {self.registration_name}")
            self.state = "face_recognition"; return

        angle = self.registration_angles[self.registration_angle_idx]
        text = f"Show {angle} face ({self.registration_angle_idx+1}/{len(self.registration_angles)}). Press 'c' to capture."
        cv2.putText(frame, text, (10, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        face_box = self.face_manager.detect_face(frame)
        if face_box:
            x,y,w,h = face_box
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            # Key press handling is done in on_key_press
    
    def on_key_press(self, event):
        key = event.char
        if key == 'r' and self.state == "face_recognition":
            name = simpledialog.askstring("Input", "Enter name for registration:", parent=self.root)
            if name:
                self.registration_name = name
                self.registration_angles = ["Front", "Left", "Right"]
                self.registration_angle_idx = 0
                self.state = "registration"
                self.update_status(f"Registering {name}...")
        
        elif key == 'c' and self.state == "registration":
            frame = self.camera.get_current_frame()
            if frame is not None:
                face_box = self.face_manager.detect_face(frame)
                if face_box:
                    x,y,w,h = face_box
                    face_roi = frame[y:y+h, x:x+w]
                    embedding = self.face_manager.get_face_embedding(face_roi)
                    if embedding is not None:
                        self.face_manager.register_face(self.registration_name, embedding)
                        self.registration_angle_idx += 1
                        self.update_status(f"Captured {self.registration_angles[self.registration_angle_idx-1]} face.")
                    else:
                        self.update_status("Failed to get embedding. Try again.")

    def update_video_feed(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo_image = ImageTk.PhotoImage(image=pil_image)
            self.video_label.config(image=photo_image)
            self.video_label.image = photo_image
        except Exception as e:
            # This can happen during shutdown, it's safe to ignore
            pass

    def process_queue(self):
        try:
            while not self.app_queue.empty():
                msg = self.app_queue.get_nowait()
                msg_type = msg[0]
                if msg_type == 'start_box_drawing': self._start_overlay()
                elif msg_type == 'update_box_drawing': self._update_overlay(*msg[1:])
                elif msg_type == 'stop_box_drawing': self._stop_overlay()
                elif msg_type == 'perform_ocr': self._perform_ocr(*msg[1:])
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.process_queue)
            
    # ... Overlay and OCR methods ...
    def _start_overlay(self):
        if hasattr(self, 'overlay_window') and self.overlay_window.winfo_exists(): return
        try:
            screenshot_path = "/tmp/fs.png"; os.system(f"scrot -o {screenshot_path}")
            self.original_screenshot = Image.open(screenshot_path)
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
            self.overlay_window.overrideredirect(True); self.overlay_window.attributes('-topmost', True)
            self.tk_screenshot = ImageTk.PhotoImage(self.original_screenshot)
            self.overlay_canvas = tk.Canvas(self.overlay_window, cursor="crosshair")
            self.overlay_canvas.pack(fill=tk.BOTH, expand=True)
            self.overlay_canvas.create_image(0, 0, image=self.tk_screenshot, anchor='nw')
        except Exception as e: print(f"✗ Overlay error: {e}")
    def _update_overlay(self, p1, p2):
        if hasattr(self, 'overlay_canvas') and self.overlay_canvas.winfo_exists():
            self.overlay_canvas.delete("selection_rect")
            self.overlay_canvas.create_rectangle(p1[0], p1[1], p2[0], p2[1], outline='red', width=3, tags="selection_rect")
    def _stop_overlay(self):
        if hasattr(self, 'overlay_window') and self.overlay_window.winfo_exists():
            self.overlay_window.destroy()
    def _perform_ocr(self, points):
        x1, y1 = points[0]; x2, y2 = points[1]
        left, top, width, height = min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)
        if width < 10 or height < 10: return
        # A bit of a hack: Hide root, take screenshot, show root.
        self.root.withdraw()
        time.sleep(0.5) # Give window manager time to hide
        try:
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            self.root.deiconify() # Show root again
            # Now run OCR in a separate thread to not freeze the GUI
            threading.Thread(target=self.run_ocr_and_show, args=(screenshot,), daemon=True).start()
        except Exception as e:
            print(f"✗ Screenshot for OCR failed: {e}")
            self.root.deiconify()
    def run_ocr_and_show(self, pil_image):
        # This part is computationally heavy
        # For now, let's just show the captured region
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("OCR Result", img_cv)
        cv2.waitKey(0)
        cv2.destroyWindow("OCR Result")

    def update_status(self, text):
        self.status_label.config(text=f"Status: {text}")
        print(f"Status: {text}")

    def update_user_label(self):
        self.user_label.config(text=f"User: {self.current_user or 'None'}")

    def on_closing(self):
        print("Closing application...")
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        self.camera.stop()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
