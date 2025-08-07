#!/usr/bin/env python3
"""
Integrated System (Final Architecture Version 10.0)
- Re-architected with a thread-safe producer-consumer pattern to prevent GUI freezes.
- Decouples Camera I/O, AI Processing, and GUI updates into separate threads.
- Ensures a non-blocking, responsive UI and stable, efficient resource utilization.
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

# --- Utility and Camera Imports ---
try:
    sys.path.append('handMini2')
    from utils import calc_landmark
    from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
except ImportError as e:
    print(f"✗ CRITICAL ERROR: Could not import from 'handMini2'. Details: {e}"); sys.exit(1)
from picamera2 import Picamera2
import pyautogui

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CAMERA_FPS = 15
FACE_RECOGNITION_DURATION = 2.0
SIMILARITY_THRESHOLD = 0.75
CONFIDENCE_THRESHOLD = 0.15
DETECTION_MODEL_PATH = os.path.join(BASE_DIR, "face/Lightweight-Face-Detection.tflite")
EMBEDDING_MODEL_PATH = os.path.join(BASE_DIR, "face/MobileFaceNet_9925_9680.tflite")
FACE_DATABASE_FILENAME = os.path.join(BASE_DIR, "face/pi_face_database_multi.pkl")
KEYPOINT_MODEL_PATH = os.path.join(BASE_DIR, 'handMini2/model/keypoint_classifier/keypoint_classifier.tflite')

class App:
    def __init__(self, root):
        self.root = root
        self.state = "idle"
        self.current_user = None

        # --- [New Architecture] Threads, Queues, and Events ---
        self.stop_event = threading.Event()
        self.camera_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.camera_thread = None
        self.ai_thread = None
        self.tflite_lock = threading.Lock()

        # --- Hardware & Libraries ---
        self.picam2 = None
        self.mouse = Controller()
        try:
            self.screen_width, self.screen_height = pyautogui.size()
        except Exception:
            self.screen_width, self.screen_height = 1920, 1080
        
        # --- Models & Managers ---
        self.face_models = {}
        self.face_db = {}
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands_model = None
        self.keypoint_classifier = None

        # --- State Variables ---
        self.current_hand_mode = "Mouse Control"
        self.recognition_start_time = None
        self.current_recognized_face = None
        self.reg_name, self.reg_angles, self.reg_angle_idx = None, [], 0
        self.capture_points = []
        self.last_finger_pos = None
        self.finger_stable_start_time = None
        self.dwell_click_duration = 1.5
        self.finger_stable_threshold = 20
        self.mode_toggle_cooldown = 0
        self.overlay_window = None

        # --- Init & Setup ---
        self.setup_gui()
        self.load_face_models()

    def setup_gui(self):
        self.root.title("Integrated System v10.2 (Final Fix)")
        self.root.configure(bg='#2c3e50')
        main_frame = tk.Frame(self.root, bg='#2c3e50'); main_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(main_frame, bg="black"); self.video_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        control_frame = tk.Frame(main_frame, bg='#34495e'); control_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        self.status_label = tk.Label(control_frame, text="Status: Ready", font=('Arial', 12), fg='white', bg='#34495e'); self.status_label.pack(side=tk.LEFT, padx=10)
        self.user_label = tk.Label(control_frame, text="User: None", font=('Arial', 12), fg='white', bg='#34495e'); self.user_label.pack(side=tk.LEFT, padx=10)
        self.start_button = tk.Button(control_frame, text="Start System", command=self.start_system); self.start_button.pack(side=tk.RIGHT, padx=10, pady=5)
        self.logout_button = tk.Button(control_frame, text="Logout", command=self.logout)
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_gui_from_queue()

    ### --- Start of Added/Fixed Helper Functions --- ###
    def update_status(self, text):
        if self.root.winfo_exists() and self.status_label:
            self.status_label.config(text=f"Status: {text}")
        print(f"Status: {text}")

    def update_user_label(self):
        if self.root.winfo_exists() and self.user_label:
            self.user_label.config(text=f"User: {self.current_user or 'None'}")

    def update_video_feed(self, frame):
        try:
            if not self.root.winfo_exists(): return
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo_image = ImageTk.PhotoImage(image=pil_image)
            self.video_label.config(image=photo_image)
            self.video_label.image = photo_image
        except (RuntimeError, tk.TclError): pass
    ### --- End of Added/Fixed Helper Functions --- ###

    # ==================== System Start/Stop Logic ====================
    def start_system(self):
        if self.camera_thread and self.camera_thread.is_alive(): return
        self.update_status("Initializing...")
        if not self.initialize_camera(): self.update_status("Camera Init Failed!"); return
        
        self.stop_event.clear()
        self.camera_thread = threading.Thread(target=self._camera_producer, daemon=True); self.camera_thread.start()
        self.ai_thread = threading.Thread(target=self._ai_consumer_producer, daemon=True); self.ai_thread.start()
        
        self.state = "face_recognition"
        self.update_status("Face Recognition Active")
        self.start_button.config(state=tk.DISABLED)
        self.logout_button.pack(side=tk.RIGHT, padx=10, pady=5)

    def logout(self):
        if not (self.camera_thread and self.camera_thread.is_alive()): return
        self.update_status("Logging out...")
        self.stop_event.set()
        self.camera_thread.join(timeout=1); self.ai_thread.join(timeout=1)
        if self.picam2: self.picam2.stop(); self.picam2.close(); self.picam2 = None
        
        while not self.camera_queue.empty(): self.camera_queue.get_nowait()
        while not self.result_queue.empty(): self.result_queue.get_nowait()
        
        self.state = "idle"; self.current_user = None
        self.update_user_label()
        self.start_button.config(state=tk.NORMAL)
        self.logout_button.pack_forget()
        black_img = ImageTk.PhotoImage(Image.new('RGB', (FRAME_WIDTH, FRAME_HEIGHT), 'black'))
        self.video_label.config(image=black_img); self.video_label.image = black_img
        self.update_status("Ready")

    def on_closing(self):
        self.stop_event.set()
        if self.camera_thread and self.camera_thread.is_alive(): self.camera_thread.join(timeout=1)
        if self.ai_thread and self.ai_thread.is_alive(): self.ai_thread.join(timeout=1)
        if self.picam2: self.picam2.close()
        self.root.destroy()
        print("Application terminated.")

    # ==================== Threading Loops (The New Architecture) ====================
    def _camera_producer(self):
        while not self.stop_event.is_set():
            try:
                frame = self.picam2.capture_array()
                frame_bgr = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 1)
                if self.camera_queue.full(): self.camera_queue.get_nowait()
                self.camera_queue.put(frame_bgr)
            except Exception as e:
                print(f"Camera capture error: {e}"); time.sleep(0.5)
        print("Camera producer stopped.")

    def _ai_consumer_producer(self):
        while not self.stop_event.is_set():
            try:
                frame = self.camera_queue.get(timeout=1)
                processed_results = {'state': self.state}
                
                if self.state == "face_recognition":
                    processed_results.update(self._process_face_ai(frame))
                elif self.state == "registration":
                    processed_results.update(self._process_reg_ai(frame))
                elif self.state == "hand_tracking":
                    if not self.hands_model: self.initialize_hand_models()
                    processed_results.update(self._process_hand_ai(frame))
                
                if self.result_queue.full(): self.result_queue.get_nowait()
                self.result_queue.put((frame, processed_results))
            except queue.Empty: continue
            except Exception as e:
                # AI 스레드에서 발생하는 에러를 여기서 잡아서 출력
                print(f"AI processing error: {e}")
        print("AI processor stopped.")

    def update_gui_from_queue(self):
        try:
            frame, results = self.result_queue.get_nowait()
            
            if results['state'] == "face_recognition" and results.get("box") is not None:
                self._draw_face_ui(frame, results)
            elif results['state'] == "registration" and results.get("box") is not None:
                self._draw_reg_ui(frame, results)
            elif results['state'] == "hand_tracking" and results.get("landmarks") is not None:
                self._draw_hand_ui(frame, results)

            self.update_video_feed(frame)
        except queue.Empty:
            pass
        finally:
            self.root.after(33, self.update_gui_from_queue)

    def initialize_camera(self):
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
                controls={"FrameDurationLimits": (int(1e6/CAMERA_FPS), int(1e6/CAMERA_FPS))}
            )
            self.picam2.configure(config); self.picam2.start()
            return True
        except Exception as e:
            print(f"✗ Camera initialization error: {e}"); return False

    # ==================== AI Processing Sub-routines ====================
    def load_face_models(self):
        # ... (이 함수는 변경 없음)
        try:
            self.face_models['detection'] = tflite.Interpreter(model_path=DETECTION_MODEL_PATH)
            self.face_models['detection'].allocate_tensors()
            self.face_models['embedding'] = tflite.Interpreter(model_path=EMBEDDING_MODEL_PATH)
            self.face_models['embedding'].allocate_tensors()
            if os.path.exists(FACE_DATABASE_FILENAME):
                with open(FACE_DATABASE_FILENAME, 'rb') as f: self.face_db = pickle.load(f)
            print("✓ Face models and DB loaded.")
        except Exception as e: print(f"✗ Face model loading error: {e}")

    def _process_face_ai(self, frame):
        face_box = self._detect_face(frame)
        
        # ### 최종 resize 에러 해결책 ###
        # 박스가 없거나, 너비/높이가 0이거나, 경계선을 벗어나는 모든 경우를 완벽하게 차단
        if not face_box:
            self.recognition_start_time = None; self.current_recognized_face = None
            return {"box": None, "text": None}
            
        x, y, w, h = face_box
        if w <= 0 or h <= 0 or x < 0 or y < 0 or (x + w) > frame.shape[1] or (y + h) > frame.shape[0]:
            return {"box": None, "text": None} # 유효하지 않은 박스는 무시

        face_roi = frame[y:y+h, x:x+w]
        embedding = self._get_face_embedding(face_roi)
        if embedding is None: return {"box": face_box, "text": None}
        
        best_name, max_sim = self._find_best_match(embedding)
        text = None
        if best_name and max_sim > SIMILARITY_THRESHOLD:
            if self.current_recognized_face != best_name:
                self.current_recognized_face = best_name; self.recognition_start_time = time.time()
            elif (time.time() - self.recognition_start_time) >= FACE_RECOGNITION_DURATION:
                self.current_user = best_name; self.state = "hand_tracking"
                self.root.after(0, self.on_login_success)
            
            if self.recognition_start_time:
                rem_time = FACE_RECOGNITION_DURATION - (time.time() - self.recognition_start_time)
                text = f"{best_name}: {max_sim:.2f} ({rem_time:.1f}s)"
        else: self.current_recognized_face = None; text = f"Unknown: {max_sim:.2f}"
        return {"box": face_box, "text": text}

    def _process_reg_ai(self, frame):
        face_box = self._detect_face(frame)
        return {"box": face_box}

    def _process_hand_ai(self, frame):
        if not self.hands_model: return {"landmarks": None, "gesture": None}
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_model.process(img_rgb)
        if not results.multi_hand_landmarks: return {"landmarks": None, "gesture": None}
        
        hand_landmarks = results.multi_hand_landmarks[0]
        lm_list = calc_landmark.calc_landmark(frame, hand_landmarks)
        pre_processed_lm = calc_landmark.pre_process_landmark(lm_list)
        with self.tflite_lock:
            gesture_id = self.keypoint_classifier(pre_processed_lm)
        gesture = self.keypoint_classifier.labels[gesture_id]
        return {"landmarks": hand_landmarks, "gesture": gesture}

    # ==================== UI Drawing Sub-routines ====================
    def _draw_face_ui(self, frame, results):
        if results.get("box"):
            x, y, w, h = results["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if results.get("text"):
                cv2.putText(frame, results["text"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def _draw_reg_ui(self, frame, results):
        if self.reg_angle_idx < len(self.reg_angles):
            angle = self.reg_angles[self.reg_angle_idx]
            text = f"Show {angle} face. Press 'c' to capture."
            cv2.putText(frame, text, (10, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if results.get("box"):
            x,y,w,h = results["box"]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    def _draw_hand_ui(self, frame, results):
        if results.get("landmarks"):
            self.mp_drawing.draw_landmarks(frame, results["landmarks"], self.mp_hands.HAND_CONNECTIONS)
            self.handle_hand_logic(results["gesture"], results["landmarks"], frame)
        
        mode_text = f"Mode: {self.current_hand_mode}"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        instruction = ""
        if self.current_hand_mode == "Screen Capture & OCR":
            instruction = "Dwell 'Pointer' to START" if not self.capture_points else "Move 'Pointer' | 'Close' to CAPTURE"
        else: instruction = "'Pointer' to move, 'Close' to Click"
        cv2.putText(frame, instruction, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # ==================== Other Helper Functions ====================
    def on_login_success(self):
        self.update_status("Hand Tracking Active")
        self.update_user_label()
        if not self.hands_model: self.initialize_hand_models()
    
    def initialize_hand_models(self):
        try:
            self.hands_model = self.mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.8)
            self.keypoint_classifier = KeyPointClassifier(model_path=KEYPOINT_MODEL_PATH)
            print("✓ Hand tracking models initialized.")
        except Exception as e: print(f"✗ Hand tracking init error: {e}")

    def on_key_press(self, event):
        key = event.char.lower()
        if key == 'r' and self.state == "face_recognition":
            name = simpledialog.askstring("Register Face", "Enter name:", parent=self.root)
            if name:
                self.reg_name = name; self.reg_angles = ["Front", "Left", "Right"]; self.reg_angle_idx = 0
                self.state = "registration"; self.update_status(f"Registering {self.reg_name}...")
        
        elif key == 'c' and self.state == "registration":
            threading.Thread(target=self._capture_registration_face, daemon=True).start()

    def _capture_registration_face(self):
        frame = self.camera_queue.get(timeout=1)
        if frame is None: return
        face_box = self._detect_face(frame)
        if face_box and face_box[2] > 0 and face_box[3] > 0:
            x,y,w,h = face_box
            face_roi = frame[y:y+h, x:x+w]
            embedding = self._get_face_embedding(face_roi)
            if embedding is not None:
                if self.reg_name in self.face_db: self.face_db[self.reg_name].append(embedding)
                else: self.face_db[self.reg_name] = [embedding]
                
                if self.reg_angle_idx < len(self.reg_angles):
                    self.update_status(f"Captured {self.reg_angles[self.reg_angle_idx]} face.")
                    self.reg_angle_idx += 1
                
                if self.reg_angle_idx >= len(self.reg_angles):
                    with open(FACE_DATABASE_FILENAME, 'wb') as f: pickle.dump(self.face_db, f)
                    self.update_status(f"Registration for {self.reg_name} complete!")
                    self.state = "face_recognition"
            else: self.update_status("Capture failed. Try again.")

    def _detect_face(self, frame):
        with self.tflite_lock:
            H0, W0 = frame.shape[:2]
            resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            det_input = (gray.astype(np.float32) / 255.0).reshape(1, FRAME_HEIGHT, FRAME_WIDTH, 1)
            interpreter = self.face_models['detection']
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], det_input); interpreter.invoke()
            out = interpreter.get_output_details()
            heatmap = interpreter.get_tensor(out[0]['index'])[0, :, :, 0]
            bbox_data = interpreter.get_tensor(out[1]['index'])[0]
        
        ys, xs = np.where(heatmap > CONFIDENCE_THRESHOLD)
        if ys.size == 0: return None
        scores = heatmap[ys, xs]; STRIDE = 8
        cx,cy = (xs + 0.5) * STRIDE, (ys + 0.5) * STRIDE
        x1,y1,x2,y2 = cx-bbox_data[ys,xs,0]*STRIDE, cy-bbox_data[ys,xs,1]*STRIDE, cx+bbox_data[ys,xs,2]*STRIDE, cy+bbox_data[ys,xs,3]*STRIDE
        boxes_pix = np.column_stack([x1, y1, x2, y2])
        boxes_pix[:, [0, 2]] *= W0 / FRAME_WIDTH; boxes_pix[:, [1, 3]] *= H0 / FRAME_HEIGHT
        bboxes_for_nms = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_pix]
        idxs = cv2.dnn.NMSBoxes(bboxes_for_nms, scores.tolist(), CONFIDENCE_THRESHOLD, 0.3)
        if idxs is not None and len(idxs) > 0:
            x1_b, y1_b, x2_b, y2_b = boxes_pix[idxs.flatten()[0]]
            return (int(x1_b), int(y1_b), int(x2_b - x1_b), int(y2_b - y1_b))
        return None

    def _get_face_embedding(self, face_roi):
        with self.tflite_lock:
            face_resized = cv2.resize(face_roi, (112, 112))
            emb_input = (cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 128.0
            interpreter = self.face_models['embedding']
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], emb_input.reshape(1, 112, 112, 3)); interpreter.invoke()
            return interpreter.get_tensor(interpreter.get_output_details()[0]['index']).flatten()

    def _find_best_match(self, embedding):
        best_name, max_sim = None, 0.0
        norm_emb = np.linalg.norm(embedding)
        if norm_emb == 0: return None, 0.0
        for name, emb_list in self.face_db.items():
            sims = [np.dot(embedding, reg_emb) / (norm_emb * np.linalg.norm(reg_emb)) for reg_emb in emb_list]
            if sims and max(sims) > max_sim:
                max_sim = max(sims); best_name = name
        return best_name, max_sim

    def handle_hand_logic(self, gesture, hand_landmarks, frame):
        if gesture == "Open" and self.mode_toggle_cooldown == 0:
            self.current_hand_mode = "Screen Capture & OCR" if self.current_hand_mode == "Mouse Control" else "Mouse Control"
            print(f"Hand mode: {self.current_hand_mode}")
            self.reset_capture_state(); self.mode_toggle_cooldown = 30
        
        tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        screen_pos = self.map_finger_to_screen(tip.x, tip.y)
        self.mouse.position = screen_pos

        if self.current_hand_mode == "Mouse Control" and gesture == "Close":
             self.mouse.click(Button.left)
        elif self.current_hand_mode == "Screen Capture & OCR":
            if len(self.capture_points) == 0 and gesture == "Pointer":
                self.handle_capture_start_point(screen_pos)
            elif len(self.capture_points) == 1:
                self._update_overlay(self.capture_points[0], screen_pos)
                if gesture == "Close":
                    self.capture_points.append(screen_pos)
                    self.perform_ocr()
                    self.reset_capture_state()
    
    def handle_capture_start_point(self, pos):
        dist = np.linalg.norm(np.array(pos) - (self.last_finger_pos if self.last_finger_pos else pos))
        if dist <= self.finger_stable_threshold:
            if self.finger_stable_start_time is None: self.finger_stable_start_time = time.time()
            elif time.time() - self.finger_stable_start_time >= self.dwell_click_duration:
                if not self.capture_points:
                    self.capture_points.append(pos); print(f"Capture start set: {pos}")
                    self._start_overlay()
                self.finger_stable_start_time = None
        else: self.finger_stable_start_time = None
        self.last_finger_pos = pos

    def reset_capture_state(self):
        self.capture_points = []; self.last_finger_pos = None; self.finger_stable_start_time = None
        self._stop_overlay()

    def map_finger_to_screen(self, x, y):
        sx = int(x * self.screen_width); sy = int(y * self.screen_height)
        return max(0, min(sx, self.screen_width - 1)), max(0, min(sy, self.screen_height - 1))
    
    def _start_overlay(self):
        if self.overlay_window and self.overlay_window.winfo_exists(): return
        try:
            self.root.withdraw(); time.sleep(0.3)
            screenshot = pyautogui.screenshot(); self.root.deiconify()
            
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
            self.overlay_window.overrideredirect(True); self.overlay_window.attributes('-topmost', True)
            self.tk_screenshot = ImageTk.PhotoImage(screenshot)
            canvas = tk.Canvas(self.overlay_window, cursor="crosshair")
            canvas.pack(fill=tk.BOTH, expand=True)
            canvas.create_image(0, 0, image=self.tk_screenshot, anchor='nw')
            self.overlay_canvas = canvas
        except Exception as e: 
            print(f"✗ Overlay error: {e}"); self.root.deiconify()

    def _update_overlay(self, p1, p2):
        if hasattr(self, 'overlay_canvas') and self.overlay_canvas and self.overlay_canvas.winfo_exists():
            self.overlay_canvas.delete("selection_rect")
            self.overlay_canvas.create_rectangle(p1[0], p1[1], p2[0], p2[1], outline='red', width=3, tags="selection_rect")

    def _stop_overlay(self):
        if self.overlay_window and self.overlay_window.winfo_exists(): self.overlay_window.destroy()
        self.overlay_window = None

    def perform_ocr(self):
        if len(self.capture_points) != 2: return
        p1, p2 = self.capture_points
        left,top,width,height = min(p1[0],p2[0]), min(p1[1],p2[1]), abs(p1[0]-p2[0]), abs(p1[1]-p2[1])
        if width < 10 or height < 10: return
        print(f"OCR would run on region: L:{left}, T:{top}, W:{width}, H:{height}")
        # OCR logic can be implemented here, running in a separate thread if heavy

def main():
    root = tk.Tk()
    app = App(root)
    root.minsize(720, 560) 
    root.mainloop()

if __name__ == "__main__":
    main()
