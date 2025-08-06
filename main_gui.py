#!/usr/bin/env python3
"""
Integrated Face Recognition and Hand Landmark Tracking System (Final Stable Version 3.0)
Resolves state management, threading, and re-login bugs for robust operation.
"""

# --- Standard Library Imports ---
import os
import pickle
import sys
import time
import threading
import queue

# --- GUI / Input Libraries ---
import tkinter as tk
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
    print(f"✗ Critical Error: Failed to import from 'handMini2'. Ensure the folder exists. Details: {e}")
    sys.exit(1)

from picamera2 import Picamera2
import pyautogui

# --- Configuration ---
# Face Recognition
CONFIDENCE_THRESHOLD = 0.15
SIMILARITY_THRESHOLD = 0.75
FACE_RECOGNITION_DURATION = 2.0
DETECTION_MODEL_PATH = "face/Lightweight-Face-Detection.tflite"
EMBEDDING_MODEL_PATH = "face/MobileFaceNet_9925_9680.tflite"
FACE_DATABASE_FILENAME = "face/pi_face_database_multi.pkl"
# Camera
FRAME_WIDTH, FRAME_HEIGHT = 480, 360 # Performance optimized
# OCR Models
EAST_MODEL_PATH = "frozen_east_text_detection.pb"
RECOGNIZER_MODEL_PATH = "recognizer_model.tflite"


class SharedCamera:
    # (이 클래스는 수정사항이 없습니다)
    def __init__(self):
        self.picam2, self.is_initialized, self.current_frame = None, False, None
        self.frame_lock, self.is_running, self.camera_thread = threading.Lock(), False, None
    def initialize(self):
        if self.is_initialized: return True
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
                controls={"FrameDurationLimits": (66666, 66666)} # ~15 FPS
            )
            self.picam2.configure(config)
            self.picam2.start()
            print("✓ Shared camera initialized successfully")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"✗ Camera initialization error: {e}")
            return False
    def start_camera_stream(self):
        if not self.is_initialized or self.is_running: return
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
    def _camera_loop(self):
        while self.is_running:
            try:
                frame = self.picam2.capture_array()
                frame = cv2.flip(frame, 1)
                with self.frame_lock: self.current_frame = frame
            except Exception as e:
                print(f"✗ Frame capture error: {e}"); time.sleep(0.1)
    def get_current_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    def stop_camera_stream(self):
        self.is_running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1)
    def close(self):
        self.stop_camera_stream()
        if self.picam2: self.picam2.close(); self.picam2 = None
        self.is_initialized = False; print("✓ Shared camera closed.")


class FaceRecognitionManager:
    # (이 클래스는 수정사항이 없습니다)
    def __init__(self, camera):
        self.camera = camera; self.models = {}; self.is_loaded = False
        self.detection_input = np.zeros((1, 480, 640, 1), dtype=np.float32)
        self.embedding_input = np.zeros((1, 112, 112, 3), dtype=np.float32)
        self.face_database = {}; self.recognition_start_time = None; self.current_recognized_face = None
        self.load_database()
    def load_models(self):
        if self.is_loaded: return True
        try:
            detection_interpreter = tflite.Interpreter(model_path=DETECTION_MODEL_PATH); detection_interpreter.allocate_tensors()
            self.models['detection'] = {'interpreter': detection_interpreter, 'input': detection_interpreter.get_input_details(), 'output': detection_interpreter.get_output_details()}
            embedding_interpreter = tflite.Interpreter(model_path=EMBEDDING_MODEL_PATH); embedding_interpreter.allocate_tensors()
            self.models['embedding'] = {'interpreter': embedding_interpreter, 'input': embedding_interpreter.get_input_details(), 'output': embedding_interpreter.get_output_details()}
            self.is_loaded = True; print("✓ Face models loaded"); return True
        except Exception as e:
            print(f"✗ Face model loading error: {e}"); return False
    def load_database(self):
        if os.path.exists(FACE_DATABASE_FILENAME):
            try:
                with open(FACE_DATABASE_FILENAME, 'rb') as f: self.face_database = pickle.load(f)
                total = sum(len(v) for v in self.face_database.values()); print(f"✓ Face DB loaded: {len(self.face_database)} people, {total} embeddings")
            except Exception as e:
                print(f"✗ Error loading face DB: {e}"); self.face_database = {}
    def save_database(self):
        try:
            with open(FACE_DATABASE_FILENAME, 'wb') as f: pickle.dump(self.face_database, f)
            print(f"✓ Face DB saved to {FACE_DATABASE_FILENAME}")
        except Exception as e: print(f"✗ Error saving face DB: {e}")
    def detect_face(self, image):
        if not self.is_loaded or image is None: return None
        interpreter = self.models['detection']['interpreter']
        H0, W0 = image.shape[:2]
        resized = cv2.resize(image, (640, 480))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self.detection_input[0, :, :, 0] = gray.astype(np.float32) / 255.0
        interpreter.set_tensor(self.models['detection']['input'][0]['index'], self.detection_input); interpreter.invoke()
        heatmap = interpreter.get_tensor(self.models['detection']['output'][0]['index'])[0, :, :, 0]
        bbox_data = interpreter.get_tensor(self.models['detection']['output'][1]['index'])[0]
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
            x1_int, y1_int, x2_int, y2_int = int(x1_b), int(y1_b), int(x2_b), int(y2_b)
            w_b, h_b = x2_int - x1_int, y2_int - y1_int
            if x1_int >= 0 and y1_int >= 0 and x2_int < W0 and y2_int < H0 and w_b > 0 and h_b > 0:
                return (x1_int, y1_int, w_b, h_b)
        return None
    def get_face_embedding(self, face_img):
        if not self.is_loaded: return None
        interpreter = self.models['embedding']['interpreter']
        face_resized = cv2.resize(face_img, (112, 112))
        self.embedding_input[0] = (cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 128.0
        interpreter.set_tensor(self.models['embedding']['input'][0]['index'], self.embedding_input); interpreter.invoke()
        return interpreter.get_tensor(self.models['embedding']['output'][0]['index']).flatten()
    def find_best_match(self, emb_check):
        if emb_check is None or not self.face_database: return None, 0.0
        best_name, max_sim = None, 0.0
        for name, emb_list in self.face_database.items():
            sims = [np.dot(emb_check, reg_emb) / (np.linalg.norm(emb_check) * np.linalg.norm(reg_emb)) for reg_emb in emb_list]
            current_max_sim = max(sims) if sims else 0.0
            if current_max_sim > max_sim: max_sim, best_name = current_max_sim, name
        return best_name, max_sim
    def register_face(self, name, embedding):
        if not name or embedding is None: return False
        if name in self.face_database: self.face_database[name].append(embedding)
        else: self.face_database[name] = [embedding]
        print(f"✓ Embedding registered for '{name}'. Total: {len(self.face_database[name])}"); return True
    def process_frame(self, frame):
        if frame is None: return frame, None, None
        face_box = self.detect_face(frame)
        if not face_box:
            self.recognition_start_time, self.current_recognized_face = None, None; return frame, None, None
        x, y, w, h = face_box; face_roi = frame[y:y+h, x:x+w]
        try:
            current_embedding = self.get_face_embedding(face_roi)
            if current_embedding is not None:
                best_match, best_sim = self.find_best_match(current_embedding)
                if best_match and best_sim > SIMILARITY_THRESHOLD:
                    if self.current_recognized_face == best_match:
                        if self.recognition_start_time is None: self.recognition_start_time = time.time()
                        if time.time() - self.recognition_start_time >= FACE_RECOGNITION_DURATION:
                            return frame, face_box, (best_match, best_sim, True)
                        else:
                            rem_time = FACE_RECOGNITION_DURATION - (time.time() - self.recognition_start_time)
                            return frame, face_box, (best_match, best_sim, False, rem_time)
                    else:
                        self.current_recognized_face, self.recognition_start_time = best_match, time.time()
                        return frame, face_box, (best_match, best_sim, False, FACE_RECOGNITION_DURATION)
                else:
                    self.recognition_start_time, self.current_recognized_face = None, None
                    return frame, face_box, (None, best_sim, False)
            return frame, face_box, (None, 0, False)
        except Exception as e:
            print(f"✗ Face recognition processing error: {e}"); return frame, face_box, (None, 0, False)


class HandTrackingManager:
    # (이 클래스는 수정사항이 없습니다)
    def __init__(self, camera, tkinter_queue=None, screen_size=None):
        self.camera, self.tkinter_queue, self.screen_size = camera, tkinter_queue, screen_size
        self.mp_hands = mp.solutions.hands; self.mp_drawing = mp.solutions.drawing_utils
        self.hands, self.keypoint_classifier = None, None; self.ocr_east_net, self.ocr_recognizer_interpreter = None, None
        self.is_initialized = False; self.current_mode = "Mouse Control"; self.mode_toggle_cooldown = 0
        self.mouse_controller = Controller()
        self.screen_width, self.screen_height = screen_size if screen_size else (1920, 1080)
        self.last_finger_pos, self.finger_stable_start_time = None, None
        self.finger_stable_threshold, self.dwell_click_duration = 20, 1.5
        self.screen_capture_points = []
    def initialize(self):
        if self.is_initialized: return True
        try:
            original_dir = os.getcwd()
            os.chdir('handMini2')
            self.hands = self.mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.8)
            self.keypoint_classifier = KeyPointClassifier()
            os.chdir(original_dir); print("✓ Hand tracking models initialized")
            self.is_initialized = True; return True
        except Exception as e:
            if 'original_dir' in locals(): os.chdir(original_dir)
            print(f"✗ Hand tracking initialization error: {e}"); return False
    def load_ocr_models(self):
        if hasattr(self, 'ocr_models_loaded') and self.ocr_models_loaded: return True
        try:
            print("Loading OCR models...")
            self.ocr_east_net = cv2.dnn.readNet(EAST_MODEL_PATH)
            self.ocr_recognizer_interpreter = tflite.Interpreter(model_path=RECOGNIZER_MODEL_PATH); self.ocr_recognizer_interpreter.allocate_tensors()
            self.ocr_models_loaded = True; print("✓ OCR models loaded successfully"); return True
        except Exception as e:
            print(f"✗ OCR model loading failed: {e}"); return False
    def process_frame(self, frame):
        if not self.is_initialized or frame is None: return
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmark_list = calc_landmark.calc_landmark(frame, hand_landmarks)
                pre_processed = calc_landmark.pre_process_landmark(landmark_list)
                gesture_id = self.keypoint_classifier(pre_processed)
                gesture = self.keypoint_classifier.labels[gesture_id]
                self.handle_gestures(gesture, hand_landmarks, frame)
        self.draw_ui(frame)
        self.mode_toggle_cooldown = max(0, self.mode_toggle_cooldown - 1)
    def handle_gestures(self, gesture, hand_landmarks, frame):
        if gesture == "Open" and self.mode_toggle_cooldown == 0:
            self.current_mode = "Screen Capture & OCR" if self.current_mode == "Mouse Control" else "Mouse Control"
            print(f"Mode changed to: {self.current_mode}")
            self.reset_mode_state()
            if self.current_mode == "Screen Capture & OCR": self.load_ocr_models()
            self.mode_toggle_cooldown = 30
        tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        finger_x, finger_y = int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0])
        screen_pos = self.map_finger_to_screen(finger_x, finger_y, frame.shape[1], frame.shape[0])
        if self.current_mode == "Mouse Control":
            if gesture == "Pointer": self.mouse_controller.position = screen_pos
            elif gesture == "Close": self.handle_dwell_click(screen_pos) # Click with 'Close' gesture
        elif self.current_mode == "Screen Capture & OCR":
            if gesture == "Pointer":
                self.mouse_controller.position = screen_pos
                if len(self.screen_capture_points) == 0: self.handle_screen_capture_start_point(screen_pos)
                elif len(self.screen_capture_points) == 1: self.update_screen_box_drawing(self.screen_capture_points[0], screen_pos)
            elif gesture == "Close" and len(self.screen_capture_points) == 1:
                self.screen_capture_points.append(screen_pos)
                print(f"Capture end set with 'Close' gesture: {screen_pos}")
                self.perform_screen_capture_and_ocr()
                self.reset_mode_state()
    def handle_dwell_click(self, pos):
        if self.last_finger_pos is None or np.linalg.norm(np.array(pos) - np.array(self.last_finger_pos)) > self.finger_stable_threshold:
            self.last_finger_pos = pos; self.finger_stable_start_time = time.time()
        if time.time() - self.finger_stable_start_time > self.dwell_click_duration:
            self.mouse_controller.click(Button.left); print(f"✓ Dwell click at {pos}"); self.finger_stable_start_time = time.time() + 999
    def handle_screen_capture_start_point(self, pos):
        dist = np.linalg.norm(np.array(pos) - np.array(self.last_finger_pos)) if self.last_finger_pos else float('inf')
        if dist <= self.finger_stable_threshold:
            if self.finger_stable_start_time is None: self.finger_stable_start_time = time.time()
            elif time.time() - self.finger_stable_start_time >= self.dwell_click_duration:
                if len(self.screen_capture_points) == 0:
                    self.screen_capture_points.append(pos); print(f"Capture start set: {pos}")
                    self.start_screen_box_drawing()
                self.finger_stable_start_time = None
        else: self.finger_stable_start_time = None
        self.last_finger_pos = pos
    def perform_screen_capture_and_ocr(self):
        if len(self.screen_capture_points) != 2: return
        x1, y1 = self.screen_capture_points[0]; x2, y2 = self.screen_capture_points[1]
        left, top, width, height = min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)
        if width < 10 or height < 10: print("✗ Capture region too small."); self.stop_screen_box_drawing(); return
        try:
            screenshot_pil = pyautogui.screenshot(region=(left, top, width, height))
            captured_image = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)
            print(f"--- Running OCR on captured region (w:{width}, h:{height}) ---")
            boxes = self._detect_text_boxes_east(captured_image, min_confidence=0.3)
            texts = []; result_image = captured_image.copy()
            for (sx, sy, ex, ey) in boxes:
                sx, sy, ex, ey = max(0, sx), max(0, sy), min(captured_image.shape[1], ex), min(captured_image.shape[0], ey)
                if ex - sx < 5 or ey - sy < 5: continue
                cropped = captured_image[sy:ey, sx:ex]
                if cropped.size == 0: continue
                text = self._recognize_single_text(cropped)
                texts.append(text); cv2.rectangle(result_image, (sx, sy), (ex, ey), (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_origin = (sx, sy - th - 5 if sy - th > 5 else sy + th + 5)
                cv2.rectangle(result_image, (text_origin[0], text_origin[1] - th), (text_origin[0] + tw, text_origin[1] + 5), (0,0,0), -1)
                cv2.putText(result_image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"✓ OCR finished. Full text: {' '.join(texts)}")
            cv2.imshow("OCR Result", result_image); cv2.waitKey(0); cv2.destroyWindow("OCR Result")
        except Exception as e: print(f"✗ Screen capture or OCR failed: {e}")
        finally: self.stop_screen_box_drawing()
    def draw_ui(self, frame):
        cv2.putText(frame, f"Mode: {self.current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        instruction = ""
        if self.current_mode == "Screen Capture & OCR":
            if len(self.screen_capture_points) == 0: instruction = "Dwell with 'Pointer' to set START point"
            else: instruction = "Move 'Pointer' to draw box | 'Close' to CAPTURE"
        elif self.current_mode == "Mouse Control": instruction = "'Pointer' to move, 'Close' to Dwell-Click"
        if instruction: cv2.putText(frame, instruction, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    def map_finger_to_screen(self, x, y, fw, fh):
        sx = int((x / fw) * self.screen_width); sy = int((y / fh) * self.screen_height)
        return max(0, min(sx, self.screen_width - 1)), max(0, min(sy, self.screen_height - 1))
    def _recognize_single_text(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY); resized = cv2.resize(gray, (200, 31))
        normalized = resized.astype(np.float32) / 255.0; input_data = normalized.reshape(1, 31, 200, 1)
        interpreter = self.ocr_recognizer_interpreter
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data); interpreter.invoke()
        y_pred = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        if tf is None: return "[ERROR: TF not found]"
        input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
        decoded, _ = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)
        return ''.join(self.CHARSET[idx] for idx in decoded[0][0].numpy() if 0 <= idx < len("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~ "))
    def _detect_text_boxes_east(self, roi, min_confidence=0.5):
        (H, W) = roi.shape[:2]; newW, newH = 320, 320
        rW, rH = W / float(newW), H / float(newH)
        blob = cv2.dnn.blobFromImage(roi, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.ocr_east_net.setInput(blob)
        (scores, geometry) = self.ocr_east_net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])
        rects, confidences = self._decode_east_predictions(scores, geometry, min_confidence)
        boxes = cv2.dnn.NMSBoxes(rects, confidences, min_confidence, 0.4)
        results = []
        if len(boxes) > 0:
            for i in boxes.flatten():
                (startX, startY, endX, endY) = rects[i]; results.append((int(startX*rW), int(startY*rH), int(endX*rW), int(endY*rH)))
        return results
    def _decode_east_predictions(self, scores, geometry, min_confidence):
        (numRows, numCols) = scores.shape[2:4]; rects, confidences = [], []
        for y in range(numRows):
            s_data, x0, x1, x2, x3, angles = scores[0,0,y], geometry[0,0,y], geometry[0,1,y], geometry[0,2,y], geometry[0,3,y], geometry[0,4,y]
            for x in range(numCols):
                if s_data[x] < min_confidence: continue
                (offX, offY) = (x * 4.0, y * 4.0); angle = angles[x]
                cos, sin = np.cos(angle), np.sin(angle)
                h, w = x0[x] + x2[x], x1[x] + x3[x]
                endX, endY = int(offX + (cos*x1[x])+(sin*x2[x])), int(offY - (sin*x1[x])+(cos*x2[x]))
                startX, startY = int(endX - w), int(endY - h)
                rects.append((startX, startY, endX, endY)); confidences.append(s_data[x])
        return rects, confidences
    def start_screen_box_drawing(self):
        if self.tkinter_queue: self.tkinter_queue.put(('start_box_drawing',))
    def update_screen_box_drawing(self, p1, p2):
        if self.tkinter_queue: self.tkinter_queue.put(('update_box_drawing', p1[0], p1[1], p2[0], p2[1]))
    def stop_screen_box_drawing(self):
        if self.tkinter_queue: self.tkinter_queue.put(('stop_box_drawing',))
    def reset_mode_state(self):
        self.screen_capture_points = []; self.last_finger_pos = None; self.finger_stable_start_time = None
        self.stop_screen_box_drawing()
    def handle_key(self, key):
        if key == ord('r'): print("Restarting capture."); self.reset_mode_state(); return False
        return True


class IntegratedGUI:
    def __init__(self):
        self.root = tk.Tk(); self.root.title("System Control"); self.root.geometry("400x200"); self.root.configure(bg='#2c3e50')
        self.screen_width, self.screen_height = 1920, 1080
        self.tkinter_queue = queue.Queue()
        self.camera = SharedCamera()
        self.face_manager = FaceRecognitionManager(self.camera)
        self.hand_manager = HandTrackingManager(self.camera, self.tkinter_queue, (self.screen_width, self.screen_height))
        self.is_logged_in, self.current_user, self.is_running, self.current_mode = False, None, False, "idle"
        self.processing_thread = None
        self.registration_name, self.registration_angles, self.registration_angle_idx = None, [], 0
        self.setup_gui()
        self.process_tkinter_queue()
    def process_tkinter_queue(self):
        try:
            while not self.tkinter_queue.empty():
                command = self.tkinter_queue.get_nowait()
                if command[0] == 'start_box_drawing': self._start_screen_box_drawing()
                elif command[0] == 'update_box_drawing': self._update_screen_box_drawing(*command[1:])
                elif command[0] == 'stop_box_drawing': self._stop_screen_box_drawing()
        except queue.Empty: pass
        self.root.after(50, self.process_tkinter_queue)
    def _start_screen_box_drawing(self):
        if hasattr(self, 'overlay_window') and self.overlay_window: return
        try:
            screenshot_path = "/tmp/fullscreen_capture.png"; os.system(f"scrot -o {screenshot_path}")
            self.original_screenshot = Image.open(screenshot_path)
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
            self.overlay_window.overrideredirect(True); self.overlay_window.attributes('-topmost', True)
            self.tk_screenshot = ImageTk.PhotoImage(self.original_screenshot)
            self.overlay_canvas = tk.Canvas(self.overlay_window, cursor="crosshair")
            self.overlay_canvas.pack(fill=tk.BOTH, expand=True)
            self.overlay_canvas.create_image(0, 0, image=self.tk_screenshot, anchor='nw')
            print("✓ Overlay started using explicit geometry.")
        except Exception as e: print(f"✗ Error starting overlay: {e}")
    def _update_screen_box_drawing(self, x1, y1, x2, y2):
        if hasattr(self, 'overlay_canvas') and self.overlay_canvas:
            self.overlay_canvas.delete("selection_rect")
            self.overlay_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=3, tags="selection_rect")
    def _stop_screen_box_drawing(self):
        if hasattr(self, 'overlay_window') and self.overlay_window:
            self.overlay_window.destroy(); self.overlay_window = None; self.overlay_canvas = None
            self.original_screenshot = None; self.tk_screenshot = None
    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg='#2c3e50'); main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        tk.Label(main_frame, text="System", font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50').pack(pady=(0, 10))
        self.status_label = tk.Label(main_frame, text="Status: Ready", font=('Arial', 12), fg='white', bg='#2c3e50'); self.status_label.pack(pady=5)
        self.user_info_label = tk.Label(main_frame, text="Not logged in", font=('Arial', 12), fg='white', bg='#2c3e50'); self.user_info_label.pack(pady=5)
        control_frame = tk.Frame(main_frame, bg='#2c3e50'); control_frame.pack(pady=10)
        self.login_button = tk.Button(control_frame, text="Start System", command=self.start_system, font=('Arial', 12), bg='#3498db', fg='white'); self.login_button.pack(side=tk.LEFT, padx=5)
        self.logout_button = tk.Button(control_frame, text="Logout", command=self.logout, font=('Arial', 12), bg='#e74c3c', fg='white');
    def start_system(self):
        if self.is_running: return
        self.update_status("Initializing..."); self.login_button.config(state=tk.DISABLED)
        if not self.camera.initialize() or not self.face_manager.load_models():
            self.update_status("Initialization Failed"); self.login_button.config(state=tk.NORMAL); return
        self.camera.start_camera_stream()
        self.is_running = True; self.current_mode = "face_recognition"
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True); self.processing_thread.start()
        self.update_status("Face Recognition Active")
    def processing_loop(self):
        try:
            while self.is_running:
                frame = self.camera.get_current_frame()
                if frame is None: time.sleep(0.01); continue
                if self.current_mode == "face_recognition": self.process_face_recognition(frame)
                elif self.current_mode == "registration": self.process_registration(frame)
                elif self.current_mode == "hand_tracking": self.process_hand_tracking(frame)
                cv2.imshow("System View", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): self.is_running = False
                elif key == ord('r') and self.current_mode == "face_recognition": self.start_registration_flow()
        finally:
            cv2.destroyAllWindows()
    def process_face_recognition(self, frame):
        frame, face_box, rec_res = self.face_manager.process_frame(frame)
        if face_box:
            x, y, w, h = face_box; cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if rec_res and isinstance(rec_res, tuple):
                name, sim, ready, *opt = rec_res
                if ready:
                    self.current_user, self.is_logged_in, self.current_mode = name, True, "hand_tracking"
                    self.root.after(0, self.on_login_success)
                else:
                    text = f"{name}: {sim:.2f} ({opt[0]:.1f}s)" if opt else (f"{name}: {sim:.2f}" if name else f"Unknown: {sim:.2f}")
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    def process_registration(self, frame):
        if self.registration_angle_idx >= len(self.registration_angles):
            print(f"✓ Registration complete for '{self.registration_name}'."); self.face_manager.save_database()
            self.current_mode = "face_recognition"; return
        angle = self.registration_angles[self.registration_angle_idx]
        main_text = f"Show {angle} face ({self.registration_angle_idx+1}/{len(self.registration_angles)}) | 'c' to capture"
        h, w, _ = frame.shape
        cv2.putText(frame, main_text, (int(w*0.1), int(h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        face_box = self.face_manager.detect_face(frame)
        if face_box:
            x, y, w, h = face_box; cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                face_roi = frame[y:y+h, x:x+w]; embedding = self.face_manager.get_face_embedding(face_roi)
                if embedding is not None:
                    self.face_manager.register_face(self.registration_name, embedding)
                    self.registration_angle_idx += 1
                else: print("✗ Failed to get embedding. Please try again.")
    def start_registration_flow(self):
        name = self.simple_input_dialog("Enter name for registration:")
        if not name: print("✗ Registration cancelled."); return
        self.registration_name = name; self.registration_angles = ["Front", "Left", "Right"]; self.registration_angle_idx = 0
        self.current_mode = "registration"
    def process_hand_tracking(self, frame):
        self.hand_manager.process_frame(frame)
        cv2.putText(frame, f"User: {self.current_user}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    def on_login_success(self):
        self.update_status(f"Hand tracking active for: {self.current_user}")
        self.user_info_label.config(text=f"Logged in as: {self.current_user}")
        self.logout_button.pack(side=tk.LEFT, padx=5)
        if not self.hand_manager.initialize(): self.update_status("Hand tracking init failed")
    def logout(self):
        self.update_status("Logging out...")
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
            print("✓ Processing thread stopped.")
        self.is_logged_in = False; self.current_user = None; self.current_mode = "idle"
        self.hand_manager.reset_mode_state()
        self.user_info_label.config(text="Not logged in")
        self.logout_button.pack_forget()
        self.login_button.config(state=tk.NORMAL)
        self.update_status("Ready")
    def simple_input_dialog(self, prompt):
        # (This function is the same as before)
        dialog = tk.Toplevel(self.root); dialog.title("Register Face"); dialog.transient(self.root); dialog.grab_set()
        result = [None]; entry = tk.Entry(dialog, width=30)
        def on_ok(): result[0] = entry.get(); dialog.destroy()
        tk.Label(dialog, text=prompt).pack(pady=10); entry.pack(pady=5); entry.focus()
        btn_frame = tk.Frame(dialog); btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        entry.bind('<Return>', lambda e: on_ok()); dialog.wait_window()
        return result[0]
    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}"); print(f"Status: {message}")
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup); self.root.mainloop()
    def cleanup(self):
        print("Cleanup initiated..."); self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)
        self.camera.close()
        cv2.destroyAllWindows()
        if self.root: self.root.destroy()
        print("✓ Application terminated.")

def main():
    app = IntegratedGUI()
    app.run()

if __name__ == "__main__":
    main()
