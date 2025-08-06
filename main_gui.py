#!/usr/bin/env python3
"""
Integrated Face Recognition and Hand Landmark Tracking System (Final Version 2.0)
Raspberry Pi 5 optimized with screenshot-on-canvas capture and integrated OCR.
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
FACE_RECOGNITION_DURATION = 3.0
DETECTION_MODEL_PATH = "face/Lightweight-Face-Detection.tflite"
EMBEDDING_MODEL_PATH = "face/MobileFaceNet_9925_9680.tflite"
FACE_DATABASE_FILENAME = "face/pi_face_database_multi.pkl"

# Camera
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

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
                controls={"FrameDurationLimits": (33333, 33333)}
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
               #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
            if not os.path.exists(DETECTION_MODEL_PATH): raise FileNotFoundError(f"{DETECTION_MODEL_PATH} not found")
            detection_interpreter = tflite.Interpreter(model_path=DETECTION_MODEL_PATH)
            detection_interpreter.allocate_tensors()
            self.models['detection'] = {'interpreter': detection_interpreter, 'input': detection_interpreter.get_input_details(), 'output': detection_interpreter.get_output_details()}
            print("✓ Face detection model loaded")
            if not os.path.exists(EMBEDDING_MODEL_PATH): raise FileNotFoundError(f"{EMBEDDING_MODEL_PATH} not found")
            embedding_interpreter = tflite.Interpreter(model_path=EMBEDDING_MODEL_PATH)
            embedding_interpreter.allocate_tensors()
            self.models['embedding'] = {'interpreter': embedding_interpreter, 'input': embedding_interpreter.get_input_details(), 'output': embedding_interpreter.get_output_details()}
            print("✓ Face embedding model loaded")
            self.is_loaded = True; return True
        except Exception as e:
            print(f"✗ Face model loading error: {e}"); return False
    def load_database(self):
        if os.path.exists(FACE_DATABASE_FILENAME):
            try:
                with open(FACE_DATABASE_FILENAME, 'rb') as f: self.face_database = pickle.load(f)
                total_embeddings = sum(len(v) for v in self.face_database.values())
                print(f"✓ Face database loaded: {len(self.face_database)} people, {total_embeddings} embeddings")
            except (Exception, EOFError) as e:
                print(f"✗ Error loading face database: {e}"); self.face_database = {}
        else: print("✓ No existing face database found")
    def save_database(self):
        try:
            with open(FACE_DATABASE_FILENAME, 'wb') as f: pickle.dump(self.face_database, f)
            print(f"✓ Face database saved to {FACE_DATABASE_FILENAME}")
        except Exception as e: print(f"✗ Error saving face database: {e}")
    def detect_face(self, image):
        if not self.is_loaded or image is None: return None
        interpreter = self.models['detection']['interpreter']
        H0, W0 = image.shape[:2]; resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self.detection_input[0, :, :, 0] = gray.astype(np.float32) / 255.0
        interpreter.set_tensor(self.models['detection']['input'][0]['index'], self.detection_input)
        interpreter.invoke()
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
        boxes_pix[:, [0, 2]] *= W0 / 640; boxes_pix[:, [1, 3]] *= H0 / 480
        bboxes_for_nms = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_pix]
        idxs = cv2.dnn.NMSBoxes(bboxes_for_nms, scores.tolist(), CONFIDENCE_THRESHOLD, 0.3)
        if idxs is not None and len(idxs) > 0:
            best_idx = idxs.flatten()[0]
            x1_b, y1_b, x2_b, y2_b = boxes_pix[best_idx]
            x1_int, y1_int, x2_int, y2_int = int(x1_b), int(y1_b), int(x2_b), int(y2_b)
            w_b, h_b = x2_int - x1_int, y2_int - y1_int
            if x1_int >= 0 and y1_int >= 0 and x2_int < W0 and y2_int < H0 and w_b > 0 and h_b > 0:
                return (x1_int, y1_int, w_b, h_b, scores[best_idx])
        return None
    def get_face_embedding(self, face_img):
        if not self.is_loaded: return None
        interpreter = self.models['embedding']['interpreter']
        face_resized = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_LINEAR)
        self.embedding_input[0] = (cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 128.0
        interpreter.set_tensor(self.models['embedding']['input'][0]['index'], self.embedding_input)
        interpreter.invoke()
        return interpreter.get_tensor(self.models['embedding']['output'][0]['index']).flatten()
    def compare_faces(self, e1, e2):
        if e1 is None or e2 is None: return 0.0, False
        sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
        return sim, sim > SIMILARITY_THRESHOLD
    def find_best_match(self, emb_check):
        if emb_check is None or not self.face_database: return None, 0.0
        best_name, max_sim = None, 0.0
        for name, emb_list in self.face_database.items():
            current_max_sim = max((self.compare_faces(emb_check, reg_emb)[0] for reg_emb in emb_list), default=0.0)
            if current_max_sim > max_sim:
                max_sim, best_name = current_max_sim, name
        return best_name, max_sim
    def register_face(self, name, embedding):
        if not name or embedding is None: return False
        if name in self.face_database: self.face_database[name].append(embedding)
        else: self.face_database[name] = [embedding]
        print(f"✓ Embedding registered for '{name}'. Total: {len(self.face_database[name])}")
        return True
    def process_frame(self, frame):
        if frame is None: return frame, None, None
        face_result = self.detect_face(frame)
        if not face_result:
            self.recognition_start_time, self.current_recognized_face = None, None
            return frame, None, None
        x, y, w, h, _ = face_result; face_roi = frame[y:y+h, x:x+w]
        try:
            current_embedding = self.get_face_embedding(face_roi)
            if current_embedding is not None:
                best_match, best_similarity = self.find_best_match(current_embedding)
                if best_match and best_similarity > SIMILARITY_THRESHOLD:
                    if self.current_recognized_face == best_match:
                        if self.recognition_start_time is None: self.recognition_start_time = time.time()
                        if time.time() - self.recognition_start_time >= FACE_RECOGNITION_DURATION:
                            return frame, face_result, (best_match, best_similarity, True)
                        else:
                            rem_time = FACE_RECOGNITION_DURATION - (time.time() - self.recognition_start_time)
                            return frame, face_result, (best_match, best_similarity, False, rem_time)
                    else:
                        self.current_recognized_face, self.recognition_start_time = best_match, time.time()
                        return frame, face_result, (best_match, best_similarity, False, FACE_RECOGNITION_DURATION)
                else:
                    self.recognition_start_time, self.current_recognized_face = None, None
                    return frame, face_result, (None, best_similarity, False)
            else: return frame, face_result, (None, 0, False)
        except Exception as e:
            print(f"✗ Face recognition processing error: {e}"); return frame, face_result, (None, 0, False)


class HandTrackingManager:
    """Handles hand tracking, gesture recognition, and actions like OCR & screen capture."""
    # ### 수정된 부분 1: __init__ 생성자에 screen_size 파라미터 추가 ###
    def __init__(self, camera, tkinter_queue=None, screen_size=None):
        self.camera = camera; self.tkinter_queue = tkinter_queue
        self.mp_hands = mp.solutions.hands; self.mp_drawing = mp.solutions.drawing_utils
        self.hands, self.keypoint_classifier = None, None
        self.ocr_east_net, self.ocr_recognizer_interpreter = None, None
        self.ocr_input_details, self.ocr_output_details = None, None
        self.CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~ "
        self.is_initialized = False; self.current_mode = "Mouse Control"
        self.mode_toggle_cooldown = 0
        self.awaiting_ocr_confirmation, self.awaiting_capture_confirmation = False, False
        self.mouse_controller = Controller()

        # ### 수정된 부분 2: screen_size를 직접 받아서 사용 ###
        if screen_size:
            self.screen_width, self.screen_height = screen_size
            print(f"✓ Screen size received: {self.screen_width}x{self.screen_height}")
        else:
            # 비상용 폴백 코드
            print("Warning: Screen size not provided, attempting fallback detection.")
            try:
                root = tk.Tk(); root.withdraw()
                self.screen_width, self.screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
                root.destroy()
            except Exception as e:
                print(f"Fallback screen size detection failed: {e}. Defaulting to 1920x1080.")
                self.screen_width, self.screen_height = 1920, 1080

        self.last_finger_pos, self.finger_stable_start_time = None, None
        self.finger_stable_threshold, self.dwell_click_duration = 20, 1.5
        self.capture_points, self.screen_capture_points = [], []

    # ### 삭제된 부분: 이 함수는 더 이상 필요 없으므로 클래스에서 완전히 제거합니다. ###
    # def get_screen_size(self):
    #     ...

    def initialize(self):
        # (이하 initialize 함수 내용은 모두 동일)
        if self.is_initialized: return True
        try:
            original_dir = os.getcwd()
            if not os.path.exists('handMini2'): raise FileNotFoundError("'handMini2' directory not found.")
            os.chdir('handMini2')
            self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.8)
            self.keypoint_classifier = KeyPointClassifier()
            os.chdir(original_dir); print("✓ Hand tracking models initialized")
            
            print("Loading OCR models...")
            if not os.path.exists(EAST_MODEL_PATH): raise FileNotFoundError(f"EAST model not found: {EAST_MODEL_PATH}")
            self.ocr_east_net = cv2.dnn.readNet(EAST_MODEL_PATH)
            self.ocr_east_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.ocr_east_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("✓ OpenCV DNN backend set to CPU.")
            
            if not os.path.exists(RECOGNIZER_MODEL_PATH): raise FileNotFoundError(f"Recognizer model not found: {RECOGNIZER_MODEL_PATH}")
            self.ocr_recognizer_interpreter = tflite.Interpreter(model_path=RECOGNIZER_MODEL_PATH)
            self.ocr_recognizer_interpreter.allocate_tensors()
            self.ocr_input_details = self.ocr_recognizer_interpreter.get_input_details()
            self.ocr_output_details = self.ocr_recognizer_interpreter.get_output_details()
            print("✓ OCR models loaded successfully")
            self.is_initialized = True; return True
        except Exception as e:
            if 'original_dir' in locals(): os.chdir(original_dir)
            print(f"✗ Hand tracking & OCR initialization error: {e}"); return False

    
    def process_frame(self, frame):
        if not self.is_initialized or frame is None: return frame, None, None
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmark_list = calc_landmark.calc_landmark(frame, hand_landmarks)
                pre_processed = calc_landmark.pre_process_landmark(landmark_list)
                gesture_id = self.keypoint_classifier(pre_processed)
                gesture = self.keypoint_classifier.labels[gesture_id]
                self.handle_gestures(gesture, hand_landmarks, frame)
        self.draw_ui(frame); self.mode_toggle_cooldown = max(0, self.mode_toggle_cooldown - 1)
        return frame, gesture, results.multi_hand_landmarks

    def handle_gestures(self, gesture, hand_landmarks, frame):
        if gesture == "Open" and self.mode_toggle_cooldown == 0:
            modes = ["Mouse Control", "Screen Capture & OCR"]
            try:
                current_index = modes.index(self.current_mode)
                self.current_mode = modes[(current_index + 1) % len(modes)]
            except ValueError: self.current_mode = "Mouse Control"
            print(f"Mode changed to: {self.current_mode}")
            self.reset_mode_state()
            self.mode_toggle_cooldown = 30
        
        elif gesture == "Pointer":
            tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_x, finger_y = int(tip.x * frame.shape[1]), int(tip.y * frame.shape[0])
            screen_pos = self.map_finger_to_screen(finger_x, finger_y, frame.shape[1], frame.shape[0])
            self.mouse_controller.position = screen_pos
            if self.current_mode == "Mouse Control": self.handle_dwell_click(screen_pos)
            elif self.current_mode == "Screen Capture & OCR": self.handle_screen_capture_pointing(screen_pos)
        
        elif gesture == "Close" and self.awaiting_capture_confirmation:
            self.perform_screen_capture_and_ocr()
            self.reset_mode_state()

    def perform_screen_capture_and_ocr(self):
        if len(self.screen_capture_points) != 2: return
        x1, y1 = self.screen_capture_points[0]
        x2, y2 = self.screen_capture_points[1]
        
        left, top, width, height = min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)

        if width < 10 or height < 10:
            print("✗ Capture region too small.")
            self.stop_screen_box_drawing()
            return
        
        try:
            # 1. 화면 캡처 (PIL Image 객체)
            screenshot_pil = pyautogui.screenshot(region=(left, top, width, height))
            
            # (선택) 캡처된 이미지를 파일로 저장
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"capture_{timestamp}.png"
            screenshot_pil.save(filename)
            print(f"✓ Screen capture saved as: {filename}")

            # 2. PIL Image를 OpenCV(Numpy) 형식으로 변환
            captured_image = cv2.cvtColor(np.array(screenshot_pil), cv2.COLOR_RGB2BGR)

            # 3. 변환된 이미지 데이터로 직접 OCR 수행
            print(f"--- Running OCR on captured region (w:{width}, h:{height}) ---")
            boxes = self._detect_text_boxes_east(captured_image, min_confidence=0.3)
            texts = []
            # 원본 이미지 복사 (결과를 그리기 위함)
            result_image = captured_image.copy()

            for (sx, sy, ex, ey) in boxes:
                sx, sy = max(0, sx), max(0, sy)
                ex, ey = min(captured_image.shape[1], ex), min(captured_image.shape[0], ey)
                if ex - sx < 5 or ey - sy < 5: continue

                cropped = captured_image[sy:ey, sx:ex]
                if cropped.size == 0: continue
                
                text = self._recognize_single_text(cropped)
                texts.append(text)

                # ### 추가된 부분: 결과 이미지에 박스와 텍스트 그리기 ###
                # 인식된 영역에 초록색 사각형 그리기
                cv2.rectangle(result_image, (sx, sy), (ex, ey), (0, 255, 0), 2)
                # 인식된 텍스트를 사각형 위에 노란색으로 쓰기
                # 텍스트 배경을 위한 사각형 추가
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_origin = (sx, sy - text_height - baseline if sy > 20 else sy + text_height + baseline)
                cv2.rectangle(result_image, (text_origin[0], text_origin[1] + baseline), (text_origin[0] + text_width, text_origin[1] - text_height), (0, 0, 0), -1)
                cv2.putText(result_image, text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            full_text = ' '.join(texts)
            print(f"✓ OCR finished. Full text: {full_text}")
            
            # ### 추가된 부분: 최종 결과 이미지를 새 창으로 표시 ###
            cv2.imshow("OCR Result", result_image)
            # 사용자가 아무 키나 누를 때까지 대기
            cv2.waitKey(0)
            # 결과 창 닫기
            cv2.destroyWindow("OCR Result")

        except Exception as e:
            print(f"✗ Screen capture or OCR failed: {e}")
        finally:
            self.stop_screen_box_drawing()

    # --- 나머지 헬퍼 함수들은 변경 없음 ---
    def get_screen_size(self):
        try:
            root = tk.Tk(); root.withdraw()
            w, h = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return w, h
        except: return 1920, 1080
    
    def map_finger_to_screen(self, x, y, fw, fh):
        sx = int((x / fw) * self.screen_width)
        sy = int((y / fh) * self.screen_height)
        return max(0, min(sx, self.screen_width - 1)), max(0, min(sy, self.screen_height - 1))

    def handle_dwell_click(self, pos):
        dist = np.linalg.norm(np.array(pos)-np.array(self.last_finger_pos)) if self.last_finger_pos else float('inf')
        if dist <= self.finger_stable_threshold:
            if self.finger_stable_start_time is None: self.finger_stable_start_time = time.time()
            elif time.time()-self.finger_stable_start_time >= self.dwell_click_duration:
                self.mouse_controller.click(Button.left); print(f"✓ Dwell click at {pos}"); self.finger_stable_start_time=None
        else: self.finger_stable_start_time = None
        self.last_finger_pos = pos

    def handle_screen_capture_pointing(self, pos):
        if len(self.screen_capture_points) == 1: self.update_screen_box_drawing(self.screen_capture_points[0], pos)
        dist = np.linalg.norm(np.array(pos) - np.array(self.last_finger_pos)) if self.last_finger_pos else float('inf')
        if dist <= self.finger_stable_threshold:
            if self.finger_stable_start_time is None: self.finger_stable_start_time = time.time()
            elif time.time() - self.finger_stable_start_time >= self.dwell_click_duration:
                if len(self.screen_capture_points) == 0:
                    self.screen_capture_points.append(pos); print(f"Capture start set: {pos}"); self.start_screen_box_drawing()
                elif len(self.screen_capture_points) == 1:
                    self.screen_capture_points.append(pos); print(f"Capture end set: {pos}"); self.awaiting_capture_confirmation = True
                self.finger_stable_start_time = None
        else: self.finger_stable_start_time = None
        self.last_finger_pos = pos

    def _decode_prediction(self, pred):
        if tf is None: return "[ERROR: TensorFlow not installed]"
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
        decoded = decoded[0][0].numpy()
        return ''.join(self.CHARSET[idx] for idx in decoded.flatten() if 0 <= idx < len(self.CHARSET))

    def _recognize_single_text(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (200, 31))
        normalized = resized.astype(np.float32) / 255.0
        input_data = normalized.reshape(1, 31, 200, 1)
        self.ocr_recognizer_interpreter.set_tensor(self.ocr_input_details[0]['index'], input_data)
        self.ocr_recognizer_interpreter.invoke()
        y_pred = self.ocr_recognizer_interpreter.get_tensor(self.ocr_output_details[0]['index'])
        return self._decode_prediction(y_pred)

    def _decode_east_predictions(self, scores, geometry, min_confidence):
        (numRows, numCols) = scores.shape[2:4]
        rects, confidences = [], []
        for y in range(numRows):
            scoresData, x0, x1, x2, x3, angles = scores[0,0,y], geometry[0,0,y], geometry[0,1,y], geometry[0,2,y], geometry[0,3,y], geometry[0,4,y]
            for x in range(numCols):
                if scoresData[x] < min_confidence: continue
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = angles[x]; cos, sin = np.cos(angle), np.sin(angle)
                h, w = x0[x] + x2[x], x1[x] + x3[x]
                endX, endY = int(offsetX + (cos*x1[x]) + (sin*x2[x])), int(offsetY - (sin*x1[x]) + (cos*x2[x]))
                startX, startY = int(endX - w), int(endY - h)
                rects.append((startX, startY, endX, endY)); confidences.append(scoresData[x])
        return rects, confidences

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
                (startX, startY, endX, endY) = rects[i]
                results.append((int(startX*rW), int(startY*rH), int(endX*rW), int(endY*rH)))
        return results

    def start_screen_box_drawing(self):
        if self.tkinter_queue: self.tkinter_queue.put(('start_box_drawing',))
    def update_screen_box_drawing(self, p1, p2):
        if self.tkinter_queue: self.tkinter_queue.put(('update_box_drawing', p1[0], p1[1], p2[0], p2[1]))
    def stop_screen_box_drawing(self):
        if self.tkinter_queue: self.tkinter_queue.put(('stop_box_drawing',))

    def reset_mode_state(self):
        self.capture_points, self.screen_capture_points = [], []
        self.awaiting_ocr_confirmation, self.awaiting_capture_confirmation = False, False
        self.last_finger_pos, self.finger_stable_start_time = None, None
        self.stop_screen_box_drawing()

    def handle_key(self, key):
        if key == ord('r'): print("Restarting capture."); self.reset_mode_state(); return False
        return True

    def draw_ui(self, frame):
        cv2.putText(frame, f"Mode: {self.current_mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        instruction = ""
        if self.current_mode == "Screen Capture & OCR":
            if self.awaiting_capture_confirmation: instruction = "'Close' gesture: CAPTURE & OCR | 'r': RESTART"
            elif len(self.screen_capture_points) == 0: instruction = "Dwell with 'Pointer' to set START point"
            else: instruction = "Dwell with 'Pointer' to set END point"
        elif self.current_mode == "Mouse Control": instruction = "Dwell for 1.5s to CLICK"
        if instruction: cv2.putText(frame, instruction, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

class IntegratedGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face & Hand Tracking System")
        # geometry는 컨트롤 창의 크기일 뿐, 전체 화면 크기와는 무관합니다.
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        
        # ### 결정적인 수정 부분 ###
        # Tkinter의 winfo_screenwidth() 대신, 더 신뢰성 있는 pyautogui.size()를 사용합니다.
        # 이것이 실제 전체 화면 해상도를 가져옵니다.
        
        screen_width, screen_height = pyautogui.size()
        
            
        self.tkinter_queue = queue.Queue()
        self.camera = SharedCamera()
        self.face_manager = FaceRecognitionManager(self.camera)
        
        # 이전 단계에서 수정한 HandTrackingManager에 올바른 크기를 전달합니다.
        self.hand_manager = HandTrackingManager(self.camera, self.tkinter_queue, screen_size=(screen_width, screen_height))
        
        self.overlay_window, self.overlay_canvas = None, None
        self.original_screenshot, self.tk_screenshot = None, None
        self.is_logged_in, self.current_user, self.is_running, self.current_mode = False, None, False, "idle"
        self.processing_thread = None
        self.setup_gui()
        self.process_tkinter_queue()
    
    def process_tkinter_queue(self):
        try:
            while not self.tkinter_queue.empty():
                command = self.tkinter_queue.get_nowait()
                cmd_type = command[0]
                if cmd_type == 'start_box_drawing': self._start_screen_box_drawing()
                elif cmd_type == 'update_box_drawing': self._update_screen_box_drawing(*command[1:])
                elif cmd_type == 'stop_box_drawing': self._stop_screen_box_drawing()
        except queue.Empty: pass
        self.root.after(50, self.process_tkinter_queue)
    
    ### FIX 1: 회색 화면 문제를 해결하는 가장 안정적인 오버레이 방식 ###
    def _start_screen_box_drawing(self):
        if self.overlay_window: return
        try:
            # 1. scrot 명령어를 시스템에서 직접 호출하여 전체 화면을 임시 파일로 저장합니다.
            #    이것이 pyautogui의 불안정한 동작을 우회하는 가장 확실한 방법입니다.
            screenshot_path = "/tmp/fullscreen_capture.png"
            os.system(f"scrot -o {screenshot_path}") # -o 옵션은 덮어쓰기를 허용합니다.

            # 2. 저장된 스크린샷 파일을 PIL 이미지 객체로 불러옵니다.
            self.original_screenshot = Image.open(screenshot_path)
            
            # --- 이하 로직은 동일합니다 ---

            # 3. 일반적인 전체 화면 창 생성
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.attributes('-topmost', True)
            self.overlay_window.attributes('-fullscreen', True)
            self.overlay_window.overrideredirect(True)
            
            # 4. 불러온 스크린샷을 캔버스 배경으로 표시
            self.tk_screenshot = ImageTk.PhotoImage(self.original_screenshot)
            self.overlay_canvas = tk.Canvas(self.overlay_window, cursor="crosshair")
            self.overlay_canvas.pack(fill=tk.BOTH, expand=True)
            self.overlay_canvas.create_image(0, 0, image=self.tk_screenshot, anchor='nw')
            print("✓ Screenshot-on-canvas overlay started using scrot.")

        except FileNotFoundError:
             print(f"✗ CRITICAL ERROR: Screenshot file not found at {screenshot_path}.")
             print("   Please ensure 'scrot' is installed ('sudo apt-get install scrot')")
        except Exception as e:
            print(f"✗ Error starting screen box drawing: {e}")
            if self.overlay_window: self.overlay_window.destroy(); self.overlay_window = None

    def _update_screen_box_drawing(self, x1, y1, x2, y2):
        if not self.overlay_canvas: return
        self.overlay_canvas.delete("selection_rect")
        self.overlay_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=3, tags="selection_rect")

    def _stop_screen_box_drawing(self):
        if self.overlay_window:
            try: self.overlay_window.destroy()
            except Exception: pass
            finally:
                self.overlay_window, self.overlay_canvas = None, None
                self.original_screenshot, self.tk_screenshot = None, None # 참조 해제
    
    # --- 나머지 GUI 함수들은 이전과 동일 ---
    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title_label = tk.Label(main_frame, text="Face & Hand Tracking System", font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=(0, 20))
        self.status_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        self.status_frame.pack(fill=tk.X, pady=(0, 20))
        self.status_label = tk.Label(self.status_frame, text="Status: Ready", font=('Arial', 12), fg='white', bg='#34495e')
        self.status_label.pack(pady=10)
        control_frame = tk.Frame(main_frame, bg='#2c3e50')
        control_frame.pack(fill=tk.X, pady=(0, 20))
        self.login_button = tk.Button(control_frame, text="Start Face Recognition", command=self.start_face_recognition, font=('Arial', 12), bg='#3498db', fg='white', relief=tk.RAISED, bd=3, padx=20, pady=10)
        self.login_button.pack(side=tk.LEFT, padx=(0, 10))
        self.logout_button = tk.Button(control_frame, text="Logout", command=self.logout, font=('Arial', 12), bg='#e74c3c', fg='white', relief=tk.RAISED, bd=3, padx=20, pady=10)
        self.user_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        self.user_info_label = tk.Label(self.user_frame, text="Not logged in", font=('Arial', 12), fg='white', bg='#34495e')
        self.user_info_label.pack(pady=10)

    def start_face_recognition(self):
        if self.is_running: return
        self.update_status("Initializing camera...")
        if not self.camera.initialize(): self.update_status("Camera initialization failed"); return
        self.update_status("Loading face models...")
        if not self.face_manager.load_models(): self.update_status("Face model loading failed"); self.camera.close(); return
        self.camera.start_camera_stream()
        self.update_status("Starting face recognition...")
        self.login_button.config(state=tk.DISABLED)
        self.is_running, self.current_mode = True, "face_recognition"
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

    def processing_loop(self):
        try:
            while self.is_running:
                frame = self.camera.get_current_frame()
                if frame is None: time.sleep(0.01); continue
                
                window_title = "System"
                if self.current_mode == "face_recognition":
                    self.process_face_recognition(frame)
                    window_title = "Face Recognition - Login"
                elif self.current_mode == "hand_tracking":
                    self.process_hand_tracking(frame)
                    window_title = f"Hand Tracking - {self.current_user}"
                
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'): self.is_running = False
                elif key == ord('r'):
                    if self.current_mode == "face_recognition": self.run_registration_flow()
                    elif self.current_mode == "hand_tracking": self.hand_manager.handle_key(key)
        except Exception as e:
            print(f"✗ Processing loop error: {e}")
        finally:
            self.is_running = False
            self.root.after(0, self.cleanup)

    def process_face_recognition(self, frame):
        frame, face_result, recognition_result = self.face_manager.process_frame(frame)
        if face_result:
            x, y, w, h, _ = face_result
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if recognition_result and isinstance(recognition_result, tuple):
                best_match, best_similarity, login_ready, *optional_time = recognition_result
                if login_ready:
                    self.current_user, self.is_logged_in, self.current_mode = best_match, True, "hand_tracking"
                    self.root.after(0, self.on_login_success)
                elif optional_time:
                    cv2.putText(frame, f"Hold for {optional_time[0]:.1f}s", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if best_match: cv2.putText(frame, f"{best_match}: {best_similarity:.2f}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif best_match:
                    cv2.putText(frame, f"Recognized: {best_match}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Unknown: {best_similarity:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process_hand_tracking(self, frame):
        self.hand_manager.process_frame(frame)
        cv2.putText(frame, f"User: {self.current_user}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def run_registration_flow(self):
        name = self.simple_input_dialog("Enter name for registration:")
        if not name: print("✗ Registration cancelled."); return
        angles = ["Front", "Left", "Right"]
        for i, angle in enumerate(angles):
            while self.is_running:
                frame = self.camera.get_current_frame()
                if frame is None: continue
                display_frame = frame.copy()
                main_text = f"Show {angle} face ({i+1}/{len(angles)})"
                sub_text = "'c': Capture | 'q': Cancel"
                h, w, _ = display_frame.shape
                cv2.rectangle(display_frame, (0, int(h*0.4)), (w, int(h*0.6)), (0,0,0), -1)
                cv2.putText(display_frame, main_text, (int(w*0.1), int(h*0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display_frame, sub_text, (int(w*0.1), int(h*0.5) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                face_result = self.face_manager.detect_face(frame)
                if face_result:
                    x, y, w, h, _ = face_result
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.imshow("Face Registration", display_frame)
                key = cv2.waitKey(20) & 0xFF
                if key == ord('c') and face_result:
                    x, y, w, h, _ = face_result
                    face_roi = frame[y:y+h, x:x+w]
                    embedding = self.face_manager.get_face_embedding(face_roi)
                    if embedding is not None: self.face_manager.register_face(name, embedding); break
                    else: print("✗ Failed to get embedding. Please try again.")
                elif key == ord('q'):
                    print("✗ Registration cancelled by user."); cv2.destroyWindow("Face Registration"); return
        print(f"✓ Registration complete for '{name}'."); self.face_manager.save_database(); cv2.destroyWindow("Face Registration")

    def simple_input_dialog(self, prompt):
        dialog = tk.Toplevel(self.root); dialog.title("Register Face"); dialog.transient(self.root); dialog.grab_set()
        result = [None]
        def on_ok(): result[0] = entry.get(); dialog.destroy()
        tk.Label(dialog, text=prompt).pack(pady=10)
        entry = tk.Entry(dialog, width=30); entry.pack(pady=5); entry.focus()
        btn_frame = tk.Frame(dialog); btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        entry.bind('<Return>', lambda e: on_ok()); dialog.wait_window()
        return result[0]

    def on_login_success(self):
        self.update_status(f"Logged in as: {self.current_user}")
        self.user_info_label.config(text=f"Logged in as: {self.current_user}")
        self.logout_button.pack(side=tk.LEFT, padx=(0, 10))
        cv2.destroyAllWindows()
        self.update_status("Starting hand tracking...")
        if not self.hand_manager.initialize(): self.update_status("Hand tracking initialization failed")
        else: self.update_status(f"Hand tracking active for: {self.current_user}")

    def logout(self):
        self.is_logged_in, self.current_user, self.current_mode, self.is_running = False, None, "idle", False
        self.hand_manager.reset_mode_state()
        self.update_status("Logged out")
        self.user_info_label.config(text="Not logged in")
        self.logout_button.pack_forget()
        self.login_button.config(state=tk.NORMAL)
        cv2.destroyAllWindows()

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        print(f"Status: {message}")
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        try: self.root.mainloop()
        finally: self.cleanup()
    
    def cleanup(self):
        if not self.is_running: return # Prevent multiple cleanup calls
        print("Cleaning up resources...")
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1)
        self.camera.close()
        self.hand_manager.stop_screen_box_drawing()
        cv2.destroyAllWindows()
        print("✓ Application terminated")
        if self.root:
            self.root.quit()

def main():
    print("=== Integrated Face & Hand Tracking System (Final Version) ===")
    app = IntegratedGUI()
    app.run()

if __name__ == "__main__":
    main()
