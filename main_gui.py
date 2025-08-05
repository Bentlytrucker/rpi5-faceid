#!/usr/bin/env python3
"""
Integrated Face Recognition and Hand Landmark Tracking System
Raspberry Pi 5 optimized with shared camera and minimal resource usage
"""

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pickle
import os
import time
import threading
import queue
from picamera2 import Picamera2
import mediapipe as mp
import sys
import tkinter as tk
from tkinter import ttk
from pynput import mouse
from pynput.mouse import Button, Controller, Listener

# Import hand tracking modules
sys.path.append('handMini2')
from utils import calc_landmark
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from ocr_bridge import recognize_text

# Configuration
CONFIDENCE_THRESHOLD = 0.15
SIMILARITY_THRESHOLD = 0.6
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
DETECTION_MODEL_PATH = "face/Lightweight-Face-Detection.tflite"
EMBEDDING_MODEL_PATH = "face/MobileFaceNet_9925_9680.tflite"
FACE_DATABASE_FILENAME = "face/pi_face_database.pkl"
FACE_RECOGNITION_DURATION = 3.0  # 3 seconds for face recognition

class SharedCamera:
    """Shared camera manager for both face and hand tracking"""
    def __init__(self):
        self.picam2 = None
        self.is_initialized = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = False
        self.camera_thread = None
    
    def initialize(self):
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
        """Start continuous camera stream"""
        if not self.is_initialized:
            return False
        
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        return True
    
    def _camera_loop(self):
        """Continuous camera capture loop"""
        try:
            while self.is_running:
                try:
                    frame = self.picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = cv2.flip(frame, 1)  # Mirror for better UX
                    
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                    
                    time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                except Exception as e:
                    print(f"✗ Frame capture error: {e}")
                    time.sleep(0.1)  # Longer delay on error
        except Exception as e:
            print(f"✗ Camera loop error: {e}")
    
    def get_current_frame(self):
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def stop_camera_stream(self):
        """Stop camera stream"""
        self.is_running = False
        if self.camera_thread:
            try:
                self.camera_thread.join(timeout=2.0)
            except:
                pass
    
    def close(self):
        self.stop_camera_stream()
        if self.picam2 is not None:
            self.picam2.close()


class FaceRecognitionManager:
    """Face recognition system with multi-angle database support."""
    def __init__(self, camera):
        self.camera = camera
        self.models = {}
        self.is_loaded = False
        self.detection_input = np.zeros((1, 480, 640, 1), dtype=np.float32)
        self.embedding_input = np.zeros((1, 112, 112, 3), dtype=np.float32)
        self.face_database = {}
        self.recognition_start_time = None
        self.current_recognized_face = None
        self.load_database()
    
    def load_models(self):
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
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"✗ Face model loading error: {e}")
            return False
    
    def load_database(self):
        if os.path.exists(FACE_DATABASE_FILENAME):
            try:
                with open(FACE_DATABASE_FILENAME, 'rb') as f:
                    self.face_database = pickle.load(f)
                total_embeddings = sum(len(v) for v in self.face_database.values())
                print(f"✓ Face database loaded: {len(self.face_database)} people, {total_embeddings} embeddings")
            except (Exception, EOFError) as e:
                print(f"✗ Error loading face database (file might be empty or corrupt): {e}")
                self.face_database = {}
        else:
            print("✓ No existing face database found")
            self.face_database = {}
    
    def save_database(self):
        try:
            with open(FACE_DATABASE_FILENAME, 'wb') as f:
                pickle.dump(self.face_database, f)
            print(f"✓ Face database saved to {FACE_DATABASE_FILENAME}")
            return True
        except Exception as e:
            print(f"✗ Error saving face database: {e}")
            return False
    
    def detect_face(self, image):
        if not self.is_loaded or image is None: return None
        interpreter = self.models['detection']['interpreter']
        H0, W0 = image.shape[:2]
        resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self.detection_input[0, :, :, 0] = gray.astype(np.float32) / 255.0
        interpreter.set_tensor(self.models['detection']['input'][0]['index'], self.detection_input)
        interpreter.invoke()
        heatmap = interpreter.get_tensor(self.models['detection']['output'][0]['index'])[0, :, :, 0]
        bbox_data = interpreter.get_tensor(self.models['detection']['output'][1]['index'])[0]
        ys, xs = np.where(heatmap > CONFIDENCE_THRESHOLD)
        if ys.size == 0: return None
        scores = heatmap[ys, xs]
        STRIDE = 8 
        cx, cy = (xs + 0.5) * STRIDE, (ys + 0.5) * STRIDE
        dx1, dy1 = bbox_data[ys, xs, 0] * STRIDE, bbox_data[ys, xs, 1] * STRIDE
        dx2, dy2 = bbox_data[ys, xs, 2] * STRIDE, bbox_data[ys, xs, 3] * STRIDE
        x1, y1, x2, y2 = cx - dx1, cy - dy1, cx + dx2, cy + dy2
        boxes_pix = np.column_stack([x1, y1, x2, y2])
        boxes_pix[:, [0, 2]] *= W0 / 640
        boxes_pix[:, [1, 3]] *= H0 / 480
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

    def compare_faces(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None: return 0.0, False
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity, similarity > SIMILARITY_THRESHOLD
    
    def find_best_match(self, embedding_to_check):
        if embedding_to_check is None or not self.face_database: return None, 0.0
        best_match_name, highest_similarity = None, 0.0
        for name, registered_embeddings_list in self.face_database.items():
            max_similarity_for_person = max(
                (self.compare_faces(embedding_to_check, reg_emb)[0] for reg_emb in registered_embeddings_list),
                default=0.0
            )
            if max_similarity_for_person > highest_similarity:
                highest_similarity = max_similarity_for_person
                best_match_name = name
        return best_match_name, highest_similarity
    
    def register_face(self, name, embedding):
        if not name or embedding is None: return False
        if name in self.face_database:
            self.face_database[name].append(embedding)
        else:
            self.face_database[name] = [embedding]
        print(f"✓ Embedding registered for '{name}'. Total for this person: {len(self.face_database[name])}")
        return True
    
    # <<< FIX: 누락되었던 process_frame 함수를 다시 추가했습니다. >>>
    def process_frame(self, frame):
        if frame is None:
            return frame, None, None
        
        face_result = self.detect_face(frame)
        if not face_result:
            self.recognition_start_time = None
            self.current_recognized_face = None
            return frame, None, None
        
        x, y, w, h, _ = face_result
        face_roi = frame[y:y+h, x:x+w]

        try:
            current_embedding = self.get_face_embedding(face_roi)
            if current_embedding is not None:
                best_match, best_similarity = self.find_best_match(current_embedding)
                
                if best_match and best_similarity > SIMILARITY_THRESHOLD:
                    if self.current_recognized_face == best_match:
                        if self.recognition_start_time is None:
                            self.recognition_start_time = time.time()
                        
                        if time.time() - self.recognition_start_time >= FACE_RECOGNITION_DURATION:
                            return frame, face_result, (best_match, best_similarity, True)
                        else:
                            remaining_time = FACE_RECOGNITION_DURATION - (time.time() - self.recognition_start_time)
                            return frame, face_result, (best_match, best_similarity, False, remaining_time)
                    else:
                        self.current_recognized_face = best_match
                        self.recognition_start_time = time.time()
                        return frame, face_result, (best_match, best_similarity, False, FACE_RECOGNITION_DURATION)
                else:
                    self.recognition_start_time = None
                    self.current_recognized_face = None
                    return frame, face_result, (None, best_similarity, False)
            else:
                return frame, face_result, (None, 0, False)
        except Exception as e:
            print(f"Face recognition processing error: {e}")
            return frame, face_result, (None, 0, False)

class HandTrackingManager:
    """Hand landmark tracking with integrated OCR capabilities."""
    def __init__(self, camera, tkinter_queue=None):
        self.camera = camera
        self.tkinter_queue = tkinter_queue
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # --- Models ---
        self.hands = None
        self.keypoint_classifier = None
        self.ocr_east_net = None
        self.ocr_recognizer_interpreter = None
        self.ocr_input_details = None
        self.ocr_output_details = None
        self.CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;?@[\\]^_`{|}~ "
        # --- End of Models ---

        self.is_initialized = False
        self.current_mode = "Mouse Control"
        self.mode_toggle_cooldown = 0
        self.awaiting_ocr_confirmation = False
        self.awaiting_capture_confirmation = False

        self.mouse_controller = Controller()
        self.screen_width, self.screen_height = self.get_screen_size()
        self.last_finger_pos, self.finger_stable_start_time = None, None
        self.finger_stable_threshold, self.dwell_click_duration = 20, 1.5
        self.capture_points, self.screen_capture_points = [], []

    def initialize(self):
        try:
            original_dir = os.getcwd()
            # --- Hand Tracking Model Loading ---
            if not os.path.exists('handMini2'):
                print("✗ Error: 'handMini2' directory not found.")
                return False
            os.chdir('handMini2')
            self.hands = self.mp_hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.8
            )
            self.keypoint_classifier = KeyPointClassifier()
            os.chdir(original_dir)
            print("✓ Hand tracking models initialized")
            
            # --- OCR Model Loading (Once) ---
            print("Loading OCR models...")
            east_model_path = "frozen_east_text_detection.pb"
            if not os.path.exists(east_model_path):
                print(f"✗ Error: EAST model not found at {east_model_path}")
                return False
            self.ocr_east_net = cv2.dnn.readNet(east_model_path)

            recognizer_model_path = "recognizer_model.tflite"
            if not os.path.exists(recognizer_model_path):
                print(f"✗ Error: Recognizer model not found at {recognizer_model_path}")
                return False
            self.ocr_recognizer_interpreter = tflite.Interpreter(model_path=recognizer_model_path)
            self.ocr_recognizer_interpreter.allocate_tensors()
            self.ocr_input_details = self.ocr_recognizer_interpreter.get_input_details()
            self.ocr_output_details = self.ocr_recognizer_interpreter.get_output_details()
            print("✓ OCR models loaded successfully")

            self.is_initialized = True
            return True
        except Exception as e:
            if 'original_dir' in locals(): os.chdir(original_dir)
            print(f"✗ Hand tracking & OCR initialization error: {e}")
            return False
    
    # <<< FIX: 누락되었던 process_frame 함수를 다시 추가했습니다. >>>
    def process_frame(self, frame):
        if not self.is_initialized or frame is None:
            return frame, None, None
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 1. 손 랜드마크 그리기
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 2. 제스처 인식
                landmark_list = calc_landmark.calc_landmark(frame, hand_landmarks)
                pre_processed = calc_landmark.pre_process_landmark(landmark_list)
                gesture_id = self.keypoint_classifier(pre_processed)
                gesture = self.keypoint_classifier.labels[gesture_id]

                # 3. 인식된 제스처에 따라 행동 결정
                self.handle_gestures(gesture, hand_landmarks, frame, frame.shape)
        
        # 4. 화면에 UI 정보 그리기
        self.draw_ui(frame)
        self.mode_toggle_cooldown = max(0, self.mode_toggle_cooldown - 1)
        
        return frame, gesture, results.multi_hand_landmarks

    # --- OCR 관련 함수들 ---
    def _decode_prediction(self, pred):
        try:
            import tensorflow as tf
            input_len = np.ones(pred.shape[0]) * pred.shape[1]
            decoded, _ = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)
            decoded = decoded[0][0].numpy()
            return ''.join(self.CHARSET[idx] for idx in decoded.flatten() if 0 <= idx < len(self.CHARSET))
        except ImportError:
            print("✗ TensorFlow not found. `pip install tensorflow` is required for OCR.")
            return "[ERROR: TensorFlow missing]"

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
            scoresData = scores[0, 0, y]
            xData0, xData1, xData2, xData3 = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(numCols):
                if scoresData[x] < min_confidence: continue
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos, sin = np.cos(angle), np.sin(angle)
                h, w = xData0[x] + xData2[x], xData1[x] + x
class IntegratedGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition & Hand Tracking System")
        self.root.geometry("800x600")
        self.root.configure(bg='#2c3e50')
        self.tkinter_queue = queue.Queue()
        self.camera = SharedCamera()
        self.face_manager = FaceRecognitionManager(self.camera) # 새로 수정된 Manager 사용
        self.hand_manager = HandTrackingManager(self.camera, self.tkinter_queue)
        self.overlay_window, self.overlay_canvas = None, None
        self.is_logged_in, self.current_user, self.is_running, self.current_mode = False, None, False, "idle"
        self.processing_thread = None
        self.setup_gui()
        self.process_tkinter_queue()
    
    # ... process_tkinter_queue, _start_..., _update_..., _stop_... 함수들은 이전과 동일 ...
    # ... setup_gui, start_face_recognition 등도 이전과 동일 ...
    
    # <<< [MODIFIED] 얼굴 등록 로직을 다중 각도 캡처 플로우로 변경
    def run_registration_flow(self):
        """Multi-angle face registration flow."""
        name = self.simple_input_dialog("Enter name for registration:")
        if not name:
            print("✗ Registration cancelled.")
            return

        angles = ["Front", "Left", "Right"]
        for i, angle in enumerate(angles):
            # Temporarily take over the main loop for registration
            while self.is_running:
                frame = self.camera.get_current_frame()
                if frame is None: continue
                
                display_frame = frame.copy()
                main_text = f"Show {angle} face ({i+1}/{len(angles)})"
                sub_text = "'c': Capture | 'q': Cancel"
                
                # Draw registration guide on the frame
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
                    if embedding is not None:
                        self.face_manager.register_face(name, embedding)
                        break # Go to next angle
                    else:
                        print("✗ Failed to get embedding. Please try again.")
                elif key == ord('q'):
                    print("✗ Registration cancelled by user.")
                    cv2.destroyWindow("Face Registration")
                    return
        
        print(f"✓ Registration complete for '{name}'.")
        self.face_manager.save_database()
        cv2.destroyWindow("Face Registration")

    # ... 나머지 함수들 ...
    # 기존 코드의 IntegratedGUI 클래스에서 나머지 함수들은 그대로 복사해 붙여넣으면 됩니다.
    # register_new_face 함수는 이제 run_registration_flow로 대체되었으므로 삭제합니다.
    # processing_loop에서 'r'키를 눌렀을 때 self.run_registration_flow()를 호출하도록 수정해야 합니다.

    # 아래는 수정이 필요한 processing_loop 함수입니다.
    def processing_loop(self):
        try:
            while self.is_running:
                frame = self.camera.get_current_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                window_title = "System"
                if self.current_mode == "face_recognition":
                    self.process_face_recognition(frame)
                    window_title = "Face Recognition - Login"
                elif self.current_mode == "hand_tracking":
                    self.process_hand_tracking(frame)
                    window_title = f"Hand Tracking - {self.current_user}"
                
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    self.is_running = False
                # <<< [MODIFIED] 'r'키를 눌렀을 때 새로운 등록 함수 호출
                elif key == ord('r') and self.current_mode == "face_recognition":
                    # Temporarily pause processing loop for registration
                    self.run_registration_flow()
                elif self.current_mode == "hand_tracking":
                    self.hand_manager.handle_key(key)
        except Exception as e:
            print(f"✗ Processing loop error: {e}")
        finally:
            self.is_running = False
            self.root.after(100, self.cleanup)

    # --- IntegratedGUI의 나머지 모든 함수들을 여기에 붙여넣어주세요 ---
    # (setup_gui, start_face_recognition, process_face_recognition, ... , cleanup)
    def process_tkinter_queue(self):
        try:
            while not self.tkinter_queue.empty():
                command = self.tkinter_queue.get_nowait()
                cmd_type = command[0]
                if cmd_type == 'start_box_drawing': self._start_screen_box_drawing(command[1], command[2])
                elif cmd_type == 'update_box_drawing': self._update_screen_box_drawing(*command[1:])
                elif cmd_type == 'stop_box_drawing': self._stop_screen_box_drawing()
        except queue.Empty: pass
        self.root.after(50, self.process_tkinter_queue)

    def _start_screen_box_drawing(self, screen_width, screen_height):
        if self.overlay_window: return
        try:
            self.overlay_window = tk.Toplevel(self.root)
            self.overlay_window.attributes('-topmost', True)
            self.overlay_window.overrideredirect(True)
            self.overlay_window.geometry(f"{screen_width}x{screen_height}+0+0")
            self.overlay_window.attributes('-alpha', 0.3) 
            self.overlay_canvas = tk.Canvas(self.overlay_window, highlightthickness=0)
            self.overlay_canvas.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            print(f"✗ Error starting screen box drawing: {e}")
            if self.overlay_window: self.overlay_window.destroy()
            self.overlay_window = None

    def _update_screen_box_drawing(self, x1, y1, x2, y2):
        if not self.overlay_canvas: return
        self.overlay_canvas.delete("all")
        self.overlay_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=3)

    def _stop_screen_box_drawing(self):
        if self.overlay_window:
            try: self.overlay_window.destroy()
            except Exception: pass
            finally: self.overlay_window, self.overlay_canvas = None, None

    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title_label = tk.Label(main_frame, text="Face Recognition & Hand Tracking System", font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
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
        if not self.camera.start_camera_stream(): self.update_status("Failed to start camera stream"); return
        self.update_status("Starting face recognition...")
        self.login_button.config(state=tk.DISABLED)
        self.is_running, self.current_mode = True, "face_recognition"
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

    def process_face_recognition(self, frame):
        _, face_result, recognition_result = self.face_manager.process_frame(frame)
        if face_result:
            x, y, w, h, _ = face_result
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if recognition_result and isinstance(recognition_result, tuple):
                best_match, best_similarity, login_ready, *optional_time = recognition_result
                if login_ready:
                    self.current_user = best_match; self.is_logged_in = True; self.current_mode = "hand_tracking"
                    self.root.after(0, self.on_login_success)
                elif optional_time:
                    cv2.putText(frame, f"Hold for {optional_time[0]:.1f}s", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"{best_match}: {best_similarity:.2f}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                elif best_match:
                    cv2.putText(frame, f"Recognized: {best_match}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Unknown: {best_similarity:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def process_hand_tracking(self, frame):
        self.hand_manager.process_frame(frame)
        cv2.putText(frame, f"User: {self.current_user}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
        self.is_logged_in = False; self.current_user = None; self.current_mode = "idle"; self.is_running = False 
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
        try: self.root.mainloop()
        finally: self.cleanup()
    
    def cleanup(self):
        print("Cleaning up resources...")
        self.is_running = False
        if self.processing_thread and self.processing_thread.is_alive(): self.processing_thread.join(timeout=1)
        self.camera.close()
        self.hand_manager.stop_screen_box_drawing()
        cv2.destroyAllWindows()
        print("✓ Application terminated")


def main():
    """Main entry point"""
    print("=== Integrated Face Recognition & Hand Tracking System ===")
    print("Starting GUI application...")
    
    # Set environment variables to prevent Qt timer issues
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
    
    app = IntegratedGUI()
    app.run()

if __name__ == "__main__":
    main() 
