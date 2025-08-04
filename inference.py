#!/usr/bin/env python3
"""
Face Recognition Inference Module
Adapted from realtime_face_recognition.py for GUI integration.
"""

import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import os
import time
import pickle
import psutil
import threading
import sys
from picamera2 import Picamera2

# Configuration
CONFIDENCE_THRESHOLD = 0.15
SIMILARITY_THRESHOLD = 0.6
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
DETECTION_MODEL_PATH = "qualcomm_lightweight/Lightweight-Face-Detection.tflite"
EMBEDDING_MODEL_PATH = "mobilefacenet_conversion/MobileFaceNet_9925_9680.tflite"
FACE_DATABASE_FILENAME = "pi_face_database.pkl"

class CameraManager:
    def __init__(self):
        self.picam2 = None
        self.is_initialized = False
    
    def initialize(self):
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
                controls={"FrameDurationLimits": (33333, 33333)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            print("✓ Picamera2 initialized successfully")
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"✗ Picamera2 initialization error: {e}")
            return False
    
    def capture_frame(self):
        if not self.is_initialized or self.picam2 is None:
            return None
        try:
            frame = self.picam2.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"✗ Frame capture error: {e}")
            return None
    
    def close(self):
        if self.picam2 is not None:
            self.picam2.close()

class ModelManager:
    def __init__(self):
        self.models = {}
        self.is_loaded = False
        # Pre-allocate arrays for better performance
        self.detection_input = np.zeros((1, 480, 640, 1), dtype=np.float32)
        self.embedding_input = np.zeros((1, 112, 112, 3), dtype=np.float32)
        # Cache for face embeddings
        self.embedding_cache = {}
        self.cache_size = 100
        self.cache_hits = 0
        self.cache_misses = 0
        # Embedding size tracking
        self.embedding_dimension = None
        self.embedding_memory_usage = 0
    
    def load_models(self):
        try:
            # Load detection model
            if not os.path.exists(DETECTION_MODEL_PATH):
                print("✗ Lightweight Face Detection model not found")
                return False
            
            detection_interpreter = tflite.Interpreter(model_path=DETECTION_MODEL_PATH)
            detection_interpreter.allocate_tensors()
            self.models['detection'] = {
                'interpreter': detection_interpreter,
                'input': detection_interpreter.get_input_details(),
                'output': detection_interpreter.get_output_details()
            }
            print("✓ Lightweight Face Detection loaded")
            
            # Load embedding model
            if not os.path.exists(EMBEDDING_MODEL_PATH):
                print("✗ MobileFaceNet model not found")
                return False
            
            embedding_interpreter = tflite.Interpreter(model_path=EMBEDDING_MODEL_PATH)
            embedding_interpreter.allocate_tensors()
            self.models['embedding'] = {
                'interpreter': embedding_interpreter,
                'input': embedding_interpreter.get_input_details(),
                'output': embedding_interpreter.get_output_details()
            }
            print("✓ MobileFaceNet loaded")
            
            # Get embedding dimension from model
            output_details = embedding_interpreter.get_output_details()
            if output_details:
                self.embedding_dimension = output_details[0]['shape'][-1]
                print(f"✓ Embedding dimension: {self.embedding_dimension}")
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"✗ Model loading error: {e}")
            return False
    
    def detect_face(self, image):
        if not self.is_loaded or 'detection' not in self.models:
            return None
        
        detection_info = self.models['detection']
        interpreter = detection_info['interpreter']
        input_details = detection_info['input']
        output_details = detection_info['output']
        
        # Fixed preprocessing
        resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        self.detection_input[0, :, :, 0] = gray.astype(np.float32) / 255.0
        
        interpreter.set_tensor(input_details[0]['index'], self.detection_input)
        interpreter.invoke()
        
        heatmap = interpreter.get_tensor(output_details[0]['index'])
        bbox = interpreter.get_tensor(output_details[1]['index'])
        
        H0, W0 = image.shape[:2]
        H_in, W_in = 480, 640
        G_H, G_W = 60, 80
        STRIDE_X, STRIDE_Y = W_in / G_W, H_in / G_H
        
        heatmap_2d = heatmap[0, :, :, 0]
        ys, xs = np.where(heatmap_2d > CONFIDENCE_THRESHOLD)
        
        if ys.size == 0:
            return None
        
        scores = heatmap_2d[ys, xs]
        cx = (xs + 0.5) * STRIDE_X
        cy = (ys + 0.5) * STRIDE_Y
        
        # Vectorized box calculation
        boxes = []
        for i in range(len(ys)):
            y, x = ys[i], xs[i]
            dy1, dx1, dy2, dx2 = bbox[0, y, x, :]
            x1 = cx[i] - dx1 * STRIDE_X
            y1 = cy[i] - dy1 * STRIDE_Y
            x2 = cx[i] + dx2 * STRIDE_X
            y2 = cy[i] + dy2 * STRIDE_Y
            boxes.append([x1, y1, x2, y2])
        
        if not boxes:
            return None
        
        boxes_pix = np.array(boxes)
        boxes_pix[:, [0,2]] *= W0 / W_in
        boxes_pix[:, [1,3]] *= H0 / H_in
        
        # Optimized NMS
        bboxes_for_nms = [[x1,y1,x2-x1,y2-y1] for x1,y1,x2,y2 in boxes_pix]
        idxs = cv2.dnn.NMSBoxes(bboxes_for_nms, scores.tolist(), CONFIDENCE_THRESHOLD, 0.3)
        
        if len(idxs) > 0:
            best_idx = idxs[np.argmax(scores[idxs])]
            x1, y1, x2, y2 = boxes_pix[best_idx]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if x1 >= 0 and y1 >= 0 and x2 < W0 and y2 < H0 and x2 > x1 and y2 > y1:
                w, h = x2 - x1, y2 - y1
                if 30 <= w <= 400 and 30 <= h <= 400:
                    return (x1, y1, w, h, scores[best_idx])
        return None
    
    def get_face_embedding(self, face_img):
        if not self.is_loaded or 'embedding' not in self.models:
            return None
        
        # Create hash for caching
        face_hash = hash(face_img.tobytes())
        if face_hash in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[face_hash]
        else:
            self.cache_misses += 1
        
        embedding_info = self.models['embedding']
        interpreter = embedding_info['interpreter']
        input_details = embedding_info['input']
        output_details = embedding_info['output']
        
        # Fixed preprocessing
        face_resized = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        self.embedding_input[0] = face_rgb.astype(np.float32) / 255.0
        
        interpreter.set_tensor(input_details[0]['index'], self.embedding_input)
        interpreter.invoke()
        embedding = interpreter.get_tensor(output_details[0]['index']).flatten()
        
        # Update memory usage tracking
        self.embedding_memory_usage = embedding.nbytes
        
        # Cache the result
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest entry
            self.embedding_cache.pop(next(iter(self.embedding_cache)))
        self.embedding_cache[face_hash] = embedding
        
        return embedding
    
    def compare_faces(self, embedding1, embedding2):
        # Fixed comparison
        if embedding1 is None or embedding2 is None:
            return 0.0, False
        
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0, False
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return similarity, similarity > SIMILARITY_THRESHOLD
    
    def get_embedding_info(self):
        """Get embedding dimension and memory usage information."""
        return {
            'dimension': self.embedding_dimension,
            'memory_bytes': self.embedding_memory_usage,
            'memory_kb': self.embedding_memory_usage / 1024,
            'cache_size': len(self.embedding_cache),
            'max_cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses
        }

class FaceDatabase:
    def __init__(self):
        self.faces = {}
        self.load_database()
    
    def load_database(self):
        if os.path.exists(FACE_DATABASE_FILENAME):
            try:
                with open(FACE_DATABASE_FILENAME, 'rb') as f:
                    self.faces = pickle.load(f)
                print(f"✓ Face database loaded: {len(self.faces)} faces")
            except Exception as e:
                print(f"✗ Error loading face database: {e}")
                self.faces = {}
        else:
            print("✓ No existing face database found, starting fresh")
            self.faces = {}
    
    def save_database(self):
        try:
            with open(FACE_DATABASE_FILENAME, 'wb') as f:
                pickle.dump(self.faces, f)
            print(f"✓ Face database saved: {FACE_DATABASE_FILENAME}")
            return True
        except Exception as e:
            print(f"✗ Error saving face database: {e}")
            return False
    
    def register_face(self, name, embedding):
        if not name or embedding is None:
            return False
        try:
            self.faces[name] = embedding
            print(f"✓ '{name}' face registered successfully")
            return True
        except Exception as e:
            print(f"✗ Error registering face: {e}")
            return False
    
    def find_best_match(self, embedding, model_manager):
        if embedding is None or not self.faces:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        # Optimized comparison loop
        for name, registered_embedding in self.faces.items():
            similarity, _ = model_manager.compare_faces(embedding, registered_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        return best_match, best_similarity
    
    def is_face_registered(self, embedding, model_manager, threshold=0.8):
        """Quick check if a face is already registered with high similarity."""
        if embedding is None or not self.faces:
            return False, None
        
        for name, registered_embedding in self.faces.items():
            similarity, is_match = model_manager.compare_faces(embedding, registered_embedding)
            if is_match and similarity > threshold:
                return True, name
        
        return False, None
    
    def get_registered_embedding(self, name):
        """Get embedding for a registered face name."""
        return self.faces.get(name, None)
    
    def get_face_count(self):
        return len(self.faces)
    
    def get_database_size(self):
        """Get database size information."""
        total_memory = 0
        for name, embedding in self.faces.items():
            total_memory += embedding.nbytes
        
        return {
            'face_count': len(self.faces),
            'total_memory_bytes': total_memory,
            'total_memory_kb': total_memory / 1024,
            'total_memory_mb': total_memory / (1024 * 1024),
            'avg_memory_per_face': total_memory / len(self.faces) if self.faces else 0
        }

class SystemMonitor:
    def __init__(self):
        self.last_system_check = time.time()
        self.system_info = {'cpu_percent': 0, 'memory_percent': 0}
        self.fps_start_time = time.time()
        self.fps_count = 0
        self.current_fps = 0
    
    def update_fps(self):
        self.fps_count += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_count / (current_time - self.fps_start_time)
            self.fps_count = 0
            self.fps_start_time = current_time
    
    def update_system_info(self):
        current_time = time.time()
        if current_time - self.last_system_check >= 10.0:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                self.system_info = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent
                }
                self.last_system_check = current_time
            except Exception as e:
                print(f"✗ System monitoring error: {e}")
    
    def get_fps(self):
        return self.current_fps
    
    def get_cpu_percent(self):
        return self.system_info.get('cpu_percent', 0)
    
    def get_memory_percent(self):
        return self.system_info.get('memory_percent', 0)

class FaceRecognitionInference:
    def __init__(self):
        self.camera = CameraManager()
        self.model_manager = ModelManager()
        self.face_database = FaceDatabase()
        self.system_monitor = SystemMonitor()
        self.is_initialized = False
        self.current_face_result = None
        self.current_recognition_result = None
    
    def initialize(self):
        """Initialize the face recognition system."""
        print("=== Initializing Face Recognition System ===")
        
        if not self.camera.initialize():
            print("✗ Failed to initialize camera")
            return False
        
        if not self.model_manager.load_models():
            print("✗ Failed to load models")
            self.camera.close()
            return False
        
        print(f"✓ Registered faces: {self.face_database.get_face_count()}")
        print("✓ System initialized successfully")
        self.is_initialized = True
        return True
    
    def capture_frame(self):
        """Capture a frame from the camera."""
        if not self.is_initialized:
            return None
        return self.camera.capture_frame()
    
    def process_frame(self, frame):
        """Process a frame for face detection and recognition."""
        if frame is None or not self.is_initialized:
            return None, None
        
        # Update system monitor
        self.system_monitor.update_fps()
        
        # Detect face
        face_result = self.model_manager.detect_face(frame)
        self.current_face_result = face_result
        
        if not face_result:
            self.current_recognition_result = None
            return None, None
        
        # Extract face ROI and get embedding
        x, y, w, h, _ = face_result
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            current_embedding = self.model_manager.get_face_embedding(face_roi)
            
            if current_embedding is not None:
                # Find best match in database
                best_match, best_similarity = self.face_database.find_best_match(
                    current_embedding, self.model_manager
                )
                self.current_recognition_result = (best_match, best_similarity)
                return face_result, (best_match, best_similarity)
            else:
                self.current_recognition_result = None
                return face_result, None
                
        except Exception as e:
            print(f"Face recognition error: {e}")
            self.current_recognition_result = None
            return face_result, None
    
    def get_current_face_result(self):
        """Get the current face detection result."""
        return self.current_face_result
    
    def register_face(self, name, face_result):
        """Register a new face with the given name."""
        if not face_result:
            return False
        
        try:
            # Get the current frame to extract face ROI
            frame = self.capture_frame()
            if frame is None:
                return False
            
            x, y, w, h, _ = face_result
            face_roi = frame[y:y+h, x:x+w]
            
            # Get embedding
            embedding = self.model_manager.get_face_embedding(face_roi)
            if embedding is None:
                return False
            
            # Register in database
            success = self.face_database.register_face(name, embedding)
            if success:
                # Save database
                self.face_database.save_database()
            
            return success
            
        except Exception as e:
            print(f"Face registration error: {e}")
            return False
    
    def is_face_registered(self, name):
        """Check if a face with the given name is registered."""
        return name in self.face_database.faces
    
    def get_fps(self):
        """Get current FPS."""
        return self.system_monitor.get_fps()
    
    def get_cpu_percent(self):
        """Get current CPU usage."""
        return self.system_monitor.get_cpu_percent()
    
    def get_memory_percent(self):
        """Get current memory usage."""
        return self.system_monitor.get_memory_percent()
    
    def get_face_count(self):
        """Get number of registered faces."""
        return self.face_database.get_face_count()
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera:
            self.camera.close()
        print("✓ Face recognition system cleaned up")

if __name__ == "__main__":
    # Test the inference system
    inference = FaceRecognitionInference()
    if inference.initialize():
        print("✓ Inference system test successful")
        inference.cleanup()
    else:
        print("✗ Inference system test failed") 
