#!/usr/bin/env python3
"""
Raspberry Pi 5 Face Recognition System
Real-time face recognition optimized for Raspberry Pi 5 with Camera Module 3.
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
DETECTION_MODEL_PATH = "./models/Lightweight-Face-Detection.tflite"
EMBEDDING_MODEL_PATH = "./models/MobileFaceNet_9925_9680.tflite"
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
        self.cache_size = 100  # Increased from 50
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
        if current_time - self.last_system_check >= 10.0:  # Reduced frequency
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

class DisplayManager:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.colors = {'green': (0, 255, 0), 'red': (0, 0, 255), 'white': (255, 255, 255)}
    
    def draw_face_box(self, frame, face_result):
        if not face_result:
            return
        x, y, w, h, confidence = face_result
        cv2.rectangle(frame, (x, y), (x+w, y+h), self.colors['green'], 2)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y-30), 
                   self.font, 0.5, self.colors['green'], 1)
    
    def draw_recognition_result(self, frame, face_result, best_match, best_similarity):
        if not face_result:
            return
        x, y, w, h, _ = face_result
        
        if best_match and best_similarity > SIMILARITY_THRESHOLD:
            cv2.putText(frame, f"{best_match}", (x, y-10), 
                       self.font, 0.7, self.colors['green'], 2)
            cv2.putText(frame, f"Sim: {best_similarity:.2f}", (x, y+h+20), 
                       self.font, 0.5, self.colors['green'], 1)
        else:
            cv2.putText(frame, "Unknown", (x, y-10), 
                       self.font, 0.7, self.colors['red'], 2)
            if best_match:
                cv2.putText(frame, f"Best: {best_similarity:.2f}", (x, y+h+20), 
                           self.font, 0.5, self.colors['red'], 1)
    
    def draw_system_info(self, frame, fps, cpu_percent, memory_percent, face_count, embedding_info=None, db_info=None):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   self.font, 0.7, self.colors['white'], 2)
        cv2.putText(frame, f"CPU: {cpu_percent:.1f}%", (10, 60), 
                   self.font, 0.5, self.colors['white'], 1)
        cv2.putText(frame, f"RAM: {memory_percent:.1f}%", (10, 80), 
                   self.font, 0.5, self.colors['white'], 1)
        cv2.putText(frame, f"Registered: {face_count}", (10, 100), 
                   self.font, 0.5, self.colors['white'], 1)
        
        # Display embedding information
        if embedding_info:
            cv2.putText(frame, f"Embedding: {embedding_info['dimension']}D", (10, 120), 
                       self.font, 0.5, self.colors['white'], 1)
            cv2.putText(frame, f"Embedding Mem: {embedding_info['memory_kb']:.1f}KB", (10, 140), 
                       self.font, 0.5, self.colors['white'], 1)
            cv2.putText(frame, f"Cache Hits: {embedding_info['cache_hits']}", (10, 160), 
                       self.font, 0.5, self.colors['white'], 1)
            cv2.putText(frame, f"Cache Misses: {embedding_info['cache_misses']}", (10, 180), 
                       self.font, 0.5, self.colors['white'], 1)
        
        # Display database information
        if db_info:
            cv2.putText(frame, f"DB Size: {db_info['total_memory_kb']:.1f}KB", (10, 200), 
                       self.font, 0.5, self.colors['white'], 1)
        
        cv2.putText(frame, "Press 'r' to register, 'i' for info, 'q' to quit", (10, 220), 
                   self.font, 0.5, self.colors['white'], 1)
    
    def show_frame(self, frame):
        cv2.imshow('Raspberry Pi Face Recognition (TFLite Runtime)', frame)
    
    def wait_for_key(self, timeout=1):
        return cv2.waitKey(timeout) & 0xFF

class FaceRecognitionSystem:
    def __init__(self):
        self.camera = CameraManager()
        self.model_manager = ModelManager()
        self.face_database = FaceDatabase()
        self.system_monitor = SystemMonitor()
        self.display_manager = DisplayManager()
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        # Advanced caching for performance
        self.last_face_result = None
        self.last_recognition_result = None
        self.frame_skip_counter = 0
        self.processing_frequency = 3  # Process every 3rd frame
    
    def initialize(self):
        print("=== Raspberry Pi 5 + Camera Module 3 Real-time Stream Face Recognition System ===")
        print("Features: Picamera2 + TFLite Runtime, ultra-optimized performance")
        print("Keys: 'r'(register), 's'(save), 'l'(load), 'i'(info), 'q'(quit)")
        
        if not self.camera.initialize():
            print("✗ Failed to initialize camera")
            return False
        
        if not self.model_manager.load_models():
            print("✗ Failed to load models")
            self.camera.close()
            return False
        
        print(f"✓ Registered faces: {self.face_database.get_face_count()}")
        print("✓ System started - press 'q' to quit")
        return True
    
    def process_frame(self, frame):
        if frame is None:
            return None, None, None
        
        # Advanced frame skipping for maximum performance
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.processing_frequency != 0:
            return frame, self.last_face_result, self.last_recognition_result
        
        face_result = self.model_manager.detect_face(frame)
        if not face_result:
            self.last_face_result = None
            self.last_recognition_result = None
            return frame, None, None
        
        x, y, w, h, _ = face_result
        face_roi = frame[y:y+h, x:x+w]
        
        try:
            # Create face hash for caching
            face_hash = hash(face_roi.tobytes())
            
            # Check if we already have this face cached
            if face_hash in self.model_manager.embedding_cache:
                current_embedding = self.model_manager.embedding_cache[face_hash]
                print("✓ Using cached embedding (no inference needed)")
            else:
                # Only generate new embedding if not cached
                current_embedding = self.model_manager.get_face_embedding(face_roi)
            
            if current_embedding is not None:
                # First, check if this face is already registered with high confidence
                is_registered, registered_name = self.face_database.is_face_registered(
                    current_embedding, self.model_manager, threshold=0.85
                )
                
                if is_registered:
                    # Use the registered embedding directly - no need for further comparison
                    print(f"✓ Fast recognition: {registered_name} (using registered embedding)")
                    self.last_face_result = face_result
                    self.last_recognition_result = (registered_name, 0.95)  # High confidence
                    return frame, face_result, (registered_name, 0.95)
                else:
                    # Perform full comparison for unknown or low-confidence matches
                    best_match, best_similarity = self.face_database.find_best_match(
                        current_embedding, self.model_manager
                    )
                    self.last_face_result = face_result
                    self.last_recognition_result = (best_match, best_similarity)
                    return frame, face_result, (best_match, best_similarity)
            else:
                self.last_face_result = face_result
                self.last_recognition_result = (None, 0)
                return frame, face_result, (None, 0)
        except Exception as e:
            print(f"Face recognition error: {e}")
            self.last_face_result = face_result
            self.last_recognition_result = (None, 0)
            return frame, face_result, (None, 0)
    
    def handle_user_input(self, key, face_result, frame):
        if key == ord('q'):
            return False
        elif key == ord('r') and face_result:
            self.register_face(face_result, frame)
        elif key == ord('s'):
            self.face_database.save_database()
        elif key == ord('l'):
            self.face_database.load_database()
        elif key == ord('i'):
            self.show_system_info()
        return True
    
    def register_face(self, face_result, frame):
        name = input("Enter name for the detected face: ").strip()
        if name:
            try:
                x, y, w, h, _ = face_result
                face_roi = frame[y:y+h, x:x+w]
                embedding = self.model_manager.get_face_embedding(face_roi)
                if embedding is not None:
                    self.face_database.register_face(name, embedding)
                    print(f"✓ Face embedding size: {embedding.nbytes} bytes ({embedding.nbytes/1024:.1f} KB)")
                    print(f"✓ Embedding dimension: {len(embedding)}")
                else:
                    print("✗ Failed to extract face embedding")
            except Exception as e:
                print(f"✗ Face registration error: {e}")
    
    def show_system_info(self):
        """Display detailed system information."""
        print("\n=== System Information ===")
        
        # Embedding information
        embedding_info = self.model_manager.get_embedding_info()
        print(f"Embedding Dimension: {embedding_info['dimension']}")
        print(f"Embedding Memory Usage: {embedding_info['memory_bytes']} bytes ({embedding_info['memory_kb']:.1f} KB)")
        print(f"Embedding Cache: {embedding_info['cache_size']}/{embedding_info['max_cache_size']}")
        print(f"Cache Hits: {embedding_info['cache_hits']}")
        print(f"Cache Misses: {embedding_info['cache_misses']}")
        
        # Database information
        db_info = self.face_database.get_database_size()
        print(f"Database Faces: {db_info['face_count']}")
        print(f"Database Size: {db_info['total_memory_bytes']} bytes ({db_info['total_memory_kb']:.1f} KB)")
        if db_info['face_count'] > 0:
            print(f"Average Memory per Face: {db_info['avg_memory_per_face']:.1f} bytes")
        
        # System information
        print(f"Current FPS: {self.system_monitor.get_fps():.1f}")
        print(f"CPU Usage: {self.system_monitor.get_cpu_percent():.1f}%")
        print(f"Memory Usage: {self.system_monitor.get_memory_percent():.1f}%")
        print("========================\n")
    
    def run(self):
        if not self.initialize():
            return
        
        self.is_running = True
        
        try:
            while self.is_running:
                frame = self.camera.capture_frame()
                
                if frame is None:
                    print("✗ Frame capture failed - retrying...")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                self.system_monitor.update_fps()
                
                # Update system info very infrequently
                if self.frame_count % 60 == 0:
                    self.system_monitor.update_system_info()
                
                frame, face_result, recognition_result = self.process_frame(frame)
                
                if face_result:
                    self.display_manager.draw_face_box(frame, face_result)
                    if recognition_result:
                        best_match, best_similarity = recognition_result
                        self.display_manager.draw_recognition_result(
                            frame, face_result, best_match, best_similarity
                        )
                
                # Get embedding and database info for display
                embedding_info = self.model_manager.get_embedding_info()
                db_info = self.face_database.get_database_size()
                
                self.display_manager.draw_system_info(
                    frame,
                    self.system_monitor.get_fps(),
                    self.system_monitor.get_cpu_percent(),
                    self.system_monitor.get_memory_percent(),
                    self.face_database.get_face_count(),
                    embedding_info,
                    db_info
                )
                
                self.display_manager.show_frame(frame)
                
                key = self.display_manager.wait_for_key(1)
                if not self.handle_user_input(key, face_result, frame):
                    break
                
        except KeyboardInterrupt:
            print("\n✓ Interrupted by user")
        except Exception as e:
            print(f"✗ System error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.is_running = False
        self.camera.close()
        cv2.destroyAllWindows()
        
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        print(f"✓ System terminated - Average FPS: {avg_fps:.1f}")
        print(f"✓ Total frames processed: {self.frame_count}")

def main():
    system = FaceRecognitionSystem()
    system.run()

if __name__ == "__main__":
    main() 