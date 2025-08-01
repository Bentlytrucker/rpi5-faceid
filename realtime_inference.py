import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import os
import time
import pickle
import psutil
import threading
import queue
from pathlib import Path
from picamera2 import Picamera2

CONFIDENCE_THRESHOLD = 0.15
SIMILARITY_THRESHOLD = 0.6
MAX_FACE_SIZE = 400
MIN_FACE_SIZE = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 15

def init_picamera2():
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameDurationLimits": (33333, 33333)}
        )
        picam2.configure(config)
        picam2.start()
        
        print("✓ Picamera2 initialized successfully")
        return picam2
        
    except Exception as e:
        print(f"✗ Picamera2 initialization error: {e}")
        return None

def capture_frame_picamera2(picam2):
    try:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame_bgr
    except Exception as e:
        print(f"✗ Frame capture error: {e}")
        return None

def load_optimized_models():
    models = {}
    
    detection_path = "./models/Lightweight-Face-Detection.tflite"
    if os.path.exists(detection_path):
        detection_interpreter = tflite.Interpreter(model_path=detection_path)
        detection_interpreter.allocate_tensors()
        models['detection'] = {
            'interpreter': detection_interpreter,
            'input': detection_interpreter.get_input_details(),
            'output': detection_interpreter.get_output_details()
        }
        print("✓ Lightweight Face Detection loaded")
    else:
        print("✗ Lightweight Face Detection model not found")
        return None
    
    embedding_path = "./models/MobileFaceNet_9925_9680.tflite"
    if os.path.exists(embedding_path):
        embedding_interpreter = tflite.Interpreter(model_path=embedding_path)
        embedding_interpreter.allocate_tensors()
        models['embedding'] = {
            'interpreter': embedding_interpreter,
            'input': embedding_interpreter.get_input_details(),
            'output': embedding_interpreter.get_output_details()
        }
        print("✓ MobileFaceNet loaded")
    else:
        print("✗ MobileFaceNet model not found")
        return None
    
    return models

def preprocess_for_detection_pi(image):
    resized = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    channeled = np.expand_dims(normalized, axis=-1)
    batched = np.expand_dims(channeled, axis=0)
    
    return batched, resized

def preprocess_face_pi(face_img, target_size=(112, 112)):
    face_resized = cv2.resize(face_img, target_size, interpolation=cv2.INTER_LINEAR)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_normalized = face_rgb.astype(np.float32) / 255.0
    face_batch = np.expand_dims(face_normalized, axis=0)
    
    return face_batch

def detect_single_face_pi(models, image):
    detection_info = models['detection']
    interpreter = detection_info['interpreter']
    input_details = detection_info['input']
    output_details = detection_info['output']
    
    input_data, resized_image = preprocess_for_detection_pi(image)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    heatmap = interpreter.get_tensor(output_details[0]['index'])
    bbox = interpreter.get_tensor(output_details[1]['index'])
    
    H0, W0 = image.shape[:2]
    H_in, W_in = 480, 640
    G_H, G_W = 60, 80
    
    STRIDE_X = W_in / G_W
    STRIDE_Y = H_in / G_H
    
    heatmap_2d = heatmap[0, :, :, 0]
    ys, xs = np.where(heatmap_2d > CONFIDENCE_THRESHOLD)
    
    if ys.size == 0:
        return None
    
    scores = heatmap_2d[ys, xs]
    
    cx = (xs + 0.5) * STRIDE_X
    cy = (ys + 0.5) * STRIDE_Y
    
    boxes = []
    for i in range(len(ys)):
        y, x = ys[i], xs[i]
        
        dy1, dx1, dy2, dx2 = bbox[0, y, x, :]
        
        x1 = cx[i] - dx1 * STRIDE_X
        y1 = cy[i] - dy1 * STRIDE_Y
        x2 = cx[i] + dx2 * STRIDE_X
        y2 = cy[i] + dy2 * STRIDE_Y
        
        boxes.append([x1, y1, x2, y2])
    
    boxes_pix = np.array(boxes).copy()
    boxes_pix[:, [0,2]] *= W0 / W_in
    boxes_pix[:, [1,3]] *= H0 / H_in
    
    if len(boxes_pix) > 0:
        idxs = np.asarray(cv2.dnn.NMSBoxes(
            bboxes=[[x1,y1,x2-x1,y2-y1] for x1,y1,x2,y2 in boxes_pix],
            scores=scores.tolist(),
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=0.3)).flatten()
        
        if len(idxs) > 0:
            best_idx = idxs[np.argmax(scores[idxs])]
            x1, y1, x2, y2 = boxes_pix[best_idx]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if x1 >= 0 and y1 >= 0 and x2 < W0 and y2 < H0 and x2 > x1 and y2 > y1:
                w, h = x2 - x1, y2 - y1
                if MIN_FACE_SIZE <= w <= MAX_FACE_SIZE and MIN_FACE_SIZE <= h <= MAX_FACE_SIZE:
                    return (x1, y1, w, h, scores[best_idx])
    
    return None

def get_face_embedding_pi(models, face_img):
    embedding_info = models['embedding']
    interpreter = embedding_info['interpreter']
    input_details = embedding_info['input']
    output_details = embedding_info['output']
    
    processed_face = preprocess_face_pi(face_img)
    interpreter.set_tensor(input_details[0]['index'], processed_face)
    interpreter.invoke()
    embedding = interpreter.get_tensor(output_details[0]['index'])
    return embedding.flatten()

def compare_faces_pi(embedding1, embedding2):
    embedding1_norm = embedding1 / np.linalg.norm(embedding1)
    embedding2_norm = embedding2 / np.linalg.norm(embedding2)
    
    similarity = np.dot(embedding1_norm, embedding2_norm)
    is_same_person = similarity > SIMILARITY_THRESHOLD
    
    return similarity, is_same_person

def save_face_database_pi(face_db, filename="pi_face_database.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(face_db, f)
    print(f"✓ Face database saved: {filename}")

def load_face_database_pi(filename="pi_face_database.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return {}

def get_system_info():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    memory_used = memory.used / (1024**3)
    memory_total = memory.total / (1024**3)
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'memory_used_gb': memory_used,
        'memory_total_gb': memory_total
    }

def main_stream():
    print("=== Raspberry Pi 5 + Camera Module 3 Real-time Stream Face Recognition System ===")
    print("Features: Picamera2 + TFLite Runtime, optimized performance")
    print("Keys: 'r'(register), 's'(save), 'l'(load), 'q'(quit)")
    
    picam2 = init_picamera2()
    if picam2 is None:
        return
    
    models = load_optimized_models()
    if models is None:
        picam2.close()
        return
    
    face_database = load_face_database_pi()
    print(f"✓ Registered faces: {len(face_database)}")
    
    frame_count = 0
    start_time = time.time()
    fps_start_time = time.time()
    fps_count = 0
    current_fps = 0
    
    last_system_check = time.time()
    system_info = {'cpu_percent': 0, 'memory_percent': 0}
    
    print("✓ System started - press 'q' to quit")
    
    try:
        while True:
            frame = capture_frame_picamera2(picam2)
            
            if frame is None:
                print("✗ Frame capture failed - retrying...")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            fps_count += 1
            
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                current_fps = fps_count / (current_time - fps_start_time)
                fps_count = 0
                fps_start_time = current_time
            
            face_result = detect_single_face_pi(models, frame)
            
            if face_result:
                x, y, w, h, confidence = face_result
                face_roi = frame[y:y+h, x:x+w]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                try:
                    current_embedding = get_face_embedding_pi(models, face_roi)
                    
                    best_match = None
                    best_similarity = 0
                    
                    for name, registered_embedding in face_database.items():
                        similarity, is_same = compare_faces_pi(current_embedding, registered_embedding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = name
                    
                    if best_match and best_similarity > SIMILARITY_THRESHOLD:
                        cv2.putText(frame, f"{best_match}", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Sim: {best_similarity:.2f}", (x, y+h+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if best_match:
                            cv2.putText(frame, f"Best: {best_similarity:.2f}", (x, y+h+20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                            
                except Exception as e:
                    print(f"Face recognition error: {e}")
                    cv2.putText(frame, "Error", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if current_time - last_system_check >= 5.0:
                system_info = get_system_info()
                last_system_check = current_time
            
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"CPU: {system_info.get('cpu_percent', 0):.1f}%", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"RAM: {system_info.get('memory_percent', 0):.1f}%", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Registered: {len(face_database)}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'r' to register, 'q' to quit", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Raspberry Pi Face Recognition (TFLite Runtime)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r') and face_result:
                name = input("Enter name for the detected face: ").strip()
                if name:
                    try:
                        x, y, w, h, _ = face_result
                        face_roi = frame[y:y+h, x:x+w]
                        embedding = get_face_embedding_pi(models, face_roi)
                        face_database[name] = embedding
                        print(f"✓ '{name}' face registered successfully")
                    except Exception as e:
                        print(f"✗ Face registration error: {e}")
            
            elif key == ord('s'):
                save_face_database_pi(face_database)
            
            elif key == ord('l'):
                face_database = load_face_database_pi()
                print(f"✓ Loaded registered faces: {len(face_database)}")
    
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
    except Exception as e:
        print(f"✗ System error: {e}")
    finally:
        picam2.close()
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"✓ System terminated - Average FPS: {avg_fps:.1f}")
        print(f"✓ Total frames processed: {frame_count}")

if __name__ == "__main__":
    main_stream() 