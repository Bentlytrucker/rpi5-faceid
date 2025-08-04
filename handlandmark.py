import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
from utils import calc_landmark
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from ocr_bridge import recognize_text

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.8
)

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

keypoint_classifier = KeyPointClassifier()
is_capturing = False
capture_points = []
prev_gesture = ""
toggle_cooldown = 0
awaiting_confirmation = False

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not awaiting_confirmation and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


            landmark_list = calc_landmark.calc_landmark(frame, hand_landmarks)
            pre_processed = calc_landmark.pre_process_landmark(landmark_list)
            gesture_id = keypoint_classifier(pre_processed)
            gesture = keypoint_classifier.labels[gesture_id]


            if gesture == "Open" and gesture != prev_gesture and toggle_cooldown == 0:
                is_capturing = not is_capturing
                print(f"capture Mode: {is_capturing}")
                toggle_cooldown = 30


                if not is_capturing and len(capture_points) == 2:
                    awaiting_confirmation = True

            elif gesture == "Pointer" and is_capturing:
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_tip.x * frame.shape[1])
                y = int(index_tip.y * frame.shape[0])

                if len(capture_points) == 0:
                    capture_points.append((x, y))
                elif len(capture_points) == 1:
                    capture_points.append((x, y))
                else:
                    capture_points[1] = (x, y)

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            prev_gesture = gesture

            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    if len(capture_points) == 2:
        cv2.rectangle(frame, capture_points[0], capture_points[1], (0, 0, 255), 2)


    if awaiting_confirmation:
        cv2.putText(frame, "[o] OCR  [r] RESTART", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Trigger OCR", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord('r') and awaiting_confirmation:
        capture_points = []
        awaiting_confirmation = False
        print("RESTART")
    elif key == ord('o') and awaiting_confirmation:
        (x1, y1), (x2, y2) = capture_points
        x1, x2 = max(0, min(x1, x2)), min(frame.shape[1], max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(frame.shape[0], max(y1, y2))

        # OCR 넘겨줄 정보
        roi = frame[y1:y2, x1:x2].copy()  # 실제 이미지 (numpy array, shape = (h, w, 3))
        bbox = (x1, y1, x2, y2) # 상자 좌표
        text = recognize_text(roi, bbox) #ocr모델함수호출. 지금은 그냥 dummy

        print("OCR Result:", text)
        capture_points = []
        awaiting_confirmation = False

    toggle_cooldown = max(toggle_cooldown - 1, 0)

cv2.destroyAllWindows()

