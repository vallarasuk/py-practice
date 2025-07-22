import cv2
import numpy as np
import mediapipe as mp
import math
import os
import face_recognition
from ultralytics import YOLO

# ---------- Load YOLOv8 for object detection ----------
model = YOLO('yolov8n.pt')

# ---------- Initialize MediaPipe Hands ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ---------- Load known faces ----------
known_encodings = []
known_names = []

for person_name in os.listdir("training_images"):
    person_folder = os.path.join("training_images", person_name)
    if not os.path.isdir(person_folder):
        continue
    for image_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        if face_locations:
            encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

# ---------- Webcam ----------
cap = cv2.VideoCapture(0)
dragging = False
drag_id = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO object detection
    results = model(frame, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()

    # Face recognition
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

    face_labels = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
        face_labels.append(name)

    # Hand detection
    hand_results = hands.process(frame_rgb)

    cursor_x, cursor_y = None, None
    pinch = False

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            index_finger = hand_landmarks.landmark[8]
            thumb_finger = hand_landmarks.landmark[4]

            h, w, _ = frame.shape
            cursor_x = int(index_finger.x * w)
            cursor_y = int(index_finger.y * h)
            thumb_x = int(thumb_finger.x * w)
            thumb_y = int(thumb_finger.y * h)

            distance = math.hypot(thumb_x - cursor_x, thumb_y - cursor_y)
            pinch = distance < 40

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw YOLO detections
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(classes[i])]
        confidence = confidences[i]

        color = (0, 255, 0) if drag_id == i else (255, 0, 0)

        if cursor_x and cursor_y:
            if x1 < cursor_x < x2 and y1 < cursor_y < y2:
                if pinch and not dragging:
                    dragging = True
                    drag_id = i
                elif not pinch and dragging and drag_id == i:
                    dragging = False
                    drag_id = None

        if dragging and drag_id == i and pinch:
            w_box = x2 - x1
            h_box = y2 - y1
            new_x1 = max(cursor_x - w_box // 2, 0)
            new_y1 = max(cursor_y - h_box // 2, 0)
            new_x2 = min(new_x1 + w_box, frame.shape[1])
            new_y2 = min(new_y1 + h_box, frame.shape[0])
            x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw face recognition results
    for (top, right, bottom, left), name in zip(face_locations, face_labels):
        color = (255, 255, 0) if name != "Unknown" else (0, 0, 255)

        if cursor_x and cursor_y:
            if left < cursor_x < right and top < cursor_y < bottom:
                if pinch and not dragging:
                    dragging = True
                    drag_id = name
                elif not pinch and dragging and drag_id == name:
                    dragging = False
                    drag_id = None

        if dragging and drag_id == name and pinch:
            w_box = right - left
            h_box = bottom - top
            new_left = max(cursor_x - w_box // 2, 0)
            new_top = max(cursor_y - h_box // 2, 0)
            new_right = min(new_left + w_box, frame.shape[1])
            new_bottom = min(new_top + h_box, frame.shape[0])
            left, top, right, bottom = new_left, new_top, new_right, new_bottom

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Live Object + Face Recognition with Hand Drag", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
