import cv2
import numpy as np
import mediapipe as mp
import math
import os
import face_recognition
from ultralytics import YOLO
import time
import threading
import queue

# Constants
TARGET_FPS = 30
# Frame skip for different tasks. These now control how often tasks are *triggered* in their own threads.
YOLO_FRAME_SKIP = 2 # Process YOLO every 2nd frame
FACE_RECOGNITION_FRAME_SKIP = 5 # Process face recognition every 5th frame
HANDS_FRAME_SKIP = 1 # Process hands on every frame (or adjust if too slow)

PINCH_THRESHOLD = 40 # Original threshold

# Thread-safe queues for communication
frame_queue = queue.Queue(maxsize=1) # To store the latest frame for processing
results_queue = queue.Queue(maxsize=1) # To pass detection results to main thread

# Shared state for results (updated by worker threads)
current_yolo_results = {'boxes': [], 'classes': [], 'confidences': []}
current_face_results = {'locations': [], 'labels': []}
current_hand_results = {'cursor_x': None, 'cursor_y': None, 'pinch': False, 'landmarks': None}

# Locks for safe access to shared state
yolo_lock = threading.Lock()
face_lock = threading.Lock()
hand_lock = threading.Lock()

# Global flags for thread control
running = True

# ---------- Load YOLOv8 for object detection ----------
# Consider yolov8s.pt for better speed/accuracy balance.
# Load model once. If GPU is available, YOLO will automatically use it.
try:
    model = YOLO('yolov8n.pt')
    print("YOLOv8 model loaded.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}. Please ensure 'yolov8n.pt' is in the same directory or accessible.")
    exit()

# ---------- Initialize MediaPipe Hands (optimized settings) ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    static_image_mode=False
)
print("MediaPipe Hands initialized.")

# ---------- Load known faces (optimized) ----------
print("Loading known faces...")
known_encodings = []
known_names = []

# Pre-load and resize training images
training_images_dir = "training_images"
if not os.path.exists(training_images_dir):
    print(f"Error: Training images directory '{training_images_dir}' not found. Face recognition will not work.")
else:
    for person_name in os.listdir(training_images_dir):
        person_folder = os.path.join(training_images_dir, person_name)
        if not os.path.isdir(person_folder):
            continue
        
        for image_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, image_name)
            try:
                image = face_recognition.load_image_file(img_path)
                # Resize image for faster encoding, or use a smaller input for face_locations directly
                small_image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5) 
                
                face_locations = face_recognition.face_locations(small_image)
                if face_locations:
                    encodings = face_recognition.face_encodings(
                        small_image, 
                        known_face_locations=face_locations,
                        num_jitters=1 # Reduced for speed, can be 0 if accuracy isn't paramount
                    )
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

print(f"Loaded {len(known_names)} face encodings")

# ---------- Webcam setup (optimized) ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # Reduced resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS) # Request target FPS

# Actual FPS measurement for display
frame_count = 0
prev_time = time.time()
fps = 0

# State variables for dragging
dragging = False
drag_id = None # Can be object index or face name
drag_type = None # "object" or "face"

# --- Worker Functions for Multi-threading ---

def frame_reader():
    """Reads frames from the camera in a separate thread."""
    global running
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            # Optionally add a small delay to prevent busy-waiting if stream is truly broken
            time.sleep(0.01)
            continue
        
        frame = cv2.flip(frame, 1) # Flip for selfie-view
        
        # Put the latest frame into the queue, overwriting if full
        if not frame_queue.full():
            try:
                frame_queue.put(frame, block=False)
            except queue.Full:
                pass # Already full, skip
        else:
            try:
                frame_queue.get(block=False) # Clear old frame
                frame_queue.put(frame, block=False) # Add new frame
            except queue.Empty:
                pass # Should not happen if full() was true
    print("Frame reader stopped.")

def yolo_detector():
    """Performs YOLO object detection in a separate thread."""
    global current_yolo_results, running
    detection_frame_count = 0
    while running:
        detection_frame_count += 1
        if detection_frame_count % YOLO_FRAME_SKIP != 0:
            time.sleep(0.001) # Small sleep to prevent busy-waiting
            continue
        
        # Get the latest frame, if available
        try:
            frame = frame_queue.get(block=False)
        except queue.Empty:
            time.sleep(0.001)
            continue # No new frame to process
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform YOLO inference
        # Using half precision for faster inference on GPU
        results = model(frame_rgb, verbose=False, half=True)[0]
        
        with yolo_lock:
            current_yolo_results['boxes'] = results.boxes.xyxy.cpu().numpy()
            current_yolo_results['classes'] = results.boxes.cls.cpu().numpy()
            current_yolo_results['confidences'] = results.boxes.conf.cpu().numpy()
        
        # Put the processed frame back if other workers need it or if it's the only copy
        # For this setup, the main thread copies the frame, so no need to put back.
        
    print("YOLO detector stopped.")

def face_recognizer():
    """Performs face recognition in a separate thread."""
    global current_face_results, running
    recognition_frame_count = 0
    while running:
        recognition_frame_count += 1
        if recognition_frame_count % FACE_RECOGNITION_FRAME_SKIP != 0:
            time.sleep(0.001) # Small sleep
            continue
        
        try:
            frame = frame_queue.get(block=False) # Get a copy for processing
            frame_queue.put(frame, block=False) # Put it back immediately so other threads can use it
        except queue.Empty:
            time.sleep(0.001)
            continue # No new frame
        
        # Use smaller image for face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_frame)
        
        face_labels = []
        if face_locations:
            # Scale back up face locations
            face_locations = [(top*2, right*2, bottom*2, left*2) 
                            for (top, right, bottom, left) in face_locations]
            
            face_encodings = face_recognition.face_encodings(
                frame, # Use original frame for encoding for better accuracy if needed, or small_frame
                known_face_locations=face_locations,
                num_jitters=1
            )
            
            for encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    known_encodings, 
                    encoding, 
                    tolerance=0.5 # Default is 0.6, 0.5 can be slightly more strict
                )
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                
                face_labels.append(name)
        
        with face_lock:
            current_face_results['locations'] = face_locations
            current_face_results['labels'] = face_labels
            
    print("Face recognizer stopped.")

def hand_tracker():
    """Performs MediaPipe hand tracking in a separate thread."""
    global current_hand_results, running
    hand_frame_count = 0
    while running:
        hand_frame_count += 1
        if hand_frame_count % HANDS_FRAME_SKIP != 0:
            time.sleep(0.001) # Small sleep
            continue
        
        try:
            frame = frame_queue.get(block=False) # Get a copy for processing
            frame_queue.put(frame, block=False) # Put it back immediately
        except queue.Empty:
            time.sleep(0.001)
            continue # No new frame
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(frame_rgb)
        
        cursor_x, cursor_y = None, None
        pinch = False
        hand_landmarks_to_draw = None # Store landmarks for drawing in main thread
        
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
                pinch = distance < PINCH_THRESHOLD
                
                hand_landmarks_to_draw = hand_landmarks # Store for drawing
                break # Only process the first hand for simplicity
        
        with hand_lock:
            current_hand_results['cursor_x'] = cursor_x
            current_hand_results['cursor_y'] = cursor_y
            current_hand_results['pinch'] = pinch
            current_hand_results['landmarks'] = hand_landmarks_to_draw
            
    print("Hand tracker stopped.")

# --- Start Worker Threads ---
frame_reader_thread = threading.Thread(target=frame_reader, daemon=True)
yolo_detector_thread = threading.Thread(target=yolo_detector, daemon=True)
face_recognizer_thread = threading.Thread(target=face_recognizer, daemon=True)
hand_tracker_thread = threading.Thread(target=hand_tracker, daemon=True)

frame_reader_thread.start()
yolo_detector_thread.start()
face_recognizer_thread.start()
hand_tracker_thread.start()

print("Worker threads started.")

# --- Main Loop (Display and Interaction) ---
while True:
    # Get the latest frame from the reader thread
    try:
        frame = frame_queue.get(block=False) # Get without blocking
    except queue.Empty:
        # If the queue is empty, wait a bit for a frame.
        # This can happen at startup or if frame reader is struggling.
        time.sleep(0.01) 
        continue

    # Create a mutable copy of the frame for drawing
    display_frame = frame.copy() 

    # Calculate FPS for the display loop
    frame_count += 1
    current_time_loop = time.time()
    if current_time_loop - prev_time >= 1.0:
        fps = frame_count
        frame_count = 0
        prev_time = current_time_loop
    
    # Get the latest results from worker threads (thread-safe access)
    with yolo_lock:
        boxes = current_yolo_results['boxes']
        classes = current_yolo_results['classes']
        confidences = current_yolo_results['confidences']

    with face_lock:
        face_locations = current_face_results['locations']
        face_labels = current_face_results['labels']

    with hand_lock:
        cursor_x = current_hand_results['cursor_x']
        cursor_y = current_hand_results['cursor_y']
        pinch = current_hand_results['pinch']
        hand_landmarks_to_draw = current_hand_results['landmarks']
        
    # Draw hand landmarks (if available)
    if hand_landmarks_to_draw:
        mp.solutions.drawing_utils.draw_landmarks(
            display_frame, 
            hand_landmarks_to_draw, 
            mp_hands.HAND_CONNECTIONS
        )

    # ---------- Object Dragging Logic ----------
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(classes[i])]
        confidence = confidences[i]
        
        color = (0, 255, 0) # Default green
        
        # Check for cursor in bounding box
        if cursor_x and cursor_y and x1 < cursor_x < x2 and y1 < cursor_y < y2:
            if pinch and not dragging:
                dragging = True
                drag_id = i
                drag_type = "object"
                print(f"Started dragging object: {label}")
            elif not pinch and dragging and drag_id == i and drag_type == "object":
                dragging = False
                drag_id = None
                drag_type = None
                print(f"Stopped dragging object: {label}")
        
        # Handle dragging
        if dragging and drag_id == i and drag_type == "object" and pinch:
            color = (0, 255, 255) # Yellow when dragging
            w_box = x2 - x1
            h_box = y2 - y1
            new_x1 = max(cursor_x - w_box // 2, 0)
            new_y1 = max(cursor_y - h_box // 2, 0)
            new_x2 = min(new_x1 + w_box, display_frame.shape[1])
            new_y2 = min(new_y1 + h_box, display_frame.shape[0])
            x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
        
        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"{label} {confidence:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 2)
    
    # ---------- Face Dragging Logic ----------
    for (top, right, bottom, left), name in zip(face_locations, face_labels):
        color = (255, 255, 0) if name != "Unknown" else (0, 0, 255) # Blue for known, Red for unknown
        
        if cursor_x and cursor_y and left < cursor_x < right and top < cursor_y < bottom:
            if pinch and not dragging:
                dragging = True
                drag_id = name # Use name as ID for faces
                drag_type = "face"
                print(f"Started dragging face: {name}")
            elif not pinch and dragging and drag_id == name and drag_type == "face":
                dragging = False
                drag_id = None
                drag_type = None
                print(f"Stopped dragging face: {name}")
        
        if dragging and drag_id == name and drag_type == "face" and pinch:
            color = (0, 255, 255) # Yellow when dragging
            w_box = right - left
            h_box = bottom - top
            new_left = max(cursor_x - w_box // 2, 0)
            new_top = max(cursor_y - h_box // 2, 0)
            new_right = min(new_left + w_box, display_frame.shape[1])
            new_bottom = min(new_top + h_box, display_frame.shape[0])
            left, top, right, bottom = new_left, new_top, new_right, new_bottom
        
        # Draw face bounding box
        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
        cv2.putText(display_frame, name, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # ---------- Display FPS and Status ----------
    cv2.putText(display_frame, f"FPS: {fps}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Drag: {'ON' if dragging else 'OFF'}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               (0, 255, 0) if dragging else (0, 0, 255), 2)
    
    # Show frame
    cv2.imshow("Optimized Object + Face + Hand Recognition", display_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False # Signal threads to stop
        break

# Cleanup
print("Main loop stopped. Waiting for threads to finish...")
cap.release()
cv2.destroyAllWindows()
# Give threads a moment to finish cleanly (optional, as they are daemon threads)
yolo_detector_thread.join(timeout=1)
face_recognizer_thread.join(timeout=1)
hand_tracker_thread.join(timeout=1)
frame_reader_thread.join(timeout=1)
print("Application closed.")