import cv2
import numpy as np
import os
import time
import pandas as pd
from math import sqrt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mediapipe as mp
import threading
from queue import Queue

class HandLandmarkDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2, detection_confidence=0.7, tracking_confidence=0.5):
        """
        Initialize the hand landmark detector with MediaPipe
        
        Parameters:
        static_image_mode -- whether to process images or video stream
        max_num_hands -- maximum number of hands to detect
        detection_confidence -- minimum confidence for hand detection
        tracking_confidence -- minimum confidence for landmark tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        # Define hand connections for custom visualization
        self.HAND_CONNECTIONS = self.mp_hands.HAND_CONNECTIONS
        
        # Dictionary to map landmark indices to names
        self.landmark_names = {
            0: "WRIST",
            1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
            5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
            9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
            13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
            17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP"
        }
        
        # Colors for different hands
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    def process_frame(self, frame):
        """
        Process a frame to detect hand landmarks
        
        Parameters:
        frame -- input image frame
        
        Returns:
        results -- MediaPipe hand detection results
        """
        # Convert to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        return results
    
    def visualize_landmarks(self, frame, results, draw_indices=True, detected_sign=None):
        """
        Visualize the detected hand landmarks on the frame
        
        Parameters:
        frame -- original frame
        results -- MediaPipe hand detection results
        draw_indices -- whether to draw landmark indices
        detected_sign -- sign that was detected (optional)
        
        Returns:
        frame_out -- frame with visualized landmarks
        """
        frame_out = frame.copy()
        img_height, img_width, _ = frame_out.shape
        
        # Count detected hands
        multi_hand_landmarks = results.multi_hand_landmarks
        multi_handedness = results.multi_handedness
        num_hands = 0 if multi_hand_landmarks is None else len(multi_hand_landmarks)
        
        # Draw status text
        status_text = f"Detected {num_hands} hand(s)"
        cv2.putText(frame_out, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display detected sign if provided
        if detected_sign:
            cv2.putText(frame_out, f"Sign: {detected_sign}", (10, img_height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        if multi_hand_landmarks:
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(multi_hand_landmarks, multi_handedness)):
                # Get handedness info (left or right hand)
                hand_label = handedness.classification[0].label
                hand_color = self.colors[hand_idx % len(self.colors)]
                
                # Draw label text
                hand_text = f"{hand_label} Hand"
                cv2.putText(frame_out, hand_text, (10, 60 + hand_idx * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
                
                # Draw landmarks and connections using MediaPipe drawing utilities
                self.mp_drawing.draw_landmarks(
                    frame_out,
                    hand_landmarks,
                    self.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Custom landmark visualization with indices
                if draw_indices:
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        x = int(landmark.x * img_width)
                        y = int(landmark.y * img_height)
                        cv2.putText(frame_out, str(i), (x + 5, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(frame_out, str(i), (x + 5, y - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_out

    def calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points
        
        Parameters:
        a, b, c -- three points forming an angle
        
        Returns:
        angle -- angle in degrees
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180.0 else 360 - angle

    def extract_features(self, landmarks, img_width, img_height):
        """
        Extract a comprehensive set of features from hand landmarks
        
        Parameters:
        landmarks -- MediaPipe hand landmarks
        img_width -- image width
        img_height -- image height
        
        Returns:
        features -- list of extracted features
        """
        # Extract normalized coordinates (MediaPipe already provides normalized coords)
        landmark_points_norm = []
        for landmark in landmarks.landmark:
            # Store the normalized coordinates (MediaPipe coordinates are already normalized)
            landmark_points_norm.append((landmark.x, landmark.y, landmark.z))
            
        # Extract 2D coordinates for feature extraction
        landmark_points_2d = [(point[0], point[1]) for point in landmark_points_norm]
        
        # Get hand size (distance from wrist to middle finger MCP)
        # Used for normalization
        hand_size = np.linalg.norm(
            np.array([landmark_points_norm[9][0] - landmark_points_norm[0][0],
                     landmark_points_norm[9][1] - landmark_points_norm[0][1]])
        )
        
        features = []
        
        # 1. Relative Distances (6 features) - Key distances in the hand
        # Normalized by hand size to be resolution independent
        distance_pairs = [(0, 8), (0, 12), (0, 16), (0, 20), (4, 8), (8, 20)]
        for start, end in distance_pairs:
            dist = np.linalg.norm(np.array(landmark_points_2d[start]) - np.array(landmark_points_2d[end]))
            # Normalize by hand size
            norm_dist = dist / hand_size if hand_size > 0 else 0
            features.append(norm_dist)
        
        # 2. Landmark Angles (5 features) - Key angles in the hand
        # Angles are already resolution independent
        angle_triplets = [(0, 5, 8), (0, 9, 12), (0, 13, 16), (0, 17, 20), (5, 9, 13)]
        for a, b, c in angle_triplets:
            angle = self.calculate_angle(
                landmark_points_2d[a], 
                landmark_points_2d[b], 
                landmark_points_2d[c]
            )
            features.append(angle)
        
        # 3. Bounding Box (4 features)
        # We use normalized coordinates directly
        x_coords = [point[0] for point in landmark_points_2d]
        y_coords = [point[1] for point in landmark_points_2d]
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        features.extend(bbox)
        
        # 4. Finger Spreads (4 features) - Distance between adjacent fingertips
        # Normalized by hand size to be resolution independent
        finger_tips = [4, 8, 12, 16, 20]
        for i in range(len(finger_tips) - 1):
            spread = np.linalg.norm(np.array(landmark_points_2d[finger_tips[i]]) - 
                                    np.array(landmark_points_2d[finger_tips[i+1]]))
            # Normalize by hand size
            norm_spread = spread / hand_size if hand_size > 0 else 0
            features.append(norm_spread)
        
        # 5. Hand Orientation (2 features)
        # Angles are already resolution independent
        wrist_to_middle = np.array(landmark_points_2d[9]) - np.array(landmark_points_2d[0])
        orientation = np.arctan2(wrist_to_middle[1], wrist_to_middle[0])
        features.append(orientation)
        
        wrist_to_index = np.array(landmark_points_2d[5]) - np.array(landmark_points_2d[0])
        orientation2 = np.arctan2(wrist_to_index[1], wrist_to_index[0])
        features.append(orientation2)
        
        # 6. Finger Flexion (5 features) - Bend angle for each finger
        # Angles are already resolution independent
        for finger_base in [1, 5, 9, 13, 17]:  # Base joints of each finger
            # Calculate the angle between three consecutive points on the finger
            flexion = self.calculate_angle(
                landmark_points_2d[finger_base], 
                landmark_points_2d[finger_base+1], 
                landmark_points_2d[finger_base+2]
            )
            features.append(flexion)
        
        # 7. Finger Tip to Palm Distances (5 features)
        # Normalized by hand size to be resolution independent
        palm_center = np.mean([landmark_points_2d[i] for i in [0, 5, 9, 13, 17]], axis=0)
        for tip in [4, 8, 12, 16, 20]:  # Finger tips
            dist = np.linalg.norm(np.array(landmark_points_2d[tip]) - palm_center)
            # Normalize by hand size
            norm_dist = dist / hand_size if hand_size > 0 else 0
            features.append(norm_dist)
        
        # 8. 3D Depth Features (3 features)
        # Normalize Z values relative to hand size
        z_values = [point[2] for point in landmark_points_norm]
        z_range = max(z_values) - min(z_values)
        features.append(z_range / hand_size if hand_size > 0 else 0)  # Z-range
        features.append(np.mean(z_values) / hand_size if hand_size > 0 else 0)  # Mean Z
        features.append(np.std(z_values) / hand_size if hand_size > 0 else 0)   # Z standard deviation
        
        return features

def get_column_names():
    """
    Generate column names for the dataset
    
    Returns:
    columns -- list of column names
    """
    columns = ['Gesture_Name', 'Is_Double_Hand']
    
    feature_categories = {
        'Distance': [
            'Wrist_to_Index_Tip', 'Wrist_to_Middle_Tip', 
            'Wrist_to_Ring_Tip', 'Wrist_to_Pinky_Tip',
            'Thumb_Tip_to_Index_Tip', 'Index_Tip_to_Pinky_Tip'
        ],
        'Angle': [
            'Wrist_Index_Angle', 'Wrist_Middle_Angle',
            'Wrist_Ring_Angle', 'Wrist_Pinky_Angle',
            'Index_Middle_Ring_Angle'
        ],
        'Bounding_Box': ['Min_X', 'Min_Y', 'Max_X', 'Max_Y'],
        'Finger_Spread': [
            'Thumb_to_Index', 'Index_to_Middle',
            'Middle_to_Ring', 'Ring_to_Pinky'
        ],
        'Orientation': ['Palm_Orientation', 'Index_Orientation'],
        'Finger_Flexion': [
            'Thumb_Flexion', 'Index_Flexion', 'Middle_Flexion', 
            'Ring_Flexion', 'Pinky_Flexion'
        ],
        'Tip_To_Palm': [
            'Thumb_Tip_To_Palm', 'Index_Tip_To_Palm', 'Middle_Tip_To_Palm',
            'Ring_Tip_To_Palm', 'Pinky_Tip_To_Palm'
        ],
        '3D_Features': ['Z_Range', 'Mean_Z', 'Z_StdDev']
    }
    
    for hand in ['Right', 'Left']:
        columns.append(f'{hand}_Hand_Present')
        for category, features in feature_categories.items():
            for feature in features:
                columns.append(f'{hand}_{category}_{feature}')
    
    return columns


class SignLanguageRecognitionSystem:
    def __init__(self):
        # Paths for data storage
        self.csv_file = 'hand_gesture_dataset.csv'
        self.model_file = 'sign_language_model3.pkl'
        
        # Initialize the hand detector
        self.detector = HandLandmarkDetector(static_image_mode=False, max_num_hands=2)
        
        # Load existing model if available
        self.model = None
        self.gestures = []
        self.sign_requirements = {}
        if os.path.exists(self.model_file):
            print("Loading existing model...")
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.gestures = model_data['gestures']
                # Load sign requirements if available
                if 'sign_requirements' in model_data:
                    self.sign_requirements = model_data['sign_requirements']
                print(f"Model loaded. Recognizes {len(self.gestures)} gestures: {', '.join(self.gestures)}")
        
        # Initialize video capture
        self.cap = None



    def get_sign_hand_requirements(self):
        """
        Create a dictionary mapping signs to their hand requirements
        
        Returns:
        sign_requirements -- dictionary with sign names as keys and hand requirement details as values
        """
        if not os.path.exists(self.csv_file):
            return {}
            
        try:
            df = pd.read_csv(self.csv_file)
            sign_requirements = {}
            
            for sign in df['Gesture_Name'].unique():
                # Get all rows for this sign
                sign_data = df[df['Gesture_Name'] == sign]
                # Check if it's a double hand sign
                is_double = bool(sign_data['Is_Double_Hand'].iloc[0])
                sign_requirements[sign] = {
                    'is_double_hand': is_double,
                    'requires_right': any(sign_data['Right_Hand_Present'] > 0),
                    'requires_left': any(sign_data['Left_Hand_Present'] > 0)
                }
            
            return sign_requirements
        except Exception as e:
            print(f"Error reading sign requirements: {e}")
            return {}
        


    
    def add_new_sign(self):
        """Capture frames for a new sign and add them to the dataset"""
        print("\n=== ADD NEW SIGN ===")
        
        # Get sign details
        sign_name = input("Enter the sign name: ")
        is_double_hand = input("Is this a double hand sign? (y/n): ").lower() == 'y'
        num_frames = int(input("How many frames to capture? (recommended: 30): "))
        
        # Initialize video capture if not already done
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return
        
        print(f"\nGet ready to show the sign '{sign_name}'.")
        print("Position your hand(s) in the camera view.")
        print("Press 'c' to start capturing, 'q' to cancel.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process and display the frame
            results = self.detector.process_frame(frame)
            vis_frame = self.detector.visualize_landmarks(frame, results)
            
            cv2.putText(vis_frame, "Press 'c' to start capture or 'q' to cancel", 
                       (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Sign Language Capture', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                print(f"Get ready! Capturing {num_frames} frames for sign: {sign_name} in 3 seconds...")
                for i in range(3, 0, -1):
                    print(f"{i}...")
                    time.sleep(1)
                    # Update display during countdown
                    ret, frame = self.cap.read()
                    if ret:
                        results = self.detector.process_frame(frame)
                        vis_frame = self.detector.visualize_landmarks(frame, results)
                        cv2.putText(vis_frame, f"Starting in {i}...", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('Sign Language Capture', vis_frame)
                        cv2.waitKey(1)
                
                self._capture_frames(sign_name, is_double_hand, num_frames)
                break
            elif key == ord('q'):
                print("Cancelled adding new sign.")
                return
        
        print(f"Successfully captured {num_frames} frames for sign: {sign_name}")
        
        # Update gesture list if not already in it
        if sign_name not in self.gestures:
            self.gestures.append(sign_name)
            
        cv2.destroyAllWindows()
    
    def _capture_frames(self, label, is_double_hand, num_frames):
        """Helper function to capture frames for dataset creation"""
        frames_captured = 0
        start_time = time.time()
        
        # Check if file exists and write header if it doesn't
        file_exists = os.path.isfile(self.csv_file)
        if not file_exists:
            columns = get_column_names()
            with open(self.csv_file, 'w') as f:
                f.write(','.join(columns) + '\n')
        
        while frames_captured < num_frames:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process the frame
            results = self.detector.process_frame(frame)
            
            if results.multi_hand_landmarks:
                right_hand_features = None
                left_hand_features = None
                right_hand_present = 0
                left_hand_present = 0
                
                img_height, img_width, _ = frame.shape
                
                # Extract features for each hand
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    is_right_hand = handedness.classification[0].label == "Right"
                    
                    features = self.detector.extract_features(hand_landmarks, img_width, img_height)
                    
                    if is_right_hand:
                        right_hand_features = features
                        right_hand_present = 1
                    else:
                        left_hand_features = features
                        left_hand_present = 1
                
                # Visualize the frame with landmarks
                vis_frame = self.detector.visualize_landmarks(frame, results)
                
                # Prepare row data
                row_data = [label, int(is_double_hand)]
                
                # Add right hand data
                row_data.append(right_hand_present)
                if right_hand_features:
                    row_data.extend(right_hand_features)
                else:
                    row_data.extend([0] * 34)  # 34 features per hand
                
                # Add left hand data
                row_data.append(left_hand_present)
                if left_hand_features:
                    row_data.extend(left_hand_features)
                else:
                    row_data.extend([0] * 34)  # 34 features per hand
                
                # Write to CSV
                with open(self.csv_file, 'a') as f:
                    f.write(','.join(map(str, row_data)) + '\n')
                
                frames_captured += 1
                
                # Display frame count
                cv2.putText(vis_frame, f'Frames: {frames_captured}/{num_frames}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Sign Language Capture', vis_frame)
            else:
                # Just show the frame without landmarks
                cv2.putText(frame, 'No hands detected', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Sign Language Capture', frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Ensure we don't capture frames too quickly
            if time.time() - start_time < frames_captured * 0.1:  # 10 FPS
                time.sleep(0.1)
    
    def train_model(self):
        """Train the sign language recognition model"""
        print("\n=== TRAIN MODEL ===")
        
        if not os.path.exists(self.csv_file):
            print(f"Error: Dataset file '{self.csv_file}' not found.")
            print("Please add some signs first using option 1.")
            return
        
        print(f"Loading dataset from {self.csv_file}...")
        df = pd.read_csv(self.csv_file)
        
        if len(df) == 0:
            print("Error: Dataset is empty. Please add signs first.")
            return
        
        # Get unique gestures
        self.gestures = df['Gesture_Name'].unique().tolist()
        print(f"Found {len(self.gestures)} gestures in the dataset: {', '.join(self.gestures)}")
        print(f"Total samples: {len(df)}")
        
        # Show distribution
        print("\nClass distribution:")
        for gesture in self.gestures:
            count = len(df[df['Gesture_Name'] == gesture])
            print(f"  {gesture}: {count} samples ({count/len(df)*100:.1f}%)")
        
        # Prepare features and target
        # Note: INCLUDE Is_Double_Hand in the features now
        X = df.drop(['Gesture_Name'], axis=1)  # Keep Is_Double_Hand
        y = df['Gesture_Name']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"\nTraining with {len(X_train)} samples, validating with {len(X_test)} samples...")
        
        # Train the model (Random Forest for good accuracy and speed)
        print("Training Random Forest classifier...")
        self.model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.2%}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Generate and save sign requirements
        sign_requirements = self.get_sign_hand_requirements()
        
        # Save the model with requirements
        print(f"Saving model to {self.model_file}...")
        with open(self.model_file, 'wb') as f:
            pickle.dump({
                'model': self.model, 
                'gestures': self.gestures,
                'sign_requirements': sign_requirements
            }, f)
        
        print("Model trained and saved successfully!")

    def start_live_detection(self):
        """Run live sign language detection"""
        print("\n=== LIVE DETECTION ===")
        
        if self.model is None:
            print("Error: No trained model available.")
            print("Please train the model first using option 2.")
            return
        
        # Load sign requirements if not already loaded
        if not self.sign_requirements:
            self.sign_requirements = self.get_sign_hand_requirements()
            print(f"Loaded requirements for {len(self.sign_requirements)} signs")
        
        # Initialize video capture if not already done
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return
        
        print("Starting live detection...")
        print("Press 'q' to quit.")
        
        # For prediction stability - use a smaller queue for quicker transitions
        prediction_queue = []
        queue_size = 3  # Reduced from 5 to 3 for faster response
        
        # Create a queue for async feature extraction with limited size
        feature_queue = Queue(maxsize=2)
        result_queue = Queue(maxsize=2)
        
        # Flag to signal thread to stop
        stop_thread = threading.Event()
        
        # Enhanced cache for recent hand features to speed up transitions
        feature_cache = {}
        cache_expiry = 0.5  # seconds
        last_cache_time = time.time()
        
        # Improved confidence threshold for unknown gesture detection
        confidence_threshold = 0.6  # Confidence threshold for valid detection
        
        # Function for prediction thread
        def predict_from_queue():
            while not stop_thread.is_set():
                try:
                    if not feature_queue.empty():
                        features_data = feature_queue.get(block=False)
                        if features_data is not None and self.model is not None:
                            features, right_hand_present, left_hand_present = features_data
                            # Make prediction
                            try:
                                prediction = self.model.predict([features])[0]
                                # Get probabilities for all classes
                                probabilities = self.model.predict_proba([features])[0]
                                max_confidence = max(probabilities)
                                
                                # Check if the predicted sign's hand requirements match detected hands
                                valid_hand_configuration = True
                                if prediction in self.sign_requirements:
                                    req = self.sign_requirements[prediction]
                                    
                                    # For double hand signs, must have both hands
                                    if req['is_double_hand'] and not (right_hand_present and left_hand_present):
                                        valid_hand_configuration = False
                                    
                                    # For single hand signs with specific hand requirements
                                    if not req['is_double_hand']:
                                        if req['requires_right'] and not right_hand_present:
                                            valid_hand_configuration = False
                                        if req['requires_left'] and not left_hand_present:
                                            valid_hand_configuration = False
                                
                                # Only consider valid if confidence exceeds threshold AND hand configuration matches
                                if max_confidence >= confidence_threshold and valid_hand_configuration:
                                    result_queue.put((prediction, max_confidence))
                                else:
                                    result_queue.put(("Unknown", max_confidence))
                            except Exception as e:
                                print(f"Prediction error: {e}")
                                result_queue.put(("Unknown", 0))
                except:
                    pass  # Handle empty queue exception
                time.sleep(0.005)  # Smaller sleep to increase responsiveness
        
        # Start prediction thread
        prediction_thread = threading.Thread(target=predict_from_queue)
        prediction_thread.daemon = True
        prediction_thread.start()
        
        current_prediction = None
        current_confidence = 0
        last_prediction_time = time.time()
        prediction_timeout = 1.0  # Reset prediction if no hands detected for this many seconds
        
        # Add FPS monitoring
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Process frame to detect hand landmarks
                results = self.detector.process_frame(frame)
                
                # Reset prediction if no hands detected for a while
                if not results.multi_hand_landmarks:
                    if time.time() - last_prediction_time > prediction_timeout:
                        current_prediction = None
                        current_confidence = 0
                        prediction_queue.clear()
                else:
                    last_prediction_time = time.time()
                    
                    right_hand_features = None
                    left_hand_features = None
                    right_hand_present = 0
                    left_hand_present = 0
                    
                    img_height, img_width, _ = frame.shape
                    
                    # Extract features for each hand
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        is_right_hand = handedness.classification[0].label == "Right"
                        
                        features = self.detector.extract_features(hand_landmarks, img_width, img_height)
                        
                        if is_right_hand:
                            right_hand_features = features
                            right_hand_present = 1
                        else:
                            left_hand_features = features
                            left_hand_present = 1
                    
                    # Determine if this is potentially a double hand sign
                    is_double_hand = (right_hand_present and left_hand_present)
                    
                    # Prepare feature row (similar to training data structure) WITH is_double_hand flag
                    combined_features = [int(is_double_hand)]  # Include Is_Double_Hand
                    
                    # Add right hand data
                    combined_features.append(right_hand_present)
                    if right_hand_features:
                        combined_features.extend(right_hand_features)
                    else:
                        combined_features.extend([0] * 34)
                    
                    # Add left hand data
                    combined_features.append(left_hand_present)
                    if left_hand_features:
                        combined_features.extend(left_hand_features)
                    else:
                        combined_features.extend([0] * 34)
                    
                    # Check feature cache to avoid redundant predictions
                    cache_key = str(combined_features[:10])  # Use first 10 features as key
                    current_time = time.time()
                    
                    if cache_key in feature_cache and current_time - feature_cache[cache_key]['time'] < cache_expiry:
                        # Use cached prediction if features are similar
                        cached_result = feature_cache[cache_key]
                        prediction = cached_result['prediction']
                        confidence = cached_result['confidence']
                        
                        prediction_queue.append(prediction)
                        if len(prediction_queue) > queue_size:
                            prediction_queue.pop(0)
                    else:
                        # Add to prediction queue if not full
                        if feature_queue.qsize() < 2:
                            # Pass hand presence info along with features
                            feature_queue.put((combined_features, right_hand_present, left_hand_present))
                    
                    # Clean up expired cache entries
                    if current_time - last_cache_time > 5.0:  # Periodic cleanup
                        for k in list(feature_cache.keys()):
                            if current_time - feature_cache[k]['time'] > cache_expiry:
                                del feature_cache[k]
                        last_cache_time = current_time
                
                # Get prediction results if available
                try:
                    while not result_queue.empty():
                        prediction, confidence = result_queue.get(block=False)
                        
                        # Update cache
                        if prediction != "Unknown" and confidence >= confidence_threshold:
                            cache_key = str(combined_features[:10])
                            feature_cache[cache_key] = {
                                'prediction': prediction,
                                'confidence': confidence,
                                'time': time.time()
                            }
                        
                        # Add to prediction queue
                        prediction_queue.append(prediction)
                        if len(prediction_queue) > queue_size:
                            prediction_queue.pop(0)
                except:
                    pass  # Handle empty queue exception
                
                # Get most common prediction from the queue
                if prediction_queue:
                    from collections import Counter
                    prediction_counts = Counter(prediction_queue)
                    most_common = prediction_counts.most_common(1)[0]
                    current_prediction, count = most_common
                    current_confidence = count / len(prediction_queue)
                
                # Display detection result with confidence
                if current_prediction == "Unknown" or current_confidence < confidence_threshold:
                    prediction_text = "Unknown"
                    confidence_display = current_confidence
                else:
                    prediction_text = current_prediction
                    confidence_display = current_confidence
                
                # Count hands to display as additional info
                hand_count = 0 if not results.multi_hand_landmarks else len(results.multi_hand_landmarks)
                hand_status = f"{hand_count} hand(s)"
                
                status_text = f"{prediction_text} ({confidence_display:.2f})"
                vis_frame = self.detector.visualize_landmarks(frame, results, detected_sign=status_text)
                
                # Add hand count display
                cv2.putText(vis_frame, hand_status, 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Add FPS display
                cv2.putText(vis_frame, f"FPS: {fps:.1f}", 
                        (10, vis_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add confidence threshold indicator
                cv2.putText(vis_frame, f"Threshold: {confidence_threshold:.2f}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Add instructions
                cv2.putText(vis_frame, "Press 'q' to quit, '+'/'-' to adjust threshold", 
                        (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow('Sign Language Detection', vis_frame)
                
                # Check for user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    # Increase threshold
                    confidence_threshold = min(confidence_threshold + 0.05, 1.0)
                    print(f"Confidence threshold increased to: {confidence_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease threshold
                    confidence_threshold = max(confidence_threshold - 0.05, 0.3)
                    print(f"Confidence threshold decreased to: {confidence_threshold:.2f}")
            
        finally:
            # Stop prediction thread and properly release resources
            stop_thread.set()
            prediction_thread.join(timeout=0.5)
            cv2.destroyAllWindows()
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            print("Detection stopped, resources released.")

    # Also modify the release_resources method to ensure proper cleanup
    def release_resources(self):
        """Release video capture resources"""
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except:
                pass
        cv2.destroyAllWindows()
        print("Resources released. Exiting...")

def main():
    """Main function for the sign language recognition system"""
    # Print welcome message
    print("=" * 50)
    print("       SIGN LANGUAGE RECOGNITION SYSTEM       ")
    print("=" * 50)
    
    # Initialize the system
    system = SignLanguageRecognitionSystem()
    
    try:
        while True:
            # Display menu
            print("\n" + "=" * 50)
            print("MENU OPTIONS:")
            print("1. Add New Sign")
            print("2. Train Model")
            print("3. Live Detection")
            print("4. Exit")
            print("=" * 50)
            
            # Get user choice
            choice = input("\nEnter your choice (1-4): ")
            
            # Process choice
            if choice == '1':
                system.add_new_sign()
            elif choice == '2':
                system.train_model()
            elif choice == '3':
                system.start_live_detection()
            elif choice == '4':
                print("\nExiting program...")
                break
            else:
                print("\nInvalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Release resources
        system.release_resources()

if __name__ == "__main__":
    main()