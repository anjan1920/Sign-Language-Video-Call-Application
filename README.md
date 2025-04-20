# Sign-language-coverter

# Sign Language Video Call Application

A web-based video call application with sign language detection and speech-to-sign conversion capabilities.

## Features

- Real-time video calls between two users
- Sign language detection using machine learning
- Real-time display of detected signs
- Speech-to-sign language conversion
- Room-based system for easy joining
- Camera and microphone controls

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- A modern web browser (Chrome, Firefox, Edge)
- Webcam and microphone access

## Installation

1. Clone this repository or download the files.

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. The application should now be running. You can:
   - Create a new room
   - Join an existing room using a room ID
   - Use the sign language detection feature by showing signs to the camera
   - Convert speech to sign language images

## Usage

### Creating a Room
1. Click the "Create Room" button on the homepage
2. Share the generated Room ID with another person you want to call

### Joining a Room
1. Enter the Room ID shared with you in the input field
2. Click the "Join Room" button

### During the Call
- Use the camera toggle button to turn your camera on/off
- Use the microphone toggle button to mute/unmute your microphone
- Show sign language gestures to the camera for detection
- Use the speech recognition feature to convert speech to sign language images
- Click "Leave Room" to end the call

## How It Works

- **Sign Language Detection**: Uses a machine learning model trained on hand landmarks detected by MediaPipe to recognize sign language gestures
- **Speech-to-Sign Conversion**: Converts spoken words to corresponding sign language images
- **WebRTC**: Enables peer-to-peer video communication
- **WebSockets**: Manages room creation/joining and real-time communication

## Project Structure

- `app.py`: Flask server with WebSocket handling and sign language processing
- `templates/index.html`: HTML template for the web interface
- `static/style.css`: CSS styling for the web interface
- `static/script.js`: JavaScript for client-side functionality
- `backend/`: Contains the sign language detection model and images
  - `sign_language_model3.pkl`: The trained model for sign language detection
  - `image/`: Directory with sign language images for different words/letters

## Credits

This application uses the following technologies:
- Flask and Flask-SocketIO for the server
- MediaPipe for hand landmark detection
- WebRTC for peer-to-peer video calls
- Web Speech API for speech recognition
- Socket.IO for real-time communication 
