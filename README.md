# Live Face Detection - Age, Gender, Emotion & Movement Speed

A browser-based application that performs real-time face detection from webcam feed, displaying:
- Face detection with bounding boxes
- Age estimation
- Gender classification
- Emotion recognition
- **Movement speed** (calculated from optical flow in polar coordinates)

## Features

- **Real-time Processing**: Live video feed from webcam with real-time face analysis
- **Client-side Only**: All processing happens in the browser, no server required
- **Multiple Detection**: Can detect and analyze multiple faces simultaneously
- **Visual Feedback**: Rectangles around faces with age, gender, emotion, and movement speed labels
- **Movement Tracking**: Tracks individual faces over time and calculates average movement speed using optical flow analysis

## Usage

1. Open `index.html` in a modern web browser (Chrome, Firefox, Edge recommended)
2. Allow camera permissions when prompted
3. Click "Start Camera" to begin face detection
4. The app will display faces with bounding boxes and information labels

## Requirements

- Modern web browser with camera access support
- HTTPS connection (or localhost) for camera access
- Internet connection for loading AI models (first time only)

## Technical Details

- Uses [face-api.js](https://github.com/justadudewhohacks/face-api.js) for face detection
- Models are loaded from CDN on first use
- All processing happens client-side using TensorFlow.js
- **Movement Speed Calculation**:
  - Tracks face positions across frames
  - Calculates velocity vectors in Cartesian coordinates
  - Converts to polar coordinates (magnitude = speed, angle = direction)
  - Averages speed over a sliding window (30 frames)
  - Displays speed in pixels per second

## Note

The first load may take a moment to download and initialize the AI models. Subsequent loads will be faster if cached by the browser.