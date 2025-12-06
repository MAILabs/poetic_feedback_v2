// Video and canvas elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const closeBtn = document.getElementById('closeBtn');
const statusEl = document.getElementById('status');
const fullScreenView = document.getElementById('fullScreenView');
const startScreen = document.getElementById('startScreen');

let isRunning = false;
let detectionInterval = null;

// Movement tracking data structure
const faceMovementData = new Map(); // faceId -> {positions: [], speeds: [], timestamps: []}
let frameCount = 0;
let previousDetections = [];
const SPEED_WINDOW_SIZE = 5; // Number of frames to average speed over
const MIN_FRAMES_FOR_SPEED = 5; // Minimum frames before showing speed

// Message playback configuration
const MESSAGE_DELAY_MS = 500; // Delay between messages (0-1000ms, easily configurable)

// Phrase display tracking
let phraseDisplayData = {
    currentPhrase: null,
    isVLM: false // Whether this phrase is from POF service
};

// Audio playback tracking
let currentAudio = null; // Currently playing Audio object
let isWaitingForNextMessage = false; // Flag to prevent multiple simultaneous message triggers
let nextMessageTimeout = null; // Timeout for delayed next message

// Phrase cycle tracking
let localPhraseCount = 0; // Counter for local phrases selected (reset after 3)
let isInPOFMode = false; // Whether we're currently displaying POF phrases
let pofPhrases = []; // Array of phrases from POF endpoint
let currentPOFPhraseIndex = 0; // Current index in POF phrases array
let pofRequestInProgress = false; // Track if POF request is currently in progress
const LOCAL_PHRASES_BEFORE_POF = 3; // Number of local phrases before calling POF

// Face tracking for reset detection
let lastFaceId = null; // Track face ID to detect new person

// Video frame buffer for blur effect (moving average over last 5 frames)
const FRAME_BUFFER_SIZE = 3;
const frameBuffer = [];
let offscreenCanvas = null;
let offscreenCtx = null;

// Load face-api models
async function loadModels() {
    try {
        statusEl.textContent = 'Loading AI models...';
        statusEl.className = 'loading';
        
        // Load all required models
        await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/'),
            faceapi.nets.faceLandmark68Net.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/'),
            faceapi.nets.faceRecognitionNet.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/'),
            faceapi.nets.faceExpressionNet.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/'),
            faceapi.nets.ageGenderNet.loadFromUri('https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/')
        ]);
        
        statusEl.textContent = 'Models loaded successfully!';
        statusEl.className = 'success';
        startBtn.disabled = false;
    } catch (error) {
        console.error('Error loading models:', error);
        statusEl.textContent = 'Error loading models. Check console for details.';
        statusEl.className = 'error';
    }
}

// Start webcam
async function startCamera() {
    try {
        startBtn.disabled = true;
        statusEl.textContent = 'Accessing camera...';
        statusEl.className = 'loading';
        
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
//                facingMode: 'user'
            }
        });
        
        video.srcObject = stream;
        
        video.addEventListener('loadedmetadata', () => {
            // Set canvas size to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Create offscreen canvas for frame buffer
            offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = video.videoWidth;
            offscreenCanvas.height = video.videoHeight;
            offscreenCtx = offscreenCanvas.getContext('2d');
            
            statusEl.textContent = 'Camera ready';
            statusEl.className = 'success';
            
            // Switch to full-screen view
            startScreen.style.display = 'none';
            fullScreenView.classList.remove('hidden');
            
            isRunning = true;
            
            // Start detection loop
            detectFaces();
        });
    } catch (error) {
        console.error('Error accessing camera:', error);
        statusEl.textContent = 'Error accessing camera. Please allow camera permissions.';
        statusEl.className = 'error';
        startBtn.disabled = false;
    }
}

// Stop webcam
function stopCamera() {
    if (video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
    
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Clear movement tracking data
    faceMovementData.clear();
    // Stop any playing audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    
    // Clear any pending timeout
    if (nextMessageTimeout) {
        clearTimeout(nextMessageTimeout);
        nextMessageTimeout = null;
    }
    
    // Reset phrase display data
    phraseDisplayData = {
        currentPhrase: null,
        isVLM: false
    };
    
    // Reset phrase cycle tracking
    isWaitingForNextMessage = false;
    localPhraseCount = 0;
    isInPOFMode = false;
    pofPhrases = [];
    currentPOFPhraseIndex = 0;
    pofRequestInProgress = false;
    lastFaceId = null;
    previousDetections = [];
    frameCount = 0;
    frameBuffer.length = 0; // Clear frame buffer
    
    // Switch back to start screen
    fullScreenView.classList.add('hidden');
    startScreen.style.display = 'flex';
    
    isRunning = false;
    startBtn.disabled = false;
    statusEl.textContent = 'Camera stopped';
    statusEl.className = '';
}

// Calculate distance between two points
function distance(p1, p2) {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
}

// Convert Cartesian velocity to polar coordinates
function cartesianToPolar(vx, vy) {
    const magnitude = Math.sqrt(vx * vx + vy * vy); // speed
    const angle = Math.atan2(vy, vx); // direction in radians
    return { magnitude, angle };
}

// Match current face to previous face based on position proximity
function matchFaceToPrevious(currentBox, previousBoxes, threshold = 100) {
    let bestMatch = -1;
    let minDistance = Infinity;
    
    const currentCenter = {
        x: currentBox.x + currentBox.width / 2,
        y: currentBox.y + currentBox.height / 2
    };
    
    previousBoxes.forEach((prevBox, index) => {
        const prevCenter = {
            x: prevBox.x + prevBox.width / 2,
            y: prevBox.y + prevBox.height / 2
        };
        
        const dist = distance(currentCenter, prevCenter);
        if (dist < threshold && dist < minDistance) {
            minDistance = dist;
            bestMatch = index;
        }
    });
    
    return bestMatch;
}

// Calculate movement speed for a face
function calculateMovementSpeed(currentBox, previousBox, timeDelta) {
    if (!previousBox || timeDelta === 0) return null;
    
    const currentCenter = {
        x: currentBox.x + currentBox.width / 2,
        y: currentBox.y + currentBox.height / 2
    };
    
    const previousCenter = {
        x: previousBox.x + previousBox.width / 2,
        y: previousBox.y + previousBox.height / 2
    };
    
    // Calculate velocity in Cartesian coordinates
    const vx = (currentCenter.x - previousCenter.x) / timeDelta;
    const vy = (currentCenter.y - previousCenter.y) / timeDelta;
    
    // Convert to polar coordinates
    const polar = cartesianToPolar(vx, vy);
    
    return {
        speed: polar.magnitude, // pixels per frame (can be converted to pixels per second)
        angle: polar.angle,
        vx: vx,
        vy: vy
    };
}

// Update movement tracking for all faces
function updateMovementTracking(detections, currentTime) {
    // Match current detections to previous ones
    detections.forEach((detection, idx) => {
        const box = detection.detection.box;
        const matchIdx = matchFaceToPrevious(box, previousDetections);
        
        let faceId;
        let timeDelta = 1; // Default: 1 frame difference
        
        if (matchIdx >= 0 && previousDetections[matchIdx].faceId !== undefined) {
            // Matched to previous face
            faceId = previousDetections[matchIdx].faceId;
            timeDelta = currentTime - (previousDetections[matchIdx].timestamp || currentTime - 1);
            if (timeDelta === 0) timeDelta = 1;
        } else {
            // New face
            faceId = `face_${frameCount}_${idx}_${Date.now()}`;
        }
        
        // Store face ID in detection for next frame
        detection.faceId = faceId;
        detection.timestamp = currentTime;
        
        // Get or create movement data for this face
        if (!faceMovementData.has(faceId)) {
            faceMovementData.set(faceId, {
                positions: [],
                speeds: [],
                timestamps: [],
                lastBox: null,
                lastTime: currentTime
            });
        }
        
        const movementData = faceMovementData.get(faceId);
        
        // Calculate speed if we have previous data
        if (movementData.lastBox && movementData.lastTime) {
            const movement = calculateMovementSpeed(
                box,
                movementData.lastBox,
                timeDelta
            );
            
            if (movement) {
                movementData.speeds.push(movement.speed);
                movementData.timestamps.push(currentTime);
                
                // Keep only recent data (sliding window)
                if (movementData.speeds.length > SPEED_WINDOW_SIZE) {
                    movementData.speeds.shift();
                    movementData.timestamps.shift();
                }
            }
        }
        
        // Update last known position
        movementData.lastBox = box;
        movementData.lastTime = currentTime;
    });
    
    // Store current detections for next frame
    previousDetections = detections.map(d => ({
        x: d.detection.box.x,
        y: d.detection.box.y,
        width: d.detection.box.width,
        height: d.detection.box.height,
        faceId: d.faceId,
        timestamp: d.timestamp
    }));
    
    // Clean up old faces that are no longer detected (after 30 frames)
    const staleThreshold = frameCount - 30;
    faceMovementData.forEach((data, faceId) => {
        if (!detections.find(d => d.faceId === faceId)) {
            // Face not in current detection, remove if stale
            const lastSeen = data.lastTime || 0;
            if (frameCount - lastSeen > 30) {
                faceMovementData.delete(faceId);
            }
        }
    });
}

// Get average movement speed for a face
function getAverageSpeed(faceId) {
    const movementData = faceMovementData.get(faceId);
    if (!movementData || movementData.speeds.length < MIN_FRAMES_FOR_SPEED) {
        return null;
    }
    
    const speeds = movementData.speeds;
    const avgSpeed = speeds.reduce((sum, speed) => sum + Math.abs(speed), 0) / speeds.length;
    
    // Convert pixels per frame to pixels per second (assuming ~30 fps)
    const fps = 30;
    const speedPerSecond = avgSpeed * fps;
    
    return {
        avgSpeed: avgSpeed,
        speedPerSecond: speedPerSecond,
        sampleCount: speeds.length
    };
}

// Reset phrase cycle when new person appears
function resetPhraseCycle() {
    // Stop any playing audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    
    // Clear any pending timeout
    if (nextMessageTimeout) {
        clearTimeout(nextMessageTimeout);
        nextMessageTimeout = null;
    }
    
    // Reset all state
    phraseDisplayData.currentPhrase = null;
    isWaitingForNextMessage = false;
    localPhraseCount = 0;
    isInPOFMode = false;
    pofPhrases = [];
    currentPOFPhraseIndex = 0;
    pofRequestInProgress = false;
}

// Helper function to select and play local phrase
function selectAndPlayLocalPhrase(dominantEmotionX, speedPixelsPerFrame) {
    const newPhrase = phraseSelector.selectPhrase(dominantEmotionX, speedPixelsPerFrame);
    if (newPhrase) {
        // Play audio for the new phrase immediately
        playPhraseAudio(newPhrase, false);
    } else {
        // If no phrase selected, trigger next message
        triggerNextMessage();
    }
}

// Convert phrase text to filename format (remove punctuation, replace spaces with underscores, add .mp3)
function phraseToFilename(phrase) {
    if (!phrase) return null;
    // Remove punctuation but keep letters (including Cyrillic), numbers, spaces, and + character
    // Remove common punctuation: . , ! ? : ; " ' ( ) [ ] { } - etc.
    let filename = phrase.replace(/[.,!?:;"'()\[\]{}\-–—…]/g, '');
    // Replace spaces with underscores
    filename = filename.replace(/\s+/g, '_');
    filename = filename.replace(/\+/g, '');
    // Add .mp3 extension
    return filename + '.mp3';
}

// Play audio for a phrase (local or POF) - event-driven
function playPhraseAudio(phrase, isVLM = false) {
    if (!phrase) {
        // If no phrase, trigger next message immediately
        triggerNextMessage();
        return;
    }
    
    // Stop any currently playing audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }
    
    // Clear any pending timeout
    if (nextMessageTimeout) {
        clearTimeout(nextMessageTimeout);
        nextMessageTimeout = null;
    }
    
    // Update display immediately
    phraseDisplayData.currentPhrase = phrase;
    phraseDisplayData.isVLM = isVLM;
    
    let audioPath;
    if (isVLM) {
        // For POF phrases, use the voice.mp3 endpoint
        audioPath = 'https://dh.ycloud.eazify.net:8443/voice.mp3';
    } else {
        // For local phrases, use the vocals directory
        const filename = phraseToFilename(phrase);
        if (!filename) {
            triggerNextMessage();
            return;
        }
        audioPath = `vocals/${filename}`;
    }
    
    const audio = new Audio(audioPath);
    isWaitingForNextMessage = false;
    let audioErrored = false; // Flag to track if audio has errored
    
    audio.onended = () => {
        // Only handle ended if audio didn't error
        if (!audioErrored && currentAudio === audio) {
            // Audio finished - trigger next message after delay
            currentAudio = null;
            isWaitingForNextMessage = false;
            
            // Wait for configured delay, then trigger next message
            nextMessageTimeout = setTimeout(() => {
                triggerNextMessage();
            }, MESSAGE_DELAY_MS);
        }
    };
    
    audio.onerror = (error) => {
        // Prevent multiple error handlers from firing
        if (audioErrored) return;
        audioErrored = true;
        
        console.warn(`Failed to load audio file: ${audioPath}`, error);
        // If audio fails to load, skip this phrase immediately
        if (currentAudio === audio) {
            currentAudio = null;
        }
        phraseDisplayData.currentPhrase = null; // Clear the failed phrase from display
        isWaitingForNextMessage = false;
        // Trigger next message immediately (no delay)
        triggerNextMessage();
    };
    
    // Set currentAudio before attempting to play
    currentAudio = audio;
    // Play the audio
    audio.play().catch(error => {
        // Prevent multiple error handlers from firing
        if (audioErrored) return;
        audioErrored = true;
        
        console.warn(`Failed to play audio: ${audioPath}`, error);
        // If play fails, skip this phrase immediately
        if (currentAudio === audio) {
            currentAudio = null;
        }
        phraseDisplayData.currentPhrase = null; // Clear the failed phrase from display
        isWaitingForNextMessage = false;
        // Trigger next message immediately (no delay)
        triggerNextMessage();
    });
}

// Trigger next message in sequence
function triggerNextMessage() {
    if (isWaitingForNextMessage) {
        // Already triggered, prevent duplicate
        return;
    }
    
    isWaitingForNextMessage = true;
    
    // If we're in POF mode, move to next POF phrase
    if (isInPOFMode && pofPhrases.length > 0) {
        currentPOFPhraseIndex++;
        
        if (currentPOFPhraseIndex < pofPhrases.length) {
            // There are more POF phrases, play the next one
            isWaitingForNextMessage = false;
            playPhraseAudio(pofPhrases[currentPOFPhraseIndex], true);
        } else {
            // All POF phrases are done, switch back to local phrases
            isInPOFMode = false;
            pofPhrases = [];
            currentPOFPhraseIndex = 0;
            localPhraseCount = 0; // Reset counter to start new cycle
            // Set flag to request next local phrase (will be handled by detectFaces)
            isWaitingForNextMessage = false;
        }
    } else {
        // We're in local phrase mode, set flag to request next local phrase
        // (will be handled by detectFaces when it has face data)
        isWaitingForNextMessage = false;
    }
}


// Capture current video frame as JPG blob
function captureFrameAsJPG() {
    return new Promise((resolve, reject) => {
        try {
            // Create a temporary canvas to capture the video frame
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = video.videoWidth;
            tempCanvas.height = video.videoHeight;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Draw current video frame to canvas
            tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
            
            // Convert to blob (JPG)
            tempCanvas.toBlob((blob) => {
                if (blob) {
                    resolve(blob);
                } else {
                    reject(new Error('Failed to create blob from canvas'));
                }
            }, 'image/jpeg', 0.9); // 90% quality
        } catch (error) {
            reject(error);
        }
    });
}

// Fetch phrases from POF endpoint
async function fetchPOFPhrases() {
    if (pofRequestInProgress) {
        return; // Don't start another request if one is in progress
    }
    
    try {
        pofRequestInProgress = true;
        
        // Capture current frame as JPG
        const imageBlob = await captureFrameAsJPG();
        
        const response = await fetch('https://dh.ycloud.eazify.net:8443/pof', {
            method: 'POST',
            headers: {
                'Content-Type': 'image/jpeg'
            },
            body: imageBlob
        });
        
        if (!response.ok) {
            throw new Error(`POF API error: ${response.statusText}`);
        }
        
        // Get the JSON array of phrases
        const phrasesArray = await response.json();
        console.log(phrasesArray);
        if (Array.isArray(phrasesArray) && phrasesArray.length > 0) {
            // Store the phrases and switch to POF mode
            pofPhrases = phrasesArray.map(phrase => phrase.trim()).filter(phrase => phrase.length > 0);
            currentPOFPhraseIndex = 0;
            isInPOFMode = true;
            
            // Display and play the first POF phrase
            if (pofPhrases.length > 0) {
                isWaitingForNextMessage = false;
                playPhraseAudio(pofPhrases[0], true);
            } else {
                // Empty array, fall back to local phrases
                isInPOFMode = false;
                isWaitingForNextMessage = false;
            }
        } else {
            throw new Error('POF API returned invalid or empty array');
        }
    } catch (error) {
        console.error('Error fetching POF phrases:', error);
        // On error, stay in local phrase mode and continue with local phrases
        isInPOFMode = false;
        pofPhrases = [];
        currentPOFPhraseIndex = 0;
        isWaitingForNextMessage = false;
    } finally {
        pofRequestInProgress = false;
    }
}

// Detect faces and draw results
async function detectFaces() {
    if (!isRunning || video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(detectFaces);
        return;
    }
    
    frameCount++;
    const currentTime = frameCount;
    
    // Capture current frame to offscreen canvas
    offscreenCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
    const currentFrameData = offscreenCtx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
    
    // Add current frame to buffer
    frameBuffer.push(currentFrameData);
    if (frameBuffer.length > FRAME_BUFFER_SIZE) {
        frameBuffer.shift(); // Remove oldest frame
    }
    
    // Calculate moving average (blur effect) and apply brightness reduction
    // Always draw the video frame (with blur if we have enough frames)
    if (frameBuffer.length > 0) {
        const blendedFrame = new ImageData(offscreenCanvas.width, offscreenCanvas.height);
        const pixelCount = blendedFrame.data.length / 4;
        const bufferLength = frameBuffer.length;
        
        for (let i = 0; i < pixelCount; i++) {
            let r = 0, g = 0, b = 0;
            
            // Average across all frames in buffer
            frameBuffer.forEach(frame => {
                const idx = i * 4;
                r += frame.data[idx];
                g += frame.data[idx + 1];
                b += frame.data[idx + 2];
            });
            
            // Calculate average
            r = Math.round(r / bufferLength);
            g = Math.round(g / bufferLength);
            b = Math.round(b / bufferLength);
            
            // Apply brightness reduction (50% = multiply by 0.5)
            r = Math.round(r * 0.5);
            g = Math.round(g * 0.5);
            b = Math.round(b * 0.5);
            
            const idx = i * 4;
            blendedFrame.data[idx] = r;
            blendedFrame.data[idx + 1] = g;
            blendedFrame.data[idx + 2] = b;
            blendedFrame.data[idx + 3] = 255; // Alpha
        }
        
        // Draw blurred and darkened video to canvas
        // ctx.putImageData(blendedFrame, 0, 0);
        drawdata = blendedFrame;
    } else {
        // If no frames in buffer yet, draw current frame directly (darkened)
        offscreenCtx.drawImage(video, 0, 0, offscreenCanvas.width, offscreenCanvas.height);
        const frameData = offscreenCtx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
        const pixelData = frameData.data;
        
        // Apply brightness reduction (50%)
        for (let i = 0; i < pixelData.length; i += 4) {
            pixelData[i] = Math.round(pixelData[i] * 0.5);     // R
            pixelData[i + 1] = Math.round(pixelData[i + 1] * 0.5); // G
            pixelData[i + 2] = Math.round(pixelData[i + 2] * 0.5); // B
        }
        
        // ctx.putImageData(frameData, 0, 0);
        drawdata = frameData;
    }
    
    // Use face-api to detect faces with all required information
    const detections = await faceapi
        .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender();
    
    // Update movement tracking
    updateMovementTracking(detections, currentTime);
    
    // Find the largest face (by width) for phrase display
    let largestFace = null;
    let largestWidth = 0;
    detections.forEach(detection => {
        const box = detection.detection.box;
        if (box.width > largestWidth) {
            largestWidth = box.width;
            largestFace = detection;
        }
    });
    
    // Manage phrase display for the largest face (if any)
    if (largestFace) {
        const box = largestFace.detection.box;
        const expressions = largestFace.expressions;
        const faceId = largestFace.faceId;
        
        // Check if this is a new person (different face ID)
        if (lastFaceId !== null && lastFaceId !== faceId) {
            // New person detected - reset everything
            // resetPhraseCycle();
        }
        lastFaceId = faceId;
        
        // Compute dominant emotion excluding 'neutral' for phrase selection
        const emotionsExNeutral = Object.keys(expressions).filter(k => k !== 'neutral');
        let dominantEmotionX = 'neutral';
        if (emotionsExNeutral.length > 0) {
            dominantEmotionX = emotionsExNeutral.reduce((a, b) => 
                expressions[a] > expressions[b] ? a : b
            );
        }
        
        // Get movement speed in pixels/frame for phrase selection
        const speedData = getAverageSpeed(faceId);
        const speedPixelsPerFrame = speedData ? speedData.avgSpeed : null;
        
        // If no audio is playing and we're not waiting for next message, start a new message
        if (!currentAudio && !isWaitingForNextMessage && phraseSelector.ready) {
            // If we're in POF mode, don't do anything here (POF phrases are handled by audio.onended)
            if (!isInPOFMode) {
                // We're in local phrase mode
                localPhraseCount++;
                
                // Check if we've shown 3 local phrases and should call POF
                if (localPhraseCount > LOCAL_PHRASES_BEFORE_POF && !pofRequestInProgress) {
                    // Reset counter and trigger POF fetch
                    localPhraseCount = 0;
                    isWaitingForNextMessage = true; // Prevent multiple triggers
                    fetchPOFPhrases().catch(error => {
                        console.error('POF fetch failed:', error);
                        // On error, continue with local phrases
                        isWaitingForNextMessage = false;
                        selectAndPlayLocalPhrase(dominantEmotionX, speedPixelsPerFrame);
                    });
                } else {
                    // Select and play local phrase
                    selectAndPlayLocalPhrase(dominantEmotionX, speedPixelsPerFrame);
                }
            }
        }
    } else {
        // No faces detected - don't start new messages, but let current audio finish playing
        // Don't reset state - continue from same point when face reappears
        // Just don't trigger new messages (handled by the if (largestFace) check above)
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(drawdata, 0, 0);
    ctx.save();

    // Draw detections
    detections.forEach(detection => {
        const box = detection.detection.box;
        const expressions = detection.expressions;
        const age = Math.round(detection.age);
        const gender = detection.gender;
        const genderProbability = detection.genderProbability;
        const faceId = detection.faceId;
        
        // Get dominant emotion
        const emotions = expressions;
        const dominantEmotion = Object.keys(emotions).reduce((a, b) => 
            emotions[a] > emotions[b] ? a : b
        );
        const emotionConfidence = Math.round(emotions[dominantEmotion] * 100);

        // Compute dominant emotion excluding 'neutral'
        const emotionsExNeutral = Object.keys(emotions).filter(k => k !== 'neutral');
        let dominantEmotionX = 'neutral';
        let dominantEmotionXConfidence = 0;
        if (emotionsExNeutral.length > 0) {
            dominantEmotionX = emotionsExNeutral.reduce((a, b) => 
                emotions[a] > emotions[b] ? a : b
            );
            dominantEmotionXConfidence = Math.round(emotions[dominantEmotionX] * 100);
        }
        
        // Get movement speed
        const speedData = getAverageSpeed(faceId);
        const speedText = speedData 
            ? `Speed: ${speedData.speedPerSecond.toFixed(1)} px/s (${phraseSelector.getSpeedCategory(speedData.speedPerSecond)})`
            : 'Speed: calculating...';
        
        // Draw rectangle around face
        // Both video and canvas are mirrored via CSS transform: scaleX(-1)
        // Face detection gives coordinates in the video's natural (non-mirrored) coordinate system
        // Since both video and canvas are mirrored the same way, the coordinates align directly
        // We draw at the same natural coordinates that face detection provides
        
        ctx.strokeStyle = '#4CAF50';
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x, box.y, box.width, box.height);
        
        ctx.restore();
        
        // Prepare text information
        // Since canvas is mirrored via CSS, we need to flip text coordinates
        const fontSize = Math.max(16, box.width / 25);
        ctx.font = `bold ${fontSize}px Arial`;
        
        // Format gender with probability
        const genderText = `${gender} (${Math.round(genderProbability * 100)}%)`;
        const emotionText = `${dominantEmotion} (${emotionConfidence}%)`;
        const ageText = `Age: ${age}`;
        
        // Draw text background for better visibility
        const textLines = [genderText, ageText, emotionText, speedText];
        const lineHeight = fontSize + 5;
        const padding = 5;
        
        // Calculate text width for background
        const maxWidth = Math.max(
            ...textLines.map(line => ctx.measureText(line).width)
        );
        
        // Calculate text position (both video and canvas are mirrored the same way, so coordinates align)
        const textX = box.x;
        const textY = box.y - 10;
        
        if (false) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(
                textX - padding,
                textY - (textLines.length * lineHeight) - padding,
                maxWidth + (padding * 2),
                (textLines.length * lineHeight) + (padding * 2)
            );
        }     
        
        // Draw text
        ctx.textBaseline = 'bottom';
        
        textLines.forEach((line, index) => {
            // Use different color for speed
            if (line.includes('Speed')) {
                ctx.fillStyle = speedData ? '#FFD700' : '#FFA500'; // Gold or orange
            } else {
                ctx.fillStyle = '#fff';
            }
            
            // Draw text at original position (coordinates align since both are mirrored the same way)
            ctx.fillText(
                line,
                textX,
                textY - (textLines.length - 1 - index) * lineHeight
            );
        });
    });
    
    // Draw phrase below the largest face (if any face is detected and phrase exists)
    if (largestFace && phraseDisplayData.currentPhrase) {
        const box = largestFace.detection.box;
        const phraseFontSize = Math.max(14, box.width / 30);
        ctx.textBaseline = 'top';
        
        // Draw phrase below the face box
        const phraseY = box.y + box.height + 10;
        const phraseX = box.x;
        
        // Draw background for better visibility
        if (false) {
            const phraseWidth = ctx.measureText(phraseDisplayData.currentPhrase).width;
            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            ctx.fillRect(
                phraseX - 5,
                phraseY - 2,
                phraseWidth + 10,
                phraseFontSize + 4
            );
        }        
        
        // Draw phrase text - use red for VLM phrases, gold for local phrases
        const phraseText = phraseDisplayData.currentPhrase.replace(/\+/g, '');
        ctx.fillStyle = phraseDisplayData.isVLM ? '#FF0000' : '#FFD700'; // Red for VLM, gold for local
        ctx.font = `italic bold ${phraseFontSize * 1.6}px Arial, sans-serif`; // Increase size by 60%
        ctx.fillText(phraseText, phraseX, phraseY);

    }
    // Continue detection loop (target ~30 FPS)
    requestAnimationFrame(detectFaces);
}

// Event listeners
startBtn.addEventListener('click', startCamera);
closeBtn.addEventListener('click', stopCamera);

// Initialize on page load
window.addEventListener('load', async () => {
    // Load phrase selector
    await phraseSelector.loadPhrases();
    // Load face detection models
    loadModels();
});

