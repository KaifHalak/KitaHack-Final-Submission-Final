import { ObjectDetector, DrawingUtils, FilesetResolver } from '@mediapipe/tasks-vision';
import { GoogleGenerativeAI } from "@google/generative-ai";

const ai = new GoogleGenerativeAI("AIzaSyBcfMUGoBbo6xX1mF0o89DMsVzqQrlXmmg");

// Object Detection Constants
const DETECTION_SCORE_THRESHOLD = 0.5;  // Confidence threshold for detections (0-1)
                                       // Higher: More confident but fewer detections
                                       // Lower: More detections but more false positives

const MAX_DETECTION_RESULTS = 20;       // Maximum number of objects to detect per frame
                                       // Higher: Detect more objects but slower performance
                                       // Lower: Faster but might miss objects
                                       // Range: 1-100

// Object Tracking Constants
const TRACKING_IOU_THRESHOLD = 0.3;     // Threshold for matching objects using IoU (0-1)
                                       // Higher (>0.5): Stricter matching, better for slow moving objects
                                       // Lower (<0.3): More lenient matching, better for fast objects
                                       // Recommended range: 0.2-0.7

const CENTER_DISTANCE_THRESHOLD = 50;    // Maximum pixel distance between object centers for matching
                                       // Higher: Better for fast moving objects but may cause ID switches
                                       // Lower: More precise but may lose fast objects
                                       // Range depends on video resolution, typically 30-100

const MAX_PREDICTION_FRAMES = 3;        // Number of frames to keep predicting missing objects
                                       // Higher: Longer object persistence but may cause ghost tracks
                                       // Lower: Quicker to drop lost objects
                                       // Range: 1-5

const MEASUREMENT_INTERVAL = 10;        // Frames between velocity/direction updates
                                       // Higher: Smoother measurements but less responsive
                                       // Lower: More responsive but may be jittery
                                       // Range: 5-30

// Proximity Warning Constants
const VERY_CLOSE_RATIO = 0.08;         // Area ratio threshold for "very close" warning (0-1)
                                       // Higher: Warnings at greater distance
                                       // Lower: Warnings only when very close
                                       // Range: 0.05-0.2

const GETTING_CLOSE_RATIO = 0.03;      // Area ratio threshold for "getting close" warning (0-1)
                                       // Should be lower than VERY_CLOSE_RATIO
                                       // Range: 0.02-0.1

// Speech Synthesis Constants
const ANNOUNCEMENT_DELAY = 5000;        // Minimum milliseconds between announcements
                                       // Higher: Less frequent announcements
                                       // Lower: More frequent announcements
                                       // Range: 1000-5000

const SPEECH_RATE = 1.2;               // Speech rate for normal announcements
                                       // Higher: Faster speech
                                       // Lower: Slower speech
                                       // Range: 0.5-2.0

const URGENT_SPEECH_RATE = 1.3;        // Speech rate for urgent announcements
                                       // Should be slightly higher than SPEECH_RATE
                                       // Range: 0.5-2.0

// Track Management Constants
const MAX_POSITION_HISTORY = 10;       // Number of past positions to keep for each track
                                       // Higher: Better prediction but more memory usage
                                       // Lower: Less memory but basic prediction
                                       // Range: 5-20

const TRACK_CLEANUP_DELAY = 1500;      // Milliseconds before removing stale tracks
                                       // Higher: Longer track persistence
                                       // Lower: Quicker cleanup
                                       // Range: 1000-3000

// Speech Warning Constants
const SPEECH_UPDATE_INTERVAL = 20;     // Frames between speech updates
                                      // Higher: Less frequent warnings
                                      // Lower: More frequent warnings
                                      // Range: 10-30

const COLLISION_PREDICTION_TIME = 500; // Time in ms to predict potential collisions
                                      // Higher: Earlier warnings but more false positives
                                      // Lower: More accurate but later warnings
                                      // Range: 1000-5000

const OVERLAP_THRESHOLD = 0.3;         // IoU threshold to consider objects overlapping
                                      // Higher: Stricter grouping
                                      // Lower: More liberal grouping
                                      // Range: 0.2-0.5

const APPROACH_SPEED_THRESHOLD = 50;   // Pixels per second to consider object approaching
                                      // Higher: Only fast approaching objects trigger
                                      // Lower: More sensitive to approaching objects
                                      // Range: 30-100

// Gemini Scene Analysis Constants
const SCENE_ANALYSIS_INTERVAL = 20000;    // Milliseconds between scene analyses
                                        // Higher: Less frequent but saves resources
                                        // Lower: More frequent but more API calls
                                        // Range: 3000-10000

const SCENE_CLIP_DURATION = 3000;        // Milliseconds of footage to analyze
                                        // Higher: More context but slower processing
                                        // Lower: Faster but less context
                                        // Range: 2000-5000

const MAX_FRAMES_PER_CLIP = 10;          // Maximum frames to send to Gemini
                                        // Higher: Better analysis but more data
                                        // Lower: Less accurate but faster
                                        // Range: 5-15

const GEMINI_PRIORITY = 'gemini';  // Highest priority level for Gemini narrations
                                  // Priority hierarchy: gemini > urgent > normal

class VisionAssistApp {
    constructor() {
        this.video = document.getElementById('video');
        this.startButton = document.getElementById('start-button');
        this.videoInput = document.getElementById('video-input');
        this.playVideoButton = document.getElementById('play-video');
        this.detectionInfo = document.getElementById('detection-info');
        this.detector = null;
        this.stream = null;
        this.isRunning = false;
        this.isCamera = false;
        this.canvas = document.createElement('canvas');
        this.canvas.style.display = 'none';
        document.body.appendChild(this.canvas);
        this.canvasCtx = this.canvas.getContext('2d');

        // Modify tracking properties
        this.previousDetections = new Map(); // Store previous frame detections
        this.trackingThreshold = TRACKING_IOU_THRESHOLD;
        this.trackedObjects = new Map(); // Store object tracking history
        this.objectIdCounter = 0; // Unique ID for each tracked object
        this.speechSynthesis = window.speechSynthesis;
        this.lastAnnouncement = 0; // To prevent too frequent announcements
        this.announcementDelay = ANNOUNCEMENT_DELAY;
        
        // Enhanced tracking properties
        this.centerDistanceThreshold = CENTER_DISTANCE_THRESHOLD;
        this.maxPredictionFrames = MAX_PREDICTION_FRAMES;
        
        // Add frame counter and update interval
        this.frameCounter = 0;
        this.measurementInterval = MEASUREMENT_INTERVAL;
        
        // Event listeners
        this.startButton.addEventListener('click', () => this.toggleCamera());
        this.videoInput.addEventListener('change', () => this.handleVideoUpload());
        this.playVideoButton.addEventListener('click', () => this.toggleVideo());
        
        // Initialize detector
        this.initializeDetector();

        // Initialize speech synthesis
        this.initializeSpeech();

        this.speechCounter = 0;
        this.lastWarnings = new Map(); // Track last warnings for each object

        // Add new properties for scene analysis
        this.frameBuffer = [];
        this.lastSceneAnalysis = 0;
        this.sceneAnalysisInterval = null;

        // Initialize scene analysis
        this.initializeSceneAnalysis();

        // Add speech queue management
        this.speechQueue = [];
        this.isGeminiSpeaking = false;
        this.currentUtterance = null;
    }

    async handleVideoUpload() {
        const file = this.videoInput.files[0];
        if (file) {
            const videoUrl = URL.createObjectURL(file);
            this.video.src = videoUrl;
            this.playVideoButton.disabled = false;
            
            // Stop camera if it's running
            if (this.isCamera) {
                this.stopCamera();
            }
        }
    }

    async toggleVideo() {
        if (!this.isRunning) {
            this.isCamera = false;
            this.isRunning = true;
            this.playVideoButton.textContent = 'Stop Video';
            await this.video.play();
            this.detectObjects();
        } else {
            this.isRunning = false;
            this.video.pause();
            this.playVideoButton.textContent = 'Play Video';
        }
    }

    async toggleCamera() {
        if (!this.isRunning) {
            await this.startCamera();
        } else {
            this.stopCamera();
        }
    }

    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' }
            });

            this.video.srcObject = this.stream;
            this.video.playsInline = true;

            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => resolve();
            });

            await this.video.play();
            this.isCamera = true;
            this.isRunning = true;
            this.startButton.textContent = 'Stop Camera';
            this.playVideoButton.disabled = true;
            this.detectObjects();
        } catch (error) {
            console.error('Error starting camera:', error);
            alert('Error accessing camera. Please ensure camera permissions are granted.');
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.video.srcObject = null;
            this.isRunning = false;
            this.isCamera = false;
            this.startButton.textContent = 'Start Camera';
            this.playVideoButton.disabled = false;
        }
    }

    async initializeDetector() {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
        );

        this.detector = await ObjectDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
                delegate: "CPU"
            },
            scoreThreshold: DETECTION_SCORE_THRESHOLD,
            maxResults: MAX_DETECTION_RESULTS
        });
    }

    async detectObjects() {
        if (!this.isRunning || !this.video.videoWidth) return;

        try {
            // Set canvas dimensions to match video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            // Draw the video frame on the canvas
            this.canvasCtx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

            // Detect objects using the canvas
            const results = this.detector.detect(this.canvas);
            this.updateDetectionInfo(results);

            // Continue detection loop with a slight delay to reduce CPU usage
            if (this.isRunning) {
                setTimeout(() => {
                    requestAnimationFrame(() => this.detectObjects());
                }, 1); // Add a 100ms delay between frames
            }
        } catch (error) {
            console.error('Error during detection:', error);
        }
    }

    updateDetectionInfo(results) {
        if (!results || !results.detections) return;
        this.frameCounter = (this.frameCounter + 1) % MEASUREMENT_INTERVAL;
        const now = Date.now();

        this.detectionInfo.innerHTML = '';
        const highlights = document.querySelectorAll('.highlighter, .detection-label');
        highlights.forEach(el => el.remove());

        const currentDetections = new Map();

        results.detections.forEach((detection) => {
            const boundingBox = detection.boundingBox;
            const center = this.getCenter(boundingBox);
            let matched = false;
            let objectId = null;

            this.trackedObjects.forEach((trackData, id) => {
                if (matched) return;

                const iou = this.calculateIoU(boundingBox, trackData.lastBox);
                const centerDistance = this.calculateDistance(center, trackData.lastCenter);
                const predictedCenter = this.getPredictedCenter(trackData);
                const distanceToPrediction = predictedCenter ? 
                    this.calculateDistance(center, predictedCenter) : Infinity;

                if (iou > this.trackingThreshold || 
                    centerDistance < this.centerDistanceThreshold ||
                    distanceToPrediction < this.centerDistanceThreshold * 1.5) {
                    matched = true;
                    objectId = id;
                    
                    // Calculate current velocity
                    const timeDelta = now - trackData.lastSeen;
                    const currentVelocity = {
                        x: (center.x - trackData.lastCenter.x) / timeDelta,
                        y: (center.y - trackData.lastCenter.y) / timeDelta
                    };
                    
                    // Calculate current speed
                    const currentSpeed = this.calculateSpeed(currentVelocity);
                    
                    // Determine if we should update measurements
                    const shouldUpdateMeasurements = 
                        // Update if current speed is 0 (stationary object)
                        currentSpeed === 0 ||
                        // Update if previous speed was 0 and now there's movement
                        (trackData.speed === 0 && currentSpeed > 0) ||
                        // Update if object is moving and we're on the measurement interval
                        (currentSpeed > 0 && this.frameCounter === 0);

                    // Update tracking data
                    currentDetections.set(objectId, {
                        ...trackData,
                        lastBox: boundingBox,
                        lastCenter: center,
                        lastSeen: now,
                        missingFrames: 0,
                        velocity: shouldUpdateMeasurements ? currentVelocity : trackData.velocity,
                        category: detection.categories[0],
                        positions: [...trackData.positions, center].slice(-MAX_POSITION_HISTORY),
                        lastMeasurement: shouldUpdateMeasurements ? now : trackData.lastMeasurement,
                        speed: shouldUpdateMeasurements ? currentSpeed : trackData.speed,
                        direction: shouldUpdateMeasurements ? 
                            this.calculateDirection(currentVelocity) : 
                            trackData.direction
                    });
                }
            });

            // Create new track if no match found
            if (!matched) {
                objectId = `obj_${this.objectIdCounter++}`;
                currentDetections.set(objectId, {
                    lastBox: boundingBox,
                    lastCenter: center,
                    lastSeen: now,
                    missingFrames: 0,
                    velocity: { x: 0, y: 0 },
                    category: detection.categories[0],
                    positions: [center],
                    lastMeasurement: now,
                    speed: 0,
                    direction: 0
                });
            }

            // Update UI with enhanced tracking info
            this.updateObjectUI(objectId, boundingBox, detection, currentDetections.get(objectId));
        });

        // Second pass: Handle missing objects and predictions
        this.trackedObjects.forEach((trackData, id) => {
            if (!currentDetections.has(id)) {
                if (trackData.missingFrames < this.maxPredictionFrames) {
                    // Predict new position
                    const predictedBox = this.getPredictedBox(trackData);
                    const predictedCenter = this.getPredictedCenter(trackData);
                    
                    currentDetections.set(id, {
                        ...trackData,
                        lastBox: predictedBox,
                        lastCenter: predictedCenter,
                        missingFrames: trackData.missingFrames + 1,
                        positions: [...trackData.positions, predictedCenter]
                    });

                    // Update UI with predicted position (semi-transparent)
                    this.updateObjectUI(id, predictedBox, { categories: [trackData.category] }, 
                        currentDetections.get(id), true);
                }
            }
        });

        // Clean up old tracks
        currentDetections.forEach((detection, id) => {
            if (now - detection.lastSeen > TRACK_CLEANUP_DELAY || detection.missingFrames >= this.maxPredictionFrames) {
                currentDetections.delete(id);
            }
        });

        this.trackedObjects = currentDetections;

        // Update speech warnings every SPEECH_UPDATE_INTERVAL frames
        this.speechCounter = (this.speechCounter + 1) % SPEECH_UPDATE_INTERVAL;
        if (this.speechCounter === 0) {
            this.updateSpeechWarnings(currentDetections);
        }
    }

    updateObjectUI(objectId, boundingBox, detection, trackData, isPredicted = false) {
        const category = detection.categories[0];
        const objectArea = boundingBox.width * boundingBox.height;
        const frameArea = this.video.videoWidth * this.video.videoHeight;
        const areaRatio = objectArea / frameArea;

        // Use stored speed and direction
        const speed = trackData.speed;
        const direction = trackData.direction;
        const directionText = speed === 0 ? '•' : this.getDirectionText(direction);
        const speedText = speed === 0 ? 'Stationary' : `${Math.round(speed * 100)}px/s`;

        // Determine proximity colors and text (existing logic)
        let proximityColor = areaRatio > VERY_CLOSE_RATIO ? 'rgba(255, 0, 0, 0.4)' : 
                           areaRatio > GETTING_CLOSE_RATIO ? 'rgba(255, 255, 0, 0.4)' : 
                           'rgba(0, 255, 0, 0.25)';
        let proximityText = areaRatio > VERY_CLOSE_RATIO ? 'VERY CLOSE!' :
                          areaRatio > GETTING_CLOSE_RATIO ? 'Getting Close' :
                          'Safe Distance';

        // Create UI elements with opacity for predicted positions
        const opacity = isPredicted ? 0.5 : 1;
        const highlighter = document.createElement('div');
        highlighter.className = 'highlighter';
        highlighter.style.left = `${boundingBox.originX}px`;
        highlighter.style.top = `${boundingBox.originY}px`;
        highlighter.style.width = `${boundingBox.width}px`;
        highlighter.style.height = `${boundingBox.height}px`;
        highlighter.style.background = proximityColor;
        highlighter.style.opacity = opacity;
        highlighter.style.borderColor = areaRatio > 0.4 ? '#ff0000' : '#ffffff';

        const label = document.createElement('div');
        label.className = 'detection-label';
        label.style.opacity = opacity;
        label.innerHTML = `
            ID: ${objectId}<br>
            ${category.categoryName} (${Math.round(category.score * 100)}%)<br>
            ${proximityText}<br>
            ${speedText}<br>
            Direction: ${directionText}
        `;
        label.style.left = `${boundingBox.originX}px`;
        label.style.top = `${boundingBox.originY - 90}px`;

        document.getElementById('container').appendChild(highlighter);
        document.getElementById('container').appendChild(label);
    }

    // Helper methods
    getCenter(box) {
        return {
            x: box.originX + box.width / 2,
            y: box.originY + box.height / 2
        };
    }

    calculateDistance(point1, point2) {
        return Math.sqrt(
            Math.pow(point2.x - point1.x, 2) + 
            Math.pow(point2.y - point1.y, 2)
        );
    }

    getPredictedCenter(trackData) {
        if (!trackData.lastCenter) return null;
        return {
            x: trackData.lastCenter.x + trackData.velocity.x * trackData.missingFrames,
            y: trackData.lastCenter.y + trackData.velocity.y * trackData.missingFrames
        };
    }

    getPredictedBox(trackData) {
        const predictedCenter = this.getPredictedCenter(trackData);
        return {
            originX: predictedCenter.x - trackData.lastBox.width / 2,
            originY: predictedCenter.y - trackData.lastBox.height / 2,
            width: trackData.lastBox.width,
            height: trackData.lastBox.height
        };
    }

    getDirectionText(degrees) {
        if (degrees === 0 && this.speed === 0) {
            return '•'; // Dot symbol for stationary objects
        }
        const directions = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘'];
        const index = Math.round(((degrees + 180) % 360) / 45) % 8;
        return directions[index];
    }

    calculateIoU(box1, box2) {
        // Calculate intersection coordinates
        const xLeft = Math.max(box1.originX, box2.originX);
        const yTop = Math.max(box1.originY, box2.originY);
        const xRight = Math.min(box1.originX + box1.width, box2.originX + box2.width);
        const yBottom = Math.min(box1.originY + box1.height, box2.originY + box2.height);

        // Check if there is no intersection
        if (xRight < xLeft || yBottom < yTop) {
            return 0;
        }

        // Calculate intersection area
        const intersectionArea = (xRight - xLeft) * (yBottom - yTop);

        // Calculate union area
        const box1Area = box1.width * box1.height;
        const box2Area = box2.width * box2.height;
        const unionArea = box1Area + box2Area - intersectionArea;

        // Return IoU
        return intersectionArea / unionArea;
    }

    // Add event listener for video ending
    setupVideoEndHandler() {
        this.video.addEventListener('ended', () => {
            if (!this.isCamera) {
                this.isRunning = false;
                this.playVideoButton.textContent = 'Play Video';
            }
        });
    }

    // Add new method for speech initialization
    initializeSpeech() {
        this.utterance = new SpeechSynthesisUtterance();
        this.utterance.rate = SPEECH_RATE;
        this.utterance.pitch = 1;
        this.utterance.volume = 1;
    }

    // Modify the speak method to handle priorities
    speak(message, priority = 'normal') {
        const now = Date.now();
        
        // Create speech item
        const speechItem = {
            message,
            priority,
            timestamp: now
        };

        // Handle Gemini priority
        if (priority === GEMINI_PRIORITY) {
            // Clear non-Gemini items from queue
            this.speechQueue = this.speechQueue.filter(item => item.priority === GEMINI_PRIORITY);
            
            // Add Gemini message to queue
            this.speechQueue.push(speechItem);
            
            // If something is currently speaking and it's not Gemini
            if (this.speechSynthesis.speaking && !this.isGeminiSpeaking) {
                // Cancel current speech
                this.speechSynthesis.cancel();
                // Current speech's onend event will trigger next speech
            } else if (!this.speechSynthesis.speaking) {
                // If nothing is speaking, start speaking
                this.speakNextInQueue();
            }
            return;
        }

        // Handle normal and urgent priorities
        if (now - this.lastAnnouncement < ANNOUNCEMENT_DELAY && priority !== 'urgent') {
            return; // Skip non-urgent messages if too soon
        }

        // Don't interrupt Gemini speech
        if (this.isGeminiSpeaking) {
            return;
        }

        // Add to queue
        this.speechQueue.push(speechItem);

        // If nothing is speaking, start speaking
        if (!this.speechSynthesis.speaking) {
            this.speakNextInQueue();
        }
    }

    // Add new method to handle speech queue
    speakNextInQueue() {
        if (this.speechQueue.length === 0) {
            this.isGeminiSpeaking = false;
            this.currentUtterance = null;
            return;
        }

        // Get next speech item
        const speechItem = this.speechQueue.shift();
        
        // Create new utterance
        const utterance = new SpeechSynthesisUtterance(speechItem.message);
        
        // Set speech parameters based on priority
        if (speechItem.priority === GEMINI_PRIORITY) {
            utterance.rate = SPEECH_RATE;
            utterance.pitch = 1;
            this.isGeminiSpeaking = true;
        } else if (speechItem.priority === 'urgent') {
            utterance.rate = URGENT_SPEECH_RATE;
            utterance.pitch = 1.2;
        } else {
            utterance.rate = SPEECH_RATE;
            utterance.pitch = 1;
        }

        // Handle speech end
        utterance.onend = () => {
            this.lastAnnouncement = Date.now();
            this.currentUtterance = null;
            if (speechItem.priority === GEMINI_PRIORITY) {
                this.isGeminiSpeaking = false;
            }
            this.speakNextInQueue(); // Speak next item in queue
        };

        // Handle speech error
        utterance.onerror = (event) => {
            console.error('Speech error:', event);
            this.currentUtterance = null;
            if (speechItem.priority === GEMINI_PRIORITY) {
                this.isGeminiSpeaking = false;
            }
            this.speakNextInQueue(); // Try next item in queue
        };

        // Speak the utterance
        this.currentUtterance = utterance;
        this.speechSynthesis.speak(utterance);
        this.lastAnnouncement = speechItem.timestamp;
    }

    // Add new helper methods for speed and direction calculations
    calculateSpeed(velocity) {
        return Math.sqrt(
            velocity.x * velocity.x + 
            velocity.y * velocity.y
        );
    }

    calculateDirection(velocity) {
        return Math.atan2(velocity.y, velocity.x) * 180 / Math.PI;
    }

    updateSpeechWarnings(currentDetections) {
        const warnings = new Map();
        const now = Date.now();

        // First pass: Collect all warnings
        currentDetections.forEach((trackData, id) => {
            const warning = this.assessThreat(trackData);
            if (warning) {
                warnings.set(id, warning);
            }
        });

        // Group overlapping warnings
        const groupedWarnings = this.groupOverlappingWarnings(warnings, currentDetections);

        // Generate and speak warnings
        groupedWarnings.forEach(group => {
            const message = this.generateWarningMessage(group);
            if (message) {
                this.speak(message, group.some(w => w.priority === 'urgent') ? 'urgent' : 'normal');
            }
        });
    }

    assessThreat(trackData) {
        const areaRatio = this.calculateAreaRatio(trackData.lastBox);
        
        // Handle very close objects immediately
        if (areaRatio > VERY_CLOSE_RATIO) {
            return {
                priority: 'urgent',
                type: 'very_close',
                position: this.getPositionDescription(trackData.lastCenter),
                category: trackData.category.categoryName
            };
        }

        // Assess getting close objects
        if (areaRatio > GETTING_CLOSE_RATIO) {
            // Predict future position
            const predictedPosition = this.predictFuturePosition(trackData, COLLISION_PREDICTION_TIME);
            if (!predictedPosition) return null;

            const predictedAreaRatio = this.calculateAreaRatio(predictedPosition);
            const approachingSpeed = -trackData.velocity.y; // Negative y means approaching

            // Warn if object is approaching and predicted to get very close
            if (approachingSpeed > APPROACH_SPEED_THRESHOLD && predictedAreaRatio > VERY_CLOSE_RATIO) {
                return {
                    priority: 'normal',
                    type: 'approaching',
                    position: this.getPositionDescription(trackData.lastCenter),
                    category: trackData.category.categoryName,
                    timeToCollision: COLLISION_PREDICTION_TIME
                };
            }
        }

        return null;
    }

    groupOverlappingWarnings(warnings, detections) {
        const groups = [];
        const processed = new Set();

        warnings.forEach((warning1, id1) => {
            if (processed.has(id1)) return;

            const group = [warning1];
            processed.add(id1);

            warnings.forEach((warning2, id2) => {
                if (id1 === id2 || processed.has(id2)) return;

                const iou = this.calculateIoU(
                    detections.get(id1).lastBox,
                    detections.get(id2).lastBox
                );

                if (iou > OVERLAP_THRESHOLD) {
                    group.push(warning2);
                    processed.add(id2);
                }
            });

            groups.push(group);
        });

        return groups;
    }

    generateWarningMessage(warningGroup) {
        if (warningGroup.length === 0) return null;

        // Sort warnings by priority (urgent first)
        warningGroup.sort((a, b) => 
            a.priority === 'urgent' ? -1 : b.priority === 'urgent' ? 1 : 0
        );

        if (warningGroup.length === 1) {
            const warning = warningGroup[0];
            if (warning.type === 'very_close') {
                return `Warning! ${warning.category} ${warning.position}!`;
            } else {
                return `Caution! ${warning.category} approaching from ${warning.position}`;
            }
        } else {
            const categories = [...new Set(warningGroup.map(w => w.category))];
            const position = warningGroup[0].position;
            
            if (warningGroup.some(w => w.type === 'very_close')) {
                return `Warning! Multiple objects including ${categories.join(' and ')} ${position}!`;
            } else {
                return `Caution! Multiple objects approaching from ${position}`;
            }
        }
    }

    getPositionDescription(center) {
        const videoCenter = {
            x: this.video.videoWidth / 2,
            y: this.video.videoHeight / 2
        };

        const horizontal = center.x < videoCenter.x * 0.7 ? 'left' :
                         center.x > videoCenter.x * 1.3 ? 'right' :
                         'center';
                         
        const vertical = center.y < videoCenter.y * 0.7 ? 'above' :
                        center.y > videoCenter.y * 1.3 ? 'below' :
                        'ahead';

        return `${vertical}${horizontal !== 'center' ? ' to the ' + horizontal : ''}`;
    }

    predictFuturePosition(trackData, timeMs) {
        if (!trackData.velocity) return null;

        const timeDelta = timeMs / 1000; // Convert to seconds
        return {
            originX: trackData.lastBox.originX + trackData.velocity.x * timeDelta,
            originY: trackData.lastBox.originY + trackData.velocity.y * timeDelta,
            width: trackData.lastBox.width,
            height: trackData.lastBox.height
        };
    }

    calculateAreaRatio(box) {
        const objectArea = box.width * box.height;
        const frameArea = this.video.videoWidth * this.video.videoHeight;
        return objectArea / frameArea;
    }

    // Add new method for scene analysis initialization
    initializeSceneAnalysis() {
        // Start scene analysis when detection starts
        this.video.addEventListener('play', () => {
            this.startSceneAnalysis();
        });

        // Stop scene analysis when video stops
        this.video.addEventListener('pause', () => {
            this.stopSceneAnalysis();
        });
    }

    startSceneAnalysis() {
        this.sceneAnalysisInterval = setInterval(() => {
            this.analyzeScene();
        }, SCENE_ANALYSIS_INTERVAL);
    }

    stopSceneAnalysis() {
        if (this.sceneAnalysisInterval) {
            clearInterval(this.sceneAnalysisInterval);
            this.sceneAnalysisInterval = null;
        }
        this.frameBuffer = [];
    }

    async captureFrame() {
        // Create a temporary canvas for frame capture
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.video.videoWidth;
        tempCanvas.height = this.video.videoHeight;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0);
        
        // Convert to base64 and return
        return tempCanvas.toDataURL('image/jpeg', 0.7);
    }

    async analyzeScene() {
        try {
            const now = Date.now();
            if (now - this.lastSceneAnalysis < SCENE_ANALYSIS_INTERVAL) {
                return;
            }

            // Capture frames for the clip duration
            const MAX_FRAMES_TO_SEND = 3; // Reduce number of frames to analyze
            const skipFrames = Math.floor(MAX_FRAMES_PER_CLIP / MAX_FRAMES_TO_SEND);

            const frames = [];
            const captureInterval = SCENE_CLIP_DURATION / MAX_FRAMES_PER_CLIP;
            
            for (let i = 0; i < MAX_FRAMES_PER_CLIP; i += skipFrames) {
                const frame = await this.captureFrame();
                frames.push(frame);
                await new Promise(resolve => setTimeout(resolve, captureInterval * skipFrames));
            }

            // Prepare the prompt for Gemini
            const prompt = `You are a vision assistant guiding someone to navigate safely. Speak directly to them. 
            Describe hazards, movement patterns, and overall safety clearly and concisely. 
            
            Format: 
            1. Start with a brief summary of the situation. 
            2. Provide clear, direct guidance on what to do.
            
            IMPORTANT: Use short, direct sentences. No extra commentary. Just give practical, spoken-style advice. 
            
            Example: 
            "Watch out for pedestrians ahead. Step slightly to your right to avoid collisions. The path looks clear, but stay alert for sudden obstacles."
            `;
            ;

            // Call Gemini API (implementation depends on your setup)
            const analysis = await this.callGeminiAPI(frames, prompt);
            
            // Convert analysis to speech
            if (analysis && analysis.trim()) {
                this.speak(analysis, this.determineWarningPriority(analysis));
            }

            this.lastSceneAnalysis = now;
        } catch (error) {
            console.error('Scene analysis error:', error);
        }
    }

    async callGeminiAPI(frames, prompt) {
        try {
            console.log("Calling Gemini API...");
            const model = ai.getGenerativeModel({ model: "gemini-2.0-flash-lite" });
            
            const imagesParts = frames.map(frame => {
                const base64Data = frame.split(',')[1];
                return {
                    inlineData: {
                        data: base64Data,
                        mimeType: "image/jpeg"
                    }
                };
            });

            const parts = [
                { text: prompt },
                ...imagesParts
            ];

            const result = await model.generateContent(parts);
            const response = await result.response;
            const text = response.text();
            
            // Use GEMINI_PRIORITY when speaking Gemini's response
            if (text && text.trim()) {
                this.speak(text, GEMINI_PRIORITY);
            }
            
            console.log(text)

            return text;
        } catch (error) {
            console.error("Gemini API error:", error);
            if (error.message) {
                console.error("Error details:", error.message);
            }
            return null;
        }
    }

    determineWarningPriority(analysis) {
        // Simple priority determination based on keywords
        const urgentKeywords = ['immediate', 'danger', 'hazard', 'warning', 'stop'];
        return urgentKeywords.some(keyword => 
            analysis.toLowerCase().includes(keyword)) ? 'urgent' : 'normal';
    }
}

// Initialize the app when the page loads
window.addEventListener('load', () => {
    new VisionAssistApp();
}); 