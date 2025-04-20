// DOM Elements
const joinContainer = document.getElementById('join-container');
const callContainer = document.getElementById('call-container');
const connectionStatus = document.getElementById('connection-status');
const roomIdDisplay = document.getElementById('room-id-display');
const roomIdInput = document.getElementById('room-id-input');
const createRoomBtn = document.getElementById('create-room-btn');
const joinRoomBtn = document.getElementById('join-room-btn');
const toggleVideoBtn = document.getElementById('toggle-video-btn');
const toggleMicBtn = document.getElementById('toggle-mic-btn');
const leaveBtn = document.getElementById('leave-btn');
const localVideo = document.getElementById('local-video');
const remoteVideo = document.getElementById('remote-video');
const localSignText = document.getElementById('local-sign-text');
const remoteSignText = document.getElementById('remote-sign-text');
const startSpeechBtn = document.getElementById('start-speech-btn');
const speechText = document.getElementById('speech-text');
const convertBtn = document.getElementById('convert-btn');
const signImagesContainer = document.getElementById('sign-images-container');

// WebRTC configuration
const servers = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
    ]
};

// Global variables
let socket;
let localStream;
let peerConnection;
let roomId;
let isCameraOn = true;
let isMicOn = true;
let isConnected = false;
let frameInterval;
let recognition;
let isRemoteVideoConnected = false;

// Initialize the app
function init() {
    // Connect to the server using Socket.IO
    socket = io();
    
    // Setup socket event listeners
    setupSocketEvents();
    
    // Setup UI event listeners
    setupUIEvents();
    
    // Setup speech recognition if available
    setupSpeechRecognition();
}

// Setup Socket.IO event listeners
function setupSocketEvents() {
    socket.on('connect', () => {
        console.log('Connected to server');
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        handleDisconnect();
    });
    
    socket.on('room_created', (data) => {
        handleRoomCreated(data.room_id);
    });
    
    socket.on('room_joined', (data) => {
        if (data.success) {
            handleRoomJoined(data.room_id);
        } else {
            alert(`Failed to join room: ${data.message}`);
            hideConnectionStatus();
        }
    });
    
    socket.on('user_joined', (data) => {
        console.log(`User joined: ${data.sid}`);
        // Start call when a new user joins
        startCall();
    });
    
    socket.on('user_left', (data) => {
        console.log(`User left: ${data.sid}`);
        handlePeerDisconnect();
    });
    
    socket.on('video_frame', (data) => {
        // Display the processed video frame from the remote user
        if (!isRemoteVideoConnected) {
            let img = new Image();
            img.onload = () => {
                // Only update if remote video is still active and WebRTC stream isn't connected
                if (!isRemoteVideoConnected) {
                    remoteVideo.style.backgroundImage = `url(${data.frame})`;
                    remoteVideo.style.backgroundSize = 'cover';
                    remoteVideo.style.backgroundPosition = 'center';
                }
                
                // Display detected sign if available - only show the sign in the blue box
                if (data.detected_sign) {
                    remoteSignText.textContent = data.detected_sign;
                    remoteSignText.classList.remove('hidden');
                } else {
                    remoteSignText.classList.add('hidden');
                }
            };
            img.src = data.frame;
        } else {
            // Just update the sign text if WebRTC stream is already connected
            if (data.detected_sign) {
                remoteSignText.textContent = data.detected_sign;
                remoteSignText.classList.remove('hidden');
            } else {
                remoteSignText.classList.add('hidden');
            }
        }
    });
    
    socket.on('speech_to_sign_result', (data) => {
        displaySpeechToSignResult(data);
    });
    
    // Handle sign detection for local video
    socket.on('sign_detected', (data) => {
        if (data.sign) {
            localSignText.textContent = data.sign;
            localSignText.classList.remove('hidden');
        } else {
            localSignText.classList.add('hidden');
        }
    });
}

// Setup UI event listeners
function setupUIEvents() {
    createRoomBtn.addEventListener('click', createRoom);
    joinRoomBtn.addEventListener('click', joinRoom);
    toggleVideoBtn.addEventListener('click', toggleVideo);
    toggleMicBtn.addEventListener('click', toggleMic);
    leaveBtn.addEventListener('click', leaveRoom);
    startSpeechBtn.addEventListener('click', startSpeechRecognition);
    convertBtn.addEventListener('click', convertSpeechToSign);
}

// Handle room creation
function createRoom() {
    showConnectionStatus();
    socket.emit('create_room');
}

// Handle joining a room
function joinRoom() {
    const roomIdToJoin = roomIdInput.value.trim();
    if (roomIdToJoin) {
        showConnectionStatus();
        socket.emit('join_room', { room_id: roomIdToJoin });
    } else {
        alert('Please enter a room ID');
    }
}

// Handle successful room creation
function handleRoomCreated(newRoomId) {
    roomId = newRoomId;
    roomIdDisplay.textContent = roomId;
    startLocalStream();
}

// Handle successful room joining
function handleRoomJoined(joinedRoomId) {
    roomId = joinedRoomId;
    roomIdDisplay.textContent = roomId;
    startLocalStream();
}

// Start local video stream
async function startLocalStream() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ 
            video: true, 
            audio: true 
        });
        
        localVideo.srcObject = localStream;
        
        hideConnectionStatus();
        hideJoinUI();
        showCallUI();
        
        // Start sending video frames for processing
        startSendingVideoFrames();
        
    } catch (error) {
        console.error('Error accessing media devices:', error);
        alert('Could not access camera or microphone. Please check permissions.');
        hideConnectionStatus();
    }
}

// Start a WebRTC call
function startCall() {
    createPeerConnection();
    
    // Add all local tracks to the peer connection
    localStream.getTracks().forEach(track => {
        peerConnection.addTrack(track, localStream);
    });
    
    // Create and send an offer
    createAndSendOffer();
}

// Create a WebRTC peer connection
function createPeerConnection() {
    peerConnection = new RTCPeerConnection(servers);
    
    // Handle ICE candidates
    peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
            socket.emit('ice_candidate', {
                candidate: event.candidate.toJSON(),
                room_id: roomId
            });
        }
    };
    
    // Handle connection state changes
    peerConnection.onconnectionstatechange = () => {
        if (peerConnection.connectionState === 'connected') {
            console.log('Peers connected');
            isConnected = true;
        }
    };
    
    // Handle incoming tracks
    peerConnection.ontrack = (event) => {
        if (remoteVideo.srcObject !== event.streams[0]) {
            console.log('Received remote stream via WebRTC');
            remoteVideo.srcObject = event.streams[0];
            isRemoteVideoConnected = true;
        }
    };
    
    // Handle ICE connection state changes
    peerConnection.oniceconnectionstatechange = () => {
        console.log('ICE state:', peerConnection.iceConnectionState);
        if (peerConnection.iceConnectionState === 'disconnected' || 
            peerConnection.iceConnectionState === 'failed' || 
            peerConnection.iceConnectionState === 'closed') {
            handlePeerDisconnect();
        }
    };
    
    // Listen for ICE candidates from the server
    socket.on('ice_candidate', (data) => {
        const candidate = new RTCIceCandidate(data.candidate);
        peerConnection.addIceCandidate(candidate)
            .catch(e => console.error('Error adding ICE candidate:', e));
    });
    
    // Handle offer and answer
    socket.on('offer', async (data) => {
        if (!peerConnection) createPeerConnection();
        
        await peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
        
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);
        
        socket.emit('answer', {
            answer: answer,
            room_id: roomId
        });
    });
    
    socket.on('answer', async (data) => {
        await peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer))
            .catch(e => console.error('Error setting remote description:', e));
    });
}

// Create and send an offer to the peer
async function createAndSendOffer() {
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    
    socket.emit('offer', {
        offer: offer,
        room_id: roomId
    });
}

// Start sending video frames to the server for sign language detection
function startSendingVideoFrames() {
    // Create a canvas to capture video frames
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = 640;
    canvas.height = 480;
    
    // Send frames every 200ms for more responsive detection
    frameInterval = setInterval(() => {
        if (localStream && isCameraOn) {
            try {
                // Make sure video is properly initialized before using it
                if (localVideo.videoWidth > 0 && localVideo.videoHeight > 0) {
                    ctx.drawImage(localVideo, 0, 0, canvas.width, canvas.height);
                    const frameData = canvas.toDataURL('image/jpeg', 0.7);
                    
                    // Send the frame to the server
                    socket.emit('video_frame', { frame: frameData });
                }
            } catch (error) {
                console.error('Error capturing video frame:', error);
            }
        }
    }, 200);
}

// Toggle video on/off
function toggleVideo() {
    isCameraOn = !isCameraOn;
    
    localStream.getVideoTracks().forEach(track => {
        track.enabled = isCameraOn;
    });
    
    // Update UI
    toggleVideoBtn.innerHTML = isCameraOn ? 
        '<i class="fas fa-video"></i>' : 
        '<i class="fas fa-video-slash"></i>';
    
    // Hide sign text when video is off
    if (!isCameraOn) {
        localSignText.classList.add('hidden');
    }
}

// Toggle microphone on/off
function toggleMic() {
    isMicOn = !isMicOn;
    
    localStream.getAudioTracks().forEach(track => {
        track.enabled = isMicOn;
    });
    
    // Update UI
    toggleMicBtn.innerHTML = isMicOn ? 
        '<i class="fas fa-microphone"></i>' : 
        '<i class="fas fa-microphone-slash"></i>';
}

// Leave the current room
function leaveRoom() {
    // Stop sending frames
    clearInterval(frameInterval);
    
    // Close peer connection
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }
    
    // Stop local media tracks
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }
    
    // Reset video elements
    localVideo.srcObject = null;
    remoteVideo.srcObject = null;
    remoteVideo.style.backgroundImage = '';
    
    // Reset UI
    showJoinUI();
    hideCallUI();
    
    // Reset variables
    roomId = null;
    isConnected = false;
    isRemoteVideoConnected = false;
    
    // Reload the page for a fresh start
    window.location.reload();
}

// Handle peer disconnect
function handlePeerDisconnect() {
    remoteVideo.srcObject = null;
    remoteSignText.classList.add('hidden');
    
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }
    
    isConnected = false;
    isRemoteVideoConnected = false;
}

// Handle server disconnect
function handleDisconnect() {
    alert('Disconnected from server. Please refresh the page.');
    leaveRoom();
}

// Setup speech recognition
function setupSpeechRecognition() {
    if ('webkitSpeechRecognition' in window) {
        recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            speechText.value = transcript;
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            startSpeechBtn.disabled = false;
        };
        
        recognition.onend = () => {
            startSpeechBtn.disabled = false;
        };
    } else {
        startSpeechBtn.style.display = 'none';
        console.warn('Speech recognition not supported in this browser');
    }
}

// Start speech recognition
function startSpeechRecognition() {
    if (recognition) {
        speechText.value = '';
        startSpeechBtn.disabled = true;
        recognition.start();
    }
}

// Convert speech to sign language
function convertSpeechToSign() {
    const text = speechText.value.trim();
    if (text) {
        socket.emit('speech_to_sign', { text: text });
    } else {
        alert('Please enter or speak some text to convert');
    }
}

// Display speech to sign language results
function displaySpeechToSignResult(data) {
    const text = data.text;
    const imagePaths = data.images;
    const wordInfo = data.word_info || [];
    
    // Clear previous images
    signImagesContainer.innerHTML = '';
    
    // If no images were found
    if (imagePaths.length === 0) {
        const noImagesMsg = document.createElement('p');
        noImagesMsg.textContent = 'No sign language images available for this text.';
        signImagesContainer.appendChild(noImagesMsg);
        return;
    }
    
    // Add images with proper word spacing
    let currentWord = '';
    
    for (let i = 0; i < imagePaths.length; i++) {
        const path = imagePaths[i];
        const info = wordInfo[i] || {};
        
        // Add spacer between words
        if (info.type === 'word' || (info.type === 'character' && info.word !== currentWord)) {
            if (currentWord !== '') {
                const spacer = document.createElement('div');
                spacer.className = 'sign-spacer';
                signImagesContainer.appendChild(spacer);
            }
            currentWord = info.word;
        }
        
        // Create and add the image
        const img = document.createElement('img');
        img.src = path;
        img.alt = info.type === 'character' ? info.character : info.word;
        img.classList.add('sign-image');
        signImagesContainer.appendChild(img);
    }
}

// Show/hide UI elements
function showJoinUI() {
    joinContainer.classList.remove('hidden');
}

function hideJoinUI() {
    joinContainer.classList.add('hidden');
}

function showCallUI() {
    callContainer.classList.remove('hidden');
}

function hideCallUI() {
    callContainer.classList.add('hidden');
}

function showConnectionStatus() {
    connectionStatus.classList.remove('hidden');
}

function hideConnectionStatus() {
    connectionStatus.classList.add('hidden');
}

// Initialize the app when the page loads
window.addEventListener('load', init); 