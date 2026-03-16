const elements = {
    btnCamera: document.getElementById('btn-camera'),
    btnStopCamera: document.getElementById('btn-stop-camera'),
    btnUpload: document.getElementById('btn-upload'),
    btnStopUpload: document.getElementById('btn-stop-upload'),
    videoUpload: document.getElementById('video-upload'),
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),

    videoContainer: document.getElementById('video-container'),
    videoPlaceholder: document.getElementById('video-placeholder'),
    sourceVideo: document.getElementById('source-video'),
    overlayCanvas: document.getElementById('overlay-canvas'),

    systemStatus: document.getElementById('system-status'),
    statusText: document.querySelector('.status-text'),

    objectCount: document.getElementById('object-count'),
    detectionList: document.getElementById('detection-list'),
    perfStats: document.getElementById('perf-stats'),

    // Iriun / Camera Selector
    cameraSelectorGroup: document.getElementById('camera-selector-group'),
    cameraSelect: document.getElementById('camera-select'),
    btnRefreshCameras: document.getElementById('btn-refresh-cameras'),

    // Dedicated Iriun Elements
    btnIriunLauncher: document.getElementById('btn-iriun-launcher'),
    iriunModal: document.getElementById('iriun-modal'),
    closeIriunModal: document.getElementById('close-iriun-modal'),
    btnIriunRetry: document.getElementById('btn-iriun-retry'),
    btnIriunTroubleshoot: document.getElementById('btn-iriun-troubleshoot'),
    iriunSearchText: document.getElementById('iriun-search-text')
};

let stream = null;
let currentMode = null; // 'camera' or 'video'
let iriunScanInterval = null;
let isConnectingIriun = false;
let lastIriunIndex = -1; // Keep track of which Iriun camera we tried last
let isManualMode = false; // Whether the user is manually picking a camera

/* ======================================================== */
/*  CAMERA DEVICE ENUMERATION (Iriun Webcam support)        */
/* ======================================================== */

/**
 * Populate the camera dropdown with all available videoinput devices.
 * Auto-selects Iriun Webcam if detected.
 */
async function populateCameraList() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');

        elements.cameraSelect.innerHTML = '';
        const debugUl = document.getElementById('detected-cameras-ul');
        if (debugUl) debugUl.innerHTML = '';

        if (videoDevices.length === 0) {
            elements.cameraSelect.innerHTML = '<option value="">No cameras found</option>';
            return;
        }

        let iriunIndices = [];
        videoDevices.forEach((device, index) => {
            const isIriun = device.label.toLowerCase().includes('iriun');
            if (isIriun) iriunIndices.push(index);
            
            const option = document.createElement('option');
            option.value = device.deviceId;
            const label = device.label || `Camera ${index + 1}`;
            option.textContent = isIriun ? `📱 ${label} (Iriun)` : `📷 ${label}`;
            option.dataset.isIriun = isIriun;
            elements.cameraSelect.appendChild(option);

            if (debugUl) {
                const li = document.createElement('li');
                li.innerHTML = isIriun ? `<b>📱 ${label}</b>` : `📷 ${label}`;
                debugUl.appendChild(li);
            }
        });

        // Smart Selection Logic:
        // If we found Iriun cameras, cycle to the next one (relative to lastIriunIndex)
        let selectedIndex = 0;
        if (iriunIndices.length > 0) {
            // Pick the next available Iriun index to try something new
            lastIriunIndex = (lastIriunIndex + 1) % iriunIndices.length;
            selectedIndex = iriunIndices[lastIriunIndex];
        }

        elements.cameraSelect.selectedIndex = selectedIndex;
        console.log(`[CameraSelector] Found ${videoDevices.length} camera(s). Iriun count: ${iriunIndices.length}. Using: ${elements.cameraSelect.options[selectedIndex].text}`);
        
        return iriunIndices.length > 0;
    } catch (err) {
        console.error('[CameraSelector] Failed to enumerate devices:', err);
        return false;
    }
}

/**
 * Open a camera stream using the currently selected device.
 */
async function openSelectedCameraStream() {
    const selectedDeviceId = elements.cameraSelect.value;

    const constraints = selectedDeviceId
        ? { video: { deviceId: { exact: selectedDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false }
        : { video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false };

    stream = await navigator.mediaDevices.getUserMedia(constraints);

    elements.sourceVideo.srcObject = stream;
    elements.sourceVideo.style.display = 'block';
    elements.videoPlaceholder.style.display = 'none';
    currentMode = 'camera';

    elements.sourceVideo.onloadedmetadata = () => {
        elements.sourceVideo.play();
        updateUIState('ready');
    };
}

/* ======================================================== */
/*  INIT                                                     */
/* ======================================================== */

function init() {
    setupEventListeners();
    updateUIState('idle');

    // Listen for new devices being plugged in (e.g. Iriun connecting via USB/Wi-Fi)
    navigator.mediaDevices.addEventListener('devicechange', async () => {
        if (currentMode === 'camera') {
            await populateCameraList();
        }
    });
}

/* ======================================================== */
/*  IRIUN AUTO-SCAN LOGIC                                   */
/* ======================================================== */

function startIriunAutoScan() {
    if (iriunScanInterval || isManualMode) return;
    elements.iriunSearchText.textContent = "Scanning for Iriun Webcam...";
    
    let attempts = 0;
    iriunScanInterval = setInterval(async () => {
        if (isConnectingIriun || isManualMode) return; // Don't scan while currently trying to connect or in manual mode
        
        attempts++;
        const found = await populateCameraList();
        if (found) {
            handleIriunFound();
        } else {
            elements.iriunSearchText.textContent = `Searching for your phone... (Scan #${attempts})`;
            if (attempts > 10) {
                elements.iriunSearchText.innerHTML = 'Still searching... <br><small>Try "Manual Select" below if it is visible in the list.</small>';
            }
        }
    }, 2000);
}

function stopIriunAutoScan() {
    if (iriunScanInterval) {
        clearInterval(iriunScanInterval);
        iriunScanInterval = null;
    }
}

async function handleIriunFound(targetDeviceId = null) {
    if (isConnectingIriun) return;
    isConnectingIriun = true;
    
    stopIriunAutoScan();
    elements.iriunSearchText.innerHTML = '<span style="color:#22c55e">✔️ Found Iriun!</span><br>Syncing video stream...';

    // If a manual ID was passed (from manual select), use it
    if (targetDeviceId) {
        elements.cameraSelect.value = targetDeviceId;
    }
    
    // Smooth delay before closing to show success state
    setTimeout(async () => {
        try {
            // Check if we are already streaming Iriun to avoid flicker
            const currentDeviceId = elements.cameraSelect.value;
            const isAlreadyActive = stream && stream.active && 
                                  elements.cameraSelect.options[elements.cameraSelect.selectedIndex]?.text.toLowerCase().includes('iriun');

            if (!isAlreadyActive) {
                console.log('[IriunLauncher] Starting Iriun stream...');
                if (stream) stopMedia();
                await openSelectedCameraStream();
            } else {
                console.log('[IriunLauncher] Iriun already active, skipping re-stream.');
            }

            // Close modal and show controls
            elements.iriunModal.classList.add('hidden');
            elements.cameraSelectorGroup.classList.remove('hidden');
            
            // Update Toggle buttons
            elements.btnCamera.classList.add('hidden');
            elements.btnStopCamera.classList.remove('hidden');
            elements.btnStopUpload.classList.add('hidden');
            elements.btnUpload.classList.remove('hidden');
            
            console.log('[IriunLauncher] Auto-connection complete.');
        } catch (err) {
            console.error('[IriunLauncher] Error during final connection:', err);
            elements.iriunSearchText.innerHTML = '<span style="color:var(--danger)">Connection Failed.</span><br>Click Retry to try again.';
            isConnectingIriun = false;
        } finally {
            isConnectingIriun = false;
        }
    }, 800);
}

/* ======================================================== */
/*  EVENT LISTENERS                                          */
/* ======================================================== */

function setupEventListeners() {

    // ── Dedicated Iriun Launcher ────────────────────────────
    elements.btnIriunLauncher.addEventListener('click', async () => {
        try {
            // Reset state
            isManualMode = false;
            const btnManualSwitch = document.getElementById('btn-manual-switch');
            if (btnManualSwitch) btnManualSwitch.textContent = "Manual Select";

            // If already searching or connecting, don't restart
            if (iriunScanInterval || isConnectingIriun) return;

            elements.iriunModal.classList.remove('hidden');
            
            // CRITICAL: Browsers hide device names (labels) until getUserMedia is called once.
            // We call a quick temp stream to "unlock" the names so we can find "Iriun".
            elements.iriunSearchText.textContent = "Waking up camera system...";
            
            // Only request if labels are missing
            const devices = await navigator.mediaDevices.enumerateDevices();
            const hasLabels = devices.some(d => d.label !== "");
            
            if (!hasLabels) {
                const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                tempStream.getTracks().forEach(t => t.stop());
            }
            
            startIriunAutoScan();
        } catch (err) {
            console.error('[IriunLauncher] Permission denied or error:', err);
            elements.iriunSearchText.innerHTML = '<span style="color:var(--danger)">Permission Denied.</span><br>Please allow camera access in the address bar.';
        }
    });

    elements.closeIriunModal.addEventListener('click', () => {
        elements.iriunModal.classList.add('hidden');
        stopIriunAutoScan();
        isManualMode = false;
    });

    // Manual Switch logic within Iriun Modal
    const btnManualSwitch = document.getElementById('btn-manual-switch');
    if (btnManualSwitch) {
        btnManualSwitch.addEventListener('click', () => {
            isManualMode = !isManualMode;
            if (isManualMode) {
                stopIriunAutoScan();
                btnManualSwitch.textContent = "Auto Scan";
                elements.iriunSearchText.textContent = "Manual Selection Active. Click a camera below.";
                populateCameraList();
            } else {
                btnManualSwitch.textContent = "Manual Select";
                startIriunAutoScan();
            }
        });
    }

    // Delegate camera list clicks in modal for manual selection
    const debugUl = document.getElementById('detected-cameras-ul');
    if (debugUl) {
        debugUl.addEventListener('click', async (e) => {
            if (!isManualMode) return;
            const li = e.target.closest('li');
            if (li) {
                const labelText = li.innerText.replace('📱 ', '').replace('📷 ', '').trim();
                console.log('[IriunLauncher] Manual selection:', labelText);
                
                // Find device ID from the selector's options
                let targetId = null;
                for (let option of elements.cameraSelect.options) {
                    if (option.text.includes(labelText)) {
                        targetId = option.value;
                        break;
                    }
                }
                
                if (targetId) handleIriunFound(targetId);
            }
        });
    }

    elements.btnIriunRetry.addEventListener('click', async () => {
        isManualMode = false;
        const btnManualSwitch = document.getElementById('btn-manual-switch');
        if (btnManualSwitch) btnManualSwitch.textContent = "Manual Select";

        elements.iriunSearchText.textContent = "Scanning again...";
        const found = await populateCameraList();
        if (found) {
            handleIriunFound();
        } else {
            elements.iriunSearchText.textContent = "Still searching... check Wi-Fi and Desktop App.";
        }
    });

    elements.btnIriunTroubleshoot.addEventListener('click', () => {
        alert("FIREWALL FIX:\n1. Open Windows Firewall Settings.\n2. Click 'Allow an app through firewall'.\n3. Ensure 'Iriun Webcam' is checked for BOTH Private and Public networks.\n4. Restart Iriun Desktop App.");
    });

    // ── Camera Mode ────────────────────────────────────────
    elements.btnCamera.addEventListener('click', async () => {
        try {
            if (stream) stopMedia();

            // Step 1: Request a temporary permission-triggering stream to unlock device labels
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            tempStream.getTracks().forEach(t => t.stop()); // release immediately

            // Step 2: Populate dropdown now that labels are available
            await populateCameraList();
            elements.cameraSelectorGroup.classList.remove('hidden');

            // Step 3: Open stream using selected device
            await openSelectedCameraStream();

            // UI state
            elements.btnCamera.classList.add('hidden');
            elements.btnStopCamera.classList.remove('hidden');
            elements.btnStopUpload.classList.add('hidden');
            elements.btnUpload.classList.remove('hidden');

        } catch (err) {
            console.error('Error accessing camera:', err);
            alert('Could not access camera. Please check permissions and ensure Iriun Webcam is running.');
        }
    });

    // ── Stop Camera ────────────────────────────────────────
    elements.btnStopCamera.addEventListener('click', () => {
        stopAnalysis();
        stopMedia();
        elements.sourceVideo.style.display = 'none';
        elements.videoPlaceholder.style.display = 'flex';
        elements.btnStopCamera.classList.add('hidden');
        elements.btnCamera.classList.remove('hidden');
        elements.cameraSelectorGroup.classList.add('hidden');
        currentMode = null;
        updateUIState('idle');
    });

    // ── Live camera switching ──────────────────────────────
    elements.cameraSelect.addEventListener('change', async () => {
        if (currentMode !== 'camera') return;
        try {
            stopMedia();
            await openSelectedCameraStream();
            console.log('[CameraSelector] Switched to:', elements.cameraSelect.options[elements.cameraSelect.selectedIndex].text);
        } catch (err) {
            console.error('[CameraSelector] Failed to switch camera:', err);
            alert('Could not switch to the selected camera.');
        }
    });

    // ── Refresh camera list ────────────────────────────────
    elements.btnRefreshCameras.addEventListener('click', async () => {
        elements.btnRefreshCameras.textContent = '⏳';
        await populateCameraList();
        elements.btnRefreshCameras.textContent = '🔄';
    });

    // ── Upload Mode ────────────────────────────────────────
    elements.btnUpload.addEventListener('click', () => {
        elements.videoUpload.click();
    });

    elements.videoUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (stream) stopMedia();

        const url = URL.createObjectURL(file);
        elements.sourceVideo.srcObject = null;
        elements.sourceVideo.src = url;
        elements.sourceVideo.style.display = 'block';
        elements.videoPlaceholder.style.display = 'none';
        elements.sourceVideo.loop = true;
        currentMode = 'video';

        elements.btnUpload.classList.add('hidden');
        elements.btnStopUpload.classList.remove('hidden');
        elements.btnStopCamera.classList.add('hidden');
        elements.btnCamera.classList.remove('hidden');
        elements.cameraSelectorGroup.classList.add('hidden'); // hide selector in upload mode

        elements.sourceVideo.onloadeddata = () => {
            elements.sourceVideo.play();
            updateUIState('ready');
        };
    });

    // ── Stop Upload ────────────────────────────────────────
    elements.btnStopUpload.addEventListener('click', () => {
        stopAnalysis();
        stopMedia();
        elements.sourceVideo.style.display = 'none';
        elements.videoPlaceholder.style.display = 'flex';
        elements.btnStopUpload.classList.add('hidden');
        elements.btnUpload.classList.remove('hidden');
        elements.videoUpload.value = '';
        currentMode = null;
        updateUIState('idle');
    });

    // ── Start / Stop Analysis ──────────────────────────────
    elements.btnStart.addEventListener('click', () => {
        if (!elements.btnStart.disabled) {
            startAnalysis();
        }
    });

    elements.btnStop.addEventListener('click', () => {
        stopAnalysis();
    });

    // ── Depth heatmap toggle ───────────────────────────────
    const depthToggle = document.getElementById('toggle-depth');
    if (depthToggle) {
        depthToggle.addEventListener('change', (e) => {
            if (window.PerceptionEngine) {
                window.PerceptionEngine.toggleDepthMap(e.target.checked);
            }
        });
    }
}

/* ======================================================== */
/*  MEDIA HELPERS                                            */
/* ======================================================== */

function stopMedia() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    elements.sourceVideo.pause();
    elements.sourceVideo.src = '';
    elements.sourceVideo.srcObject = null;
}

/* ======================================================== */
/*  UI STATE                                                 */
/* ======================================================== */

function updateUIState(state) {
    if (state === 'idle') {
        elements.statusText.textContent = 'System Idle';
        elements.systemStatus.classList.remove('active');
        elements.btnStart.disabled = true;
        elements.btnStart.classList.add('disabled');
        elements.btnStop.classList.add('hidden');
        elements.btnStart.classList.remove('hidden');
        elements.videoContainer.classList.remove('active');
        elements.perfStats.style.display = 'none';
    } else if (state === 'ready') {
        elements.statusText.textContent = 'Ready for Analysis';
        elements.systemStatus.classList.remove('active');
        elements.btnStart.disabled = false;
        elements.btnStart.classList.remove('disabled');
        elements.btnStop.classList.add('hidden');
        elements.btnStart.classList.remove('hidden');
    } else if (state === 'running') {
        elements.statusText.textContent = 'Perception Engine Active';
        elements.systemStatus.classList.add('active');
        elements.btnStart.classList.add('hidden');
        elements.btnStop.classList.remove('hidden');
        elements.videoContainer.classList.add('active');
        elements.perfStats.style.display = 'block';
    }
}

function startAnalysis() {
    updateUIState('running');
    if (window.PerceptionEngine) {
        window.PerceptionEngine.start(elements.sourceVideo, elements.overlayCanvas);
    } else {
        console.error('Perception Engine not loaded.');
    }
}

function stopAnalysis() {
    updateUIState('ready');
    if (window.PerceptionEngine) {
        window.PerceptionEngine.stop();
    }
}

// Resize canvas when window resizes
window.addEventListener('resize', () => {
    if (window.PerceptionEngine && window.PerceptionEngine.isRunning) {
        window.PerceptionEngine.resizeCanvas();
    }
});

// Initialize app when DOM loads
document.addEventListener('DOMContentLoaded', init);
