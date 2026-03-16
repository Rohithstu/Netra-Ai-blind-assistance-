/**
 * Netra — Perception Engine (Layer 1 + Layer 2 + Layer 3 + Layer 4 integration)
 * 
 * Runs COCO-SSD object detection, feeds results to the Spatial Engine,
 * then to the Event Engine, and finally to the Priority Decision Engine.
 * Draws risk-aware overlays and updates all intelligence panels.
 */
class PerceptionEngine {
    constructor() {
        this.model = null;
        this.video = null;
        this.canvas = null;
        this.ctx = null;
        this.isRunning = false;
        this.animationId = null;

        // Performance tracking
        this.lastTime = 0;
        this.frameCount = 0;
        this.fps = 0;

        // UI Panels references
        this.voicePanel = document.getElementById('voice-panel-body');
        this.memoryPanel = document.getElementById('memory-panel-body');
        this.scenePanel = document.getElementById('scene-panel-body');
        this.navPanel = document.getElementById('nav-panel-body');
        this.adaptivePanel = document.getElementById('adaptive-panel-body');
        this.autonomousPanel = document.getElementById('autonomous-panel-body');

        // Initialize Intelligence Engines
        this.spatial = window.SpatialEngine;
        this.events = window.EventEngine;
        this.priority = window.PriorityEngine;
        this.memory = window.MemoryEngine;
        this.voice = window.VoiceEngine;
        this.scene = window.SceneEngine;
        this.navigation = window.NavigationEngine;
        this.adaptive = window.AdaptiveEngine;
        this.autonomous = window.AutonomousEngine;

        // UI Listeners for Layer 6
        this.initVoiceUI();

        // Motion tracking history
        this.previousDetections = [];
        this.MOTION_THRESHOLD = 15; // pixels

        // Depth heatmap toggle
        this.showDepthMap = false;

        // Backend Core Engine support
        this.engine = 'browser'; // 'browser' | 'core'
        this.socket = null;
        this.isProcessingCore = false;
        this.coreStatusEl = document.getElementById('core-status');

        this.init();
        this.initEngineSelector();
    }

    async init() {
        try {
            console.log("Loading COCO-SSD model...");
            this.model = await cocoSsd.load();
            console.log("Model loaded successfully.");
        } catch (error) {
            console.error("Failed to load model:", error);
            alert("Error loading perception model. Please check connection.");
        }
    }

    start(videoElement, canvasElement) {
        if (!this.model) {
            console.warn("Model not loaded yet. Waiting...");
            setTimeout(() => this.start(videoElement, canvasElement), 1000);
            return;
        }

        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = this.canvas.getContext('2d');
        this.isRunning = true;

        this.resizeCanvas();
        this.lastTime = performance.now();
        this.detectFrame();
    }

    stop() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    resizeCanvas() {
        if (this.video && this.canvas) {
            this.canvas.width = this.video.videoWidth || this.video.clientWidth;
            this.canvas.height = this.video.videoHeight || this.video.clientHeight;
        }
    }

    toggleDepthMap(value) {
        this.showDepthMap = value;
    }

    async detectFrame() {
        if (!this.isRunning) return;

        if (this.video.readyState === this.video.HAVE_ENOUGH_DATA) {
            this.resizeCanvas();

            // Calculate FPS
            const now = performance.now();
            this.frameCount++;
            if (now - this.lastTime >= 1000) {
                this.fps = this.frameCount;
                document.getElementById('perf-stats').innerText = `FPS: ${this.fps}`;
                this.frameCount = 0;
                this.lastTime = now;
            }

            try {
                let detections = [];
                
                if (this.engine === 'core' && this.socket && this.socket.readyState === WebSocket.OPEN) {
                    // Send frame to Python Backend
                    if (!this.isProcessingCore) {
                        this.sendFrameToCore();
                    }
                    // Detections will be updated via WebSocket onmessage
                    return; 
                } else {
                    // Use local COCO-SSD
                    const predictions = await this.model.detect(this.video);
                    detections = this.processDetections(predictions);
                    this.runIntelligencePipeline(detections);
                }
            } catch (error) {
                console.error("Detection error:", error);
            }
        }

        this.animationId = requestAnimationFrame(() => this.detectFrame());
    }

    /**
     * The main pipeline that feeds detections through all layers.
     */
    runIntelligencePipeline(processed, emotions = [], behaviors = []) {
        // --- Layer 2: Spatial analysis --------------------------------
        let spatialResult = null;
        if (window.SpatialEngine) {
            spatialResult = window.SpatialEngine.analyse(
                processed,
                this.canvas.width,
                this.canvas.height,
                this.video,
                emotions,
                behaviors
            );
        }

        // --- Layer 3: Event intelligence ------------------------------
        let eventResult = null;
        if (window.EventEngine && spatialResult) {
            eventResult = window.EventEngine.process(
                spatialResult.objects,
                this.canvas.width,
                this.canvas.height,
                emotions,
                behaviors
            );
        }

        // Layer 4: Priority & Notifications
        const priorityState = this.priority ? this.priority.process(eventResult ? eventResult.events : []) : { active: [], history: [] };

        // Layer 5: Memory & Person Recognition
        const memoryState = this.memory ? this.memory.process(spatialResult ? spatialResult.objects : processed, priorityState.active) : { recognized: [], journal: [] };

        // Layer 6: Voice Interaction (Automated alerts)
        if (this.voice && priorityState.active.length > 0) {
            this.voice.speakAlert(priorityState.active[0]);
        }

        // Layer 7: Scene Awareness
        const sceneState = this.scene ? this.scene.analyse(spatialResult ? spatialResult.objects : processed, spatialResult) : null;

        // Layer 8: Navigation Guidance
        const navState = this.navigation ? this.navigation.process(spatialResult, eventResult, priorityState, sceneState) : null;
        if (this.voice && navState) {
            this.voice.speakNavigation(navState);
        }

        // Layer 9: Adaptive Learning
        const adaptiveState = this.adaptive ? this.adaptive.learn(sceneState, navState, memoryState, spatialResult ? spatialResult.objects : processed) : null;

        // Layer 10: Autonomous Master Control
        const autonomousState = this.autonomous ? this.autonomous.coordinate({
            detections: processed,
            spatial: spatialResult,
            priority: priorityState,
            navigation: navState,
            scene: sceneState,
            memory: memoryState
        }) : null;

        // Update UI
        this.updateUI(spatialResult ? spatialResult.objects : processed, spatialResult, eventResult, priorityState, memoryState, sceneState, navState, adaptiveState, autonomousState, emotions, behaviors);
    }

    /* ------------------------------------------------------------------ */
    /*  ENGINE SELECTION & BACKEND COMMUNICATION                          */
    /* ------------------------------------------------------------------ */

    initEngineSelector() {
        const btnBrowser = document.getElementById('engine-browser');
        const btnCore = document.getElementById('engine-core');

        if (!btnBrowser || !btnCore) return;

        btnBrowser.onclick = () => {
            this.engine = 'browser';
            btnBrowser.classList.add('active');
            btnCore.classList.remove('active');
            if (this.coreStatusEl) this.coreStatusEl.classList.add('hidden');
            console.log("AI Engine switched to: Browser (COCO-SSD)");
        };

        btnCore.onclick = () => {
            this.engine = 'core';
            btnCore.classList.add('active');
            btnBrowser.classList.remove('active');
            if (this.coreStatusEl) this.coreStatusEl.classList.remove('hidden');
            this.connectToCore();
            console.log("AI Engine switched to: Netra Core (YOLOv8)");
        };
    }

    connectToCore() {
        if (this.socket && this.socket.readyState <= 1) return;

        console.log("Connecting to Netra Core Server (ws://localhost:8001/ws/vision)...");
        if (this.coreStatusEl) {
            this.coreStatusEl.className = 'core-status connecting';
        }

        this.socket = new WebSocket('ws://localhost:8001/ws/vision');

        this.socket.onopen = () => {
            console.log("✅ Connected to Netra Core Server.");
            if (this.coreStatusEl) this.coreStatusEl.className = 'core-status online';
        };

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.isProcessingCore = false;
            
            // Map the results to the intelligence pipeline
            if (data.detections) {
                // Enrich detections with centroid and position logic that Perception expects
                const processed = data.detections.map(det => {
                    const [x, y, w, h] = det.bbox;
                    const centerX = x + w / 2;
                    const centerY = y + h / 2;
                    let position = 'center';
                    if (centerX < this.canvas.width * 0.33) position = 'left';
                    else if (centerX > this.canvas.width * 0.66) position = 'right';
                    
                    return {
                        ...det,
                        score: det.score,
                        centerX,
                        centerY,
                        position,
                        isMoving: false // Simple stationary default
                    };
                });
                
                this.runIntelligencePipeline(processed, data.emotions || [], data.behaviors || []);
            }
            
            // Continue detection loop if running
            if (this.isRunning) {
                requestAnimationFrame(() => this.detectFrame());
            }
        };

        this.socket.onclose = () => {
            console.warn("❌ Disconnected from Netra Core Server.");
            if (this.coreStatusEl) this.coreStatusEl.className = 'core-status';
            if (this.engine === 'core') {
                setTimeout(() => this.connectToCore(), 3000);
            }
        };

        this.socket.onerror = (err) => {
            console.error("WebSocket Error:", err);
        };
    }

    sendFrameToCore() {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
        
        // Capture frame from video and convert to base64
        const offscreenCanvas = document.createElement('canvas');
        offscreenCanvas.width = 640;
        offscreenCanvas.height = 480;
        const octx = offscreenCanvas.getContext('2d');
        octx.drawImage(this.video, 0, 0, 640, 480);
        
        const base64Img = offscreenCanvas.toDataURL('image/jpeg', 0.7);
        
        this.isProcessingCore = true;
        this.socket.send(JSON.stringify({ // Changed json.dumps to JSON.stringify
            image: base64Img,
            timestamp: Date.now()
        }));
    }

    updateUI(detections, spatial, events, priority, memory, scene, nav, adaptive, autonomous, emotions = [], behaviors = []) {
        this.drawOverlays(detections, spatial, nav, emotions, behaviors);
        this.updateObjectsPanel(detections);
        this.updateSpatialPanel(spatial, behaviors);
        this.updateEventsPanel(events);
        this.updatePriorityPanel(priority);
        this.updateMemoryPanel(memory);
        this.updateScenePanel(scene);
        this.updateNavPanel(nav);
        this.updateAdaptivePanel(adaptive);
        this.updateAutonomousPanel(autonomous);
        // Voice UI is updated via events in VoiceEngine
    }

    /* ------------------------------------------------------------------ */
    /*  PROCESS DETECTIONS (Layer 1)                                       */
    /* ------------------------------------------------------------------ */

    processDetections(predictions) {
        const processedDetections = [];
        const canvasWidth = this.canvas.width;

        predictions.forEach(pred => {
            const [x, y, width, height] = pred.bbox;
            const centerX = x + width / 2;
            const centerY = y + height / 2;

            let position = 'center';
            if (centerX < canvasWidth * 0.33) position = 'left';
            else if (centerX > canvasWidth * 0.66) position = 'right';

            let isMoving = false;
            for (let prev of this.previousDetections) {
                if (prev.class === pred.class) {
                    const dist = Math.sqrt((centerX - prev.centerX) ** 2 + (centerY - prev.centerY) ** 2);
                    if (dist > this.MOTION_THRESHOLD) { isMoving = true; break; }
                }
            }

            processedDetections.push({ ...pred, centerX, centerY, position, isMoving });
        });

        this.previousDetections = processedDetections;
        return processedDetections;
    }

    /* ------------------------------------------------------------------ */
    /*  DRAW OVERLAYS — now risk-colour-coded + distance labels            */
    /* ------------------------------------------------------------------ */

    _riskColor(risk) {
        if (risk === 'immediate') return { box: '#EF4444', bg: 'rgba(239,68,68,0.85)', glow: '#EF4444' };
        if (risk === 'potential') return { box: '#EAB308', bg: 'rgba(234,179,8,0.85)', glow: '#EAB308' };
        return { box: '#10B981', bg: 'rgba(16,185,129,0.85)', glow: '#10B981' };
    }

    drawOverlays(objects, spatialResult, navState, emotions = [], behaviors = []) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Optional depth heatmap underlay
        if (this.showDepthMap && spatialResult && spatialResult.depthMapData) {
            this.ctx.drawImage(spatialResult.depthMapData, 0, 0, this.canvas.width, this.canvas.height);
        }

        // --- Layer 8: Navigation Path Overlay ---
        if (navState && navState.direction !== 'STOP') {
            this._drawNavPath(navState);
        }

        objects.forEach(obj => {
            const [x, y, w, h] = obj.bbox;
            const colors = this._riskColor(obj.risk || 'safe');

            // Bounding box
            this.ctx.lineWidth = 3;
            this.ctx.strokeStyle = colors.box;
            if (obj.risk === 'immediate') {
                this.ctx.shadowColor = colors.glow;
                this.ctx.shadowBlur = 14;
            } else if (obj.risk === 'potential') {
                this.ctx.shadowColor = colors.glow;
                this.ctx.shadowBlur = 8;
            } else {
                this.ctx.shadowBlur = 0;
            }
            this.ctx.strokeRect(x, y, w, h);
            this.ctx.shadowBlur = 0;

            // Label background
            let label = obj.class;
            const distLabel = obj.distance ? `${obj.distance}m` : '';
            const riskIcon = obj.risk === 'immediate' ? '⚠ ' : obj.risk === 'potential' ? '! ' : '';

            if (obj.class === 'person' && obj.identity) {
                label = `${obj.identity.name} (${Math.round(obj.score * 100)}%)`;
                this.ctx.fillStyle = obj.identity.isKnown ? '#6366F1' : '#2563EB'; // Use ctx for fillStyle
            } else {
                this.ctx.fillStyle = colors.bg; // Use colors.bg for non-person objects
            }

            const text = `${riskIcon}${label} ${distLabel}`;
            const textW = this.ctx.measureText(text).width + 16;

            this.ctx.beginPath();
            this._roundRect(x, y - 28, textW, 26, 4);
            this.ctx.fill();

            // Label text
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.font = 'bold 14px -apple-system, sans-serif';
            this.ctx.fillText(text, x + 6, y - 9);

            // Emotion Label (Top-left of box)
            if (obj.emotion) {
                const emotionText = `🎭 ${obj.emotion.toUpperCase()}`;
                const etw = this.ctx.measureText(emotionText).width + 12;
                this.ctx.fillStyle = 'rgba(124, 58, 237, 0.9)'; // Purple for emotion
                this._roundRect(x, y + 2, etw, 20, 3);
                this.ctx.fill();
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.font = 'bold 12px sans-serif';
                this.ctx.fillText(emotionText, x + 6, y + 16);
            }

            // Behavior icons (Top-right of box)
            if (obj.behaviors && obj.behaviors.length > 0) {
                const behaviorIcons = { waving: '👋', pointing: '👉', conversation: '💬', leaning_forward: '🤸' };
                const icons = obj.behaviors.map(b => behaviorIcons[b] || '👤').join(' ');
                const itw = this.ctx.measureText(icons).width + 12;
                this.ctx.fillStyle = 'rgba(59, 130, 246, 0.9)'; // Blue for behavior
                this._roundRect(x + w - itw, y + 2, itw, 20, 3);
                this.ctx.fill();
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.fillText(icons, x + w - itw + 6, y + 16);
            }

            // Depth zone badge (bottom-right of box)
            if (obj.depthZone) {
                const zoneText = obj.depthZone.toUpperCase();
                const ztw = this.ctx.measureText(zoneText).width + 12;
                this.ctx.fillStyle = 'rgba(0,0,0,0.55)';
                this._roundRect(x + w - ztw - 4, y + h - 22, ztw, 20, 3);
                this.ctx.fill();
                this.ctx.fillStyle = '#FFFFFF';
                this.ctx.font = '12px monospace';
                this.ctx.fillText(zoneText, x + w - ztw + 2, y + h - 7);
            }
        });
    }

    _roundRect(x, y, w, h, r) {
        this.ctx.moveTo(x + r, y);
        this.ctx.lineTo(x + w - r, y);
        this.ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        this.ctx.lineTo(x + w, y + h - r);
        this.ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        this.ctx.lineTo(x + r, y + h);
        this.ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        this.ctx.lineTo(x, y + r);
        this.ctx.quadraticCurveTo(x, y, x + r, y);
    }

    /* ------------------------------------------------------------------ */
    /*  SIDE PANEL — Detection list (enhanced with spatial data)           */
    /* ------------------------------------------------------------------ */

    updateObjectsPanel(detections) {
        const listEl = document.getElementById('detection-list');
        const countEl = document.getElementById('object-count');
        countEl.innerText = detections.length;

        if (detections.length === 0) {
            listEl.innerHTML = '<div class="empty-state">No objects detected</div>';
            return;
        }

        // Sort: immediate risk first, then potential, then safe; within each by distance ascending
        const sorted = [...detections].sort((a, b) => {
            const riskOrder = { immediate: 0, potential: 1, safe: 2, undefined: 3 };
            const ra = riskOrder[a.risk] ?? 3, rb = riskOrder[b.risk] ?? 3;
            if (ra !== rb) return ra - rb;
            return (a.distance || 99) - (b.distance || 99);
        });

        listEl.innerHTML = sorted.map(det => {
            const conf = Math.round(det.score * 100);
            const riskClass = det.risk === 'immediate' ? 'risk-immediate'
                : det.risk === 'potential' ? 'risk-potential' : 'risk-safe';

            return `
                <div class="detection-item ${riskClass}">
                    <div class="detection-header">
                        <span class="detection-name">${det.class}</span>
                        <span class="detection-conf">${conf}%</span>
                    </div>
                    <div class="detection-meta">
                        <span class="meta-tag pos-tag">${det.position}</span>
                        <span class="meta-tag motion-tag">${det.movement || (det.isMoving ? 'moving' : 'stationary')}</span>
                        ${det.distance ? `<span class="meta-tag dist-tag">${det.distance}m</span>` : ''}
                        ${det.depthZone ? `<span class="meta-tag zone-tag zone-${det.depthZone}">${det.depthZone}</span>` : ''}
                        ${det.emotion ? `<span class="meta-tag emotion-tag" style="background:#7C3AED">${det.emotion}</span>` : ''}
                        ${det.behaviors ? det.behaviors.map(b => `<span class="meta-tag behavior-tag" style="background:#2563EB">${b}</span>`).join('') : ''}
                    </div>
                </div>`;
        }).join('');
    }

    /* ------------------------------------------------------------------ */
    /*  SPATIAL PANEL — Path status + obstacle warnings                    */
    /* ------------------------------------------------------------------ */

    updateSpatialPanel(spatialResult) {
        const panel = document.getElementById('spatial-panel-body');
        if (!panel || !spatialResult) return;

        const ps = spatialResult.pathStatus;
        const statusIcon = s => s === 'blocked' ? '🔴' : s === 'caution' ? '🟡' : '🟢';
        const statusLabel = s => s === 'blocked' ? 'Blocked' : s === 'caution' ? 'Caution' : 'Clear';

        let html = `
            <div class="spatial-section">
                <h4>Navigation Corridors</h4>
                <div class="path-grid">
                    <div class="path-cell ${ps.left}">
                        <span class="path-icon">${statusIcon(ps.left)}</span>
                        <span class="path-label">Left</span>
                        <span class="path-status">${statusLabel(ps.left)}</span>
                    </div>
                    <div class="path-cell ${ps.center}">
                        <span class="path-icon">${statusIcon(ps.center)}</span>
                        <span class="path-label">Center</span>
                        <span class="path-status">${statusLabel(ps.center)}</span>
                    </div>
                    <div class="path-cell ${ps.right}">
                        <span class="path-icon">${statusIcon(ps.right)}</span>
                        <span class="path-label">Right</span>
                        <span class="path-status">${statusLabel(ps.right)}</span>
                    </div>
                </div>
            </div>`;

        if (spatialResult.obstacleZone) {
            html += `
            <div class="spatial-section obstacle-alert">
                <span class="alert-icon">⚠️</span>
                <span>${spatialResult.obstacleZone}</span>
            </div>`;
        }

        // Approaching objects
        const approaching = spatialResult.objects.filter(o => o.movement === 'approaching');
        if (approaching.length) {
            html += `<div class="spatial-section"><h4>Approaching</h4>`;
            approaching.forEach(o => {
                html += `<div class="approach-item">🚶 ${o.class} — ${o.distance}m — ${o.position}</div>`;
            });
            html += `</div>`;
        }

        panel.innerHTML = html;
    }

    /* ------------------------------------------------------------------ */
    /*  EVENTS PANEL (Layer 3) — Detected Situations                       */
    /* ------------------------------------------------------------------ */

    updateEventsPanel(eventResult) {
        const panel = document.getElementById('events-panel-body');
        if (!panel) return;

        if (!eventResult || eventResult.events.length === 0) {
            panel.innerHTML = '<div class="empty-state">No active situations</div>';
            return;
        }

        const severityOrder = { high: 0, medium: 1, low: 2 };
        const sorted = [...eventResult.events].sort((a, b) =>
            (severityOrder[a.severity] ?? 2) - (severityOrder[b.severity] ?? 2)
        );

        panel.innerHTML = sorted.map(evt => {
            const icon = this._eventIcon(evt.type);
            const sevClass = `severity-${evt.severity}`;
            return `
                <div class="event-item ${sevClass}">
                    <div class="event-icon">${icon}</div>
                    <div class="event-body">
                        <div class="event-label">${evt.label}</div>
                        <div class="event-meta">
                            <span class="meta-tag pos-tag">${evt.position}</span>
                            ${evt.distance ? `<span class="meta-tag dist-tag">${evt.distance}m</span>` : ''}
                            ${evt.depthZone ? `<span class="meta-tag zone-tag zone-${evt.depthZone}">${evt.depthZone}</span>` : ''}
                        </div>
                    </div>
                </div>`;
        }).join('');
    }

    /* ------------------------------------------------------------------ */
    /*  PRIORITY PANEL (Layer 4) — Priority Decision                       */
    /* ------------------------------------------------------------------ */

    updatePriorityPanel(priorityResult) {
        const panel = document.getElementById('priority-panel-body');
        if (!panel) return;

        if (!priorityResult || (priorityResult.active.length === 0 && priorityResult.history.length === 0)) {
            panel.innerHTML = '<div class="empty-state">System evaluating priority...</div>';
            return;
        }

        let html = '';
        if (priorityResult.active.length > 0) {
            html += priorityResult.active.map(alert => this._renderPriorityAlert(alert)).join('');
        } else {
            html += '<div class="empty-state">No immediate priority alerts</div>';
        }

        if (priorityResult.history.length > 0) {
            html += '<div style="margin: 1rem 0 .5rem; font-size: .7rem; text-transform: uppercase; color: var(--text-sub); border-bottom: 1px solid var(--border)">Recent History</div>';
            const activeIds = new Set(priorityResult.active.map(a => a.id));
            const historyToShow = priorityResult.history
                .filter(a => !activeIds.has(a.id))
                .slice(0, 3);
            html += historyToShow.map(alert => this._renderPriorityAlert(alert, true)).join('');
        }
        panel.innerHTML = html;
    }

    _renderPriorityAlert(alert, isHistory = false) {
        const icon = this._eventIcon(alert.type);
        const classification = alert.classification || 'Low';

        return `
            <div class="priority-alert ${classification} ${isHistory ? 'history-fade' : ''}">
                <div class="priority-header">
                    <span class="priority-label">${icon} ${alert.label}</span>
                    <span class="priority-badge">${classification}</span>
                </div>
                <div class="priority-meta">
                    <span class="score-tag">PRIORITY ${alert.priorityScore}/10</span>
                    <span class="meta-tag pos-tag">${alert.position}</span>
                    <span class="meta-tag dist-tag">${alert.distance}m</span>
                </div>
            </div>`;
    }

    /* ======== MEMORY PANEL (Layer 5) ======== */
    updateMemoryPanel(state) {
        if (!this.memoryPanel || !state) return;

        let html = '';

        // Section: People Recognition
        const peopleList = [];
        if (state.people) {
            if (state.people.recognized) peopleList.push(...state.people.recognized);
            if (state.people.unknown) peopleList.push(...state.people.unknown);
        }

        const topPeople = peopleList.slice(0, 2);
        if (topPeople.length > 0) {
            html += `<div style="margin-bottom: 0.75rem; display: flex; flex-direction: column; gap: 0.5rem;">`;
            topPeople.forEach(p => html += this._renderIdentityCard(p));
            html += `</div><div style="height: 1px; background: var(--border); margin: 0.5rem 0"></div>`;
        }

        // Section: Recent Memory Log
        if (state.recentLog && state.recentLog.length > 0) {
            html += `<div class="memory-log" style="display: flex; flex-direction: column; gap: 0.4rem;">`;
            state.recentLog.slice(0, 5).forEach(log => html += this._renderLogEntry(log));
            html += `</div>`;
        }

        if (html === '') {
            html = '<div class="empty-state">Knowledge base expanding...</div>';
        }

        this.memoryPanel.innerHTML = html;
    }

    _renderIdentityCard(person) {
        const icon = person.isKnown ? '👤' : '❓';
        const status = person.isKnown ? 'Recognized' : 'Unknown';
        const typeClass = person.isKnown ? 'known' : 'unknown';
        const name = person.name || 'Unknown Person';

        return `
            <div class="memory-identity ${typeClass}">
                <div class="identity-avatar">${icon}</div>
                <div class="identity-info">
                    <span class="identity-name">${name}</span>
                    <span class="identity-meta">${status} — ${person.encounterCount || 1} encounters</span>
                </div>
            </div>`;
    }

    _renderLogEntry(log) {
        return `
            <div class="memory-log-entry priority-${log.classification}">
                <span class="log-time">${log.timestamp}</span>
                <span class="log-content">
                    <b>${log.type.replace('_', ' ')}</b>: ${log.label} (${log.position})
                </span>
            </div>`;
    }

    _eventIcon(type) {
        const icons = {
            approach: '🚶',
            obstacle_appear: '🚧',
            path_blocked: '⛔',
            entry: '👤',
            exit: '🚪',
            proximity_change: '⚡'
        };
        return icons[type] || '📌';
    }
    /* ======== SCENE PANEL (Layer 7) ======== */
    updateScenePanel(state) {
        if (!this.scenePanel || !state) return;

        const structureHtml = state.structure
            .map(s => `<span class="struct-tag">${s}</span>`)
            .join('');

        this.scenePanel.innerHTML = `
            <div class="scene-main">
                <span class="scene-cat">${state.category}</span>
                <span class="scene-desc">${state.description}</span>
                <div class="scene-structure">
                    ${structureHtml}
                </div>
            </div>`;
    }

    /* ======== NAVIGATION PANEL (Layer 8) ======== */
    updateNavPanel(state) {
        if (!this.navPanel || !state) return;

        const iconMap = {
            'Continue Forward': '⬆️',
            'Move Slightly Right': '↗️',
            'Move Slightly Left': '↖️',
            'STOP': '🛑',
            'Stop and Scan': '⏹️'
        };

        const icon = iconMap[state.direction] || '🧭';

        this.navPanel.innerHTML = `
            <div class="nav-card ${state.status}">
                <div class="nav-dir">
                    <span class="nav-dir-icon">${icon}</span>
                    <span>${state.direction}</span>
                </div>
                <div class="nav-reason">${state.reason}</div>
                <div class="nav-meta-row">
                    <span class="meta-tag dist-tag">Conf: ${Math.round(state.confidence * 100)}%</span>
                </div>
            </div>`;
    }

    _drawNavPath(nav) {
        const w = this.canvas.width;
        const h = this.canvas.height;
        this.ctx.save();

        this.ctx.strokeStyle = 'rgba(6, 182, 212, 0.6)';
        this.ctx.lineWidth = 40;
        this.ctx.lineCap = 'round';
        this.ctx.shadowBlur = 15;
        this.ctx.shadowColor = 'rgba(6, 182, 212, 0.8)';

        const centerX = w / 2;
        const bottomY = h;
        const targetX = nav.direction.includes('Left') ? w * 0.2 :
            nav.direction.includes('Right') ? w * 0.8 : w / 2;
        const targetY = h * 0.3;

        this.ctx.beginPath();
        this.ctx.moveTo(centerX, bottomY);
        this.ctx.bezierCurveTo(centerX, h * 0.7, targetX, h * 0.7, targetX, targetY);
        this.ctx.stroke();

        this.ctx.restore();
    }

    /* ======== ADAPTIVE PANEL (Layer 9) ======== */
    updateAdaptivePanel(state) {
        if (!this.adaptivePanel || !state) return;

        this.adaptivePanel.innerHTML = `
            <div class="adaptive-insight">
                <div class="insight-label">Learned Environment</div>
                <div class="insight-value">${state.learnedScene}</div>
            </div>
            
            <div class="insight-grid">
                <div class="adaptive-insight insight-mini">
                    <div class="insight-label">Common Path</div>
                    <div class="insight-value">${state.commonPath}</div>
                </div>
                <div class="adaptive-insight insight-mini">
                    <div class="insight-label">Frequent Person</div>
                    <div class="insight-value">${state.frequentPerson}</div>
                </div>
            </div>

            <div class="adaptive-insight">
                <div class="insight-label">Persistent Obstacles</div>
                <div class="insight-value">${state.topObstacles.length > 0 ? state.topObstacles.map(o => `<span>#${o}</span>`).join(' ') : 'Learning...'}</div>
            </div>
            
            <div style="font-size:0.7rem; color:var(--text-sub); text-align:right">
                Knowledge Base: ${state.totalScenes} environments tracked
            </div>`;
    }

    /* ======== AUTONOMOUS PANEL (Layer 10) ======== */
    updateAutonomousPanel(state) {
        if (!this.autonomousPanel || !state) return;

        const healthColor = h => h === 'stable' || h === 'online' ? '#10B981' : '#F59E0B';
        const decision = state.decision;

        this.autonomousPanel.innerHTML = `
            <div class="master-status-row">
                <div class="status-indicator active"></div>
                <div class="status-text">MISSION: ${state.status.toUpperCase()}</div>
                <div class="conf-badge">CONF: ${Math.round(state.confidence * 100)}%</div>
            </div>

            <div class="neural-decision">
                <div class="decision-header">BEHAVIORAL MODEL DECISION</div>
                <div class="decision-action">${decision.action}</div>
                <div class="decision-meta">${decision.reason}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${decision.score * 100}%"></div>
                </div>
            </div>

            <div class="health-grid">
                <div class="health-item">
                    <span class="health-dot" style="background:${healthColor(state.health.vision)}"></span>
                    <span>VISION</span>
                </div>
                <div class="health-item">
                    <span class="health-dot" style="background:${healthColor(state.health.spatial)}"></span>
                    <span>SPATIAL</span>
                </div>
                <div class="health-item">
                    <span class="health-dot" style="background:${healthColor(state.health.memory)}"></span>
                    <span>MEMORY</span>
                </div>
                <div class="health-item">
                    <span class="health-dot" style="background:${healthColor(state.health.logic)}"></span>
                    <span>LOGIC</span>
                </div>
            </div>`;
    }

    /* ======== VOICE INTERACTION (Layer 6) ======== */
    initVoiceUI() {
        const micBtn = document.getElementById('mic-btn');
        if (!micBtn || !this.voice) return;

        micBtn.onclick = () => {
            if (this.voice.isListening) return;
            this.voice.startListening();
            micBtn.classList.add('active');
        };

        // Hook into VoiceEngine events to update Panel
        this.voice.onUIUpdate = (data) => {
            if (data.alert) {
                const el = document.getElementById('voice-last-alert');
                if (el) el.innerText = data.alert;
            }
            if (data.command) {
                const el = document.getElementById('voice-command');
                if (el) el.innerText = data.command;
            }
            if (data.response) {
                const el = document.getElementById('voice-response');
                if (el) el.innerText = data.response;
            }

            if (!this.voice.isListening) micBtn.classList.remove('active');
        };
    }
}

// Instantiate and expose globally
window.PerceptionEngine = new PerceptionEngine();
