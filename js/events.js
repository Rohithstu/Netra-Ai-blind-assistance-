/**
 * Netra — Event Intelligence Engine (Layer 3)
 *
 * Transforms raw perception + spatial data into meaningful situational events.
 * Tracks objects over time, validates events across multiple frames,
 * and outputs structured event reports for the Priority Decision layer.
 */
class EventEngine {
    constructor() {
        /* ---- object tracking memory ---- */
        this.trackedObjects = new Map();   // id → history[]
        this.nextTrackId    = 1;
        this.MATCH_DIST     = 80;          // max pixel distance to match across frames
        this.HISTORY_LEN    = 15;          // frames of history to keep per object

        /* ---- event buffers ---- */
        this.activeEvents   = [];          // current frame's validated events
        this.pendingEvents  = new Map();   // eventKey → { count, data }
        this.CONFIRM_FRAMES = 3;           // frames needed to confirm event
        this.EXPIRE_FRAMES  = 8;           // frames before pending event expires

        /* ---- scene memory (entry / exit) ---- */
        this.previousIds    = new Set();
        this.frameCounter   = 0;
    }

    /* ================================================================== */
    /*  PUBLIC API                                                         */
    /* ================================================================== */

    /**
     * Process one frame of spatial objects.
     * @param {Array}  spatialObjects – objects enriched by SpatialEngine
     * @param {number} frameW
     * @param {number} frameH
     * @returns {{ events: Array, trackedObjects: Map }}
     */
    process(spatialObjects, frameW, frameH, emotions = [], behaviors = []) {
        this.frameCounter++;

        // 1. Update tracking
        this._updateTracking(spatialObjects, frameW, frameH);

        // 2. Detect raw events
        const rawEvents = [];
        this._detectApproachEvents(rawEvents);
        this._detectObstacleAppearance(rawEvents, frameW);
        this._detectPathBlockEvents(rawEvents, spatialObjects, frameW);
        this._detectProximityChangeEvents(rawEvents);
        this._detectEmotionalEvents(rawEvents, emotions);
        this._detectSocialInteractionEvents(rawEvents, behaviors);
        this._detectCrowdEvents(rawEvents, emotions, spatialObjects);

        // 3. Validate events (multi-frame confirmation)
        this._validateEvents(rawEvents);

        // 4. Expire stale pending events
        this._expireStaleEvents();

        return {
            events         : [...this.activeEvents],
            trackedObjects : this.trackedObjects
        };
    }

    /* ================================================================== */
    /*  OBJECT TRACKING                                                    */
    /* ================================================================== */

    _updateTracking(objects, frameW, frameH) {
        const used = new Set();
        const currentIds = new Set();

        objects.forEach(obj => {
            let bestId   = null;
            let bestDist = Infinity;

            // Find closest tracked object of same class
            for (const [id, history] of this.trackedObjects) {
                if (used.has(id)) continue;
                const last = history[history.length - 1];
                if (last.class !== obj.class) continue;
                const d = Math.hypot(obj.centerX - last.centerX, obj.centerY - last.centerY);
                if (d < this.MATCH_DIST && d < bestDist) {
                    bestDist = d;
                    bestId   = id;
                }
            }

            if (bestId !== null) {
                // Update existing track
                const history = this.trackedObjects.get(bestId);
                history.push({ ...obj, frame: this.frameCounter });
                if (history.length > this.HISTORY_LEN) history.shift();
                used.add(bestId);
                obj._trackId = bestId;
                currentIds.add(bestId);
            } else {
                // New track
                const id = this.nextTrackId++;
                this.trackedObjects.set(id, [{ ...obj, frame: this.frameCounter }]);
                obj._trackId = id;
                currentIds.add(id);
            }
        });

        // Mark disappeared tracks (keep for a few frames for exit detection)
        for (const [id, history] of this.trackedObjects) {
            if (!currentIds.has(id)) {
                const age = this.frameCounter - history[history.length - 1].frame;
                if (age > 10) this.trackedObjects.delete(id);
            }
        }

        // Save current IDs for entry/exit
        this._currentIds = currentIds;
    }

    /* ================================================================== */
    /*  EVENT DETECTORS                                                    */
    /* ================================================================== */

    /** Approach events — object distance shrinking over recent history */
    _detectApproachEvents(events) {
        for (const [id, history] of this.trackedObjects) {
            if (history.length < 4) continue;
            const recent = history.slice(-4);
            if (!recent[0].distance || !recent[recent.length - 1].distance) continue;

            const distDelta = recent[0].distance - recent[recent.length - 1].distance;
            if (distDelta > 0.3) {
                const last = recent[recent.length - 1];
                events.push({
                    type     : 'approach',
                    label    : `${last.class} approaching`,
                    object   : last.class,
                    position : last.position,
                    distance : last.distance,
                    depthZone: last.depthZone,
                    movement : 'toward user',
                    severity : last.depthZone === 'near' ? 'high' : 'medium',
                    _key     : `approach_${id}`
                });
            }
        }
    }

    /** Obstacle appearance — object appears in center near zone recently */
    _detectObstacleAppearance(events, frameW) {
        for (const [id, history] of this.trackedObjects) {
            if (history.length < 2 || history.length > 6) continue;   // appeared recently
            const last = history[history.length - 1];
            const inCenter = last.centerX > frameW * 0.25 && last.centerX < frameW * 0.75;
            if (inCenter && (last.depthZone === 'near' || last.depthZone === 'medium')) {
                events.push({
                    type     : 'obstacle_appear',
                    label    : `${last.class} appeared ahead`,
                    object   : last.class,
                    position : last.position,
                    distance : last.distance,
                    depthZone: last.depthZone,
                    movement : last.movement || 'stationary',
                    severity : last.depthZone === 'near' ? 'high' : 'medium',
                    _key     : `obstacle_${id}`
                });
            }
        }
    }

    /** Path block — multiple objects in center near/medium zone */
    _detectPathBlockEvents(events, spatialObjects, frameW) {
        const centerObstacles = spatialObjects.filter(o => {
            const inCenter = o.centerX > frameW * 0.3 && o.centerX < frameW * 0.7;
            return inCenter && (o.depthZone === 'near' || o.depthZone === 'medium');
        });

        if (centerObstacles.length >= 2) {
            events.push({
                type     : 'path_blocked',
                label    : 'Walking path is blocked',
                object   : centerObstacles.map(o => o.class).join(', '),
                position : 'center',
                distance : Math.min(...centerObstacles.map(o => o.distance || 99)),
                depthZone: 'near',
                movement : 'stationary',
                severity : 'high',
                _key     : 'path_blocked'
            });
        } else if (centerObstacles.length === 1 && centerObstacles[0].risk === 'immediate') {
            events.push({
                type     : 'path_blocked',
                label    : `${centerObstacles[0].class} blocking path`,
                object   : centerObstacles[0].class,
                position : 'center',
                distance : centerObstacles[0].distance,
                depthZone: centerObstacles[0].depthZone,
                movement : centerObstacles[0].movement || 'stationary',
                severity : 'high',
                _key     : 'path_blocked_single'
            });
        }
    }

    /** Proximity change — sudden change in object distance */
    _detectProximityChangeEvents(events) {
        for (const [id, history] of this.trackedObjects) {
            if (history.length < 3) continue;
            const last = history[history.length - 1];
            const prev = history[history.length - 2];

            if (!last.distance || !prev.distance) continue;

            const change = Math.abs(last.distance - prev.distance);
            if (change > 0.5) { // 0.5m change in one frame is significant
                events.push({
                    type     : 'proximity_change',
                    label    : `${last.class} distance changed rapidly`,
                    object   : last.class,
                    position : last.position,
                    distance : last.distance,
                    depthZone: last.depthZone,
                    movement : last.distance < prev.distance ? 'moving closer' : 'moving away',
                    severity : last.depthZone === 'near' ? 'medium' : 'low',
                    _key     : `proximity_${id}`
                });
            }
        }
    }

    /** Emotional events — strong facial expressions detected by core backend */
    _detectEmotionalEvents(events, emotions) {
        emotions.forEach((e, idx) => {
            if (e.reportable && e.intensity !== 'low') {
                events.push({
                    type     : 'face_emotion',
                    label    : `${e.person} looks ${e.emotion}`,
                    object   : e.person,
                    position : e.direction,
                    distance : e.distance,
                    severity : e.intensity === 'high' ? 'medium' : 'low',
                    _key     : `emotion_${e.person}_${idx}`
                });
            }
        });
    }

    /** Social interaction events — waving, conversation, etc. */
    _detectSocialInteractionEvents(events, behaviors) {
        behaviors.forEach((b, idx) => {
            b.behaviors.forEach(behavior => {
                if (['waving', 'pointing', 'conversation', 'leaning_forward'].includes(behavior)) {
                    events.push({
                        type     : 'social_interaction',
                        label    : `Social cue: ${behavior.replace('_', ' ')}`,
                        object   : 'person',
                        position : b.movement || 'center',
                        distance : b.distance,
                        severity : 'medium',
                        _key     : `social_${behavior}_${idx}`
                    });
                }
            });
        });
    }

    /** Crowd detection — high density of people in frame */
    _detectCrowdEvents(events, emotions, spatialObjects) {
        const peopleCount = spatialObjects.filter(o => o.class === 'person').length;
        if (peopleCount >= 5) {
            events.push({
                type     : 'crowd_alert',
                label    : `Crowded area: ${peopleCount} people`,
                object   : 'crowd',
                position : 'ahead',
                severity : 'high',
                _key     : 'crowd_density'
            });
        }
    }

    /* ================================================================== */
    /*  EVENT VALIDATION (multi-frame confirmation)                        */
    /* ================================================================== */

    _validateEvents(rawEvents) {
        this.activeEvents = [];

        rawEvents.forEach(evt => {
            const key = evt._key;
            if (this.pendingEvents.has(key)) {
                const pending = this.pendingEvents.get(key);
                pending.count++;
                pending.lastFrame = this.frameCounter;
                pending.data = evt;

                if (pending.count >= this.CONFIRM_FRAMES) {
                    // Confirmed — promote to active
                    this.activeEvents.push(evt);
                }
            } else {
                // First sighting — add to pending
                this.pendingEvents.set(key, {
                    count     : 1,
                    firstFrame: this.frameCounter,
                    lastFrame : this.frameCounter,
                    data      : evt
                });
            }
        });

        // Also keep recently-confirmed events that are still being seen
        for (const [key, pending] of this.pendingEvents) {
            if (pending.count >= this.CONFIRM_FRAMES &&
                pending.lastFrame === this.frameCounter &&
                !this.activeEvents.find(e => e._key === key)) {
                this.activeEvents.push(pending.data);
            }
        }
    }

    _expireStaleEvents() {
        for (const [key, pending] of this.pendingEvents) {
            if (this.frameCounter - pending.lastFrame > this.EXPIRE_FRAMES) {
                this.pendingEvents.delete(key);
            }
        }
    }
}

// Expose globally
window.EventEngine = new EventEngine();
