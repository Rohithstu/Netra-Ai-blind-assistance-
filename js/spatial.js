/**
 * Netra — Spatial Intelligence Engine (Layer 2)
 * 
 * Transforms flat object detections into depth-aware spatial understanding.
 * Uses monocular depth heuristics (bounding-box size + vertical position)
 * because we only have a single laptop camera — no hardware depth sensor.
 */
class SpatialEngine {
    constructor() {
        // ----- reference sizes (fraction of frame height) for common objects ----
        // When a known object fills roughly this fraction of the frame it is
        // approximately 1 m away.  Larger → closer, smaller → farther.
        this.REF_HEIGHTS = {
            person  : 0.70,
            bicycle : 0.40,
            car     : 0.35,
            dog     : 0.25,
            cat     : 0.20,
            chair   : 0.30,
            couch   : 0.30,
            bed     : 0.25,
            tv      : 0.30,
            laptop  : 0.20,
            door    : 0.60,
            default : 0.30
        };

        // Depth-zone thresholds (estimated metres)
        this.NEAR_THRESHOLD   = 1.5;   // < 1.5 m  → near
        this.MEDIUM_THRESHOLD = 3.5;   // 1.5 – 3.5 m → medium

        // Movement / approaching tracking
        this.previousSpatial  = [];    // last-frame spatial data
        this.APPROACH_DELTA   = 0.15;  // distance-change to flag "approaching"

        // Safe-path analysis
        this.pathStatus = { left: 'clear', center: 'clear', right: 'clear' };
        this.obstacleZone = null;

        // Depth heat-map (generated per frame)
        this.depthMapData = null;
    }

    /* ------------------------------------------------------------------ */
    /*  PUBLIC API                                                         */
    /* ------------------------------------------------------------------ */

    /**
     * Analyse one frame of detections coming from Layer 1.
     * @param {Array}  detections  – processed detections from PerceptionEngine
     * @param {number} frameW      – current frame / canvas width
     * @param {number} frameH      – current frame / canvas height
     * @param {HTMLVideoElement} video – the source video for depth-map generation
     * @returns {Object} spatialResult
     */
    analyse(detections, frameW, frameH, video, emotions = [], behaviors = []) {
        const spatialObjects = detections.map(det => {
            // Prefer backend distance if available, otherwise heuristic
            const distance   = det.distance !== undefined ? det.distance : this._estimateDistance(det, frameH);
            const depthZone  = this._classifyDepthZone(distance);
            // Prefer backend risk if available
            const risk       = det.risk || this._assessRisk(det, depthZone, frameW);
            const movement   = det.movement || this._trackMovement(det, distance);

            // Attach personal metadata if it matches this detection (heuristic match by position/depth)
            const myEmotion = emotions.find(e => Math.abs(e.distance - distance) < 0.5 && e.direction === det.position);
            const myBehavior = behaviors.find(b => Math.abs(b.distance - distance) < 0.5 && b.movement === det.position);

            return { 
                ...det, 
                distance, 
                depthZone, 
                risk, 
                movement,
                emotion: myEmotion ? myEmotion.emotion : null,
                behaviors: myBehavior ? myBehavior.behaviors : []
            };
        });

        // Update path status
        this._analyseSafePath(spatialObjects, frameW);

        // Generate pseudo depth-map
        this._generateDepthMap(spatialObjects, frameW, frameH, video);

        // Store for next-frame comparison
        this.previousSpatial = spatialObjects;

        return {
            objects       : spatialObjects,
            pathStatus    : { ...this.pathStatus },
            obstacleZone  : this.obstacleZone,
            depthMapData  : this.depthMapData
        };
    }

    /* ------------------------------------------------------------------ */
    /*  DISTANCE ESTIMATION                                                */
    /* ------------------------------------------------------------------ */

    _estimateDistance(det, frameH) {
        const [, , , h] = det.bbox;
        const heightRatio = h / frameH;
        const ref = this.REF_HEIGHTS[det.class] || this.REF_HEIGHTS.default;

        // inverse-proportional: distance ≈ ref / heightRatio
        let distance = ref / (heightRatio + 0.001);

        // Vertical-position adjustment: objects near the bottom of the frame
        // are generally closer (ground-plane assumption).
        const bottomY = (det.bbox[1] + det.bbox[3]) / frameH;
        const verticalFactor = 1.0 - (bottomY * 0.3);  // up to 30 % adjustment
        distance *= verticalFactor;

        return Math.max(0.2, Math.round(distance * 10) / 10); // clamp & round
    }

    _classifyDepthZone(distance) {
        if (distance < this.NEAR_THRESHOLD)   return 'near';
        if (distance < this.MEDIUM_THRESHOLD) return 'medium';
        return 'far';
    }

    /* ------------------------------------------------------------------ */
    /*  COLLISION / RISK ASSESSMENT                                        */
    /* ------------------------------------------------------------------ */

    _assessRisk(det, depthZone, frameW) {
        const centerX = det.centerX;
        const inCenter = centerX > frameW * 0.25 && centerX < frameW * 0.75;

        if (depthZone === 'near' && inCenter)   return 'immediate';
        if (depthZone === 'near')               return 'potential';
        if (depthZone === 'medium' && inCenter) return 'potential';
        return 'safe';
    }

    /* ------------------------------------------------------------------ */
    /*  MOVEMENT / APPROACH TRACKING                                       */
    /* ------------------------------------------------------------------ */

    _trackMovement(det, currentDistance) {
        if (!det.isMoving) return 'stationary';

        // Find same object in previous frame
        for (const prev of this.previousSpatial) {
            if (prev.class === det.class && prev.position === det.position) {
                const delta = prev.distance - currentDistance;
                if (delta > this.APPROACH_DELTA)      return 'approaching';
                if (delta < -this.APPROACH_DELTA)     return 'moving away';
                // Lateral movement
                const lateralDelta = Math.abs(det.centerX - prev.centerX);
                if (lateralDelta > 20) return 'crossing';
            }
        }
        return 'moving';
    }

    /* ------------------------------------------------------------------ */
    /*  SAFE WALKING PATH                                                  */
    /* ------------------------------------------------------------------ */

    _analyseSafePath(objects, frameW) {
        // Reset
        this.pathStatus = { left: 'clear', center: 'clear', right: 'clear' };
        this.obstacleZone = null;

        let nearestCenterDist = Infinity;

        objects.forEach(obj => {
            const zone = obj.position; // left / center / right
            if (obj.depthZone === 'near' || obj.depthZone === 'medium') {
                if (obj.risk === 'immediate') {
                    this.pathStatus[zone] = 'blocked';
                } else if (obj.risk === 'potential' && this.pathStatus[zone] !== 'blocked') {
                    this.pathStatus[zone] = 'caution';
                }
            }
            // Track nearest center obstacle
            if (zone === 'center' && obj.distance < nearestCenterDist) {
                nearestCenterDist = obj.distance;
            }
        });

        if (nearestCenterDist < this.NEAR_THRESHOLD) {
            this.obstacleZone = 'Near obstacle detected ahead';
        } else if (nearestCenterDist < this.MEDIUM_THRESHOLD) {
            this.obstacleZone = 'Obstacle at medium range ahead';
        }
    }

    /* ------------------------------------------------------------------ */
    /*  DEPTH HEATMAP GENERATION (pseudo — painted from detection data)    */
    /* ------------------------------------------------------------------ */

    _generateDepthMap(objects, w, h, video) {
        // We create an offscreen canvas that higher-level code can composite
        if (!this._offscreen || this._offscreen.width !== w || this._offscreen.height !== h) {
            this._offscreen = document.createElement('canvas');
            this._offscreen.width  = w;
            this._offscreen.height = h;
        }
        const ctx = this._offscreen.getContext('2d');
        ctx.clearRect(0, 0, w, h);

        // Base gradient: ground-plane heuristic (bottom = near/warm, top = far/cool)
        const grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0,   'rgba(59,130,246,0.18)');  // blue-ish (far)
        grad.addColorStop(0.5, 'rgba(34,211,238,0.12)');  // cyan (medium)
        grad.addColorStop(1,   'rgba(249,115,22,0.20)');  // orange (near)
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);

        // Paint warm blobs at near-object locations
        objects.forEach(obj => {
            const [x, y, bw, bh] = obj.bbox;
            const cx = x + bw / 2;
            const cy = y + bh / 2;
            const radius = Math.max(bw, bh) * 0.6;

            let color;
            if (obj.depthZone === 'near')        color = 'rgba(239,68,68,0.30)';
            else if (obj.depthZone === 'medium') color = 'rgba(234,179,8,0.22)';
            else                                  color = 'rgba(59,130,246,0.15)';

            const rGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
            rGrad.addColorStop(0, color);
            rGrad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = rGrad;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.fill();
        });

        this.depthMapData = this._offscreen;
    }
}

// Expose globally
window.SpatialEngine = new SpatialEngine();
