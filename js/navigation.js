/**
 * Netra — Navigation Intelligence Engine (Layer 8)
 * 
 * Determines safe movement directions by scoring walking corridors
 * and avoiding obstacles detected in previous layers.
 */
class NavigationEngine {
    constructor() {
        this.currentGuidance = {
            direction: 'Analyzing...',
            reason: 'Determining safe path...',
            status: 'neutral', // neutral, safe, caution, danger
            confidence: 0
        };

        this.lastInstruction = '';
        this.frameCounter = 0;
        this.updateInterval = 10; // Frames between recalculations for stability
    }

    /**
     * Process all inputs to generate navigation guidance
     */
    process(spatialData, eventData, priorityData, sceneData) {
        this.frameCounter++;
        if (this.frameCounter % this.updateInterval !== 0 && this.currentGuidance.direction !== 'Analyzing...') {
            return this.currentGuidance;
        }

        // 1. Safety Filter (Highest Priority)
        const stopAlert = this._checkImmediateSafety(priorityData);
        if (stopAlert) {
            this.currentGuidance = {
                direction: 'STOP',
                reason: stopAlert.label,
                status: 'danger',
                confidence: 1.0
            };
            return this.currentGuidance;
        }

        // 2. Corridor Analysis
        if (!spatialData || !spatialData.pathStatus) {
            return this.currentGuidance;
        }

        const ps = spatialData.pathStatus;
        const guidance = this._decideDirection(ps, spatialData.objects);

        this.currentGuidance = {
            ...guidance,
            confidence: 0.8 // Base confidence
        };

        return this.currentGuidance;
    }

    /* ================================================================== */
    /*  PRIVATE METHODS                                                   */
    /* ================================================================== */

    _checkImmediateSafety(priority) {
        if (!priority || !priority.active) return null;
        // Immediate stop for high priority hazards in center
        return priority.active.find(a => 
            a.classification === 'High' && 
            (a.position === 'center' || a.distance < 1.0)
        );
    }

    _decideDirection(pathStatus, objects) {
        // Preference: Center > Right > Left (arbitrary heuristic)
        
        if (pathStatus.center === 'clear') {
            return {
                direction: 'Continue Forward',
                reason: 'Center path is clear',
                status: 'safe'
            };
        }

        if (pathStatus.right === 'clear') {
            return {
                direction: 'Move Slightly Right',
                reason: 'Obstacle ahead, right side clear',
                status: 'caution'
            };
        }

        if (pathStatus.left === 'clear') {
            return {
                direction: 'Move Slightly Left',
                reason: 'Obstacle ahead, left side clear',
                status: 'caution'
            };
        }

        // If nothing is clear, find the least blocked or stop
        const obstacleNearby = objects.find(o => o.distance < 2.0);
        return {
            direction: 'Stop and Scan',
            reason: obstacleNearby ? `Obstacle ${obstacleNearby.class} too close` : 'Path fully obstructed',
            status: 'danger'
        };
    }
}

// Global instance
window.NavigationEngine = new NavigationEngine();
