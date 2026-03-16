/**
 * Netra — Priority Decision Engine (Layer 4)
 * 
 * Evaluates events from Layer 3 to determine their urgency.
 * Manages an alert queue, filters repetitive information, and enforces 
 * cooldowns to ensure the system communicates clearly and efficiently.
 */
class PriorityEngine {
    constructor() {
        // State management
        this.activeAlerts = [];       // Higher-priority events that should be active
        this.alertHistory = [];       // Log of previous alerts
        
        // Configuration
        this.ALERT_COOLDOWN = 5000;   // 5 seconds cooldown for repetitive alerts
        this.MAX_HISTORY = 20;

        // Tracks last time an alert was "fired" for a specific event key
        this.lastFiredTime = new Map(); // eventKey -> timestamp
    }

    /**
     * Process list of situational events from Layer 3
     * @param {Array} events - List of events detected in the current frame
     * @returns {Object} - Prioritized alerts for the UI
     */
    process(events) {
        const currentTime = Date.now();
        const prioritized = [];

        events.forEach(event => {
            // 1. Calculate Priority Score (0 - 10)
            const score = this._calculatePriorityScore(event);
            
            // 2. Classify (High, Medium, Low)
            const classification = this._classifyAlert(score, event);

            // 3. Filtering & Cooldown Logic
            if (this._shouldFireAlert(event, classification, currentTime)) {
                const alert = {
                    ...event,
                    priorityScore: score,
                    classification: classification,
                    timestamp: currentTime,
                    id: `alert_${currentTime}_${Math.random().toString(36).substr(2, 5)}`
                };

                prioritized.push(alert);
                this.lastFiredTime.set(event._key, currentTime);
                
                this._addToHistory(alert);
            }
        });

        // Current active "brain focus" (sorted by priority)
        this.activeAlerts = prioritized.sort((a, b) => b.priorityScore - a.priorityScore);

        return {
            active: this.activeAlerts,
            history: this.alertHistory
        };
    }

    /* ================================================================== */
    /*  PRIVATE METHODS                                                   */
    /* ================================================================== */

    /**
     * Official Layer 4 Scoring Logic:
     * - Safety Risk (Vehicles/Path Blocked)
     * - Proximity (Near/Medium/Far)
     * - Motion (Toward User)
     * - Direction (Center is highest)
     */
    _calculatePriorityScore(event) {
        let score = 4; // Base score

        // Hazard Risk Weighting
        const highRiskObjects = ['car', 'bus', 'truck', 'motorcycle', 'bicycle'];
        if (highRiskObjects.includes(event.class)) score += 3;
        if (event.type === 'path_blocked') score += 4;
        if (event.type === 'obstacle_appear' && event.depthZone === 'near') score += 3;

        // Proximity Factor
        if (event.depthZone === 'near') score += 3;
        else if (event.depthZone === 'medium') score += 1;
        else if (event.depthZone === 'far') score -= 2;

        // Direction/Position Factor (Relative to User)
        if (event.position === 'center') score += 2;

        // Motion Factor
        if (event.type === 'approach' || (event.movement === 'approaching')) score += 2;
        if (event.movement === 'moving away') score -= 1;

        // Clamp 0-10
        return Math.max(0, Math.min(10, score));
    }

    _classifyAlert(score, event) {
        if (score >= 8 || event.severity === 'high') return 'High';
        if (score >= 5) return 'Medium';
        return 'Low';
    }

    _shouldFireAlert(event, classification, currentTime) {
        // High priority events (Danger/Collision) trigger INSTANT alerts
        if (classification === 'High') return true;

        // Low priority events (Informational) are mostly filtered unless significant
        if (classification === 'Low') {
            const lastTime = this.lastFiredTime.get(event._key);
            if (!lastTime) return event.severity === 'medium' || event.severity === 'high';
            return false; // Suppress repetitive low-priority noise
        }

        // Medium priority (Awareness needed)
        const lastTime = this.lastFiredTime.get(event._key);
        if (!lastTime) return true;

        return (currentTime - lastTime) > this.ALERT_COOLDOWN;
    }

    _addToHistory(alert) {
        this.alertHistory.unshift(alert);
        if (this.alertHistory.length > this.MAX_HISTORY) {
            this.alertHistory.pop();
        }
    }
}

// Expose globally
window.PriorityEngine = new PriorityEngine();
