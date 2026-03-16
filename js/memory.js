/**
 * Netra — Memory Intelligence Engine (Layer 5)
 * 
 * Manages person recognition database, encounter history, and event journaling.
 * Allows the system to identify known individuals and recall significant situations.
 * Uses localStorage for persistence across sessions.
 */
class MemoryEngine {
    constructor() {
        // Load database from storage or initialize empty
        this.KNOWN_PEOPLE_KEY = 'netra_known_people';
        this.EVENT_LOG_KEY = 'netra_event_log';
        
        this.people = this._loadFromStorage(this.KNOWN_PEOPLE_KEY, {
            'p1': { id: 'p1', name: 'Rohit', encounterCount: 0, lastSeen: null, firstSeen: Date.now() },
            'p2': { id: 'p2', name: 'Ravi', encounterCount: 0, lastSeen: null, firstSeen: Date.now() }
        });
        
        this.eventLog = this._loadFromStorage(this.EVENT_LOG_KEY, []);
        
        // Active tracking session (track unique identity across frames)
        this.activeIdentities = new Map(); // trackId -> person object
        
        this.MAX_EVENT_LOG = 30;
    }

    /**
     * Process detected objects and situational events for memory storage/retrieval
     */
    process(detections, events) {
        const currentTime = Date.now();
        const recognized = [];
        const unknown = [];

        // 1. Person Recognition and Identity Matching
        detections.forEach(obj => {
            if (obj.class === 'person') {
                const identity = this._identifyPerson(obj, currentTime);
                obj.identity = identity; // Attach identity to object
                
                if (identity.isKnown) {
                    recognized.push(identity);
                } else {
                    unknown.push(identity);
                }
            }
        });

        // 2. Event Journaling (Store high priority events)
        if (events && events.length > 0) {
            events.forEach(event => {
                // Only store High priority or significant Medium priority events
                if (event.classification === 'High' || 
                   (event.classification === 'Medium' && !this._eventRecentlyLogged(event))) {
                    this._logEvent(event);
                }
            });
        }

        return {
            people: { recognized, unknown },
            recentLog: this.eventLog.slice(0, 10),
            database: this.people
        };
    }

    /* ================================================================== */
    /*  PRIVATE METHODS                                                   */
    /* ================================================================== */

    /**
     * Logic to match a detection to a person ID.
     * In a full implementation, this would use face embeddings.
     * Here, we use temporal track IDs and simulate recognition.
     */
    _identifyPerson(obj, currentTime) {
        const trackId = obj._trackId; // set by events.js tracking
        // If we already tagged this trackId in this session
        if (this.activeIdentities.has(trackId)) {
            return this.activeIdentities.get(trackId);
        }

        // --- SIMULATED RECOGNITION LOGIC ---
        // We simulate that some trackIds match known people
        // In a real app, this is where face-api.js or similar would run.
        let identity = {
            id: `unknown_${trackId}`,
            name: 'Unknown Person',
            isKnown: false,
            lastSeen: currentTime
        };

        // For demonstration purposes: map specific track IDs to names
        // or just pick one randomly if it's the first time seeing them.
        const knownIds = Object.keys(this.people);
        if (trackId && trackId % 5 === 0) { // Simulate match every 5th track
            const personId = knownIds[trackId % knownIds.length];
            const p = this.people[personId];

            p.encounterCount++;
            p.lastSeen = currentTime;

            identity = {
                id: p.id,
                name: p.name,
                isKnown: true,
                encounterCount: p.encounterCount,
                lastSeen: p.lastSeen
            };

            this._saveToStorage(this.KNOWN_PEOPLE_KEY, this.people);
        }

        this.activeIdentities.set(trackId, identity);
        return identity;
    }

    _logEvent(event) {
        const entry = {
            id: `log_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            type: event.type,
            label: event.label,
            classification: event.classification,
            position: event.position
        };

        this.eventLog.unshift(entry);
        if (this.eventLog.length > this.MAX_EVENT_LOG) this.eventLog.pop();
        
        this._saveToStorage(this.EVENT_LOG_KEY, this.eventLog);
    }

    _eventRecentlyLogged(event) {
        // Prevent spamming the memory log with the same event in < 10 seconds
        return this.eventLog.some(e => 
            e.label === event.label && 
            e.type === event.type && 
            (Date.now() - new Date().setHours(...e.timestamp.split(':').map(Number))) < 10000
        );
    }

    /* --- Storage Helpers --- */
    _loadFromStorage(key, defaultValue) {
        try {
            const data = localStorage.getItem(key);
            return data ? JSON.parse(data) : defaultValue;
        } catch (e) { return defaultValue; }
    }

    _saveToStorage(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) { /* ignore */ }
    }
}

// Expose globally
window.MemoryEngine = new MemoryEngine();
