/**
 * Netra — Adaptive Intelligence Engine (Layer 9)
 * 
 * Implements long-term learning and pattern recognition.
 * Tracks recurring environments, obstacles, and navigation paths.
 * Uses localStorage for persistent learning across sessions.
 */
class AdaptiveEngine {
    constructor() {
        this.STORAGE_KEY = 'netra_adaptive_kb';
        this.kb = this._loadKB();
        
        // Active session state
        this.sessionData = {
            currentScene: null,
            frameBuffer: [],
            lastUpdateTime: Date.now()
        };
    }

    /**
     * Update knowledge base based on current perception
     */
    learn(scene, navigation, memory, detections) {
        if (!scene || scene.category === 'Initializing...') return;

        // 1. Scene Frequency Learning
        this._updateSceneFrequency(scene.category);

        // 2. Navigation Path Learning
        if (navigation && navigation.status === 'safe') {
            this._updatePathPattern(scene.category, navigation.direction);
        }

        // 3. Obstacle Pattern Learning
        if (detections && detections.length > 0) {
            this._updateObstaclePatterns(scene.category, detections);
        }

        // 4. Social Association Learning
        if (memory && memory.people && memory.people.recognized) {
            this._updateSocialContext(scene.category, memory.people.recognized);
        }

        // Periodic save (every 5 seconds or major events)
        if (Date.now() - this.sessionData.lastUpdateTime > 5000) {
            this._saveKB();
        }

        return this._getTopPatterns();
    }

    _loadKB() {
        const data = localStorage.getItem(this.STORAGE_KEY);
        if (data) {
            try {
                return JSON.parse(data);
            } catch (e) {
                console.error("Failed to parse adaptive KB:", e);
            }
        }
        return {
            scenes: {},      // category -> {visitCount, firstSeen, lastSeen}
            paths: {},       // scene -> {direction -> count}
            obstacles: {},   // scene -> {class_pos -> {count, label}}
            social: {}       // scene -> {personName -> count}
        };
    }

    _saveKB() {
        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.kb));
        this.sessionData.lastUpdateTime = Date.now();
    }

    /* --- Learning Logic --- */

    _updateSceneFrequency(category) {
        if (!this.kb.scenes[category]) {
            this.kb.scenes[category] = { visitCount: 0, firstSeen: new Date().toISOString() };
        }
        this.kb.scenes[category].visitCount++;
        this.kb.scenes[category].lastSeen = new Date().toISOString();
    }

    _updatePathPattern(scene, direction) {
        if (!this.kb.paths[scene]) this.kb.paths[scene] = {};
        this.kb.paths[scene][direction] = (this.kb.paths[scene][direction] || 0) + 1;
    }

    _updateObstaclePatterns(scene, detections) {
        if (!this.kb.obstacles[scene]) this.kb.obstacles[scene] = {};
        
        detections.forEach(d => {
            if (d.risk !== 'safe') {
                const key = `${d.class}_${d.position}`;
                if (!this.kb.obstacles[scene][key]) {
                    this.kb.obstacles[scene][key] = { count: 0, label: d.class };
                }
                this.kb.obstacles[scene][key].count++;
            }
        });
    }

    _updateSocialContext(scene, recognizedPeople) {
        if (!this.kb.social[scene]) this.kb.social[scene] = {};
        recognizedPeople.forEach(p => {
            this.kb.social[scene][p.name] = (this.kb.social[scene][p.name] || 0) + 1;
        });
    }

    /* --- Insight Generation --- */

    _getTopPatterns() {
        // Return structured insights for UI
        const topSceneCat = Object.keys(this.kb.scenes).sort((a,b) => this.kb.scenes[b].visitCount - this.kb.scenes[a].visitCount)[0];
        
        let pathInsight = "Analyzing paths...";
        if (topSceneCat && this.kb.paths[topSceneCat]) {
            const bestPath = Object.keys(this.kb.paths[topSceneCat]).sort((a,b) => this.kb.paths[topSceneCat][b] - this.kb.paths[topSceneCat][a])[0];
            pathInsight = `${bestPath} (usually)`;
        }

        let obstacleInsight = [];
        if (topSceneCat && this.kb.obstacles[topSceneCat]) {
            obstacleInsight = Object.values(this.kb.obstacles[topSceneCat])
                .sort((a,b) => b.count - a.count)
                .slice(0, 2)
                .map(o => o.label);
        }

        let frequentPerson = "None";
        if (topSceneCat && this.kb.social[topSceneCat]) {
            frequentPerson = Object.keys(this.kb.social[topSceneCat]).sort((a,b) => this.kb.social[topSceneCat][b] - this.kb.social[topSceneCat][a])[0] || "None";
        }

        return {
            learnedScene: topSceneCat || "Learning...",
            commonPath: pathInsight,
            topObstacles: obstacleInsight,
            frequentPerson: frequentPerson,
            totalScenes: Object.keys(this.kb.scenes).length
        };
    }
}

// Global instance
window.AdaptiveEngine = new AdaptiveEngine();
