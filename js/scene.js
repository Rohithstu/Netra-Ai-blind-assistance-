/**
 * Netra — Scene Intelligence Engine (Layer 7)
 * 
 * Analyzes object density, spatial layout, and environmental context
 * to classify the overall scene (Indoor, Corridor, Street, etc.)
 * and detect transitions between areas.
 */
class SceneEngine {
    constructor() {
        this.currentScene = {
            category: 'Initializing...',
            description: 'Analyzing environment...',
            confidence: 0,
            structure: [],
            features: []
        };

        this.history = []; // Scene history
        this.frameCounter = 0;
        this.analysisWindow = 5; // Frames for stable classification
        this.classificationBuffer = [];

        // Predefined Scene Categories (Expanded for "Universal" Awareness)
        this.CATEGORIES = {
            INDOOR_ROOM: 'Indoor Room',
            OFFICE: 'Office Workspace',
            KITCHEN_DINING: 'Kitchen or Dining Area',
            BEDROOM: 'Bedroom',
            BATHROOM: 'Bathroom',
            CORRIDOR: 'Corridor or Hallway',
            STREET: 'Street Environment',
            OUTDOOR_WALKWAY: 'Outdoor Walkway',
            PARK_GARDEN: 'Park or Garden Area',
            PUBLIC_TRANSIT: 'Public Transit Area',
            RETAIL_SHOP: 'Shopping or Retail Space',
            SPORT_ARENA: 'Sports or Activity Space',
            OPEN_SPACE: 'Open Landscape'
        };
    }

    /**
     * Process frame data to understand the overall scene
     */
    analyse(detections, spatialData) {
        this.frameCounter++;
        
        // 1. Scene Classification Logic (Multi-variate Heuristics)
        const newCategory = this._classifyEnvironment(detections, spatialData);
        
        // Buffer for stability (avoid rapid switching)
        this.classificationBuffer.push(newCategory);
        if (this.classificationBuffer.length > this.analysisWindow) {
            this.classificationBuffer.shift();
        }

        const stableCategory = this._getMostFrequent(this.classificationBuffer);
        
        // Detect Transitions
        const isTransition = stableCategory !== this.currentScene.category && this.currentScene.category !== 'Initializing...';

        // 2. Structural Analysis
        const structure = this._analyzeStructure(detections, spatialData);

        // Update State
        const prevCategory = this.currentScene.category;
        this.currentScene = {
            category: stableCategory,
            description: this._getSceneDescription(stableCategory, structure),
            structure: structure,
            isTransition: isTransition
        };

        if (isTransition) {
            console.log(`Universal Scene Transition: ${prevCategory} → ${stableCategory}`);
            this._logSceneMemory(this.currentScene);
        }

        return this.currentScene;
    }

    /* ================================================================== */
    /*  PRIVATE METHODS                                                   */
    /* ================================================================== */

    _classifyEnvironment(detections, spatial) {
        const counts = this._getObjectCounts(detections);
        
        // --- 1. SPECIALIZED INDOOR AREAS ---
        
        // Kitchen / Dining
        if (counts.sink > 0 || counts.bottle > 0 || counts.cup > 0 || counts.knife > 0 || counts.spoon > 0 || counts.fork > 0 || counts.bowl > 0) {
            return this.CATEGORIES.KITCHEN_DINING;
        }

        // Bedroom
        if (counts.bed > 0) return this.CATEGORIES.BEDROOM;

        // Bathroom
        if (counts.toilet > 0 || counts.toothbrush > 0) return this.CATEGORIES.BATHROOM;

        // Office Workspace (Electronics density)
        if (counts.laptop > 0 || counts.keyboard > 0 || counts.mouse > 0 || counts.monitor > 0) return this.CATEGORIES.OFFICE;

        // --- 2. OUTDOOR & TRANSIT ---

        // Street Environment
        if (counts.car > 0 || counts.bus > 0 || counts.truck > 0 || counts.bicycle > 0 || counts.traffic_light > 0 || counts.stop_sign > 0 || counts.fire_hydrant > 0) {
            return this.CATEGORIES.STREET;
        }

        // Park or Garden
        if (counts.dog > 0 || counts.bird > 0 || counts.bench > 0 || counts.tie > 0 || counts.frisbee > 0) {
            // Note: bencher + plant = garden
            const natureDensity = (counts.potted_plant || 0) + (counts.bird || 0);
            if (natureDensity > 0 || counts.bench > 0) return this.CATEGORIES.PARK_GARDEN;
        }

        // Public Transit / Crowded Spaces
        if (counts.person > 4 || counts.backpack > 1 || counts.handbag > 1) {
            return this.CATEGORIES.PUBLIC_TRANSIT;
        }

        // --- 3. STRUCTURE-BASED (Corridors / Open Spaces) ---

        // Corridor detection (Structure analysis)
        if (spatial && spatial.pathStatus) {
            const ps = spatial.pathStatus;
            // Narrow path with walls
            if (ps.center === 'clear' && ps.left === 'blocked' && ps.right === 'blocked') {
                return this.CATEGORIES.CORRIDOR;
            }
        }

        // Retail / Commercial
        if (counts.tv > 0 || (counts.person > 2 && counts.handbag > 0)) {
            return this.CATEGORIES.RETAIL_SHOP;
        }

        // Sport / Sports Arena
        if (counts.sports_ball > 0 || counts.tennis_racket > 0 || counts.baseball_bat > 0 || counts.baseball_glove > 0 || counts.skis > 0 || counts.snowboard > 0) {
            return this.CATEGORIES.SPORT_ARENA;
        }

        // Generic Indoor Room
        if (counts.chair > 0 || counts.table > 0 || counts.sofa > 0 || counts.potted_plant > 0) {
            return this.CATEGORIES.INDOOR_ROOM;
        }

        // Default or Open Space
        if (detections.length === 0 && spatial && spatial.safeWalkingSpace > 0.6) {
            return this.CATEGORIES.OPEN_SPACE;
        }

        return this.currentScene.category === 'Initializing...' ? this.CATEGORIES.INDOOR_ROOM : this.currentScene.category;
    }

    _analyzeStructure(detections, spatial) {
        const structuralFeatures = [];
        
        // Doors indicate entry points
        if (detections.some(d => d.class === 'door')) {
            const door = detections.find(d => d.class === 'door');
            structuralFeatures.push(`Entry point (${door.position})`);
        }

        // Obstacles arrangement
        if (spatial && spatial.pathStatus) {
            const ps = spatial.pathStatus;
            if (ps.center === 'clear') structuralFeatures.push('Clear walking path');
            else structuralFeatures.push('Central obstacles ahead');
        }

        // Furniture density
        const furnitureCount = detections.filter(d => ['chair', 'table', 'sofa', 'bed'].includes(d.class)).length;
        if (furnitureCount > 2) structuralFeatures.push('Densely furnished area');
        else if (furnitureCount > 0) structuralFeatures.push('Furnished space');

        return structuralFeatures;
    }

    _getSceneDescription(category, structure) {
        let desc = category;
        if (structure.length > 0) {
            desc += ` with ${structure[0].toLowerCase()}`;
        }
        return desc;
    }

    _getObjectCounts(detections) {
        const counts = {};
        detections.forEach(d => {
            counts[d.class] = (counts[d.class] || 0) + 1;
        });
        return counts;
    }

    _getMostFrequent(arr) {
        const map = {};
        let mostFreq = arr[0];
        arr.forEach(val => {
            map[val] = (map[val] || 0) + 1;
            if (map[val] > map[mostFreq]) mostFreq = val;
        });
        return mostFreq;
    }

    _logSceneMemory(scene) {
        if (window.MemoryEngine) {
            window.MemoryEngine._logEvent({
                type: 'scene_transition',
                label: `New area: ${scene.category}`,
                classification: 'Medium',
                position: 'surroundings'
            });
        }
    }
}

// Global instance
window.SceneEngine = new SceneEngine();
