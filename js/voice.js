/**
 * Netra — Voice Intelligence Engine (Layer 6)
 * 
 * Handles bidirectional voice communication:
 * 1. Speech Synthesis (TTS): Spoken alerts for high-priority events.
 * 2. Voice Recognition (STT): Understanding user commands.
 */
class VoiceEngine {
    constructor() {
        this.synth = window.speechSynthesis;
        this.recognition = null;
        this.isListening = false;
        this.lastSpokenText = "";
        this.voiceCooldown = 3000; // 3 seconds between alerts
        this.lastSpeakTime = 0;

        this.onCommand = null; // Callback for commands
        this.initRecognition();
    }

    initRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn("Speech recognition not supported in this browser.");
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = false;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this.isListening = true;
            this.updateUIPanel({ command: "Listening..." });
        };

        this.recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript.toLowerCase();
            console.log("Voice Input Received:", transcript);
            
            // Wake Word Detection
            if (transcript.includes("netra") || transcript.includes("natra") || transcript.includes("ultra")) {
                const command = transcript.replace(/netra|natra|ultra/g, "").trim();
                
                if (command.length === 0) {
                    // Just the wake word
                    this.speak("Yes, I am listening.");
                    this.updateUIPanel({ command: "Netra", response: "Listening..." });
                } else {
                    // Combined wake word + command
                    this.handleCommand(command);
                }
            } else if (!this.isListening) {
                // If not activated via button, but speech detected, ignore unless wake word present
                console.log("Ignoring speech without wake word.");
            } else {
                // Traditional button-activated listening
                this.handleCommand(transcript);
            }
        };

        this.recognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            this.isListening = false;
            this.updateUIPanel({ command: "Error: " + event.error });
        };

        this.recognition.onend = () => {
            this.isListening = false;
        };
    }

    startListening() {
        if (this.recognition && !this.isListening) {
            try {
                this.recognition.start();
            } catch (e) {
                console.error("Failed to start recognition:", e);
            }
        }
    }

    /**
     * Converts a priority alert into a spoken message
     */
    speakAlert(alert) {
        const currentTime = Date.now();
        
        // Safety: Only speak High/Medium alerts and respect cooldown
        if (alert.classification === 'Low') return;
        if (currentTime - this.lastSpeakTime < this.voiceCooldown) return;

        // Construct a natural message
        // e.g., "Obstacle ahead" or "Ravi is approaching from the left"
        let msg = "";
        const name = (alert.class === 'person' && alert.identity) ? alert.identity.name : alert.label;
        
        if (alert.type === 'path_blocked') msg = "Path blocked ahead.";
        else if (alert.type === 'obstacle_appear') msg = `${name} appeared ahead.`;
        else if (alert.type === 'approach') msg = `${name} approaching from the ${alert.position}.`;
        else if (alert.type === 'face_emotion') msg = alert.label;
        else if (alert.type === 'social_interaction') msg = alert.label;
        else if (alert.type === 'crowd_alert') msg = `${alert.label}. Move slowly.`;
        else msg = `${name} ${alert.position}.`;

        this.speak(msg);
    }

    /**
     * Speaks navigation instructions if they represent a significant change
     */
    speakNavigation(nav) {
        if (!nav || nav.status === 'safe') return;

        const currentTime = Date.now();
        if (currentTime - this.lastSpeakTime < 4000) return; // Higher cooldown for nav

        // Only speak if it's a critical instruction (STOP or specific Move)
        if (nav.direction === 'STOP' || nav.direction.includes('Move')) {
            const msg = `${nav.direction}. ${nav.reason}.`;
            this.speak(msg);
        }
    }

    speak(text) {
        if (!text || text === this.lastSpokenText) return;
        
        // Stop current speech if priority
        if (this.synth.speaking) {
            this.synth.cancel();
        }

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        
        utterance.onstart = () => {
            this.updateUIPanel({ alert: text });
        };

        this.synth.speak(utterance);
        this.lastSpokenText = text;
        this.lastSpeakTime = Date.now();
    }

    handleCommand(command) {
        this.updateUIPanel({ command: command });
        let response = "";

        if (command.includes("what is in front") || command.includes("describe")) {
            response = this._getEnvironmentDescription();
        } else if (command.includes("who is") || command.includes("nearby")) {
            response = this._getNearbyPeople();
        } else if (command.includes("is the path clear") || command.includes("where") && command.includes("walk")) {
            response = this._getNavigationGuidance();
        } else {
            response = "I didn't quite catch that. Try asking what is in front of you or where you should walk.";
        }

        this.speak(response);
        this.updateUIPanel({ response: response });
    }

    /* --- Context-Aware Response Helpers --- */

    _getEnvironmentDescription() {
        let sceneContext = "";
        if (window.SceneEngine && window.SceneEngine.currentScene.category !== 'Initializing...') {
            sceneContext = `You are in ${window.SceneEngine.currentScene.category}. `;
        }

        if (!window.PerceptionEngine || !window.PerceptionEngine.previousDetections.length) {
            return sceneContext + "The environment is currently clear.";
        }
        const counts = {};
        window.PerceptionEngine.previousDetections.forEach(d => {
            counts[d.class] = (counts[d.class] || 0) + 1;
        });
        const items = Object.entries(counts).map(([name, count]) => `${count} ${name}${count > 1 ? 's' : ''}`).join(", ");
        return `${sceneContext}In front of you, I see ${items}.`;
    }

    _getNearbyPeople() {
        const people = window.PerceptionEngine.previousDetections.filter(d => d.class === 'person');
        if (people.length === 0) return "No one is nearby.";
        
        const names = people.map(p => {
            const name = p.identity ? p.identity.name : "A person";
            const emotion = p.emotion ? ` who appears ${p.emotion}` : "";
            return `${name}${emotion} at the ${p.position}`;
        }).join(", ");
        
        return `Nearby, there is ${names}.`;
    }

    _getNavigationGuidance() {
        if (!window.NavigationEngine) return "I cannot determine a safe path right now.";
        const nav = window.NavigationEngine.currentGuidance;
        if (nav.direction === 'Analyzing...') return "I am still analyzing the environment. Please wait.";
        
        return `${nav.direction}. ${nav.reason}.`;
    }

    updateUIPanel(data) {
        // This will be called by perception.js to update the L6 UI
        if (this.onUIUpdate) this.onUIUpdate(data);
    }
}

// Global instance
window.VoiceEngine = new VoiceEngine();
