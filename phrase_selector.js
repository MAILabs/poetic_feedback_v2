// Phrase selector module
// Loads phrases from phrases.json and provides selection mechanism with priority system

class PhraseSelector {
    constructor() {
        this.phrases = null;
        this.recentPhrases = []; // Track recently used phrases to avoid repetition
        this.maxRecentHistory = 5; // Keep track of last 5 phrases
        this.ready = false;
    }

    // Load phrases from global PHRASES_DATA variable (loaded via phrases.js script)
    async loadPhrases() {
        try {
            // Check if PHRASES_DATA is available (loaded from phrases.js)
            if (typeof PHRASES_DATA === 'undefined') {
                throw new Error('PHRASES_DATA is not defined. Make sure phrases.js is loaded before phrase_selector.js');
            }
            this.phrases = PHRASES_DATA;
            this.ready = true;
            console.log('Phrases loaded successfully');
            return true;
        } catch (error) {
            console.error('Error loading phrases:', error);
            this.ready = false;
            return false;
        }
    }

    // Convert speed value to category (lo/med/hi)
    getSpeedCategory(speed) {
        if (speed === null || speed === undefined) {
            return 'med'; // Default to medium if speed is not available
        }
        if (speed < 300) {
            return 'lo';
        } else if (speed <= 1000) {
            return 'med';
        } else {
            return 'hi';
        }
    }

    // Get all phrases for a given emotion and speed category
    getPhrasesForCombination(emotion, speedCategory) {
        if (!this.ready || !this.phrases) {
            return [];
        }

        const allPhrases = [];

        // Get phrases specific to emotion and speed
        if (this.phrases[emotion] && this.phrases[emotion][speedCategory]) {
            allPhrases.push(...this.phrases[emotion][speedCategory]);
        }

        // Add phrases from emotion's "all" category
        if (this.phrases[emotion] && this.phrases[emotion].all) {
            allPhrases.push(...this.phrases[emotion].all);
        }

        // Add phrases from global "all" category for this speed
        if (this.phrases.all && this.phrases.all[speedCategory]) {
            allPhrases.push(...this.phrases.all[speedCategory]);
        }

        // Add phrases from global "all" category's "all" speed
        if (this.phrases.all && this.phrases.all.all) {
            allPhrases.push(...this.phrases.all.all);
        }

        return allPhrases;
    }

    // Select a random phrase with priority mechanism
    selectPhrase(emotion, speed) {
        if (!this.ready) {
            return null;
        }

        const speedCategory = this.getSpeedCategory(speed);
        const availablePhrases = this.getPhrasesForCombination(emotion, speedCategory);

        if (availablePhrases.length === 0) {
            return null;
        }

        // Create priority-weighted list
        // Recently used phrases get lower weight
        const weightedPhrases = availablePhrases.map(phrase => {
            const recentIndex = this.recentPhrases.indexOf(phrase);
            // If phrase was recently used, give it lower weight
            // Weight decreases based on how recent it was (0 = most recent, gets lowest weight)
            const weight = recentIndex >= 0 ? 1 / (recentIndex + 2) : 1;
            return { phrase, weight };
        });

        // Calculate total weight
        const totalWeight = weightedPhrases.reduce((sum, item) => sum + item.weight, 0);

        // Select random value between 0 and totalWeight
        let random = Math.random() * totalWeight;
        let selectedPhrase = null;

        // Select phrase based on weighted probability
        for (const item of weightedPhrases) {
            random -= item.weight;
            if (random <= 0) {
                selectedPhrase = item.phrase;
                break;
            }
        }

        // Fallback to random selection if something went wrong
        if (!selectedPhrase) {
            selectedPhrase = availablePhrases[Math.floor(Math.random() * availablePhrases.length)];
        }

        // Update recent phrases history
        // Remove phrase if it already exists in history
        const existingIndex = this.recentPhrases.indexOf(selectedPhrase);
        if (existingIndex >= 0) {
            this.recentPhrases.splice(existingIndex, 1);
        }
        // Add to front of history
        this.recentPhrases.unshift(selectedPhrase);
        // Keep only recent history
        if (this.recentPhrases.length > this.maxRecentHistory) {
            this.recentPhrases.pop();
        }

        return selectedPhrase;
    }

    // Reset recent phrases history (useful when starting new session)
    resetHistory() {
        this.recentPhrases = [];
    }
}

// Create global instance
const phraseSelector = new PhraseSelector();

