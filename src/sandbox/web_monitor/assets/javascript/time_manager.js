/**
 * time_manager.js - Time Manager
 * ===============================
 * Gestionnaire de temps et timestamps pour UltraMotion IGT Dashboard
 * 
 * Responsabilités:
 * - Mise à jour de l'horloge en temps réel
 * - Gestion des timestamps "dernière MAJ"
 * - Calcul du temps écoulé depuis dernière mise à jour
 * - Formatage des dates et heures
 */

class TimeManager {
    constructor(config = {}) {
        this.config = {
            updateInterval: config.updateInterval || 1000,
            locale: config.locale || 'fr-FR',
            timeFormat: config.timeFormat || {
                hour12: false,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            },
            ...config
        };
        
        this.clockTimer = null;
        this.lastUpdateTime = null;
        this.lastUpdateTimer = null;
        
        this.init();
    }

    /**
     * Initialise le gestionnaire de temps
     */
    init() {
        this.startClock();
        this.startLastUpdateTracker();
        console.log('⏰ TimeManager initialisé');
    }

    /**
     * Démarre l'horloge en temps réel
     */
    startClock() {
        // Mise à jour immédiate
        this.updateClock();
        
        // Puis mise à jour régulière
        this.clockTimer = setInterval(() => {
            this.updateClock();
        }, this.config.updateInterval);
    }

    /**
     * Met à jour l'affichage de l'horloge
     */
    updateClock() {
        const now = new Date();
        const timeString = now.toLocaleTimeString(this.config.locale, this.config.timeFormat);
        
        // Éléments d'horloge
        const timeElements = ['current-time', 'footer-timestamp'];
        timeElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = timeString;
            }
        });
    }

    /**
     * Démarre le suivi du temps écoulé depuis dernière MAJ
     */
    startLastUpdateTracker() {
        this.lastUpdateTimer = setInterval(() => {
            this.updateLastUpdateIndicators();
        }, 1000);
    }

    /**
     * Marque le moment de la dernière mise à jour
     */
    markLastUpdate() {
        this.lastUpdateTime = new Date();
        this.updateLastUpdateIndicators();
    }

    /**
     * Met à jour les indicateurs "il y a X secondes"
     */
    updateLastUpdateIndicators() {
        if (!this.lastUpdateTime) {
            // Pas encore de mise à jour
            const elements = ['gpu-last-update', 'sidebar-last-update'];
            elements.forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    element.textContent = '-- s';
                }
            });
            return;
        }

        const now = new Date();
        const secondsElapsed = Math.floor((now - this.lastUpdateTime) / 1000);
        const timeText = this.formatElapsedTime(secondsElapsed);
        
        const elements = ['gpu-last-update', 'sidebar-last-update'];
        elements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = timeText;
                
                // Changement de couleur selon l'ancienneté
                if (secondsElapsed > 30) {
                    element.style.color = 'var(--error-color)';
                } else if (secondsElapsed > 10) {
                    element.style.color = 'var(--warning-color)';
                } else {
                    element.style.color = 'var(--text-muted)';
                }
            }
        });
    }

    /**
     * Formate le temps écoulé en texte lisible
     */
    formatElapsedTime(seconds) {
        if (seconds < 60) {
            return `${seconds} s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            return `${minutes} min`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}min`;
        }
    }

    /**
     * Retourne un timestamp formaté pour l'heure actuelle
     */
    getCurrentTimestamp() {
        return new Date().toLocaleTimeString(this.config.locale, this.config.timeFormat);
    }

    /**
     * Retourne la date formatée
     */
    getCurrentDate() {
        return new Date().toLocaleDateString(this.config.locale, {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }

    /**
     * Retourne le temps écoulé depuis la dernière mise à jour
     */
    getTimeSinceLastUpdate() {
        if (!this.lastUpdateTime) {
            return null;
        }
        
        const now = new Date();
        return Math.floor((now - this.lastUpdateTime) / 1000);
    }

    /**
     * Arrête l'horloge
     */
    stopClock() {
        if (this.clockTimer) {
            clearInterval(this.clockTimer);
            this.clockTimer = null;
        }
    }

    /**
     * Arrête le suivi des mises à jour
     */
    stopLastUpdateTracker() {
        if (this.lastUpdateTimer) {
            clearInterval(this.lastUpdateTimer);
            this.lastUpdateTimer = null;
        }
    }

    /**
     * Arrête tous les timers
     */
    destroy() {
        this.stopClock();
        this.stopLastUpdateTracker();
        console.log('⏰ TimeManager arrêté');
    }

    /**
     * Redémarre le gestionnaire de temps
     */
    restart() {
        this.destroy();
        this.init();
    }

    /**
     * Retourne des statistiques sur le temps
     */
    getStats() {
        return {
            currentTime: this.getCurrentTimestamp(),
            currentDate: this.getCurrentDate(),
            lastUpdateTime: this.lastUpdateTime,
            secondsSinceLastUpdate: this.getTimeSinceLastUpdate(),
            isClockRunning: this.clockTimer !== null,
            isTrackerRunning: this.lastUpdateTimer !== null
        };
    }
}

// Export pour utilisation globale
window.TimeManager = TimeManager;