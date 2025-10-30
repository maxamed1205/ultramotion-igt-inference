/**
 * main.js - Application Orchestrator
 * ===================================
 * Point d'entrée principal modulaire pour UltraMotion IGT Dashboard
 * Hôpitaux Universitaires de Genève (HUG)
 */

class UltraMotionApp {
    constructor() {
        this.currentPage = this.detectCurrentPage();
        this.currentModule = null;
        this.globalConfig = this.loadGlobalConfig();
        
        console.log(` UltraMotion IGT Application - Page: ${this.currentPage}`);
        this.init();
    }

    detectCurrentPage() {
        const path = window.location.pathname;
        
        if (path.includes('/dashboard') || path === '/') {
            return 'dashboard';
        } else if (path.includes('/compare')) {
            return 'compare';
        } else if (path.includes('/frames')) {
            return 'frames';
        } else if (path.includes('/trends')) {
            return 'trends';
        }
        
        return 'dashboard';
    }

    loadGlobalConfig() {
        return {
            websocket: {
                url: 'ws://localhost:8050/ws/v1/pipeline',
                reconnectInterval: 5000,
                maxReconnectAttempts: 10
            },
            ui: {
                animationDuration: 300,
                theme: 'dark'
            },
            thresholds: {
                gpu: { warning: 70, critical: 90 },
                latency: { warning: 50, critical: 100 },
                queue: { warning: 80, critical: 95 }
            }
        };
    }

    async init() {
        try {
            this.setupGlobalStyles();
            await this.loadPageModule();
            this.setupGlobalEventHandlers();
            
            console.log(' UltraMotion IGT Application initialisée');
        } catch (error) {
            console.error(' Erreur lors de l\'initialisation:', error);
        }
    }

    setupGlobalStyles() {
        const root = document.documentElement;
        root.style.setProperty('--animation-duration', `${this.globalConfig.ui.animationDuration}ms`);
        document.body.classList.add(`theme-${this.globalConfig.ui.theme}`);
    }

    async loadPageModule() {
        console.log(` Chargement du module: ${this.currentPage}`);
        
        switch (this.currentPage) {
            case 'dashboard':
                if (window.dashboardModule) {
                    this.currentModule = window.dashboardModule;
                    console.log(' Module Dashboard activé');
                }
                break;
            default:
                console.warn(` Module non reconnu: ${this.currentPage}`);
        }
    }

    setupGlobalEventHandlers() {
        window.addEventListener('error', (event) => {
            console.error(' Erreur globale:', event.error);
        });

        document.addEventListener('keydown', (event) => {
            if (event.ctrlKey && event.shiftKey && event.key === 'D') {
                event.preventDefault();
                this.showDebugInfo();
            }
        });
    }

    showDebugInfo() {
        const debugInfo = {
            page: this.currentPage,
            config: this.globalConfig,
            timestamp: new Date().toISOString()
        };
        
        console.log(' Debug Info:', debugInfo);
    }

    getState() {
        return {
            currentPage: this.currentPage,
            config: this.globalConfig,
            moduleState: this.currentModule ? this.currentModule.getState() : null
        };
    }
}

// Initialisation automatique
document.addEventListener('DOMContentLoaded', () => {
    window.ultraMotionApp = new UltraMotionApp();
});

// Export pour utilisation externe
window.UltraMotionApp = UltraMotionApp;
