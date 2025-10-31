/**
 * dashboard.js - Dashboard Module
 * ===============================
 * Module spécifique au dashboard principal UltraMotion IGT
 * 
 * Responsabilités:
 * - Orchestration spécifique au dashboard
 * - Gestion des interactions dashboard
 * - Logique métier dashboard
 */

class DashboardModule {
    constructor() {
        this.wsManager = null;
        this.uiManager = null;
        this.timeManager = null;
        this.isInitialized = false;
        
        this.init();
    }

    /**
     * Initialise le module dashboard
     */
    init() {
        console.log('📊 Initialisation du module Dashboard');
        
        // Vérification que les managers sont disponibles
        if (!window.WebSocketManager || !window.UIManager || !window.TimeManager) {
            console.error('❌ Managers requis non disponibles');
            return;
        }

        // Initialisation des managers
        this.initializeManagers();
        this.setupEventListeners();
        this.setupDashboardInteractions();
        
        this.isInitialized = true;
        console.log('✅ Module Dashboard initialisé');
    }

    /**
     * Initialise les gestionnaires
     */
    initializeManagers() {
        // Configuration WebSocket
        const wsConfig = {
            url: 'ws://localhost:8050/ws/v1/pipeline'
        };

        // Configuration UI
        const uiConfig = {
            thresholds: {
                gpu: { warning: 70, critical: 90 },
                latency: { warning: 50, critical: 100 },
                queue: { warning: 80, critical: 95 }
            }
        };

        // Configuration Time
        const timeConfig = {
            updateInterval: 1000,
            locale: 'fr-FR'
        };

        // Instanciation
        this.wsManager = new WebSocketManager(wsConfig);
        this.uiManager = new UIManager(uiConfig);
        this.timeManager = new TimeManager(timeConfig);
    }

    /**
     * Configure les listeners d'événements
     */
    setupEventListeners() {
        // Événements WebSocket
        this.wsManager.on('connection_status', (data) => {
            this.uiManager.updateConnectionStatus(data.status);
        });

        this.wsManager.on('gpu_metrics', (data) => {
            this.uiManager.updateGPUMetrics(data);
            this.timeManager.markLastUpdate();
        });

        this.wsManager.on('pipeline_metrics', (data) => {
            this.uiManager.updatePipelineMetrics(data);
            this.timeManager.markLastUpdate();
        });

        this.wsManager.on('queue_metrics', (data) => {
            this.uiManager.updateQueueMetrics(data);
            this.timeManager.markLastUpdate();
        });

        this.wsManager.on('system_health', (data) => {
            this.uiManager.updateSystemHealth(data);
            this.timeManager.markLastUpdate();
        });

        // Ajout du listener manquant
        this.wsManager.on('system_metrics', (data) => {
            // console.log('🎯 [DEBUG] Événement system_metrics reçu dans dashboard:', data);
            this.uiManager.updateSystemMetrics(data);
            this.timeManager.markLastUpdate();
        });


        // Gestion de la fermeture de page
        window.addEventListener('beforeunload', () => {
            this.destroy();
        });
    }

    /**
     * Configure les interactions spécifiques au dashboard
     */
    setupDashboardInteractions() {
        // Boutons d'action (si présents)
        const exportBtn = document.querySelector('.btn[title="Exporter les données"]');
        if (exportBtn) {
            exportBtn.addEventListener('click', this.handleExport.bind(this));
        }

        const settingsBtn = document.querySelector('.btn[title="Paramètres du dashboard"]');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', this.handleSettings.bind(this));
        }

        // Raccourcis clavier
        document.addEventListener('keydown', (event) => {
            if (event.ctrlKey && event.key === 'e') {
                event.preventDefault();
                this.handleExport();
            }
            if (event.ctrlKey && event.key === ',') {
                event.preventDefault();
                this.handleSettings();
            }
        });
    }

    /**
     * Gère l'export des données
     */
    handleExport() {
        console.log('📤 Export des données demandé');
        
        const data = {
            timestamp: new Date().toISOString(),
            dashboard: 'ultramotion-igt',
            timeStats: this.timeManager.getStats(),
            connectionState: this.wsManager.getConnectionState()
        };
        
        // Création du fichier de export
        const blob = new Blob([JSON.stringify(data, null, 2)], { 
            type: 'application/json' 
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ultramotion-metrics-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('✅ Export terminé');
    }

    /**
     * Gère l'ouverture des paramètres
     */
    handleSettings() {
        console.log('⚙️ Ouverture des paramètres');
        // TODO: Implémenter modal de paramètres
        alert('Paramètres du dashboard\n\nFonctionnalité en développement...');
    }

    /**
     * Démarre la connexion WebSocket
     */
    start() {
        if (this.wsManager) {
            this.wsManager.connect();
        }
    }

    /**
     * Arrête le dashboard
     */
    stop() {
        if (this.wsManager) {
            this.wsManager.disconnect();
        }
    }

    /**
     * Redémarre le dashboard
     */
    restart() {
        this.stop();
        setTimeout(() => {
            this.start();
        }, 1000);
    }

    /**
     * Détruit le module
     */
    destroy() {
        console.log('🗑️ Destruction du module Dashboard');
        
        if (this.wsManager) {
            this.wsManager.disconnect();
        }
        
        if (this.timeManager) {
            this.timeManager.destroy();
        }
        
        this.isInitialized = false;
    }

    /**
     * Retourne l'état du dashboard
     */
    getState() {
        return {
            isInitialized: this.isInitialized,
            wsState: this.wsManager ? this.wsManager.getConnectionState() : null,
            timeStats: this.timeManager ? this.timeManager.getStats() : null
        };
    }
}

// Auto-initialisation quand le DOM est prêt
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardModule = new DashboardModule();
    window.dashboardModule.start();
});

// Export pour utilisation externe
window.DashboardModule = DashboardModule;