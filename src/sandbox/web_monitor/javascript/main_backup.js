/**
 * main.js - UltraMotion IGT Dashboard
 * ===================================
 * Point d'entrée principal pour le dashboard professionnel
 * Hôpitaux Universitaires de Genève (HUG)
 * Laboratoire de Cinésiologie
 * 
 * Gère les WebSockets, animations, et interactions utilisateur
 */

// Configuration du dashboard
const DASHBOARD_CONFIG = {
    websocket: {
        url: 'ws://localhost:8050/ws',
        reconnectInterval: 5000,
        maxReconnectAttempts: 10
    },
    update: {
        interval: 1000,
        animationDuration: 300
    },
    thresholds: {
        gpu: {
            warning: 70,
            critical: 90
        },
        latency: {
            warning: 50,
            critical: 100
        },
        queue: {
            warning: 80,
            critical: 95
        }
    }
};

// État global du dashboard
class DashboardState {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.isConnected = false;
        this.lastUpdate = null;
        this.metrics = {
            gpu: {},
            pipeline: {},
            queues: {},
            system: {}
        };
    }

    updateMetric(category, key, value) {
        if (!this.metrics[category]) {
            this.metrics[category] = {};
        }
        this.metrics[category][key] = value;
        this.lastUpdate = new Date();
    }

    getMetric(category, key) {
        return this.metrics[category]?.[key] || null;
    }
}

// Instance globale de l'état
const dashboardState = new DashboardState();

// Gestionnaire de WebSocket
class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectTimer = null;
    }

    connect() {
        try {
            this.ws = new WebSocket(DASHBOARD_CONFIG.websocket.url);
            
            this.ws.onopen = () => {
                console.log('✅ WebSocket connecté');
                dashboardState.isConnected = true;
                dashboardState.reconnectAttempts = 0;
                this.updateConnectionStatus('connected');
                this.clearReconnectTimer();
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('❌ Erreur parsing message WebSocket:', error);
                }
            };

            this.ws.onclose = () => {
                console.log('⚠️ WebSocket fermé');
                dashboardState.isConnected = false;
                this.updateConnectionStatus('disconnected');
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('❌ Erreur WebSocket:', error);
                this.updateConnectionStatus('error');
            };

        } catch (error) {
            console.error('❌ Erreur connexion WebSocket:', error);
            this.scheduleReconnect();
        }
    }

    handleMessage(data) {
        console.log('📨 Message reçu:', data);
        
        // Mise à jour des métriques selon le type de message
        if (data.type === 'gpu_metrics') {
            this.updateGPUMetrics(data.data);
        } else if (data.type === 'pipeline_metrics') {
            this.updatePipelineMetrics(data.data);
        } else if (data.type === 'queue_metrics') {
            this.updateQueueMetrics(data.data);
        } else if (data.type === 'system_health') {
            this.updateSystemHealth(data.data);
        }

        // Mise à jour de l'indicateur de dernière MAJ
        this.updateLastUpdateIndicators();
    }

    updateGPUMetrics(data) {
        // Utilisation GPU avec animation du cercle
        if (data.utilization !== undefined) {
            dashboardState.updateMetric('gpu', 'utilization', data.utilization);
            this.animateGPUUtilization(data.utilization);
            document.getElementById('overview-gpu-util').textContent = `${data.utilization}%`;
        }

        // Mémoire GPU
        if (data.memory_used !== undefined) {
            dashboardState.updateMetric('gpu', 'memory_used', data.memory_used);
            document.getElementById('gpu-memory').textContent = `${data.memory_used.toFixed(1)} MB`;
        }

        if (data.memory_reserved !== undefined) {
            dashboardState.updateMetric('gpu', 'memory_reserved', data.memory_reserved);
            document.getElementById('gpu-memory-reserved').textContent = `${data.memory_reserved.toFixed(1)} MB`;
        }

        // Informations device
        if (data.device) {
            document.getElementById('gpu-device').textContent = data.device;
            document.getElementById('sidebar-gpu-device').textContent = data.device;
        }

        if (data.driver) {
            document.getElementById('gpu-driver').textContent = data.driver;
            document.getElementById('sidebar-gpu-driver').textContent = data.driver;
        }

        if (data.streams !== undefined) {
            document.getElementById('gpu-streams').textContent = data.streams;
            document.getElementById('sidebar-gpu-streams').textContent = data.streams;
        }

        // Performance metrics
        if (data.frames !== undefined) {
            document.getElementById('gpu-frames').textContent = data.frames;
        }

        if (data.avg_latency !== undefined) {
            document.getElementById('gpu-avg-lat').textContent = `${data.avg_latency.toFixed(1)} ms`;
            document.getElementById('overview-latency').textContent = `${data.avg_latency.toFixed(1)} ms`;
        }

        if (data.throughput !== undefined) {
            document.getElementById('gpu-throughput').textContent = `${data.throughput.toFixed(1)} FPS`;
        }

        if (data.breakdown) {
            document.getElementById('gpu-breakdown').textContent = 
                `${data.breakdown.norm}/${data.breakdown.pin}/${data.breakdown.copy}`;
        }
    }

    updatePipelineMetrics(data) {
        // Latences inter-étapes - mise à jour des valeurs individuelles
        if (data.rx_to_gpu !== undefined) {
            document.getElementById('interstage-rx-gpu').textContent = `${data.rx_to_gpu.toFixed(1)} ms`;
        }
        if (data.gpu_to_proc !== undefined) {
            document.getElementById('interstage-gpu-proc').textContent = `${data.gpu_to_proc.toFixed(1)} ms`;
        }
        if (data.proc_to_cpu !== undefined) {
            document.getElementById('interstage-proc-cpu').textContent = `${data.proc_to_cpu.toFixed(1)} ms`;
        }
        if (data.cpu_to_tx !== undefined) {
            document.getElementById('interstage-cpu-tx').textContent = `${data.cpu_to_tx.toFixed(1)} ms`;
        }
        if (data.total_latency !== undefined) {
            document.getElementById('interstage-total').textContent = `${data.total_latency.toFixed(1)} ms`;
        }
        if (data.samples !== undefined) {
            document.getElementById('interstage-samples').textContent = data.samples;
        }

        // Nouvelle barre de progression colorée
        this.updateLatencyBar(data);
    }

    updateLatencyBar(data) {
        // Calcul des proportions pour la barre colorée
        const rxGpu = data.rx_to_gpu || 0;
        const gpuProc = data.gpu_to_proc || 0;
        const procCpu = data.proc_to_cpu || 0;
        const cpuTx = data.cpu_to_tx || 0;
        const total = rxGpu + gpuProc + procCpu + cpuTx;

        if (total > 0) {
            // Calcul des pourcentages
            const rxGpuPercent = (rxGpu / total) * 100;
            const gpuProcPercent = (gpuProc / total) * 100;
            const procCpuPercent = (procCpu / total) * 100;
            const cpuTxPercent = (cpuTx / total) * 100;

            // Mise à jour des segments de la barre
            const segmentRxGpu = document.getElementById('segment-rx-gpu');
            const segmentGpuProc = document.getElementById('segment-gpu-proc');
            const segmentProcCpu = document.getElementById('segment-proc-cpu');
            const segmentCpuTx = document.getElementById('segment-cpu-tx');

            if (segmentRxGpu) {
                segmentRxGpu.style.width = `${Math.max(rxGpuPercent, 2)}%`;
                segmentRxGpu.title = `RX→GPU: ${rxGpu.toFixed(1)}ms (${rxGpuPercent.toFixed(1)}%)`;
            }
            if (segmentGpuProc) {
                segmentGpuProc.style.width = `${Math.max(gpuProcPercent, 2)}%`;
                segmentGpuProc.title = `GPU→PROC: ${gpuProc.toFixed(1)}ms (${gpuProcPercent.toFixed(1)}%)`;
            }
            if (segmentProcCpu) {
                segmentProcCpu.style.width = `${Math.max(procCpuPercent, 2)}%`;
                segmentProcCpu.title = `PROC→CPU: ${procCpu.toFixed(1)}ms (${procCpuPercent.toFixed(1)}%)`;
            }
            if (segmentCpuTx) {
                segmentCpuTx.style.width = `${Math.max(cpuTxPercent, 2)}%`;
                segmentCpuTx.title = `CPU→TX: ${cpuTx.toFixed(1)}ms (${cpuTxPercent.toFixed(1)}%)`;
            }

            // Animation de mise à jour
            const latencyBar = document.getElementById('latency-bar');
            if (latencyBar) {
                latencyBar.classList.add('updating');
                setTimeout(() => latencyBar.classList.remove('updating'), 1000);
            }
        }
    }

    updateQueueMetrics(data) {
        // Files d'attente avec barres animées
        if (data.queue_rt !== undefined) {
            document.getElementById('queue-rt').textContent = data.queue_rt;
            this.animateQueueBar('queue-rt-bar', data.queue_rt, 10);
        }
        
        if (data.queue_gpu !== undefined) {
            document.getElementById('queue-gpu').textContent = data.queue_gpu;
            this.animateQueueBar('queue-gpu-bar', data.queue_gpu, 10);
        }
        
        if (data.drops !== undefined) {
            document.getElementById('queue-drops').textContent = data.drops;
        }

        // Mise à jour overview queue
        const totalQueue = (data.queue_rt || 0) + (data.queue_gpu || 0);
        document.getElementById('overview-queue').textContent = totalQueue;
    }

    updateSystemHealth(data) {
        if (data.status) {
            const healthElement = document.getElementById('system-health');
            const statusMap = {
                'operational': 'Opérationnel',
                'warning': 'Attention',
                'critical': 'Critique',
                'unknown': 'Inconnu'
            };
            healthElement.textContent = statusMap[data.status] || data.status;
            healthElement.className = `overview-value status-${data.status}`;
        }
    }

    animateGPUUtilization(percentage) {
        // Animation du cercle de progression
        const circle = document.getElementById('gpu-util-circle');
        const utilValue = document.getElementById('gpu-util');
        
        if (circle && utilValue) {
            const circumference = 2 * Math.PI * 15.9155;
            const dashArray = (percentage / 100) * circumference;
            
            circle.style.transition = 'stroke-dasharray 0.5s ease-out';
            circle.style.strokeDasharray = `${dashArray}, ${circumference}`;
            
            // Animation du texte
            utilValue.style.transition = 'color 0.3s ease';
            utilValue.textContent = Math.round(percentage);
            
            // Couleur dynamique selon seuils
            let color = 'var(--accent-color)';
            if (percentage > DASHBOARD_CONFIG.thresholds.gpu.critical) {
                color = 'var(--error-color)';
            } else if (percentage > DASHBOARD_CONFIG.thresholds.gpu.warning) {
                color = 'var(--warning-color)';
            }
            
            circle.style.stroke = color;
            utilValue.style.color = color;
        }
    }

    animateQueueBar(barId, value, maxValue) {
        const bar = document.getElementById(barId);
        if (!bar) return;
        
        const fill = bar.querySelector('.queue-fill');
        if (!fill) return;
        
        const percentage = Math.min((value / maxValue) * 100, 100);
        fill.style.transition = 'width 0.4s ease-out, background-color 0.3s ease';
        fill.style.width = `${percentage}%`;
        
        // Couleur selon le niveau
        let color = 'var(--accent-color)';
        if (percentage > DASHBOARD_CONFIG.thresholds.queue.critical) {
            color = 'var(--error-color)';
        } else if (percentage > DASHBOARD_CONFIG.thresholds.queue.warning) {
            color = 'var(--warning-color)';
        }
        fill.style.backgroundColor = color;
    }

    updateConnectionStatus(status) {
        const statusDot = document.getElementById('connection-status');
        const statusText = document.getElementById('footer-status-text');
        const statusIcon = document.getElementById('footer-status-icon');
        
        const statusConfig = {
            connected: {
                class: 'status-connected',
                text: 'Connecté',
                icon: 'fas fa-circle text-success'
            },
            disconnected: {
                class: 'status-disconnected',
                text: 'Déconnecté',
                icon: 'fas fa-circle text-error'
            },
            error: {
                class: 'status-error',
                text: 'Erreur',
                icon: 'fas fa-circle text-error'
            }
        };
        
        const config = statusConfig[status] || statusConfig.error;
        
        if (statusDot) {
            statusDot.className = `status-dot ${config.class}`;
        }
        if (statusText) {
            statusText.textContent = config.text;
        }
        if (statusIcon) {
            statusIcon.className = config.icon;
        }
    }

    updateLastUpdateIndicators() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('fr-FR', { 
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        
        // Mise à jour des timestamps
        const elements = ['gpu-last-update', 'sidebar-last-update'];
        elements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = '0 s';
            }
        });
    }

    scheduleReconnect() {
        if (dashboardState.reconnectAttempts >= DASHBOARD_CONFIG.websocket.maxReconnectAttempts) {
            console.error('❌ Nombre maximum de tentatives de reconnexion atteint');
            return;
        }

        dashboardState.reconnectAttempts++;
        console.log(`🔄 Tentative de reconnexion ${dashboardState.reconnectAttempts}/${DASHBOARD_CONFIG.websocket.maxReconnectAttempts} dans ${DASHBOARD_CONFIG.websocket.reconnectInterval}ms`);
        
        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, DASHBOARD_CONFIG.websocket.reconnectInterval);
    }

    clearReconnectTimer() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }

    disconnect() {
        this.clearReconnectTimer();
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Gestionnaire d'animations et interactions
class UIManager {
    constructor() {
        this.initializeAnimations();
        this.initializeInteractions();
    }

    initializeAnimations() {
        // Animation d'entrée progressive des cartes
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    setTimeout(() => {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }, index * 100);
                }
            });
        });

        document.querySelectorAll('.card').forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            card.style.transition = 'all 0.6s ease-out';
            observer.observe(card);
        });
    }

    initializeInteractions() {
        // Boutons d'action
        const exportBtn = document.querySelector('.btn[title="Exporter les données"]');
        if (exportBtn) {
            exportBtn.addEventListener('click', this.handleExport.bind(this));
        }

        const settingsBtn = document.querySelector('.btn[title="Paramètres du dashboard"]');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', this.handleSettings.bind(this));
        }

        // Hover effects sur les cartes métriques
        document.querySelectorAll('.metric-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-2px)';
                card.style.boxShadow = 'var(--shadow-md)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0)';
                card.style.boxShadow = 'none';
            });
        });
    }

    handleExport() {
        console.log('📤 Export des données demandé');
        // TODO: Implémenter l'export des métriques
        const data = {
            timestamp: new Date().toISOString(),
            metrics: dashboardState.metrics
        };
        
        // Simulation d'export (à remplacer par vraie logique)
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ultramotion-metrics-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    handleSettings() {
        console.log('⚙️ Ouverture des paramètres');
        // TODO: Implémenter modal de paramètres
        alert('Paramètres - Fonctionnalité en développement');
    }
}

// Gestionnaire de temps
class TimeManager {
    constructor() {
        this.startClock();
    }

    startClock() {
        this.updateTime();
        setInterval(() => this.updateTime(), 1000);
    }

    updateTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('fr-FR', { 
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        
        const elements = ['current-time', 'footer-timestamp'];
        elements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = timeString;
            }
        });
    }
}

// Initialisation principale
class Dashboard {
    constructor() {
        this.wsManager = null;
        this.uiManager = null;
        this.timeManager = null;
    }

    async init() {
        console.log('🚀 Initialisation du Dashboard UltraMotion IGT');
        
        // Attendre que le DOM soit prêt
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.start());
        } else {
            this.start();
        }
    }

    start() {
        console.log('✨ Démarrage du dashboard');
        
        // Initialisation des managers
        this.timeManager = new TimeManager();
        this.uiManager = new UIManager();
        this.wsManager = new WebSocketManager();
        
        // Connexion WebSocket
        this.wsManager.connect();
        
        // Gestion de la fermeture de la page
        window.addEventListener('beforeunload', () => {
            if (this.wsManager) {
                this.wsManager.disconnect();
            }
        });
        
        console.log('✅ Dashboard initialisé avec succès');
    }
}

// Instance globale et démarrage
const dashboard = new Dashboard();
dashboard.init();

// Export pour utilisation externe
window.UltraMotionDashboard = {
    dashboard,
    state: dashboardState,
    config: DASHBOARD_CONFIG
};

