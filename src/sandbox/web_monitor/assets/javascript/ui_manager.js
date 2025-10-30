/**
 * ui_manager.js - UI Manager
 * ===========================
 * Gestionnaire d'interface utilisateur pour UltraMotion IGT Dashboard
 * 
 * Responsabilités:
 * - Mise à jour des éléments DOM
 * - Animations et transitions
 * - Gestion des seuils et alertes
 * - Interactions utilisateur
 */

class UIManager {
    constructor(config = {}) {
        this.config = {
            thresholds: {
                gpu: { warning: 70, critical: 90 },
                latency: { warning: 50, critical: 100 },
                queue: { warning: 80, critical: 95 }
            },
            animationDuration: 300,
            ...config
        };
        
        this.lastUpdate = null;
        this.init();
    }

    /**
     * Initialise le gestionnaire UI
     */
    init() {
        this.initializeAnimations();
        this.initializeInteractions();
        console.log('✨ UIManager initialisé');
    }

    /**
     * Met à jour les métriques GPU
     */
    updateGPUMetrics(data) {
        // Utilisation GPU avec cercle de progression
        if (data.utilization !== undefined) {
            this.animateGPUUtilization(data.utilization);
            this.updateElement('overview-gpu-util', `${data.utilization}%`);
            this.updateElement('gpu-util', Math.round(data.utilization));
        }

        // Mémoire GPU
        if (data.memory_used !== undefined) {
            this.updateElement('gpu-memory', `${data.memory_used.toFixed(1)} MB`);
        }

        if (data.memory_reserved !== undefined) {
            this.updateElement('gpu-memory-reserved', `${data.memory_reserved.toFixed(1)} MB`);
        }

        // Informations device
        if (data.device) {
            this.updateElement('gpu-device', data.device);
            this.updateElement('sidebar-gpu-device', data.device);
        }

        if (data.driver) {
            this.updateElement('gpu-driver', data.driver);
            this.updateElement('sidebar-gpu-driver', data.driver);
        }

        if (data.streams !== undefined) {
            this.updateElement('gpu-streams', data.streams);
            this.updateElement('sidebar-gpu-streams', data.streams);
        }

        // Performance metrics
        if (data.frames !== undefined) {
            this.updateElement('gpu-frames', data.frames);
        }

        if (data.avg_latency !== undefined) {
            this.updateElement('gpu-avg-lat', `${data.avg_latency.toFixed(1)} ms`);
            this.updateElement('overview-latency', `${data.avg_latency.toFixed(1)} ms`);
        }

        if (data.throughput !== undefined) {
            this.updateElement('gpu-throughput', `${data.throughput.toFixed(1)} FPS`);
        }

        if (data.breakdown) {
            this.updateElement('gpu-breakdown', 
                `${data.breakdown.norm}/${data.breakdown.pin}/${data.breakdown.copy}`);
        }
    }

    /**
     * Met à jour les métriques de pipeline
     */
    updatePipelineMetrics(data) {
        // Valeurs individuelles
        if (data.rx_to_gpu !== undefined) {
            this.updateElement('interstage-rx-gpu', `${data.rx_to_gpu.toFixed(1)} ms`);
        }
        if (data.gpu_to_proc !== undefined) {
            this.updateElement('interstage-gpu-proc', `${data.gpu_to_proc.toFixed(1)} ms`);
        }
        if (data.proc_to_cpu !== undefined) {
            this.updateElement('interstage-proc-cpu', `${data.proc_to_cpu.toFixed(1)} ms`);
        }
        if (data.cpu_to_tx !== undefined) {
            this.updateElement('interstage-cpu-tx', `${data.cpu_to_tx.toFixed(1)} ms`);
        }
        if (data.total_latency !== undefined) {
            this.updateElement('interstage-total', `${data.total_latency.toFixed(1)} ms`);
        }
        if (data.samples !== undefined) {
            this.updateElement('interstage-samples', data.samples);
        }

        // Barre de progression colorée
        this.updateLatencyBar(data);
    }

    /**
     * Met à jour la barre de latence colorée
     */
    updateLatencyBar(data) {
        const rxCpu = data.rx_cpu || 0;
        const cpuGpu = data.cpu_gpu || 0;
        const procGpu = data.proc_gpu || 0;
        const gpuCpu = data.gpu_cpu || 0;
        const cpuTx = data.cpu_tx || 0;
        const total = rxCpu + cpuGpu + procGpu + gpuCpu + cpuTx;

        if (total > 0) {
            // Calcul des pourcentages
            const rxCpuPercent = (rxCpu / total) * 100;
            const cpuGpuPercent = (cpuGpu / total) * 100;
            const procGpuPercent = (procGpu / total) * 100;
            const gpuCpuPercent = (gpuCpu / total) * 100;
            const cpuTxPercent = (cpuTx / total) * 100;

            // Mise à jour des segments
            this.updateSegment('segment-rx-cpu', rxCpuPercent, `RX→CPU: ${rxCpu.toFixed(1)}ms`);
            this.updateSegment('segment-cpu-gpu', cpuGpuPercent, `CPU→GPU: ${cpuGpu.toFixed(1)}ms`);
            this.updateSegment('segment-proc-gpu', procGpuPercent, `PROC(GPU): ${procGpu.toFixed(1)}ms`);
            this.updateSegment('segment-gpu-cpu', gpuCpuPercent, `GPU→CPU: ${gpuCpu.toFixed(1)}ms`);
            this.updateSegment('segment-cpu-tx', cpuTxPercent, `CPU→TX: ${cpuTx.toFixed(1)}ms`);

            // Animation de mise à jour
            this.animateLatencyBar();
        }
    }

    /**
     * Met à jour un segment de la barre de latence
     */
    updateSegment(elementId, percentage, tooltip) {
        const element = document.getElementById(elementId);
        if (element) {
            element.style.width = `${Math.max(percentage, 2)}%`;
            element.title = tooltip;
        }
    }

    /**
     * Met à jour les métriques de files d'attente
     */
    updateQueueMetrics(data) {
        if (data.queue_rt !== undefined) {
            this.updateElement('queue-rt', data.queue_rt);
            this.animateQueueBar('queue-rt-bar', data.queue_rt, 10);
        }
        
        if (data.queue_gpu !== undefined) {
            this.updateElement('queue-gpu', data.queue_gpu);
            this.animateQueueBar('queue-gpu-bar', data.queue_gpu, 10);
        }
        
        if (data.drops !== undefined) {
            this.updateElement('queue-drops', data.drops);
        }

        // Mise à jour overview queue
        const totalQueue = (data.queue_rt || 0) + (data.queue_gpu || 0);
        this.updateElement('overview-queue', totalQueue);
    }

    /**
     * Met à jour l'état de santé du système
     */
    updateSystemHealth(data) {
        if (data.status) {
            const healthElement = document.getElementById('system-health');
            const statusMap = {
                'operational': 'Opérationnel',
                'warning': 'Attention',
                'critical': 'Critique',
                'unknown': 'Inconnu'
            };
            
            if (healthElement) {
                healthElement.textContent = statusMap[data.status] || data.status;
                healthElement.className = `overview-value status-${data.status}`;
            }
        }
    }

    /**
     * Anime le cercle d'utilisation GPU
     */
    animateGPUUtilization(percentage) {
        const circle = document.getElementById('gpu-util-circle');
        if (!circle) return;
        
        const circumference = 2 * Math.PI * 15.9155;
        const dashArray = (percentage / 100) * circumference;
        
        circle.style.transition = 'stroke-dasharray 0.5s ease-out';
        circle.style.strokeDasharray = `${dashArray}, ${circumference}`;
        
        // Couleur dynamique selon seuils
        let color = 'var(--primary-color)';
        if (percentage > this.config.thresholds.gpu.critical) {
            color = 'var(--error-color)';
        } else if (percentage > this.config.thresholds.gpu.warning) {
            color = 'var(--warning-color)';
        }
        
        circle.style.stroke = color;
    }

    /**
     * Anime les barres de files d'attente
     */
    animateQueueBar(barId, value, maxValue) {
        const bar = document.getElementById(barId);
        if (!bar) return;
        
        const fill = bar.querySelector('.queue-fill');
        if (!fill) return;
        
        const percentage = Math.min((value / maxValue) * 100, 100);
        fill.style.transition = 'width 0.4s ease-out, background-color 0.3s ease';
        fill.style.width = `${percentage}%`;
        
        // Couleur selon le niveau
        let color = 'var(--primary-color)';
        if (percentage > this.config.thresholds.queue.critical) {
            color = 'var(--error-color)';
        } else if (percentage > this.config.thresholds.queue.warning) {
            color = 'var(--warning-color)';
        }
        fill.style.backgroundColor = color;
    }

    /**
     * Anime la barre de latence lors de la mise à jour
     */
    animateLatencyBar() {
        const latencyBar = document.getElementById('latency-bar');
        if (latencyBar) {
            latencyBar.classList.add('updating');
            setTimeout(() => latencyBar.classList.remove('updating'), 1000);
        }
    }

    /**
     * Initialise les animations d'entrée
     */
    initializeAnimations() {
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

    /**
     * Initialise les interactions utilisateur
     */
    initializeInteractions() {
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

    /**
     * Met à jour un élément du DOM
     */
    updateElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
    }

    /**
     * Met à jour les indicateurs de statut de connexion
     */
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

    /**
     * Marque le dernier temps de mise à jour
     */
    markLastUpdate() {
        this.lastUpdate = new Date();
        const elements = ['gpu-last-update', 'sidebar-last-update'];
        elements.forEach(id => {
            this.updateElement(id, '0 s');
        });
    }

    /**
     * Met à jour les métriques système globales (overview + GPU)
     * Appelée à chaque message `system_metrics` reçu depuis le backend.
     */
    updateSystemMetrics(data) {
        if (!data) return;

        // ===========================
        //  🧩 GPU - Utilisation
        // ===========================
        if (data.gpu) {
            const gpu = data.gpu;

            // Utilisation principale (%)
            if (gpu.usage !== undefined) {
                this.updateElement('overview-gpu-util', `${gpu.usage.toFixed(1)}%`);
                this.updateElement('gpu-util', Math.round(gpu.usage));
                this.animateGPUUtilization(gpu.usage);
            }

            // Température
            if (gpu.temp !== undefined) {
                this.updateElement('gpu-temperature', `${gpu.temp.toFixed(1)}°C`);
            }

            // Mémoire VRAM (utilisée)
            if (gpu.vram_used !== undefined) {
                this.updateElement('gpu-memory-usage', `${gpu.vram_used.toFixed(0)} MB`);
            }

            // Streams actifs
            if (gpu.streams !== undefined) {
                this.updateElement('gpu-streams', gpu.streams);
                this.updateElement('sidebar-gpu-streams', gpu.streams);
            }

            // Nom du device
            if (gpu.device) {
                this.updateElement('gpu-device', gpu.device);
                this.updateElement('sidebar-gpu-device', gpu.device);
            }

            // Driver
            if (gpu.driver) {
                this.updateElement('gpu-driver', gpu.driver);
                this.updateElement('sidebar-gpu-driver', gpu.driver);
            }
        }

        // ===========================
        //  ⚙️ CPU / FPS / Pipeline
        // ===========================
        if (data.fps) {
            const fps = data.fps;
            // Latence moyenne (proc)
            if (fps.proc !== undefined) {
                this.updateElement('overview-latency', `${fps.proc.toFixed(1)} ms`);
            }
        }

        if (data.cpu) {
            // on pourrait ajouter des updates CPU ici plus tard
        }

        // ===========================
        //  🧱 Queues / Latence globale
        // ===========================
        if (data.queue) {
            const q = data.queue;
            this.updateElement('overview-queue', q.total || 0);
        }

        // ===========================
        //  🔄 Latences Inter-étapes (Pipeline GPU-Résident)
        // ===========================
        if (data.interstage) {
            const interstage = data.interstage;
            
            // RX → CPU
            if (interstage.rx_cpu !== undefined) {
                this.updateElement('lat-rx-cpu', `${interstage.rx_cpu.toFixed(1)} ms`);
            }
            
            // CPU → GPU
            if (interstage.cpu_gpu !== undefined) {
                this.updateElement('lat-cpu-gpu', `${interstage.cpu_gpu.toFixed(1)} ms`);
            }
            
            // PROC(GPU)
            if (interstage.proc_gpu !== undefined) {
                this.updateElement('lat-proc-gpu', `${interstage.proc_gpu.toFixed(1)} ms`);
            }
            
            // GPU → CPU
            if (interstage.gpu_cpu !== undefined) {
                this.updateElement('lat-gpu-cpu', `${interstage.gpu_cpu.toFixed(1)} ms`);
            }
            
            // CPU → TX
            if (interstage.cpu_tx !== undefined) {
                this.updateElement('lat-cpu-tx', `${interstage.cpu_tx.toFixed(1)} ms`);
            }
            
            // Total RX→TX
            if (interstage.total !== undefined) {
                this.updateElement('interstage-total', `${interstage.total.toFixed(1)} ms`);
            }
            
            // Frame number
            if (interstage.frame_id !== undefined) {
                this.updateElement('interstage-frame-id', interstage.frame_id);
            }

            // Mise à jour de la barre de latence colorée
            this.updateLatencyBar(interstage);
        }

        // ===========================
        //  ❤️ Santé système
        // ===========================
        if (data.status) {
            this.updateSystemHealth({ status: data.status });
        }

        // ===========================
        //  ⏱️ Dernière mise à jour
        // ===========================
        this.markLastUpdate();
    }




}

// Export pour utilisation globale
window.UIManager = UIManager;