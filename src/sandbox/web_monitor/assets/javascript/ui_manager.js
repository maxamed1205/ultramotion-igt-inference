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
        this.diagnosticDOM(); // Ajout du diagnostic
        console.log('✨ UIManager initialisé');
    }

    /**
     * Diagnostic de la structure DOM
     */
    diagnosticDOM() {
        console.log('🔧 [DIAGNOSTIC] Vérification de la structure DOM...');
        
        // Vérification de la barre de latence principale
        const latencyBar = document.getElementById('latency-bar');
        console.log('🔧 [DIAGNOSTIC] Barre de latence:', latencyBar ? 'trouvée' : 'NON TROUVÉE');
        
        // Vérification de chaque segment
        const segmentIds = [
            'segment-rx-cpu',
            'segment-cpu-gpu', 
            'segment-proc-gpu',
            'segment-gpu-cpu',
            'segment-cpu-tx'
        ];
        
        segmentIds.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                const computedStyle = getComputedStyle(element);
                console.log(`🔧 [DIAGNOSTIC] ${id}:`, {
                    trouvé: 'OUI',
                    classes: element.className,
                    styleWidth: element.style.width,
                    computedWidth: computedStyle.width,
                    display: computedStyle.display,
                    backgroundColor: computedStyle.backgroundColor,
                    visibility: computedStyle.visibility
                });
            } else {
                console.error(`❌ [DIAGNOSTIC] ${id}: NON TROUVÉ`);
            }
        });
        
        // Vérification du parent container
        const container = document.querySelector('.c-latency-bar-container');
        console.log('🔧 [DIAGNOSTIC] Container:', container ? 'trouvé' : 'NON TROUVÉ');
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
        console.log('🔄 [DEBUG] updatePipelineMetrics appelée avec:', JSON.stringify(data, null, 2));
        
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

        // Vérification des clés de données pour la barre
        console.log('🔍 [DEBUG] Clés de données disponibles:', Object.keys(data));
        console.log('🔍 [DEBUG] Valeurs de pipeline détectées:', {
            rx_cpu: data.rx_cpu,
            cpu_gpu: data.cpu_gpu,
            proc_gpu: data.proc_gpu,
            gpu_cpu: data.gpu_cpu,
            cpu_tx: data.cpu_tx,
            total: data.total
        });

        // Barre de progression colorée
        this.updateLatencyBar(data);
    }

    updateLatencyBar(data) {
        if (!data) return;
        console.log('📊 [DEBUG] updateLatencyBar appelée avec:', data);

        // Extraction avec sécurité
        const rxCpu  = parseFloat(data.rx_cpu  || 0);
        const cpuGpu = parseFloat(data.cpu_gpu || 0);
        const procGpu= parseFloat(data.proc_gpu|| 0);
        const gpuCpu = parseFloat(data.gpu_cpu || 0);
        const cpuTx  = parseFloat(data.cpu_tx  || 0);

        // Debug des valeurs extraites
        console.log('📊 [DEBUG] Valeurs extraites:', {
            rxCpu, cpuGpu, procGpu, gpuCpu, cpuTx
        });

        // Utiliser le total transmis si présent, sinon recalculer
        const total = parseFloat(data.total || (rxCpu + cpuGpu + procGpu + gpuCpu + cpuTx));
        console.log('📊 [DEBUG] Total calculé:', total);
        
        if (total <= 0) {
            console.warn('⚠️ [DEBUG] Total <= 0, pas de mise à jour');
            return;
        }

        // Calcul des pourcentages exacts
        const ratios = {
            rx_cpu:  (rxCpu  / total) * 100,
            cpu_gpu: (cpuGpu / total) * 100,
            proc_gpu:(procGpu/ total) * 100,
            gpu_cpu: (gpuCpu/ total) * 100,
            cpu_tx:  (cpuTx  / total) * 100
        };

        console.log('📊 [DEBUG] Pourcentages calculés:', ratios);

        // Mise à jour DOM dans un ordre stable
        const segments = [
            ['segment-rx-cpu',  ratios.rx_cpu,  `RX→CPU: ${rxCpu.toFixed(1)}ms`],
            ['segment-cpu-gpu', ratios.cpu_gpu, `CPU→GPU: ${cpuGpu.toFixed(1)}ms`],
            ['segment-proc-gpu',ratios.proc_gpu,`PROC(GPU): ${procGpu.toFixed(1)}ms`],
            ['segment-gpu-cpu', ratios.gpu_cpu, `GPU→CPU: ${gpuCpu.toFixed(1)}ms`],
            ['segment-cpu-tx',  ratios.cpu_tx,  `CPU→TX: ${cpuTx.toFixed(1)}ms`],
        ];

        console.log('📊 [DEBUG] Configuration des segments:', segments);

        segments.forEach(([id, pct, tooltip]) => {
            const el = document.getElementById(id);
            console.log(`📊 [DEBUG] Traitement segment ${id}:`, {
                element: el ? 'trouvé' : 'NON TROUVÉ',
                percentage: pct,
                tooltip: tooltip
            });
            
            if (el) {
                const w = Math.max(pct, 1.5); // min 1.5% visible
                console.log(`📊 [DEBUG] Définition largeur ${id}: ${w}%`);
                el.style.width = `${w}%`;
                el.title = tooltip;
                el.style.flexGrow = 0; // évite les déformations
                
                // Debug des styles appliqués
                const computedStyles = getComputedStyle(el);
                console.log(`📊 [DEBUG] Styles finaux ${id}:`, {
                    width: el.style.width,
                    display: computedStyles.display,
                    background: computedStyles.background,
                    backgroundColor: computedStyles.backgroundColor,
                    visibility: computedStyles.visibility,
                    borderColor: computedStyles.borderColor
                });
            } else {
                console.error(`❌ [DEBUG] Élément ${id} non trouvé dans le DOM!`);
            }
        });

        // Ajoute une légère animation de rafraîchissement
        const latencyBar = document.getElementById('latency-bar');
        if (latencyBar) {
            latencyBar.classList.add('updating');
            setTimeout(() => latencyBar.classList.remove('updating'), 800);
            console.log('📊 [DEBUG] Animation de rafraîchissement ajoutée');
        } else {
            console.error('❌ [DEBUG] Barre de latence non trouvée!');
        }
    }


    /**
     * Met à jour un segment de la barre de latence
     */
    updateSegment(elementId, percentage, tooltip) {
        console.log(`📊 [DEBUG] updateSegment: ${elementId} = ${percentage.toFixed(1)}% (${tooltip})`);
        const element = document.getElementById(elementId);
        if (element) {
            const finalPercentage = Math.max(percentage, 2); // Minimum 2% pour visibilité
            element.style.width = `${finalPercentage}%`;
            element.title = tooltip;
            console.log(`📊 [DEBUG] Segment ${elementId} mis à jour: ${finalPercentage}%`);
        } else {
            console.warn(`⚠️ [DEBUG] Élément de segment non trouvé: ${elementId}`);
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
            console.log(`🔧 [DEBUG] updateElement: ${elementId} = ${value}`);
            element.textContent = value;
        } else {
            console.warn(`⚠️ [DEBUG] Élément non trouvé: ${elementId}`);
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
        console.log('🎨 [DEBUG] updateSystemMetrics appelée avec:', data);
        if (!data) {
            console.log('⚠️ [DEBUG] Données vides dans updateSystemMetrics');
            return;
        }

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
            console.log('🔧 [DEBUG] Traitement des données interstage:', data.interstage);
            const interstage = data.interstage;
            
            // RX → CPU
            if (interstage.rx_cpu !== undefined) {
                console.log(`🔧 [DEBUG] Mise à jour lat-rx-cpu: ${interstage.rx_cpu.toFixed(1)} ms`);
                this.updateElement('lat-rx-cpu', `${interstage.rx_cpu.toFixed(1)} ms`);
            }
            
            // CPU → GPU
            if (interstage.cpu_gpu !== undefined) {
                console.log(`🔧 [DEBUG] Mise à jour lat-cpu-gpu: ${interstage.cpu_gpu.toFixed(1)} ms`);
                this.updateElement('lat-cpu-gpu', `${interstage.cpu_gpu.toFixed(1)} ms`);
            }
            
            // PROC(GPU)
            if (interstage.proc_gpu !== undefined) {
                console.log(`🔧 [DEBUG] Mise à jour lat-proc-gpu: ${interstage.proc_gpu.toFixed(1)} ms`);
                this.updateElement('lat-proc-gpu', `${interstage.proc_gpu.toFixed(1)} ms`);
            }
            
            // GPU → CPU
            if (interstage.gpu_cpu !== undefined) {
                console.log(`🔧 [DEBUG] Mise à jour lat-gpu-cpu: ${interstage.gpu_cpu.toFixed(1)} ms`);
                this.updateElement('lat-gpu-cpu', `${interstage.gpu_cpu.toFixed(1)} ms`);
            }
            
            // CPU → TX
            if (interstage.cpu_tx !== undefined) {
                console.log(`🔧 [DEBUG] Mise à jour lat-cpu-tx: ${interstage.cpu_tx.toFixed(1)} ms`);
                this.updateElement('lat-cpu-tx', `${interstage.cpu_tx.toFixed(1)} ms`);
            }
            
            // Total RX→TX
            if (interstage.total !== undefined) {
                console.log(`🔧 [DEBUG] Mise à jour interstage-total: ${interstage.total.toFixed(1)} ms`);
                this.updateElement('interstage-total', `${interstage.total.toFixed(1)} ms`);
            }
            
            // Frame number
            if (interstage.frame_id !== undefined) {
                console.log(`🔧 [DEBUG] Mise à jour interstage-frame-id: ${interstage.frame_id}`);
                this.updateElement('interstage-frame-id', interstage.frame_id);
            }

            // Mise à jour de la barre de latence colorée
            this.updateLatencyBar(interstage);
        } else {
            console.log('⚠️ [DEBUG] Aucune données interstage trouvées dans:', data);
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