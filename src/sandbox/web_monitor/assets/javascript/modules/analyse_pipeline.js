// analyse_pipeline.js
// Analyse Pipeline (page /analyse-pipeline)
// Affiche 2 graphiques :
//  1. GPU Transfer (Norm / Pin / Copy) -> barres empilées par frame
//  2. Pipeline GPU-Résident (RX → CPU→GPU → PROC(GPU) → GPU→CPU → TX) -> lignes par étape

class AnalysePipelineModule {
    constructor() {
        // Limite du nombre de frames qu'on garde visibles
        this.maxFrames = 100;

        // Références Plotly (on les stocke après init)
        this.gpuTransferDiv = null;
        this.interstageDiv = null;

        this.isReady = false;
    }

    init() {
        // Sécurité : ne pas double-init
        if (this.isReady) return;

        // Vérif environnement
        if (typeof Plotly === "undefined") {
            console.error("❌ Plotly non chargé");
            return;
        }
        if (!window.wsManager) {
            console.error("❌ wsManager non dispo");
            return;
        }

        // Récupère les divs où on va dessiner
        this.gpuTransferDiv = document.getElementById("chart-gpu-transfer");
        this.interstageDiv = document.getElementById("chart-interstage-latency");

        if (!this.gpuTransferDiv || !this.interstageDiv) {
            console.error("❌ Les conteneurs de graphiques ne sont pas présents dans le DOM");
            return;
        }

        // 1. Créer les graphiques vides
        this.initGpuTransferChart();
        this.initInterstageLatencyChart();

        // 2. Abonnements WebSocket
        this.subscribeWebSocket();

        // 3. Marquer prêt
        this.isReady = true;
        console.log("✅ AnalysePipelineModule initialisé");
    }

    // -- 1. Création des graphiques ---------------------------------

    initGpuTransferChart() {
        // Barres empilées : Norm / Pin / Copy
        const traces = [
            { x: [], y: [], name: "Norm", type: "bar" },
            { x: [], y: [], name: "Pin",  type: "bar" },
            { x: [], y: [], name: "Copy", type: "bar" }
        ];

        const layout = {
            title: "GPU Transfer - Décomposition par Frame (Norm / Pin / Copy)",
            xaxis: { title: "Numéro de Frame" },
            yaxis: { title: "Latence (ms)", rangemode: "tozero" },
            barmode: "stack",
            margin: { t: 40, r: 20, b: 40, l: 50 },
            showlegend: true,
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            font: { color: "#e6edf3" }
        };

        const config = {
            displayModeBar: false,
            responsive: true
        };

        Plotly.newPlot(this.gpuTransferDiv, traces, layout, config);
    }

    initInterstageLatencyChart() {
        // Courbes par étape : RX / CPU→GPU / PROC(GPU) / GPU→CPU / TX
        const traces = [
            { x: [], y: [], name: "RX",         mode: "lines+markers", type: "scatter" },
            { x: [], y: [], name: "CPU→GPU",    mode: "lines+markers", type: "scatter" },
            { x: [], y: [], name: "PROC(GPU)",  mode: "lines+markers", type: "scatter" },
            { x: [], y: [], name: "GPU→CPU",    mode: "lines+markers", type: "scatter" },
            { x: [], y: [], name: "TX",         mode: "lines+markers", type: "scatter" }
        ];

        const layout = {
            title: "🎯 Pipeline GPU-Résident - Latences Inter-Étapes Détaillées",
            xaxis: { title: "Numéro de Frame" },
            yaxis: { title: "Latence (ms)", rangemode: "tozero" },
            margin: { t: 50, r: 20, b: 40, l: 50 },
            showlegend: true,
            annotations: [{
                text: "RX → CPU-to-GPU → PROC(GPU) → GPU-to-CPU → TX",
                showarrow: false,
                xref: "paper", yref: "paper",
                x: 0.5, xanchor: "center",
                y: 1.02, yanchor: "bottom",
                font: { size: 12, color: "#999" }
            }],
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            font: { color: "#e6edf3" }
        };

        const config = {
            displayModeBar: false,
            responsive: true
        };

        Plotly.newPlot(this.interstageDiv, traces, layout, config);
    }

    // -- 2. Abonnement WS -------------------------------------------

    subscribeWebSocket() {
        // On écoute UNIQUEMENT pipeline_metrics
        window.wsManager.on("pipeline_metrics", (payload) => {
            this.handlePipelineFrame(payload);
        });
    }

    // -- 3. Réception d'une frame pipeline --------------------------

    handlePipelineFrame(payload) {
        // Sécurité
        if (!payload) return;
        if (!payload.frame_id) return;
        if (!payload.latencies) return;

        const f = payload.frame_id;
        const t = payload.latencies;

        // exemple t = { norm, pin, copy, rx, cpu2gpu, proc, gpu2cpu, tx }

        // Mettre à jour les deux graphiques
        this.updateGpuTransferChart(f, t);
        this.updateInterstageLatencyChart(f, t);
    }

    // -- 4. Mise à jour des graphiques ------------------------------

    updateGpuTransferChart(frameId, timings) {
        // timings.norm / timings.pin / timings.copy
        const xArr = [[frameId], [frameId], [frameId]];
        const yArr = [
            [timings.norm ?? 0],
            [timings.pin  ?? 0],
            [timings.copy ?? 0]
        ];

        Plotly.extendTraces(
            this.gpuTransferDiv,
            { x: xArr, y: yArr },
            [0, 1, 2],
            this.maxFrames
        );
    }

    updateInterstageLatencyChart(frameId, timings) {
        // RX, CPU→GPU, PROC(GPU), GPU→CPU, TX
        const xArr = [[frameId], [frameId], [frameId], [frameId], [frameId]];
        const yArr = [
            [timings.rx       ?? 0],
            [timings.cpu2gpu  ?? 0],
            [timings.proc     ?? 0],
            [timings.gpu2cpu  ?? 0],
            [timings.tx       ?? 0]
        ];

        Plotly.extendTraces(
            this.interstageDiv,
            { x: xArr, y: yArr },
            [0, 1, 2, 3, 4],
            this.maxFrames
        );
    }
}

// Expose globalement pour debug / réinit manuelle si besoin
window.AnalysePipelineModule = new analysePipelineModule();

// Init automatique quand le DOM est prêt
document.addEventListener("DOMContentLoaded", () => {
    const mainEl = document.querySelector('main[data-page="analyse-pipeline"]');
    if (!mainEl) {
        // On ne lance pas le module si on n'est pas sur la bonne page
        return;
    }
    window.analysePipelineModule.init();
});
