// pipeline_card.js
// ============================================================================
// Carte : "GPU-Résident — Latence totale"
// Affiche les latences inter-étapes de la dernière frame (interstage)
// Se met à jour en temps réel via WebSocketManager (event: "pipeline_snapshot")
// ============================================================================

class PipelineCard {
    constructor() {
        this.initialized = false;
    }

    init() {
        if (this.initialized) return;

        // Récupère les éléments DOM
        this.ids = {
            total: document.getElementById("interstage-total"),
            frame: document.getElementById("interstage-frame-id"),
            rx_cpu: document.getElementById("lat-rx-cpu"),
            cpu_gpu: document.getElementById("lat-cpu-gpu"),
            proc_gpu: document.getElementById("lat-proc-gpu"),
            gpu_cpu: document.getElementById("lat-gpu-cpu"),
            cpu_tx: document.getElementById("lat-cpu-tx"),
            bars: {
                rx_cpu: document.getElementById("segment-rx-cpu"),
                cpu_gpu: document.getElementById("segment-cpu-gpu"),
                proc_gpu: document.getElementById("segment-proc-gpu"),
                gpu_cpu: document.getElementById("segment-gpu-cpu"),
                cpu_tx: document.getElementById("segment-cpu-tx"),
            },
        };

        // Vérifie la présence du gestionnaire WebSocket
        if (!window.wsManager) {
            console.warn("⚠️ WebSocketManager non initialisé — pipeline_card.js en attente");
            return;
        }

        // Abonnement à l'événement pipeline_snapshot
        window.wsManager.on("pipeline_snapshot", (snapshot) => {
            this.update(snapshot);
        });

        this.initialized = true;
        console.log("✅ PipelineCard initialisée et en attente de données WS");
    }

    // ---------------------------------------------------------------------
    // Met à jour la carte avec le dernier snapshot reçu
    // ---------------------------------------------------------------------
    update(snapshot) {
        if (!snapshot || !snapshot.latest) return;

        const latest = snapshot.latest;
        const lat = latest.interstage || {};
        const total = lat.total || 0;

        // 🧩 Mise à jour des valeurs textuelles
        if (this.ids.frame) this.ids.frame.textContent = latest.frame_id ?? "-";
        if (this.ids.total) this.ids.total.textContent = `${total.toFixed(2)} ms`;

        const keys = ["rx_cpu", "cpu_gpu", "proc_gpu", "gpu_cpu", "cpu_tx"];
        for (const k of keys) {
            const val = lat[k] ?? 0;
            if (this.ids[k]) this.ids[k].textContent = `${val.toFixed(2)} ms`;
        }

        // 🎨 Mise à jour des largeurs de segments proportionnelles
        this.updateBarWidths(lat, total);
    }

    // ---------------------------------------------------------------------
    // Ajuste la largeur (%) des segments de latence
    // ---------------------------------------------------------------------
    updateBarWidths(lat, total) {
        if (total <= 0) return;

        const keys = ["rx_cpu", "cpu_gpu", "proc_gpu", "gpu_cpu", "cpu_tx"];
        for (const k of keys) {
            const v = lat[k] ?? 0;
            const pct = Math.min((v / total) * 100, 100);
            const el = this.ids.bars[k];
            if (el) el.style.width = `${pct.toFixed(1)}%`;
        }
    }
}

// -------------------------------------------------------------------------
// Enregistrement global
// -------------------------------------------------------------------------
window.PipelineCard = new PipelineCard();

// Lancement auto au chargement du DOM
document.addEventListener("DOMContentLoaded", () => {
    const card = document.querySelector("#latency-bar");
    if (card) window.PipelineCard.init();
});
