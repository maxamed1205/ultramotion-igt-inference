# 📊 SYNTHÈSE ARCHITECTURE ULTRAMOTION IGT (Épurée)

**Date :** 29 octobre 2025  
**Version :** Post-nettoyage (fichiers deprecated supprimés)

---

## 🏗️ Architecture Globale (A→B→C→D)

```
┌──────────────────────────────────────────────────────────────┐
│                   PIPELINE TEMPS RÉEL                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  A. ACQUISITION (RX)                                         │
│     ├─ IGTGateway._rx_thread                                │
│     ├─ Réception PlusServer (IGTLink)                       │
│     └─ Buffer: _mailbox (AdaptiveDeque maxlen=2)           │
│                                                               │
│  B. PREPROCESSING (CPU→GPU)                                  │
│     ├─ core/preprocessing/cpu_to_gpu.py                     │
│     ├─ Transfert asynchrone (pinned memory)                 │
│     └─ Buffer: Queue_GPU (queue.Queue)                      │
│                                                               │
│  C. INFERENCE (GPU)                                          │
│     ├─ core/inference/engine/orchestrator.py                │
│     ├─ D-FINE (détection bbox)                              │
│     ├─ MobileSAM (segmentation mask)                        │
│     ├─ Postprocess (poids spatiaux)                         │
│     └─ Buffer: Queue_Out (queue.Queue)                      │
│                                                               │
│  D. OUTPUT (TX)                                              │
│     ├─ IGTGateway._tx_thread                                │
│     ├─ service/slicer_server.py                             │
│     └─ Buffer: _outbox (AdaptiveDeque maxlen=8)            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Structure des Fichiers (Actifs Uniquement)

### **`src/core/` — Cœur de la Pipeline**

```
core/
├── acquisition/          # Étape A (Décodage uniquement)
│   ├── decode.py        # Décodage IGTLink
│   └── __init__.py
│
├── preprocessing/        # Étape B (CPU→GPU)
│   ├── cpu_to_gpu.py    # ✅ prepare_frame_for_gpu(), transfert async
│   └── __init__.py
│
├── inference/            # Étape C (Inférence GPU)
│   ├── engine/
│   │   ├── model_loader.py       # ✅ initialize_models()
│   │   ├── orchestrator.py       # ✅ run_inference(), prepare_inference_inputs()
│   │   ├── inference_dfine.py    # ✅ run_detection()
│   │   ├── inference_sam.py      # ✅ run_segmentation()
│   │   ├── postprocess.py        # ✅ compute_mask_weights()
│   │   └── gpu_optim.py          # Optimisations GPU (FP16, etc.)
│   ├── MobileSAM/                # Modèle SAM
│   ├── d_fine/                   # Modèle D-FINE
│   ├── dfine_infer.py            # ✅ run_dfine_detection()
│   └── sam_infer.py              # Wrapper SAM
│
├── queues/               # Gestion des buffers
│   ├── buffers.py       # ✅ Queue_GPU, Queue_Out, init_queues()
│   ├── adaptive.py      # ✅ AdaptiveDeque (drop-oldest, resize)
│   └── gpu_buffers.py   # Pool GPU (phase 4)
│
├── monitoring/           # Logs + KPI
│   ├── async_logging.py # ✅ Logging asynchrone
│   ├── kpi.py           # ✅ safe_log_kpi(), format_kpi()
│   ├── monitor.py       # ✅ start_monitor_thread()
│   └── filters.py       # Filtres logs
│
├── state_machine/        # FSM (VISIBLE/RELOCALIZING/LOST)
│   └── visibility_fsm.py # Machine à états (non utilisée pour l'instant)
│
├── model/                # Poids des modèles
│   ├── Dfine_last.pt    # ✅ D-FINE RGB
│   ├── Dfine_last_mono.pth  # D-FINE mono-canal
│   └── mobile_sam.pt    # ✅ MobileSAM
│
├── types.py             # ✅ RawFrame, GpuFrame, ResultPacket, FrameMeta
└── utils/               # Utilitaires
```

### **`src/service/` — Orchestration & Réseau**

```
service/
├── gateway/              # Gestionnaire principal
│   ├── manager.py       # ✅ IGTGateway (orchestrateur RX/TX)
│   ├── supervisor.py    # ✅ Thread supervision (KPI, FPS, latence)
│   ├── stats.py         # ✅ GatewayStats (collecteur métriques)
│   ├── events.py        # ✅ EventEmitter
│   ├── config.py        # ✅ GatewayConfig
│   └── __init__.py
│
├── slicer_server.py     # ✅ run_slicer_server() (Thread TX)
├── dashboard_service.py # ✅ Dashboard Web (FastAPI + Plotly)
├── plus_client.py       # Client PlusServer (legacy, peu utilisé)
├── igthelper.py         # Helper IGTLink
├── registry.py          # Registre des threads
├── autotune.py          # Auto-tuning paramètres
└── heartbeat.py         # Mesure latence réseau
```

### **`tests/` — Tests**

```
tests/
├── test_pipeline_full.py                        # ✅ Test inférence isolée
├── tests_gateway/
│   └── test_gateway_real_pipeline_mock.py      # ✅ Test pipeline complète
├── conftest.py                                  # Configuration pytest
└── (autres tests unitaires...)
```

---

## 🔄 Flux de Données (Production)

### **Point d'Entrée : `main.py`**

```python
main.py
 ├── Configure logging (async, KPI)
 ├── Charge config/gateway.yaml → GatewayConfig
 ├── Crée IGTGateway(config)
 ├── gateway.start()
 │     ├── _rx_thread → PlusServer → _mailbox
 │     ├── _tx_thread → _outbox → 3D Slicer
 │     └── _supervisor_thread → Monitoring
 └── start_monitor_thread() → KPI globaux
```

### **Circulation des Données**

```
PlusServer (port 18944)
    ↓ [IGTLink IMAGE]
IGTGateway._rx_thread
    ↓ [decode + RawFrame]
_mailbox (AdaptiveDeque, maxlen=2)
    ↓ [receive_image()]
Thread Processing Local
    ├─ cpu_to_gpu.prepare_frame_for_gpu()
    │     ↓ [GpuFrame]
    ├─ Queue_GPU
    │     ↓ [try_dequeue()]
    ├─ run_inference(GpuFrame)
    │     ├─ run_detection() → bbox, conf
    │     ├─ run_segmentation() → mask
    │     └─ compute_mask_weights() → W_edge, W_in, W_out
    │     ↓ [ResultPacket]
    └─ Queue_Out
        ↓ [enqueue_nowait_out()]
gateway.send_mask(mask, meta)
    ↓
_outbox (AdaptiveDeque, maxlen=8)
    ↓ [_tx_ready.set()]
IGTGateway._tx_thread / run_slicer_server()
    ↓ [pyigtl IMAGE]
3D Slicer (port 18945)
```

---

## 🎯 Modules Clés & Responsabilités

### **1. IGTGateway** (`service/gateway/manager.py`)
**Rôle :** Orchestrateur principal réseau + threads  
**Responsabilités :**
- Lancer threads RX/TX/Supervisor
- Gérer buffers `_mailbox` et `_outbox`
- Exposer API : `receive_image()`, `send_mask()`
- Collecte stats (FPS RX/TX, latence, bytes)

### **2. cpu_to_gpu** (`core/preprocessing/cpu_to_gpu.py`)
**Rôle :** Transfert asynchrone CPU→GPU  
**Responsabilités :**
- Normalisation image (float32, [0,1])
- Pinned memory allocation
- Transfert CUDA async (via stream)
- Création `GpuFrame`

### **3. orchestrator** (`core/inference/engine/orchestrator.py`)
**Rôle :** Coordination inférence D-FINE + SAM  
**Responsabilités :**
- `run_inference(GpuFrame)` → (ResultPacket, latency_ms)
- `prepare_inference_inputs()` → bbox + mask + weights
- Appel séquentiel : D-FINE → SAM → postprocess
- Gestion erreurs (state="LOST" si échec)

### **4. slicer_server** (`service/slicer_server.py`)
**Rôle :** Thread TX (envoi vers Slicer)  
**Responsabilités :**
- Lire `_outbox` (event-based wakeup)
- Convertir mask → pyigtl.ImageMessage
- Envoyer via IGTLink
- Update stats TX

### **5. dashboard_service** (`service/dashboard_service.py`)
**Rôle :** Monitoring temps réel  
**Responsabilités :**
- Collecte métriques (FPS, latence, GPU, queues)
- Serveur FastAPI sur :8050
- Graphiques Plotly interactifs
- Historique glissant (300 points)

---

## 🧪 Tests Disponibles

### **1. `test_pipeline_full.py`** (Inférence Isolée)
**Objectif :** Tester D-FINE + MobileSAM sur image statique  
**Utilise :**
- `initialize_models()` → Charge D-FINE + SAM
- `run_inference(GpuFrame)` → Inférence complète
- Assertions : state="VISIBLE", mask/bbox présents
- Visualisation : 3 images (original, bbox, mask)

**Flux :**
```
00157.jpg (640x640 RGB)
    ↓ [PIL + torch]
GpuFrame [1,3,640,640] CUDA
    ↓ [run_inference]
ResultPacket {state, bbox, mask, score}
    ↓ [Visualisation]
test_pipeline_result.png
```

### **2. `test_gateway_real_pipeline_mock.py`** (Pipeline Complète)
**Objectif :** Tester chaîne RX→PROC→TX avec fausses données  
**Utilise :**
- `IGTGateway` (vraie instance)
- `simulate_frame_source()` → Génère images random 100 Hz
- `simulate_processing()` → Seuillage simple (mask = img > 128)
- `run_slicer_server()` → Envoi simulé
- Dashboard → Métriques temps réel

**Flux :**
```
simulate_frame_source (100 Hz)
    ↓ [RawFrame random]
_mailbox
    ↓ [receive_image()]
simulate_processing (seuillage)
    ↓ [send_mask()]
_outbox
    ↓ [run_slicer_server]
Envoi simulé
    ↓ [Dashboard]
Métriques (FPS, latence)
```

---

## 📊 Buffers & Queues

### **Buffers Actifs**

| Nom | Type | Taille | Rôle | Localisation |
|-----|------|--------|------|--------------|
| `_mailbox` | AdaptiveDeque | 2 | Buffer RX (images reçues) | IGTGateway |
| `Queue_GPU` | queue.Queue | 10 | Buffer avant inférence | core/queues/buffers.py |
| `Queue_Out` | queue.Queue | 10 | Buffer après inférence | core/queues/buffers.py |
| `_outbox` | AdaptiveDeque | 8 | Buffer TX (masques à envoyer) | IGTGateway |

### **Politiques de Gestion**

- **AdaptiveDeque :** Drop-oldest automatique, resize dynamique
- **queue.Queue :** Drop-oldest manuel si plein (via `enqueue_nowait_*`)
- **Backpressure :** Suppression frames anciennes (> 500ms lag)

---

## ⚠️ Modules Non Utilisés (À Décider)

### **1. `state_machine/visibility_fsm.py`**
**Statut :** Implémenté mais jamais appelé  
**Raison :** FSM VISIBLE/RELOCALIZING/LOST pas encore intégrée  
**Action :** À intégrer dans phase suivante

### **2. `core/queues/gpu_buffers.py`**
**Statut :** Partiellement implémenté (phase 4)  
**Raison :** Pool GPU avancé pour optimisation mémoire  
**Action :** Activer quand phase 4 (optimisations GPU)

### **3. `service/plus_client.py`**
**Statut :** Legacy, peu utilisé  
**Raison :** Remplacé par IGTGateway intégré  
**Action :** Garder pour compatibilité, mais documenter comme legacy

---

## 🎯 Prochaine Étape : Test avec Dataset Réel

**Objectif :** Créer `test_gateway_dataset_inference.py`

**Spécifications :**
1. Lire 213 images depuis `JPEGImages/Video_001/`
2. Injecter dans `IGTGateway._mailbox`
3. Thread processing avec **vraie inférence** :
   - `cpu_to_gpu.prepare_frame_for_gpu()`
   - `run_inference()` (D-FINE + MobileSAM)
4. Envoi via `gateway.send_mask()`
5. Dashboard temps réel
6. Métriques : FPS, latence, GPU usage

**Références :**
- Base : `test_gateway_real_pipeline_mock.py`
- Inférence : `test_pipeline_full.py`
- Dataset : `C:\Users\maxam\Desktop\TM\dataset\HUMERUS LATERAL XG SW_cropped\JPEGImages\Video_001\`

---

## ✅ Résumé Post-Nettoyage

**Fichiers supprimés :** 5 fichiers + 2 dossiers  
**Architecture :** Épurée et claire  
**Tests actifs :** 2 tests fonctionnels  
**Prêt pour :** Intégration dataset réel + vraie inférence

---

**Architecture validée ! Prêt pour la suite. 🚀**
