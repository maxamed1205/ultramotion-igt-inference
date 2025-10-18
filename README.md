# ultramotion-igt-inference

Prototype de service d’inférence OpenIGTLink déployable

## Objectif
Ce dépôt contient une structure minimale, compatible avec les conteneurs Docker, pour un **service d’inférence en temps réel**.  
Ce service s’abonne à un flux d’images OpenIGTLink (provenant de **PlusServer**),  
exécute une pipeline de segmentation (**D-FINE + MobileSAM**)  
et republie une **carte binaire de labels (`BoneMask`)** vers **3D Slicer** via OpenIGTLink.

---

## État actuel
Ce dépôt est une **ébauche initiale** (scaffold) :
- service squelette fonctionnel,  
- Dockerfile prêt pour le build,  
- workflow CI configuré.  

L’intégration des modèles réels et les tests GPU intensifs seront réalisés ultérieurement.

---

## Démarrage rapide (mode développement)
1. **Construire le conteneur**  
   (nécessite **NVIDIA Docker** sur Linux ou **WSL2** sous Windows) :

   ```bash
   # La commande de build est indiquée dans la section Dockerfile

```

## Contrats des queues

| Queue | Type | Taille max | Politique | Producteur | Consommateur |
|--------|------|-------------|------------|--------------|---------------|
| Queue_Raw | RawFrame | illimitée | no-drop | receiver | archive |
| Queue_RT_dyn | RawFrame | 3 | drop-oldest | receiver | cpu_to_gpu |
| Queue_GPU | GpuFrame | 3 | drop-oldest | cpu_to_gpu | segmentation_engine |
| Queue_Out | ResultPacket | 3 | drop-oldest | segmentation_engine | slicer_sender |

### KPI exposés
- **drops_rt**, **drops_gpu**, **drops_out**
- **size**, **maxsize**, **last_backpressure_ts**
- accessibles via `monitor.get_pipeline_metrics()` (retourne `{"queues": ..., "timestamp": ...}`)

### 📊 Canal KPI (Monitoring temps réel)

- Logger : `igt.kpi`
- Fichier : `logs/kpi.log`
- Fréquence : 1 Hz (configurable)
- Format : `ts=<float> fps_in=<float> fps_out=<float> latency_ms=<float> gpu_util=<float> q_rt=<int> q_gpu=<int> drops_rt=<int> drops_gpu=<int>`
- Contrôle : configurer le niveau via `src/config/logging.yaml`

Exemple de ligne KPI :

```
2025-10-17 11:00:00 | igt.kpi | ts=1739800210.456 fps_in=25.0 fps_out=24.9 latency_ms=38.2 gpu_util=51.0 q_rt=2 q_gpu=1 drops_rt=0 drops_gpu=0
```

### ⚙️ Mode Performance et Logging Asynchrone

- **Activer le mode performance (réduit la verbosité à WARNING+):**

```bash
LOG_MODE=perf python src/main.py
```

- **Désactiver les métriques KPI :**

```bash
KPI_LOGGING=0 python src/main.py
```

- **Logging asynchrone :** Tous les logs sont envoyés à une `queue.Queue` centrale et écrits par un thread (QueueListener). Cela évite toute contention disque depuis les threads de pipeline.

- **Fichiers :**
   - `logs/pipeline.log` — logs fonctionnels (rotation automatique)
   - `logs/kpi.log` — métriques temps réel (rotation automatique)

- **Mesure GPU :** Si `pynvml` est installée, la charge GPU (%) est incluse dans chaque ligne KPI.

---


### Schéma typé des queues (rappel rapide)

Le module `src/core/queues/buffers.py` expose des alias et helpers typés :

- `QRaw`   : archive (list[RawFrame])
- `QRTDyn` : runtime queue (Queue[RawFrame]) — accès via `get_queue_rt_dyn()`
- `QGPU`   : runtime queue (Queue[GpuFrame])  — accès via `get_queue_gpu()`
- `QOut`   : runtime queue (Queue[ResultPacket]) — accès via `get_queue_out()`

Principales fonctions d'API non-bloquantes :

- `init_queues(config)` : initialise le registre des 4 queues
- `get_queue_*()` : helpers typés pour récupérer chaque queue
- `enqueue_nowait_rt(q_rt, item)`, `enqueue_nowait_gpu(q_gpu, item)`, `enqueue_nowait_out(q_out, item)` : tentatives d'enqueue non-bloquantes
- `try_dequeue(q)` : consommation non-bloquante (retourne `None` si vide)
- `apply_rt_backpressure(q_rt, now=None, max_lag_ms=500)` : applique drop-oldest sur RT et renvoie stats `{removed, remaining}`

Invariants : FIFO, non-blocage, priorité temps-réel > exhaustivité (drop-oldest sur RT quand lag trop important).


### Contrat IGTLink ↔ FrameMeta

| Champ IGTLink       | Champ interne   | Type              | Exemple             |
|---------------------|-----------------|-------------------|---------------------|
| DeviceName          | device_name     | str               | "Image"             |
| Timestamp           | ts              | float64 (s)       | 1739582334.23       |
| FrameNumber         | frame_id        | int               | 142                 |
| Spacing             | spacing         | tuple(float32×3)  | (0.5, 0.5, 1.0)     |
| ImageOrientation    | orientation     | str               | "UN"                |
| CoordinateFrame     | coord_frame     | str               | "Echographique"     |
| Matrix (TRANSFORM)  | pose.matrix     | np.ndarray(4×4)   | float32 homogène    |
