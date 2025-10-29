# Dashboard Unifié - Ultramotion IGT

## 📊 Vue d'ensemble

Le **Dashboard Unifié** combine les fonctionnalités de deux dashboards précédents :

1. **GPU Transfer Dashboard** (`dashboard_gpu_transfer.py`) - Métriques détaillées du transfert CPU→GPU
2. **Pipeline Dashboard** (`dashboard_service.py`) - Métriques complètes de la pipeline RX→PROC→TX

## 🎯 Fonctionnalités

### Métriques Pipeline (RX → PROC → TX)
- **FPS** par étape (RX, PROC, TX)
- **Latences inter-étapes** :
  - RX → PROC
  - PROC → TX
  - RX → TX (total)
- **Synchronisation** TX/PROC
- **Statistiques** : moyenne, dernière valeur, compteurs de frames

### Métriques GPU Transfer (CPU → GPU)
- **Décomposition du transfert** :
  - `norm_ms` : Normalisation [0,255] → [0,1]
  - `pin_ms` : Allocation/copie vers pinned memory
  - `copy_ms` : Transfert asynchrone GPU
- **Throughput** GPU (frames/sec)
- **Statistiques** : moyenne, min, max

### Métriques Système
- **Utilisation GPU** (%)
- **Mémoire GPU** (MB)
- **Files d'attente** (RT, GPU, OUT)
- **Drops** et alertes

### Graphiques Interactifs
- **FPS temps réel** (RX vs TX)
- **Latences pipeline** par frame
- **GPU Transfer** - Décomposition en barres empilées
- **Latences inter-étapes** - Évolution temporelle

## 🚀 Utilisation

### Lancement Simple

```powershell
.\run_unified_dashboard.ps1
```

Puis ouvrir : **http://localhost:8050**

### Lancement avec Test de Pipeline

```powershell
.\run_test_with_unified_dashboard.ps1
```

Cette commande :
1. Nettoie les anciens logs
2. Démarre le dashboard en arrière-plan
3. Lance un test de pipeline
4. Affiche les métriques en temps réel
5. Arrête proprement le dashboard à la fin

### Lancement Manuel

```powershell
python -m service.dashboard_unified --port 8050 --host 0.0.0.0 --interval 1.0
```

**Options :**
- `--port` : Port du serveur (défaut: 8050)
- `--host` : Hôte (défaut: 0.0.0.0)
- `--interval` : Intervalle de collecte en secondes (défaut: 1.0)

## 📁 Structure

```
dashboard_unified.py
├── UnifiedMetricsCollector
│   ├── Pipeline metrics (RX/PROC/TX)
│   ├── GPU Transfer metrics (norm/pin/copy)
│   ├── GPU utilization
│   └── Queue metrics
│
├── FastAPI Application
│   ├── GET /                    # Interface HTML
│   ├── GET /api/metrics/latest  # Dernières métriques JSON
│   ├── GET /api/metrics/history # Historique JSON
│   ├── GET /api/health          # Health check
│   └── WS  /ws/metrics          # WebSocket temps réel
│
└── CollectorThread
    └── Collecte périodique en arrière-plan
```

## 🔍 Sources de Données

Le dashboard parse automatiquement les logs suivants :

1. **`logs/kpi.log`** :
   - Événements `copy_async` pour GPU transfer
   - Métriques globales (FPS, latences, drops)

2. **`logs/pipeline.log`** :
   - Événements RX/PROC/TX avec timestamps
   - Calcul des latences inter-étapes

3. **Monitoring temps réel** :
   - `core.monitoring.monitor.get_aggregated_metrics()`
   - `core.monitoring.monitor.get_gpu_utilization()`
   - `core.queues.buffers.collect_queue_metrics()`

## 📊 Exemple de Logs Parsés

### GPU Transfer (`kpi.log`)
```
event=copy_async device=cuda:0 H=512 W=512 norm_ms=0.8 pin_ms=1.2 copy_ms=0.5 total_ms=2.5 frame=42
```

### Pipeline (`pipeline.log`)
```
[2025-10-29 14:30:15,123] RX: Received frame #42
[2025-10-29 14:30:15,145] PROC: Processing frame #42
[2025-10-29 14:30:15,167] TX: Sent frame #42
```

## 🎨 Interface

Le dashboard affiche **6 cartes de métriques** :

1. **État Général** - Statut (OK/WARNING/CRITICAL), dernière mise à jour
2. **Pipeline** - RX/PROC/TX frames et FPS
3. **Latences** - Inter-étapes (avg/last)
4. **GPU Transfer** - Décomposition norm/pin/copy
5. **GPU Utilisation** - % et mémoire
6. **Files d'attente** - Tailles et drops

**4 graphiques interactifs** :

1. **FPS Pipeline** - Ligne temporelle RX vs TX
2. **Latence Pipeline** - RX→TX globale
3. **GPU Transfer** - Barres empilées par frame
4. **Latences par Frame** - RX→PROC, PROC→TX, RX→TX

## 🔧 Configuration des Seuils

Modifiez les seuils d'alerte dans `DashboardConfig` :

```python
@dataclass
class DashboardConfig:
    # Seuils FPS
    fps_warning: float = 70.0
    fps_critical: float = 50.0
    
    # Seuils latence (ms)
    latency_warning: float = 30.0
    latency_critical: float = 50.0
    
    # Seuils GPU (%)
    gpu_warning: float = 90.0
    gpu_critical: float = 95.0
```

## 🔗 API REST

### GET /api/metrics/latest

Retourne les dernières métriques au format JSON :

```json
{
  "timestamp": 1698589015.123,
  "datetime": "2025-10-29T14:30:15",
  "fps_rx": 85.2,
  "fps_tx": 84.1,
  "latency_rxtx_avg": 18.5,
  "gpu_util": 78.3,
  "gpu_transfer": {
    "stats": {
      "avg_norm": 0.8,
      "avg_pin": 1.2,
      "avg_copy": 0.5,
      "throughput_fps": 83.5
    }
  },
  "health": "OK"
}
```

### GET /api/metrics/history?last_n=100

Retourne les N dernières métriques :

```json
{
  "history": [...],
  "count": 100
}
```

### GET /api/health

Health check rapide :

```json
{
  "status": "OK",
  "fps_in": 85.2,
  "latency_ms": 18.5,
  "gpu_util": 78.3
}
```

### WebSocket /ws/metrics

Stream temps réel JSON toutes les secondes.

## 🎯 Intégration

Le dashboard unifié remplace :
- `dashboard_gpu_transfer.py` (port 8051)
- `dashboard_service.py` (port 8050)

Il fonctionne sur un **seul port** (8050 par défaut) et affiche **toutes les métriques**.

## 📝 Notes

- **Auto-refresh** : 1 seconde par défaut
- **Buffer** : 500 frames GPU, 300 snapshots pipeline
- **Graphiques** : 100 derniers points pour lisibilité
- **WebSocket** : Mise à jour en temps réel sans polling

## 🐛 Dépannage

### Le dashboard ne démarre pas

Vérifiez les dépendances :

```powershell
pip install fastapi uvicorn torch
```

### Pas de métriques GPU Transfer

Vérifiez que `logs/kpi.log` contient des événements `copy_async`.

### Pas de métriques Pipeline

Vérifiez que `logs/pipeline.log` contient des logs RX/PROC/TX.

### Port déjà utilisé

Changez le port :

```powershell
python -m service.dashboard_unified --port 8888
```

## 📚 Voir Aussi

- `docs/Dashboard_Guide.md` - Guide complet des dashboards
- `docs/Monitoring_and_KPI.md` - Documentation des KPI
- `ARCHITECTURE_CLEAN.md` - Architecture globale
