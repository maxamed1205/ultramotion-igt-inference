# Dashboard Unifié - Fusion Complète ✅

## 📅 Date : 29 Octobre 2025

## 🎯 Objectif

Fusionner deux dashboards distincts en un seul dashboard unifié :
- `dashboard_service.py` (port 8050) - Métriques Pipeline RX/PROC/TX
- `dashboard_gpu_transfer.py` (port 8051) - Métriques GPU Transfer détaillées

## ✨ Résultat

Un nouveau dashboard unifié : `dashboard_unified.py` sur **port 8050 unique**

## 📁 Fichiers Créés

### 1. Service Principal

**`src/service/dashboard_unified.py`** (1100+ lignes)

Architecture :
- `UnifiedMetricsCollector` : Collecte GPU + Pipeline
- FastAPI app avec 5 endpoints REST + WebSocket
- HTML interactif avec 4 graphiques Plotly
- Thread de collecte en arrière-plan

Fonctionnalités :
- ✅ Métriques Pipeline (RX/PROC/TX, latences inter-étapes)
- ✅ Métriques GPU Transfer (norm/pin/copy avec décomposition)
- ✅ Utilisation GPU et mémoire
- ✅ Files d'attente et drops
- ✅ Système de santé (OK/WARNING/CRITICAL)
- ✅ API REST complète
- ✅ WebSocket temps réel
- ✅ Interface HTML responsive

### 2. Scripts de Lancement

**`run_unified_dashboard.ps1`**
- Lance uniquement le dashboard
- Vérifie les dépendances
- Crée le répertoire logs si nécessaire

**`run_test_with_unified_dashboard.ps1`**
- Nettoie les logs
- Lance le dashboard en arrière-plan
- Exécute un test de pipeline
- Arrête proprement le dashboard à la fin

### 3. Documentation

**`docs/Dashboard_Unified_README.md`**
- Guide complet d'utilisation
- Description des fonctionnalités
- Exemples de logs parsés
- Configuration des seuils
- API REST documentation
- Troubleshooting

**`docs/Dashboard_Migration_Guide.md`**
- Guide de migration complet
- Tableau de correspondance (scripts, URLs, endpoints)
- Migration étape par étape
- Adaptation du code API
- Tests de validation
- Procédure de rollback

### 4. Tests

**`tests/test_dashboard_unified.py`**
- 7 tests de validation :
  1. Initialisation du collecteur
  2. Collection des métriques
  3. Historique
  4. Endpoints API
  5. Parsing GPU metrics
  6. Parsing Pipeline metrics
  7. Calcul du statut de santé

## 🎨 Interface Utilisateur

### Cartes de Métriques (6)

1. **État Général**
   - Statut (OK/WARNING/CRITICAL)
   - Dernière mise à jour

2. **Pipeline (RX → PROC → TX)**
   - Frame # et FPS par étape
   - Synchronisation TX/PROC

3. **Latences Inter-Étapes**
   - RX → PROC (avg/last)
   - PROC → TX (avg/last)
   - RX → TX total (avg/last)

4. **GPU Transfer (CPU→GPU)** ⭐ NOUVEAU
   - Frames traitées
   - Latence moyenne
   - Décomposition norm/pin/copy
   - Throughput (FPS)

5. **GPU Utilisation**
   - Utilisation (%)
   - Mémoire (MB)

6. **Files d'attente**
   - Queue RT, GPU, OUT
   - Drops RT

### Graphiques Interactifs (4)

1. **FPS Pipeline**
   - Ligne temporelle : FPS RX vs FPS TX
   - 100 derniers points

2. **Latence Pipeline**
   - Latence RX→TX globale
   - Remplissage sous la courbe

3. **GPU Transfer - Décomposition** ⭐ NOUVEAU
   - Barres empilées par frame
   - norm_ms / pin_ms / copy_ms
   - 100 dernières frames

4. **Latences par Frame**
   - 3 courbes : RX→PROC, PROC→TX, RX→TX
   - Zoom/Pan interactif

## 🔌 API REST

### Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface HTML |
| `/api/metrics/latest` | GET | Dernières métriques JSON |
| `/api/metrics/history` | GET | Historique (optionnel: ?last_n=N) |
| `/api/health` | GET | Health check rapide |
| `/ws/metrics` | WebSocket | Stream temps réel |

### Exemple JSON

```json
{
  "timestamp": 1698589015.123,
  "datetime": "2025-10-29T14:30:15",
  
  "fps_rx": 85.2,
  "fps_proc": 84.8,
  "fps_tx": 84.1,
  
  "latency_rxproc_avg": 8.5,
  "latency_proctx_avg": 10.0,
  "latency_rxtx_avg": 18.5,
  
  "gpu_util": 78.3,
  "gpu_memory_mb": 2048.5,
  
  "gpu_transfer": {
    "frames": [1, 2, 3],
    "norm_ms": [0.8, 0.7, 0.9],
    "pin_ms": [1.2, 1.1, 1.2],
    "copy_ms": [0.5, 0.5, 0.5],
    "stats": {
      "avg_total": 2.47,
      "avg_norm": 0.8,
      "avg_pin": 1.15,
      "avg_copy": 0.5,
      "throughput_fps": 83.5
    }
  },
  
  "health": "OK"
}
```

## 🔍 Sources de Données

Le dashboard parse 3 sources :

1. **`logs/kpi.log`**
   - Événements `copy_async` pour GPU transfer
   - Format : `event=copy_async device=cuda:0 H=512 W=512 norm_ms=0.8 pin_ms=1.2 copy_ms=0.5 total_ms=2.5 frame=1`

2. **`logs/pipeline.log`**
   - Événements RX/PROC/TX avec timestamps
   - Calcul automatique des latences inter-étapes

3. **Monitoring temps réel**
   - `core.monitoring.monitor.get_aggregated_metrics()`
   - `core.monitoring.monitor.get_gpu_utilization()`
   - `core.queues.buffers.collect_queue_metrics()`

## 📊 Avantages de la Fusion

| Aspect | Avant | Après |
|--------|-------|-------|
| **Dashboards** | 2 services séparés | 1 service unifié |
| **Ports** | 8050 + 8051 | 8050 uniquement |
| **Interface** | 2 fenêtres | 1 fenêtre |
| **Corrélation** | Manuelle | Automatique |
| **Ressources** | 2 serveurs | 1 serveur |
| **API** | 2 APIs distinctes | 1 API unifiée |
| **Maintenance** | 2 codebases | 1 codebase |

## 🚀 Utilisation

### Lancement Simple

```powershell
.\run_unified_dashboard.ps1
```

Puis ouvrir : **http://localhost:8050**

### Avec Test de Pipeline

```powershell
.\run_test_with_unified_dashboard.ps1
```

### Commande Directe

```powershell
python -m service.dashboard_unified --port 8050 --host 0.0.0.0 --interval 1.0
```

## ✅ Tests de Validation

Exécuter les tests :

```powershell
python tests/test_dashboard_unified.py
```

Tests couverts :
1. ✅ Initialisation du collecteur
2. ✅ Collection des métriques
3. ✅ Historique
4. ✅ Endpoints API
5. ✅ Parsing GPU metrics
6. ✅ Parsing Pipeline metrics
7. ✅ Calcul du statut de santé

## 📝 Migration depuis les Anciens Dashboards

### Tableau de Correspondance

| Ancien | Nouveau |
|--------|---------|
| `run_test_with_dashboard.ps1` | `run_test_with_unified_dashboard.ps1` |
| `run_gpu_dashboard_simple.ps1` | `run_unified_dashboard.ps1` |
| `python -m service.dashboard_service` | `python -m service.dashboard_unified` |
| `python -m service.dashboard_gpu_transfer` | `python -m service.dashboard_unified` |
| `http://localhost:8050` (Pipeline) | `http://localhost:8050` (Tout) |
| `http://localhost:8051` (GPU) | `http://localhost:8050` (Tout) |

### Migration API

**Avant :**
```python
# 2 requêtes séparées
pipeline_data = requests.get("http://localhost:8050/api/metrics/latest").json()
gpu_data = requests.get("http://localhost:8051/api/metrics").json()
```

**Après :**
```python
# 1 seule requête
data = requests.get("http://localhost:8050/api/metrics/latest").json()
fps_rx = data["fps_rx"]
gpu_stats = data["gpu_transfer"]["stats"]
```

## 🔧 Configuration

### Seuils d'Alerte

Modifiables dans `DashboardConfig` :

```python
@dataclass
class DashboardConfig:
    fps_warning: float = 70.0
    fps_critical: float = 50.0
    latency_warning: float = 30.0
    latency_critical: float = 50.0
    gpu_warning: float = 90.0
    gpu_critical: float = 95.0
```

### Paramètres

- `--port` : Port du serveur (défaut: 8050)
- `--host` : Hôte (défaut: 0.0.0.0)
- `--interval` : Intervalle de collecte en secondes (défaut: 1.0)

## 📚 Documentation

- `docs/Dashboard_Unified_README.md` - Guide complet
- `docs/Dashboard_Migration_Guide.md` - Guide de migration
- `docs/Dashboard_Guide.md` - Guide général
- `docs/Monitoring_and_KPI.md` - KPI documentation

## 🎯 Points Clés

1. **Un seul dashboard** pour toutes les métriques
2. **Corrélation directe** entre GPU et Pipeline
3. **API unifiée** avec toutes les données
4. **Interface moderne** avec Plotly
5. **WebSocket temps réel** pour toutes les métriques
6. **Compatibilité** avec les logs existants
7. **Tests automatisés** pour validation
8. **Documentation complète** pour migration

## ✨ Fonctionnalités Avancées

- Auto-refresh configurable (1s par défaut)
- Buffer circulaire (500 frames GPU, 300 snapshots pipeline)
- Graphiques interactifs (zoom, pan, hover)
- Health monitoring avec alertes visuelles
- WebSocket pour push temps réel
- API REST pour intégration externe
- Historique glissant configurable

## 🎉 Conclusion

Le dashboard unifié est **opérationnel et testé** :

✅ Fusion réussie de `dashboard_service.py` + `dashboard_gpu_transfer.py`
✅ Interface unique sur port 8050
✅ Toutes les métriques GPU + Pipeline visibles
✅ API REST complète
✅ WebSocket temps réel
✅ Documentation complète
✅ Scripts de lancement
✅ Tests de validation
✅ Guide de migration

**Prêt à l'emploi ! 🚀**
