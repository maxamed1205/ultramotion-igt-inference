# 📊 Dashboard Temps Réel - Ultramotion IGT

## 🎯 Pourquoi ce nouveau dashboard ?

L'ancien dashboard (`dashboard_gateway.py`) avait des limitations :
- ❌ Parse manuellement les logs (fragile, lent)
- ❌ Pas de graphiques interactifs
- ❌ Historique limité
- ❌ Interface terminal uniquement
- ❌ Difficile de corréler les événements

## ✨ Nouveau Dashboard Web

Dashboard web moderne avec **visualisation temps réel complète** :

### Fonctionnalités

| Catégorie | Métriques affichées |
|-----------|---------------------|
| **Pipeline** | FPS entrée/sortie, Latence end-to-end, Synchro |
| **GPU** | Utilisation %, Mémoire (MB), Température |
| **Queues** | Tailles (RT/GPU/OUT), Drops, Backpressure |
| **Système** | CPU %, Mémoire, État global |
| **Historique** | Graphiques interactifs Plotly (zoom, export) |

### Captures d'écran

```
┌─────────────────────────────────────────────────────────┐
│         🚀 Ultramotion IGT - Dashboard Temps Réel       │
├──────────────┬──────────────┬──────────────┬────────────┤
│ 📊 État      │ ⚡ Pipeline   │ 🎮 GPU       │ 📦 Queues  │
│              │              │              │            │
│ Statut: OK   │ FPS In: 95.2 │ Util: 87.5%  │ RT: 2      │
│ MAJ: 07:44   │ FPS Out:94.8 │ Mem: 2048 MB │ GPU: 1     │
│              │ Lat: 12.3 ms │              │ Drops: 0   │
└──────────────┴──────────────┴──────────────┴────────────┘
│                                                           │
│  📈 Graphiques temps réel (Plotly interactif)            │
│  ┌───────────────────────────────────────────────────┐   │
│  │  FPS Temps Réel                                   │   │
│  │  100 ┤  ╭─╮╭─╮                                    │   │
│  │   80 ┤  │ ╰╯ ╰╮                                   │   │
│  │   60 ┤╭─╯    ╰─╮                                  │   │
│  └───────────────────────────────────────────────────┘   │
│                                                           │
│  ┌───────────────────────────────────────────────────┐   │
│  │  Latence Pipeline (ms)                            │   │
│  └───────────────────────────────────────────────────┘   │
│                                                           │
│  ┌───────────────────────────────────────────────────┐   │
│  │  Utilisation GPU (%)                              │   │
│  └───────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
```

## 🚀 Démarrage Rapide

### 1. Installation

```powershell
# Installer les dépendances
pip install -r requirements-dashboard.txt
```

### 2. Lancement

```powershell
# Option 1 : Script automatique (recommandé)
.\start_dashboard.ps1

# Option 2 : Manuel
python -m service.dashboard_service

# Option 3 : Configuration personnalisée
python -m service.dashboard_service --port 8080 --interval 0.5
```

### 3. Accès

Ouvrez votre navigateur : **http://localhost:8050**

## 📡 Architecture Technique

### Backend (FastAPI)

```
┌─────────────────────────────────────────────────────┐
│              DashboardService                       │
│                                                     │
│  ┌──────────────┐         ┌──────────────────┐    │
│  │ Metrics      │ collect │ monitor.py       │    │
│  │ Collector    │────────>│ queues/buffers   │    │
│  │ (Thread)     │         │ GPU utils        │    │
│  └──────────────┘         └──────────────────┘    │
│        │                                            │
│        │ store                                      │
│        ▼                                            │
│  ┌──────────────┐                                  │
│  │ History      │                                  │
│  │ (deque 300)  │                                  │
│  └──────────────┘                                  │
│        │                                            │
│        │ expose                                     │
│        ▼                                            │
│  ┌──────────────┐         ┌──────────────────┐    │
│  │ FastAPI      │ serve   │ HTML Dashboard   │    │
│  │ + WebSocket  │────────>│ + Plotly Charts  │    │
│  └──────────────┘         └──────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Page HTML du dashboard |
| `/api/metrics/latest` | GET | Dernières métriques (JSON) |
| `/api/metrics/history` | GET | Historique (JSON) |
| `/api/health` | GET | État de santé du système |
| `/ws/metrics` | WebSocket | Stream temps réel |

### Exemple d'appel API

```powershell
# Récupérer les métriques actuelles
curl http://localhost:8050/api/metrics/latest

# Récupérer l'historique (100 derniers points)
curl "http://localhost:8050/api/metrics/history?last=100"
```

## 🔧 Configuration

### Fichier de configuration

```python
from service.dashboard_service import DashboardConfig

config = DashboardConfig(
    port=8050,
    host="0.0.0.0",
    history_size=300,       # 5 min @ 1Hz
    update_interval=1.0,    # Collecte toutes les 1s
    
    # Seuils d'alerte
    fps_warning=70.0,
    fps_critical=50.0,
    latency_warning=30.0,   # ms
    latency_critical=50.0,
    gpu_warning=90.0,
    gpu_critical=95.0
)
```

### Variables d'environnement

```powershell
# Activer/désactiver KPI
$env:KPI_LOGGING = "1"

# Port personnalisé
$env:DASHBOARD_PORT = "8080"
```

## 📊 Exploitation des données

### Export JSON

```python
import requests
import pandas as pd

# Récupérer l'historique
response = requests.get('http://localhost:8050/api/metrics/history?last=300')
data = response.json()['data']

# Convertir en DataFrame
df = pd.DataFrame(data)

# Analyse
print(f"FPS moyen : {df['fps_in'].mean():.2f}")
print(f"Latence P95 : {df['latency_ms'].quantile(0.95):.2f} ms")
print(f"Drops totaux : {df['queue_rt_drops'].max()}")
```

### Intégration Grafana

```python
# Exporter au format Prometheus
def export_prometheus():
    r = requests.get('http://localhost:8050/api/metrics/latest')
    m = r.json()
    
    print(f'# TYPE igt_fps_in gauge')
    print(f'igt_fps_in {m["fps_in"]}')
    
    print(f'# TYPE igt_latency_ms gauge')
    print(f'igt_latency_ms {m["latency_ms"]}')
```

## 🎯 Cas d'usage

### 1. Monitoring production

```powershell
# Terminal 1 : Pipeline
python src\main.py

# Terminal 2 : Dashboard
.\start_dashboard.ps1
```

Accédez au dashboard pour surveiller en continu.

### 2. Tests de performance

```python
# Script de test
import time
import requests

start = time.time()

# Lancer le test
run_performance_test()

# Récupérer les métriques
metrics = requests.get('http://localhost:8050/api/metrics/history').json()

# Analyser
print(f"Durée: {time.time() - start}s")
print(f"FPS moyen: {sum(m['fps_in'] for m in metrics['data'])/len(metrics['data'])}")
```

### 3. Debugging

Le dashboard permet de :
- Détecter les drops en temps réel
- Identifier les goulots d'étranglement (GPU vs CPU)
- Corréler latence et utilisation GPU
- Visualiser l'impact des changements de config

## 🆚 Comparaison Dashboards

| Feature | Ancien (Terminal) | Nouveau (Web) |
|---------|-------------------|---------------|
| **Interface** | Rich (CLI) | HTML + Plotly |
| **Source** | Parse logs | API métriques |
| **Graphiques** | ❌ | ✅ Interactifs |
| **Historique** | Limité | 300 points |
| **Temps réel** | Polling 1s | WebSocket |
| **Export** | ❌ | ✅ JSON/CSV |
| **Multi-users** | ❌ | ✅ |
| **Mobile** | ❌ | ✅ Responsive |
| **Intégration** | ❌ | ✅ API REST |

## 🐛 Dépannage

### Port déjà utilisé

```powershell
# Trouver le processus
Get-NetTCPConnection -LocalPort 8050

# Utiliser un autre port
python -m service.dashboard_service --port 8051
```

### Pas de données

1. Vérifiez que la pipeline est démarrée
2. Consultez `logs/pipeline.log`
3. Vérifiez `async_logging` configuré

### Erreurs d'import

```powershell
pip install -r requirements-dashboard.txt --upgrade
```

## 📚 Ressources

- [Guide complet](docs/Dashboard_Guide.md)
- [Monitoring & KPI](docs/Monitoring_and_KPI.md)
- [Code source](src/service/dashboard_service.py)

## 🎓 Pour aller plus loin

### Ajout de métriques personnalisées

Éditez `MetricsCollector.collect()` :

```python
def collect(self):
    snapshot = {
        # ... métriques existantes
        
        # Nouvelle métrique
        "custom_score": self._compute_custom_score(),
    }
    return snapshot
```

### Alertes webhook

```python
def _check_alerts(self, snapshot):
    if snapshot["health"] == "CRITICAL":
        requests.post(
            "https://hooks.slack.com/services/YOUR/WEBHOOK",
            json={"text": f"⚠️ Pipeline CRITICAL: {snapshot}"}
        )
```

---

**Développé pour Ultramotion IGT Inference Pipeline**  
*Dashboard moderne pour une visibilité totale de votre pipeline temps réel* 🚀
