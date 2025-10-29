# 📊 Guide du Dashboard Temps Réel Ultramotion IGT

## Vue d'ensemble

Le nouveau dashboard web offre une visualisation complète et interactive de la pipeline d'inférence IGT en temps réel.

### ✨ Fonctionnalités principales

1. **Métriques temps réel** :
   - FPS entrée/sortie
   - Latence end-to-end
   - Utilisation GPU et mémoire
   - État des queues (RT, GPU, OUT)
   - Compteurs de drops

2. **Graphiques interactifs** :
   - Historique FPS (entrée vs sortie)
   - Évolution de la latence
   - Utilisation GPU dans le temps
   - Zoom, export, hover tooltips

3. **Alertes visuelles** :
   - Statut global (OK / WARNING / CRITICAL)
   - Codage couleur automatique selon seuils
   - Mise à jour automatique toutes les secondes

4. **Architecture moderne** :
   - Backend FastAPI (API REST + WebSocket)
   - Frontend HTML/JS avec Plotly
   - Pas de parsing de logs (exploite directement les métriques)

## 🚀 Installation

### Prérequis

- Python 3.9+
- Pipeline Ultramotion IGT fonctionnelle
- (Optionnel) NVIDIA GPU avec drivers CUDA

### Installation des dépendances

```powershell
# Depuis la racine du projet
pip install -r requirements-dashboard.txt
```

## 📖 Utilisation

### Démarrage rapide

```powershell
# Option 1 : Démarrage direct
python -m service.dashboard_service

# Option 2 : Configuration personnalisée
python -m service.dashboard_service --port 8080 --interval 0.5

# Option 3 : Import dans votre code
```

```python
from service.dashboard_service import DashboardService, DashboardConfig

config = DashboardConfig(
    port=8050,
    host="0.0.0.0",
    update_interval=1.0,
    history_size=300,
    
    # Seuils personnalisés
    fps_warning=70.0,
    fps_critical=50.0,
    latency_warning=30.0,  # ms
    latency_critical=50.0
)

service = DashboardService(config)
service.start()
```

### Accès au dashboard

Une fois démarré, ouvrez votre navigateur :

```
http://localhost:8050
```

## 🎯 Architecture

### Backend (FastAPI)

Le backend expose plusieurs endpoints :

#### API REST

```bash
# Dernières métriques
GET http://localhost:8050/api/metrics/latest

# Historique (60 derniers points)
GET http://localhost:8050/api/metrics/history?last=60

# Santé du système
GET http://localhost:8050/api/health
```

#### WebSocket

```javascript
// Streaming temps réel
const ws = new WebSocket('ws://localhost:8050/ws/metrics');

ws.onmessage = (event) => {
    const metrics = JSON.parse(event.data);
    console.log(metrics);
};
```

### Métriques collectées

Le dashboard collecte automatiquement :

```json
{
  "timestamp": 1761720240.123,
  "datetime": "2025-10-29T07:44:00.123",
  
  "fps_in": 95.2,
  "fps_out": 94.8,
  "latency_ms": 12.3,
  
  "gpu_util": 87.5,
  "gpu_memory_mb": 2048.0,
  
  "queue_rt_size": 2,
  "queue_rt_drops": 0,
  "queue_gpu_size": 1,
  "queue_out_size": 3,
  
  "health": "OK"
}
```

## 🔧 Configuration avancée

### Seuils d'alerte

Modifiez `DashboardConfig` pour ajuster les seuils :

```python
config = DashboardConfig(
    # FPS
    fps_warning=70.0,      # Jaune si < 70 fps
    fps_critical=50.0,     # Rouge si < 50 fps
    
    # Latence
    latency_warning=30.0,  # Jaune si > 30 ms
    latency_critical=50.0, # Rouge si > 50 ms
    
    # GPU
    gpu_warning=90.0,      # Jaune si > 90%
    gpu_critical=95.0      # Rouge si > 95%
)
```

### Historique

```python
config = DashboardConfig(
    history_size=300,      # 300 points (5 min @ 1Hz)
    update_interval=1.0    # Collecte toutes les 1s
)
```

## 📊 Comparaison : Ancien vs Nouveau Dashboard

| Fonctionnalité | Ancien (`dashboard_gateway.py`) | Nouveau (`dashboard_service.py`) |
|----------------|--------------------------------|----------------------------------|
| **Type** | Terminal (Rich) | Web interactif |
| **Source données** | Parse logs manuellement | API métriques directes |
| **Graphiques** | Aucun | Plotly interactifs |
| **Historique** | Limité (fenêtre de lecture) | Configurable (300 points) |
| **Temps réel** | Polling 1-2s | WebSocket streaming |
| **Export** | Impossible | JSON API disponible |
| **Multi-utilisateurs** | Non | Oui (web) |
| **Zoom/Analyse** | Non | Oui (Plotly) |
| **Mobile** | Non | Responsive |

## 🔌 Intégration

### Avec Grafana/Prometheus

Le dashboard expose des métriques JSON facilement exploitables :

```python
# Script d'export Prometheus
import requests

metrics = requests.get('http://localhost:8050/api/metrics/latest').json()

# Format Prometheus
print(f'igt_fps_in{{}} {metrics["fps_in"]}')
print(f'igt_fps_out{{}} {metrics["fps_out"]}')
print(f'igt_latency_ms{{}} {metrics["latency_ms"]}')
print(f'igt_gpu_util{{}} {metrics["gpu_util"]}')
```

### Logging automatique

Les métriques sont déjà loggées via `kpi.log` et `pipeline.log` sans modification.

## 🐛 Dépannage

### Le dashboard ne démarre pas

```powershell
# Vérifiez les dépendances
pip install -r requirements-dashboard.txt

# Vérifiez le port
netstat -an | findstr "8050"

# Changez le port si occupé
python -m service.dashboard_service --port 8051
```

### Pas de données affichées

- Assurez-vous que la pipeline est démarrée
- Vérifiez que `async_logging` est configuré
- Consultez les logs : `logs/pipeline.log`

### WebSocket déconnecté

- Vérifiez votre firewall
- Rechargez la page web
- Vérifiez les logs du service

## 📈 Exemples d'utilisation

### Monitoring continu

```powershell
# Terminal 1 : Pipeline principale
python src\main.py

# Terminal 2 : Dashboard
python -m service.dashboard_service

# Terminal 3 : Tests
python tests\tests_gateway\test_gateway_real_pipeline_mock.py
```

### Mode démo/présentation

```python
# Démarrage avec interval réduit pour démo fluide
python -m service.dashboard_service --interval 0.5
```

### Export de données

```powershell
# Récupération de l'historique via API
curl http://localhost:8050/api/metrics/history?last=300 > metrics_export.json

# Analyse ultérieure
python -c "import json; data=json.load(open('metrics_export.json')); print(f'FPS moyen: {sum(m['fps_in'] for m in data['data'])/len(data['data'])}')"
```

## 🎨 Personnalisation

Le dashboard est entièrement personnalisable. Éditez `generate_dashboard_html()` dans `dashboard_service.py` pour :

- Ajouter de nouveaux graphiques
- Modifier les couleurs/thèmes
- Ajouter des métriques personnalisées
- Intégrer des alertes email/Slack

## 📝 TODO / Améliorations futures

- [ ] Export CSV/Excel des métriques
- [ ] Alertes par email/webhook
- [ ] Comparaison A/B entre sessions
- [ ] Heatmap de performance
- [ ] Prédiction de dégradation (ML)
- [ ] Mode sombre/clair
- [ ] Authentification utilisateur

## 🤝 Support

Pour toute question ou problème :
1. Consultez `logs/pipeline.log` et `logs/error.log`
2. Vérifiez la section Dépannage ci-dessus
3. Contactez l'équipe de développement
