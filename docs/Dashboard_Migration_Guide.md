# Guide de Migration vers le Dashboard Unifié

## 🎯 Pourquoi migrer ?

Le **Dashboard Unifié** remplace deux dashboards séparés par une interface unique :

| Ancien | Port | Métriques |
|--------|------|-----------|
| `dashboard_service.py` | 8050 | Pipeline RX/PROC/TX, latences, queues |
| `dashboard_gpu_transfer.py` | 8051 | GPU Transfer (norm/pin/copy) |

⬇️

| Nouveau | Port | Métriques |
|---------|------|-----------|
| `dashboard_unified.py` | 8050 | **Toutes les métriques** des deux dashboards |

**Avantages :**
- ✅ Une seule interface
- ✅ Corrélation directe entre GPU et Pipeline
- ✅ Moins de ressources (un seul serveur)
- ✅ API unifiée

## 🔄 Tableau de Correspondance

### Scripts de Lancement

| Ancien Script | Nouveau Script |
|---------------|----------------|
| `run_test_with_dashboard.ps1` | `run_test_with_unified_dashboard.ps1` |
| `run_gpu_dashboard_simple.ps1` | `run_unified_dashboard.ps1` |
| `run_gpu_test_with_dashboard.ps1` | `run_test_with_unified_dashboard.ps1` |

### Commandes Python

| Ancien | Nouveau |
|--------|---------|
| `python -m service.dashboard_service` | `python -m service.dashboard_unified` |
| `python -m service.dashboard_gpu_transfer` | `python -m service.dashboard_unified` |

### URLs

| Ancien | Nouveau |
|--------|---------|
| `http://localhost:8050` (Pipeline) | `http://localhost:8050` (Tout) |
| `http://localhost:8051` (GPU) | `http://localhost:8050` (Tout) |

### Endpoints API

Les endpoints restent identiques, mais retournent maintenant **toutes les métriques** :

| Endpoint | Dashboard Service | GPU Transfer | Unifié |
|----------|-------------------|--------------|---------|
| `/api/metrics/latest` | ✅ Pipeline | ❌ | ✅ Pipeline + GPU |
| `/api/metrics` | ❌ | ✅ GPU | ✅ Pipeline + GPU |
| `/api/metrics/history` | ✅ | ❌ | ✅ |
| `/api/health` | ✅ | ✅ | ✅ |
| `/ws/metrics` | ✅ | ❌ | ✅ |

## 📝 Structure JSON

### Ancien Dashboard Service

```json
{
  "fps_in": 85.2,
  "fps_rx": 85.2,
  "fps_tx": 84.1,
  "latency_rxtx_avg": 18.5,
  "gpu_util": 78.3
}
```

### Ancien GPU Transfer Dashboard

```json
{
  "frames": [1, 2, 3],
  "total_ms": [2.5, 2.3, 2.6],
  "norm_ms": [0.8, 0.7, 0.9],
  "pin_ms": [1.2, 1.1, 1.2],
  "copy_ms": [0.5, 0.5, 0.5],
  "stats": {
    "avg_total": 2.47,
    "avg_norm": 0.8,
    "throughput_fps": 83.5
  }
}
```

### Nouveau Dashboard Unifié

```json
{
  // Métriques Pipeline (identiques)
  "fps_in": 85.2,
  "fps_rx": 85.2,
  "fps_tx": 84.1,
  "latency_rxtx_avg": 18.5,
  "gpu_util": 78.3,
  
  // Métriques GPU Transfer (nouveau)
  "gpu_transfer": {
    "frames": [1, 2, 3],
    "total_ms": [2.5, 2.3, 2.6],
    "norm_ms": [0.8, 0.7, 0.9],
    "pin_ms": [1.2, 1.1, 1.2],
    "copy_ms": [0.5, 0.5, 0.5],
    "stats": {
      "avg_total": 2.47,
      "avg_norm": 0.8,
      "throughput_fps": 83.5
    }
  }
}
```

**⚠️ Migration API :** Si vous utilisez l'API, les métriques GPU sont maintenant dans `data.gpu_transfer` au lieu de `data` directement.

## 🔧 Migration Étape par Étape

### 1. Arrêter les anciens dashboards

```powershell
# Si lancés manuellement
Ctrl+C dans les terminaux

# Si lancés en arrière-plan
Get-Process python | Where-Object {$_.CommandLine -like "*dashboard*"} | Stop-Process
```

### 2. Tester le nouveau dashboard

```powershell
.\run_unified_dashboard.ps1
```

Ouvrir http://localhost:8050 et vérifier que :
- ✅ Les métriques Pipeline s'affichent
- ✅ Les métriques GPU Transfer s'affichent
- ✅ Les graphiques sont tous présents

### 3. Tester avec pipeline

```powershell
.\run_test_with_unified_dashboard.ps1
```

Vérifier que :
- ✅ Le dashboard démarre automatiquement
- ✅ Les métriques se mettent à jour en temps réel
- ✅ Le dashboard s'arrête proprement à la fin

### 4. Adapter vos scripts

Si vous avez des scripts personnalisés qui appellent les anciens dashboards :

**Avant :**
```powershell
python -m service.dashboard_service --port 8050 &
python -m service.dashboard_gpu_transfer --port 8051 &
python my_test.py
```

**Après :**
```powershell
python -m service.dashboard_unified --port 8050 &
python my_test.py
```

### 5. Adapter votre code API (si applicable)

**Avant :**
```python
# Récupérer métriques Pipeline
response = requests.get("http://localhost:8050/api/metrics/latest")
pipeline_data = response.json()

# Récupérer métriques GPU
response = requests.get("http://localhost:8051/api/metrics")
gpu_data = response.json()
```

**Après :**
```python
# Récupérer TOUTES les métriques
response = requests.get("http://localhost:8050/api/metrics/latest")
data = response.json()

# Accéder aux métriques Pipeline (inchangé)
fps_rx = data["fps_rx"]
latency = data["latency_rxtx_avg"]

# Accéder aux métriques GPU (nouveau chemin)
gpu_stats = data["gpu_transfer"]["stats"]
avg_norm = gpu_stats["avg_norm"]
throughput = gpu_stats["throughput_fps"]
```

## 📊 Interface Graphique

### Ancien Dashboard Service

**Cartes :**
- État Général
- Pipeline (RX/PROC/TX)
- Latences
- KPI
- GPU Utilisation
- Files d'attente

**Graphiques :**
- FPS temps réel
- Latences pipeline
- Utilisation GPU
- Latences par frame

### Nouveau Dashboard Unifié

**Cartes :** (identiques + 1 nouvelle)
- État Général
- Pipeline (RX/PROC/TX)
- Latences
- **GPU Transfer** ← NOUVEAU
- GPU Utilisation
- Files d'attente

**Graphiques :** (identiques + 1 nouveau)
- FPS temps réel
- Latences pipeline
- **GPU Transfer - Décomposition** ← NOUVEAU (barres empilées)
- Latences par frame

## ⚠️ Points d'Attention

### 1. Port par défaut

Le dashboard unifié utilise le **port 8050** par défaut (comme l'ancien dashboard_service).

Si vous aviez des configurations spécifiques pour le port 8051, elles ne sont plus nécessaires.

### 2. WebSocket

L'ancien `dashboard_gpu_transfer` n'avait pas de WebSocket. Le nouveau dashboard unifié utilise WebSocket pour **toutes** les métriques.

Si vous aviez du code polling l'ancien endpoint GPU, vous pouvez maintenant utiliser le WebSocket :

```javascript
const ws = new WebSocket('ws://localhost:8050/ws/metrics');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Toutes les métriques (Pipeline + GPU) arrivent ici
};
```

### 3. Logs

Les deux dashboards lisent les mêmes logs :
- `logs/kpi.log`
- `logs/pipeline.log`

Aucun changement nécessaire dans votre génération de logs.

### 4. Dépendances

Les dépendances sont identiques :

```txt
fastapi
uvicorn
torch (optionnel, pour mémoire GPU)
```

## 🧪 Tests de Validation

### Test 1 : Dashboard seul

```powershell
.\run_unified_dashboard.ps1
```

✅ Dashboard démarre sur http://localhost:8050
✅ Interface s'affiche correctement
✅ Aucune erreur dans la console

### Test 2 : Dashboard + Pipeline

```powershell
.\run_test_with_unified_dashboard.ps1
```

✅ Dashboard démarre automatiquement
✅ Test s'exécute
✅ Métriques s'affichent en temps réel
✅ Dashboard s'arrête proprement

### Test 3 : API REST

```powershell
# Terminal 1
.\run_unified_dashboard.ps1

# Terminal 2
curl http://localhost:8050/api/metrics/latest
curl http://localhost:8050/api/health
```

✅ Endpoints répondent
✅ JSON valide
✅ Toutes les métriques présentes

## 🔄 Rollback

Si vous rencontrez des problèmes et devez revenir aux anciens dashboards :

```powershell
# Relancer l'ancien dashboard service
python -m service.dashboard_service --port 8050

# ET/OU relancer l'ancien GPU transfer
python -m service.dashboard_gpu_transfer --port 8051
```

Les anciens fichiers sont toujours présents et fonctionnels.

## 📚 Ressources

- `docs/Dashboard_Unified_README.md` - Documentation complète
- `docs/Dashboard_Guide.md` - Guide général des dashboards
- `src/service/dashboard_unified.py` - Code source
- `run_unified_dashboard.ps1` - Script de lancement
- `run_test_with_unified_dashboard.ps1` - Script de test

## 🆘 Support

En cas de problème :

1. Vérifier les logs du dashboard dans la console
2. Vérifier que les fichiers `logs/kpi.log` et `logs/pipeline.log` existent
3. Tester avec `.\run_test_with_unified_dashboard.ps1`
4. Consulter `docs/Dashboard_Unified_README.md`

## ✅ Checklist de Migration

- [ ] Arrêter les anciens dashboards
- [ ] Tester le nouveau dashboard seul
- [ ] Tester avec pipeline
- [ ] Adapter les scripts personnalisés (si applicable)
- [ ] Adapter le code API (si applicable)
- [ ] Mettre à jour la documentation interne
- [ ] Valider avec tous les tests
- [ ] Supprimer les anciens scripts (optionnel)

**Bonne migration ! 🚀**
