# 🚀 DÉMARRAGE RAPIDE - Dashboard Ultramotion IGT

## ✅ SOLUTION AU PROBLÈME DES LOGS DUPLIQUÉS

Le problème de duplication des logs a été **RÉSOLU** ! 

### 🔧 Corrections appliquées :

1. **`async_logging.py`** : Modifié pour utiliser `propagate=True` au lieu d'ajouter des handlers multiples
2. **Tests** : Commenté l'appel à `dictConfig()` avant `setup_async_logging()`

Les logs ne devraient plus apparaître en double maintenant ! ✅

---

## 📊 NOUVEAU DASHBOARD WEB

Un **dashboard web moderne** a été créé pour remplacer l'ancien dashboard terminal.

### 🎯 Fonctionnalités

- ✅ Interface web interactive (plus de parsing de logs !)
- ✅ Graphiques Plotly temps réel
- ✅ API REST + WebSocket
- ✅ Métriques complètes : FPS, latence, GPU, queues
- ✅ Alertes visuelles automatiques
- ✅ Export JSON/CSV

### 🚀 Installation et Démarrage

```powershell
# 1. Installer les dépendances
pip install -r requirements-dashboard.txt

# 2. Lancer le dashboard
.\start_dashboard.ps1

# 3. Ouvrir dans le navigateur
# http://localhost:8050
```

### 📖 Documentation complète

- **Guide utilisateur** : `docs/Dashboard_Guide.md`
- **Référence technique** : `docs/Dashboard_README.md`

### 🎮 Utilisation

```powershell
# Terminal 1 : Pipeline principale
python src\main.py

# Terminal 2 : Dashboard
.\start_dashboard.ps1

# Terminal 3 : Tests
python tests\tests_gateway\test_gateway_real_pipeline_mock.py
```

Consultez le dashboard à **http://localhost:8050** pour voir toutes les métriques en temps réel !

---

## 📝 RÉSUMÉ DES CHANGEMENTS

### Fichiers modifiés

1. **`src/core/monitoring/async_logging.py`**
   - Corrigé la duplication en utilisant `propagate=True`
   - Suppression des handlers multiples

2. **`tests/tests_gateway/test_gateway_real_pipeline_mock.py`**
   - Commenté `logging.config.dictConfig(cfg)`
   - Laissé `setup_async_logging()` gérer la configuration

3. **`tests/tests_gateway/test_gateway_offline_mock.py`**
   - Même correction que ci-dessus

### Fichiers créés

1. **`src/service/dashboard_service.py`** - Dashboard web complet
2. **`requirements-dashboard.txt`** - Dépendances
3. **`start_dashboard.ps1`** - Script de démarrage
4. **`docs/Dashboard_Guide.md`** - Guide complet
5. **`docs/Dashboard_README.md`** - Référence technique
6. **`QUICKSTART.md`** - Ce fichier !

---

## 🎉 C'est tout !

Vous avez maintenant :
- ✅ Des logs sans duplication
- ✅ Un dashboard web moderne et complet
- ✅ Une visibilité totale sur votre pipeline

**Bon monitoring ! 🚀**
