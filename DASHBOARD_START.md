# 🎯 GUIDE RAPIDE - Dashboard en Action

## ✅ Solution Simple : 2 Terminaux

Le dashboard affiche les métriques **en temps réel** depuis les logs KPI.  
Voici comment le voir fonctionner :

### Terminal 1 : Pipeline Continue

```powershell
python tests\tests_gateway\test_continuous_for_dashboard.py
```

Cela génère des frames en continu et écrit les métriques dans `logs/kpi.log`.

### Terminal 2 : Dashboard

```powershell
.\start_dashboard.ps1
```

Puis ouvrez **http://localhost:8050** dans votre navigateur.

---

## 📊 Ce que vous verrez

Le dashboard lit automatiquement les métriques depuis `logs/kpi.log` et affiche :

- **FPS** : Frames par seconde (entrée/sortie)
- **Latence** : Délai de traitement en ms
- **GPU** : Utilisation si disponible
- **Queues** : État des files d'attente
- **Graphiques** : Historique temps réel

---

## 🔍 Vérification

Si le dashboard affiche `0.0` partout :

### 1. Vérifier que la pipeline génère des logs

```powershell
Get-Content logs\kpi.log -Tail 5
```

Vous devriez voir des lignes comme :
```
ts=1761720240.123 event=rx_update fps_rx=30.5 ...
```

### 2. Vérifier que le dashboard tourne

```powershell
curl http://localhost:8050/api/metrics/latest
```

Vous devriez voir un JSON avec les métriques.

### 3. Si toujours rien

Le dashboard lit les logs EXISTANTS. Si vous venez de démarrer la pipeline, attendez 5-10 secondes que des données soient écrites.

---

## 🚀 Script Automatique (RECOMMANDÉ)

Au lieu de 2 terminaux, utilisez le script automatique :

```powershell
.\launch_pipeline_and_dashboard.ps1
```

Ce script :
1. Lance la pipeline en arrière-plan
2. Lance le dashboard
3. Arrête tout proprement avec Ctrl+C

---

## 💡 Astuce

Pour voir plus de données dans les graphiques, laissez la pipeline tourner 1-2 minutes avant de consulter le dashboard.

Les graphiques affichent les 100 derniers points (environ 2 minutes à 1Hz).

---

**C'est tout ! Consultez http://localhost:8050 et profitez de la visualisation temps réel ! 🎉**
