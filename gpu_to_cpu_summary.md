# 🧭 AUDIT COMPLET DES TRANSFERTS GPU→CPU

## 📊 Résumé Exécutif

**Date du rapport:** 29 octobre 2025  
**Périmètre:** `src/` et `tests/` - 25 conversions GPU→CPU détectées  
**Impact critique:** 4 transferts à fort impact performance identifiés

## 📈 Statistiques Globales

### Répartition par Catégorie
- 🔵 **Nécessaires (gardez):** 15 conversions (60%)
- 🟠 **Redondants (optimisables):** 5 conversions (20%)  
- 🔴 **Dangereux pour perf (priorité):** 4 conversions (16%)
- 🧪 **Tests uniquement:** 1 conversion (4%)

### Impact Performance
- **Élevé:** 4 conversions (masques SAM, image preprocessing)
- **Moyen:** 8 conversions (bounding boxes, matrices)
- **Faible:** 13 conversions (scalaires, tests, cleanup)

## 🚨 TOP 5 des Conversions Critiques

### 1. 🔴 **MobileSAM Predictor** (Priorité P1)
**Fichier:** `src/core/inference/MobileSAM/mobile_sam/predictor.py`  
**Lignes:** 164-166  
**Problème:** Triple conversion massive `masks`, `iou_predictions`, `low_res_masks`
```python
masks_np = masks[0].detach().cpu().numpy()           # ⚠️ GROS IMPACT
iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
low_res_masks_np = low_res_masks[0].detach().cpu().numpy()  # Potentiellement inutile
```
**Impact:** Transfert de gros tenseurs (masques haute résolution) à chaque frame  
**Recommandation:** Garder en tenseurs GPU jusqu'au `ResultPacket` final

### 2. 🔴 **Orchestrator SAM Input** (Priorité P1)
**Fichier:** `src/core/inference/engine/orchestrator.py`  
**Ligne:** 88  
**Problème:** Conversion RGB premature pour SAM
```python
arr_np = arr[0].permute(1, 2, 0).detach().cpu().numpy()  # ⚠️ GPU→CPU→GPU chain
```
**Impact:** Chain GPU→CPU→GPU si SAM traite l'image ensuite  
**Recommandation:** Passer directement le tensor GPU à SAM ou convertir SAM vers GPU

### 3. 🟠 **D-FINE Frame Debug** (Priorité P2)
**Fichier:** `src/core/inference/dfine_infer.py`  
**Ligne:** 145  
**Problème:** Conversion pour debug uniquement
```python
frame_cpu = frame_mono.detach().to("cpu", non_blocking=False)  # Debug only?
```
**Recommandation:** Supprimer si utilisé uniquement pour debug

### 4. 🟠 **Matcher Cost Matrix** (Priorité P2)
**Fichier:** `src/core/inference/d_fine/matcher.py`  
**Ligne:** 112  
**Problème:** Matrice de coût sur CPU
```python
C = C.view(bs, num_queries, -1).cpu()  # Hungarian matching on CPU
```
**Recommandation:** Vérifier si Hungarian matching peut rester sur GPU

### 5. 🔵 **D-FINE Final Output** (Garder - Nécessaire)
**Fichiers:** `src/core/inference/dfine_infer.py`  
**Lignes:** 178, 243  
**Justification:** Conversion finale pour résultat Slicer - appropriée
```python
return box_xyxy.to(dtype=torch.float32).cpu().numpy(), conf  # ✅ Final output OK
```

## 🔄 Chaînes de Conversions Détectées

### Chaîne Problématique Identifiée:
```
orchestrator.py:88   →  arr.detach().cpu().numpy()     (GPU→CPU)
                     →  [passage vers SAM]
mobile_sam/predictor.py  →  masks.detach().cpu().numpy()  (GPU→CPU)
```

**Problème:** Double transfert GPU→CPU au lieu d'une pipeline entièrement GPU

## 💡 Plan de Refactor Recommandé

### Phase 1: Optimisations Critiques (P1)
1. **Modifier MobileSAM Predictor**
   - Retourner tenseurs GPU au lieu de NumPy arrays
   - Différer `.cpu().numpy()` au moment du `ResultPacket`
   
2. **Pipeline GPU-to-GPU pour SAM**
   - Éliminer conversion dans `orchestrator.py:88`  
   - Configurer SAM pour accepter tenseurs GPU directement

### Phase 2: Optimisations Moyennes (P2)
3. **Nettoyer D-FINE Debug**
   - Supprimer `frame_cpu` si debug uniquement
   
4. **Optimiser Hungarian Matching**
   - Investiguer si matching peut rester sur GPU

### Phase 3: Consolidation Finale
5. **Architecture GPU-Resident**
   ```python
   # Objectif: Pipeline entièrement GPU jusqu'à ResultPacket
   D-FINE (GPU) → SAM (GPU) → PostProcess (GPU) → ResultPacket.cpu().numpy()
   ```

## 🎯 Objectif Final

### Pipeline Optimisée Cible:
```
[Input Image] → GPU tensor
    ↓
[D-FINE] → GPU tensors (bbox, scores)
    ↓  
[SAM] → GPU tensors (masks, iou)
    ↓
[Post-processing] → GPU tensors
    ↓
[ResultPacket création] → ⚡ UN SEUL .cpu().numpy() final
```

### Bénéfices Attendus:
- **Réduction transferts:** 4→1 conversions par frame
- **Latence réduite:** Élimination GPU↔CPU overhead  
- **Débit amélioré:** Pipeline continue sur GPU
- **Mémoire optimisée:** Moins de copies temporaires

## ✅ Critères de Validation

- [ ] MobileSAM retourne tenseurs GPU
- [ ] Orchestrator passe tenseurs GPU à SAM  
- [ ] Un seul `.cpu().numpy()` final dans ResultPacket
- [ ] Tests de régression passent
- [ ] Benchmarks montrent amélioration latence >20%

## 📋 Actions Immédiates

1. **Modifier** `mobile_sam/predictor.py` pour retourner tenseurs
2. **Analyser** si SAM peut traiter input GPU directement
3. **Tester** impact sur latence avec profiling détaillé
4. **Implémenter** conversion unique finale

---
**🏁 En résumé:** 4 conversions critiques identifiées, plan de refactor vers pipeline GPU-resident prêt à implémenter.