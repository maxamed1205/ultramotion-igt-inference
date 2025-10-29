# 🚀 SamPredictor GPU-Resident Refactoring

## 📋 Résumé des changements

La méthode `SamPredictor.predict()` a été refactorisée pour éliminer les transferts GPU→CPU inutiles qui causaient des spikes de latence.

## ⚡ Avant/Après

### ❌ Avant (version originale)
```python
# Forçait 3 transferts GPU→CPU à chaque prédiction
masks_np = masks[0].detach().cpu().numpy()
iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
return masks_np, iou_predictions_np, low_res_masks_np
```

### ✅ Après (version optimisée)
```python
# Mode GPU-resident par défaut (as_numpy=False)
masks, iou, low_res = predictor.predict()  # Retourne des tensors CUDA

# Mode compatibilité (as_numpy=True) 
masks, iou, low_res = predictor.predict(as_numpy=True)  # Retourne des numpy arrays
```

## 🎯 Utilisation

### Mode GPU-resident (recommandé)
```python
# Nouveau comportement par défaut - garde tout sur GPU
masks, iou_predictions, low_res_masks = predictor.predict(
    point_coords=coords,
    point_labels=labels,
    # as_numpy=False par défaut
)

# Les résultats sont des torch.Tensor sur CUDA
assert isinstance(masks, torch.Tensor)
assert masks.is_cuda  # Si GPU disponible
```

### Mode compatibilité (legacy)
```python
# Pour la rétrocompatibilité avec le code existant
masks, iou_predictions, low_res_masks = predictor.predict(
    point_coords=coords,
    point_labels=labels,
    as_numpy=True  # Force la conversion en numpy
)

# Les résultats sont des np.ndarray
assert isinstance(masks, np.ndarray)
```

## 📊 Monitoring KPI

L'instrumentation KPI log automatiquement:
```json
{
    "ts": 1698765432.0,
    "event": "sam_predict_output", 
    "as_numpy": 0,  // 0=GPU-resident, 1=numpy mode
    "mask_device": "cuda:0",
    "iou_device": "cuda:0", 
    "low_device": "cuda:0"
}
```

## 🧭 Migration pour les modules existants

### 1. Orchestrator (temporaire)
```python
# Migration temporaire - ajouter as_numpy=True
masks, iou, low = predictor.predict(..., as_numpy=True)
```

### 2. Modules optimisés (recommandé)
```python
# Garder sur GPU jusqu'au ResultPacket final
masks, iou, low = predictor.predict(...)  # as_numpy=False par défaut
# ... traitement sur GPU ...
# Conversion finale seulement si nécessaire
final_result = masks.detach().cpu().numpy() if need_numpy else masks
```

## ⚠️ Gestion d'erreurs

Si vous voyez `TypeError: Object of type Tensor is not JSON serializable`:
- C'est normal: le code s'attend encore à du numpy
- Solution temporaire: ajoutez `as_numpy=True`
- Solution permanente: refactorisez le module pour accepter les tensors

## 🧪 Tests

Les tests sont dans `tests/test_sam_predictor_gpu_resident.py`:
```bash
# Lancer les tests
cd ultramotion-igt-inference
python -m pytest tests/test_sam_predictor_gpu_resident.py -v
```

## 📈 Bénéfices attendus

1. **🚀 Latence réduite**: Plus de synchronisation GPU→CPU forcée
2. **📊 Pics éliminés**: Plus de spikes "GPU Copy" dans les métriques  
3. **⚡ Pipeline fluide**: SAM reste entièrement sur GPU
4. **🔄 Compatibilité**: Code existant continue de fonctionner avec `as_numpy=True`

## 🔗 Étapes suivantes

1. ✅ **Refactoring SamPredictor** (cette étape - TERMINÉ)
2. ⏳ **Refactoring Orchestrator**: Modifier pour accepter les tensors
3. ⏳ **Tests d'intégration**: Valider le pipeline complet
4. ⏳ **Monitoring**: Vérifier les KPI en production

## 🔄 Rollback

En cas de problème:
```python
# Forcer le mode legacy en changeant la valeur par défaut
def predict(..., as_numpy: bool = True):  # Temporaire
```

Ou revenir à la version précédente du fichier.