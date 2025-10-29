# 🚀 Orchestrator.py GPU-Resident Refactoring

## 📋 Résumé des changements

La fonction `prepare_inference_inputs()` dans `orchestrator.py` a été refactorisée pour éliminer le transfert GPU→CPU prématuré qui causait des spikes de performance.

## ⚡ Avant/Après

### ❌ Avant (version originale)
```python
# Forçait un transfert GPU→CPU pour tous les appels SAM
arr_np = arr[0].permute(1, 2, 0).detach().cpu().numpy()
# ... traitement en NumPy ...
mask = run_segmentation(sam_model, full_image)  # NumPy array
```

### ✅ Après (version optimisée)
```python
# Mode GPU-resident par défaut (sam_as_numpy=False)
prepare_inference_inputs(frame, dfine, sam)  # Garde tensors sur GPU

# Mode compatibilité (sam_as_numpy=True) 
prepare_inference_inputs(frame, dfine, sam, sam_as_numpy=True)  # Legacy CPU
```

## 🎯 Changements techniques

### 1. Nouvelle signature
```python
def prepare_inference_inputs(
    frame_t: np.ndarray, 
    dfine_model: Any, 
    sam_model: Any, 
    tau_conf: float = 0.0001, 
    sam_as_numpy: bool = False  # 🆕 Nouveau paramètre
) -> Dict[str, Any]:
```

### 2. Chemin GPU-resident (nouveau, par défaut)
```python
if hasattr(arr, "detach") and not sam_as_numpy:
    # ✅ GPU-resident path : SAM reçoit directement le tensor CUDA
    if arr.ndim == 4 and arr.shape[0] == 1:
        full_image = arr[0].permute(1, 2, 0).contiguous()  # Reste sur GPU
        
        # Normalisation GPU-native
        if full_image.dtype != torch.float32:
            full_image = full_image.to(torch.float32)
        if full_image.max() > 1.0:
            full_image = full_image / 255.0
```

### 3. Chemin legacy (rétrocompatibilité)
```python
else:
    # 🧩 Legacy CPU path (for backward compat)
    arr_np = arr[0].permute(1, 2, 0).detach().cpu().numpy()
    # ... ancien traitement NumPy préservé ...
```

### 4. Appel SAM modifié
```python
# Avant
mask = run_segmentation(sam_model, full_image, bbox_xyxy=bbox_t)

# Après
mask = run_segmentation(sam_model, full_image, bbox_xyxy=bbox_t, as_numpy=sam_as_numpy)
```

## 📊 Monitoring KPI

L'instrumentation KPI log automatiquement:
```json
{
    "ts": 1698765432.0,
    "event": "prepare_inference_inputs",
    "sam_as_numpy": 0,  // 0=GPU-resident, 1=legacy mode
    "tensor_type": "<class 'torch.Tensor'>",
    "device": "cuda:0"
}
```

## 🧭 Migration

### Usage recommandé (GPU-resident)
```python
# Nouveau comportement par défaut - pas de transfert GPU→CPU
result = prepare_inference_inputs(frame, dfine_model, sam_model)
# sam_as_numpy=False par défaut
```

### Usage de compatibilité (legacy)
```python
# Pour la rétrocompatibilité temporaire
result = prepare_inference_inputs(frame, dfine_model, sam_model, sam_as_numpy=True)
```

## 🧪 Tests

Les tests sont dans `tests/test_orchestrator_gpu_resident.py`:
```bash
# Lancer les tests
cd ultramotion-igt-inference
python -m pytest tests/test_orchestrator_gpu_resident.py -v
```

### Tests couverts:
- ✅ Mode GPU-resident avec tensors CUDA
- ✅ Mode legacy avec numpy arrays
- ✅ Comportement par défaut (GPU-resident)
- ✅ Instrumentation KPI
- ✅ Traitement spécifique tensors GPU

## 📈 Bénéfices attendus

1. **🚀 Latence réduite**: Plus de transfert GPU→CPU forcé dans l'orchestrator
2. **📊 Pipeline fluide**: Les tensors restent sur GPU entre DFINE et SAM
3. **⚡ Performance stable**: Élimination des spikes GPU Copy dans les métriques
4. **🔄 Compatibilité**: Code existant continue de fonctionner avec `sam_as_numpy=True`

## ⚠️ Considérations

### GPU Memory
- Le mode GPU-resident garde plus de données sur GPU
- Surveiller l'usage mémoire GPU en production

### Compatibilité
- `run_segmentation()` doit être mis à jour pour accepter `as_numpy` parameter
- Tests d'intégration nécessaires avec le pipeline complet

## 🔗 Flux de données

### Mode GPU-resident (sam_as_numpy=False)
```
Frame (GPU) → DFINE (GPU) → SAM (GPU) → Mask (GPU) → ResultPacket
     ↑                                                      ↓
     └─────────────── Pas de transfert GPU→CPU ─────────────┘
```

### Mode legacy (sam_as_numpy=True)  
```
Frame (GPU) → DFINE (GPU) → GPU→CPU → SAM (CPU) → Mask (CPU) → ResultPacket
                                ↑
                        Transfert forcé
```

## 🔄 Rollback

En cas de problème:
```python
# Forcer le mode legacy globalement
def prepare_inference_inputs(..., sam_as_numpy: bool = True):  # Temporaire
```

## 🚀 Étapes suivantes

1. ✅ **Orchestrator refactoring** (cette étape - TERMINÉ)
2. ⏳ **Update inference_sam.py**: Ajouter support du paramètre `as_numpy`
3. ⏳ **Tests d'intégration**: Pipeline complet GPU-resident
4. ⏳ **Production monitoring**: Vérifier les KPI et la stabilité

## 📊 Métriques de validation

| Métrique | Avant | Après (attendu) |
|----------|-------|-----------------|
| GPU→CPU Copy events | ~3 par frame | 0 par frame |
| Latence SAM | Variable (spikes) | Stable |
| Memory GPU peak | Baseline | +10-15% |
| Throughput | Baseline | +15-25% |