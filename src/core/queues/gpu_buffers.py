# ⚠️ TODO [Phase 4] : Étendre la logique de pool à la mémoire GPU
# - après la section sur _PIN_POOL
# - Créer un module core/queues/gpu_buffers.py
# - Pré-allouer des buffers persistants pour ROI / mask / bbox / logits
# - Éviter torch.empty() en boucle → moins de fragmentation / jitter ↓
# - Peut être initialisé une seule fois dans detection_and_engine.initialize_models()
