"""
service.autotune
----------------
Boucle d'auto-régulation du gateway.

Ce module contient un contrôleur léger de type PID simplifié
qui ajuste la cadence (FPS cible) et la fréquence du superviseur
selon la latence réseau et la charge observée sur les buffers.
"""

import logging
import time
from typing import Any

LOG = logging.getLogger("igt.autotune")  # Initialise un logger spécifique pour le module d'auto-régulation


class AutoTuner:
    """Contrôleur auto-adaptatif du pipeline."""

    def __init__(self, cfg: Any, hysteresis_ms: float = 10.0):
        self.cfg = cfg  # Référence à la configuration active du Gateway (objet GatewayConfig)
        self.hysteresis_ms = hysteresis_ms  # Seuil d’hystérésis (en millisecondes) pour éviter les oscillations de réglage
        self._last_update = 0.0  # Horodatage du dernier ajustement effectué
        self._prev_latency = 0.0  # Latence mesurée lors du cycle précédent
        self._trend = 0.0  # Tendance lissée des variations de latence

    def tune(self, latency_ms: float, queue_len: int) -> None:  # Ajuste les paramètres de la passerelle selon les métriques
        """Ajuste les paramètres du gateway selon les métriques."""
        now = time.time()  # Récupère le temps actuel (en secondes)
        if now - self._last_update < 5.0:  # Limite les ajustements à une fois toutes les 5 secondes maximum
            return

        target_fps = self.cfg.target_fps  # FPS cible actuel issu de la configuration
        interval = self.cfg.supervise_interval_s  # Intervalle actuel du thread superviseur

        diff = latency_ms - self._prev_latency  # Différence entre la latence actuelle et la précédente
        self._trend = 0.7 * self._trend + 0.3 * diff  # Lissage exponentiel de la tendance (comportement proche d’un PID)
        self._prev_latency = latency_ms  # Met à jour la latence précédente pour le prochain cycle


        # ⚠️ TODO: valeur seuil fixe (80 ms) — à adapter dynamiquement selon la latence moyenne réelle observée.
        #          Par exemple, la moyenne + 30 % serait plus pertinente pour ton pipeline GPU.
        # ⚠️ TODO: valeur seuil basse (40 ms) — probablement trop stricte pour ton traitement IA temps réel.
        #          Ces bornes devraient venir d’une configuration YAML ou d’une estimation adaptative.
        if latency_ms > 80 + self.hysteresis_ms:  # Si la latence est supérieure à 80 ms + marge d’hystérésis
            target_fps = max(10.0, target_fps - 2.0)  # Réduit la fréquence cible, mais jamais en dessous de 10 FPS
        elif latency_ms < 40 - self.hysteresis_ms:  # Si la latence est inférieure à 40 ms - marge d’hystérésis
            target_fps = min(30.0, target_fps + 1.0)  # Augmente légèrement la fréquence cible, max 30 FPS


        # ⚠️ TODO: seuils codés en dur (12 / 4) — incohérents avec la taille actuelle des queues (_mailbox=2, _outbox=8).
        #          Adapter ces valeurs en proportion de la taille max des buffers, ou les charger depuis GatewayConfig.
        #          Exemple: high_threshold = 0.75 * outbox.maxlen ; low_threshold = 0.25 * outbox.maxlen
        if queue_len > 12:  # Si la taille de la file (outbox ou mailbox) dépasse 12 éléments
            interval = min(5.0, interval + 0.5)  # Allonge la période du superviseur (moins fréquent)
        elif queue_len < 4:  # Si la file est presque vide
            interval = max(1.0, interval - 0.5)  # Raccourcit l’intervalle pour réagir plus vite

        changed = (target_fps != self.cfg.target_fps) or (interval != self.cfg.supervise_interval_s)  # Vérifie si un changement est nécessaire
        if changed:
            LOG.info("Auto-tune -> fps_cible=%0.1f, intervalle=%0.1f, latence=%0.1f ms",
                     target_fps, interval, latency_ms)  # Journalise l’ajustement effectué
            self.cfg.target_fps = target_fps  # Met à jour le FPS cible dans la configuration
            self.cfg.supervise_interval_s = interval  # Met à jour l’intervalle de supervision

        self._last_update = now  # Mémorise le moment du dernier ajustement pour limiter la fréquence des modifications
