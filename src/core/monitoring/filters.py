import logging   # module standard Python pour la journalisation
import os        # utilisé pour lire les variables d'environnement du système


class PerfFilter(logging.Filter):  # filtre personnalisé dérivé de logging.Filter
    """Filtre les logs pour réduire la verbosité en mode performance.

    Si la variable d'environnement LOG_MODE=perf, seuls les niveaux WARNING et supérieurs sont autorisés.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # méthode principale appelée à chaque message log
        mode = os.getenv("LOG_MODE", "").lower()  # lit la variable d'environnement LOG_MODE (par ex. "dev" ou "perf")
        if mode == "perf" and record.levelno < logging.WARNING:  # si on est en mode performance et que le niveau est inférieur à WARNING (donc INFO/DEBUG)
            return False  # le message est ignoré (non affiché / non écrit)
        return True  # sinon, le message passe le filtre (il sera traité par le handler suivant)


class NoErrorFilter(logging.Filter):  # second filtre, aussi dérivé de logging.Filter
    """Filtre excluant les messages d'erreur (ERROR et supérieurs) d'un handler.

    Ce filtre est utilisé sur les handlers "pipeline" pour que les erreurs soient envoyées
    uniquement vers le fichier d'erreurs dédié (error.log) et non dupliquées ailleurs.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # méthode appliquée à chaque log
        return record.levelno < logging.ERROR  # retourne True uniquement pour les messages inférieurs à ERROR (DEBUG, INFO, WARNING)
