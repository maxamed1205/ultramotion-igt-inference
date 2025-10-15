Param(
    [string]$ImageName = "maxamed1205/ultramotion-igt-inference:latest",
    [int]$HostPort = 18944
)
####################################################################################################
# 🔍 LIGNE SCRIPT
# Param(
#     [string]$ImageName = "maxamed1205/ultramotion-igt-inference:latest",
#     [int]$HostPort = 18944
# )
#
# 🧠 RÔLE GÉNÉRAL :
# Déclare les paramètres d’entrée du script `run.ps1`, utilisés pour lancer le conteneur Docker.
# On définit ici deux variables configurables :
#   1️⃣ `$ImageName` → nom et tag de l’image Docker à exécuter
#   2️⃣ `$HostPort`  → port réseau à exposer sur la machine hôte
#
# ⚙️ DÉCOMPOSITION :
#   • `[string]$ImageName` → texte (nom complet de l’image)
#   • `[int]$HostPort` → entier (port TCP local)
#   • `Param()` → directive PowerShell qui permet de définir les arguments passables à l’exécution :
#
#       ▶ Exemple :
#           .\run.ps1
#           → lancera l’image par défaut : 18944 → 18944
#
#           .\run.ps1 -ImageName "huglab/umi:gpu" -HostPort 19000
#           → lancera le conteneur sur le port 19000
#
# 🧩 UTILITÉ :
#   • Permet de réutiliser le même script pour différentes versions ou environnements (local, test, prod).
#   • Évite d’éditer le code à chaque exécution : tout passe par des arguments simples.
#
# 🧠 BONNES PRATIQUES :
#   • Toujours typer les paramètres (`[string]`, `[int]`) pour éviter les erreurs d’interprétation.
#   • Fournir des valeurs par défaut cohérentes avec la configuration du projet.
#
# 📘 EN SYNTHÈSE :
#   ➜ Définit les paramètres d’exécution du conteneur
#   ➜ `$ImageName` = image Docker à lancer
#   ➜ `$HostPort`  = port local à exposer (redirection  <host>:<container>)
####################################################################################################


Write-Host "Running Docker image: $ImageName"
####################################################################################################
# 🔍 LIGNE SCRIPT
# Write-Host "Running Docker image: $ImageName"
#
# 🧠 RÔLE GÉNÉRAL :
# Affiche un message informatif dans la console indiquant quelle image Docker
# est en cours d’exécution.
#
# ⚙️ DÉCOMPOSITION :
#   • `Write-Host` → affiche un message simple dans PowerShell (stdout).
#   • `$ImageName` → inséré dynamiquement dans la chaîne de texte.
#
# 🧩 UTILITÉ :
#   • Aide à suivre les logs et l’état de la commande lancée.
#   • Pratique pour les scripts CI/CD ou l’exécution manuelle dans le terminal.
#
# 📘 EN SYNTHÈSE :
#   ➜ Affiche le nom complet de l’image Docker exécutée
#   ➜ Fournit un retour utilisateur clair avant le lancement
####################################################################################################

docker run --gpus all -it --rm -p ${HostPort}:18944 --name umi_service $ImageName
####################################################################################################
# 🔍 LIGNE SCRIPT
# docker run --gpus all -it --rm -p ${HostPort}:18944 --name umi_service $ImageName
#
# 🧠 RÔLE GÉNÉRAL :
# Lance un conteneur Docker basé sur l’image spécifiée et configure les paramètres
# d’exécution nécessaires pour le service d’inférence Ultramotion.
#
# ⚙️ DÉCOMPOSITION :
#
#   1️⃣ `docker run`
#       → Commande Docker pour exécuter une image sous forme de conteneur actif.
#
#   2️⃣ `--gpus all`
#       → Donne accès à **tous les GPU NVIDIA** disponibles sur la machine hôte.
#       → Nécessite l’installation du **NVIDIA Container Toolkit**.
#       → Indispensable pour les modèles Torch / CUDA (exécution GPU).
#
#   3️⃣ `-it`
#       → `-i` = mode interactif (stdin gardé ouvert)
#       → `-t` = allocation d’un pseudo-terminal pour afficher les logs en direct.
#       → Ensemble, ces deux options permettent de suivre l’inférence en temps réel.
#
#   4️⃣ `--rm`
#       → Supprime automatiquement le conteneur une fois arrêté.
#       → Évite d’accumuler des conteneurs “exited” inutiles.
#
#   5️⃣ `-p ${HostPort}:18944`
#       → Redirige le port du conteneur (`18944`) vers le port local (`$HostPort`).
#       → Exemple :
#           $HostPort = 18944 →  localhost:18944 → conteneur:18944
#           $HostPort = 19000 →  localhost:19000 → conteneur:18944
#       → Utile pour éviter les conflits si plusieurs conteneurs tournent simultanément.
#
#   6️⃣ `--name umi_service`
#       → Donne un nom lisible au conteneur en cours (ici “umi_service”).
#       → Facilite la gestion : on peut le stopper, inspecter ou redémarrer via ce nom.
#
#   7️⃣ `$ImageName`
#       → Nom complet de l’image Docker à exécuter (défini plus haut).
#
# 🧩 COMPORTEMENT :
#   • Démarre le service IA dans un environnement GPU complet.
#   • Ouvre un port TCP pour OpenIGTLink (par défaut : 18944).
#   • À l’arrêt, le conteneur est automatiquement supprimé.
#
# 🧠 BONNES PRATIQUES :
#   • Conserver la correspondance des ports 18944:18944 pour Slicer/PlusServer.
#   • Utiliser `--name` unique pour éviter les conflits en environnement multi-container.
#   • Ajouter `-d` (detach) pour lancer le conteneur en arrière-plan si nécessaire.
#
# 📘 EN SYNTHÈSE :
#   ➜ Exécute le conteneur d’inférence Ultramotion avec accès GPU
#   ➜ Mappe le port hôte `$HostPort` vers le port OpenIGTLink 18944
#   ➜ Lance le service interactif nommé `umi_service`, supprimé à l’arrêt
####################################################################################################
