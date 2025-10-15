Param(
    [string]$ImageName = "maxamed1205/ultramotion-igt-inference:latest"
)
####################################################################################################
# 🔍 LIGNE SCRIPT
# Param(
#     [string]$ImageName = "maxamed1205/ultramotion-igt-inference:latest"
# )
#
# 🧠 RÔLE GÉNÉRAL :
# Déclare le ou les paramètres d’entrée du script PowerShell.
# Ici, on définit une variable `$ImageName` permettant de nommer l’image Docker à construire.
# Si l’utilisateur n’en fournit pas, une valeur par défaut est utilisée.
#
# ⚙️ DÉCOMPOSITION :
#   1️⃣ Param() :
#       → mot-clé PowerShell pour déclarer des paramètres en entrée de script.
#   2️⃣ [string] :
#       → spécifie que le paramètre est de type texte.
#   3️⃣ $ImageName :
#       → nom de la variable.
#   4️⃣ "maxamed1205/ultramotion-igt-inference:latest" :
#       → valeur par défaut utilisée si aucun argument n’est passé.
#
# 🧩 UTILISATION :
#   - Exécution simple :
#         .\build.ps1
#       → utilisera le nom par défaut.
#
#   - Exécution personnalisée :
#         .\build.ps1 -ImageName "huglab/ultramotion-inference:test"
#       → construira une image sous ce nom alternatif.
#
# 🧠 BONNES PRATIQUES :
#   • Toujours définir un paramètre par défaut pour éviter les erreurs d’exécution.
#   • Conserver un tag explicite `:latest` ou versionné (`:v1.0`, `:gpu-cu121`, etc.).
#
# 📘 EN SYNTHÈSE :
#   ➜ Crée un paramètre $ImageName pour nommer dynamiquement l’image Docker
#   ➜ Utilise une valeur par défaut standardisée
#   ➜ Permet d’exécuter le script sans argument ou avec un nom personnalisé
####################################################################################################

Write-Host "Building Docker image: $ImageName"

####################################################################################################
# 🔍 LIGNE SCRIPT
# Write-Host "Building Docker image: $ImageName"
#
# 🧠 RÔLE GÉNÉRAL :
# Affiche dans la console un message informatif avant le lancement du build.
#
# ⚙️ DÉCOMPOSITION :
#   • `Write-Host` → fonction PowerShell d’affichage dans la console (stdout).
#   • `"Building Docker image: $ImageName"` → message dynamique indiquant quelle image sera construite.
#
# 🧩 UTILITÉ :
#   • Donne un retour visuel immédiat à l’utilisateur.
#   • Utile dans les logs CI/CD (GitHub Actions, Azure, etc.) pour identifier quelle image est buildée.
#
# 📘 EN SYNTHÈSE :
#   ➜ Affiche le nom de l’image en cours de build
#   ➜ Améliore la lisibilité et la traçabilité des builds
####################################################################################################

docker build -t $ImageName ..


####################################################################################################
# 🔍 LIGNE SCRIPT
# docker build -t $ImageName ..
#
# 🧠 RÔLE GÉNÉRAL :
# Lance la construction de l’image Docker à partir du Dockerfile.
#
# ⚙️ DÉCOMPOSITION :
#   • `docker build` → commande principale pour construire une image Docker.
#   • `-t $ImageName` → tague l’image avec le nom fourni en paramètre.
#   • `..` → indique que le Dockerfile se trouve dans le dossier parent (`/docker/..`).
#
# 🧩 CONTEXTE TYPIQUE :
#   Arborescence :
#     ultramotion-igt-inference/
#     ├── Dockerfile
#     ├── pyproject.toml
#     └── docker/
#         └── build.ps1
#
#   Ici, le `..` fait remonter PowerShell d’un dossier pour accéder au Dockerfile racine.
#
# 🧠 BONNES PRATIQUES :
#   • Toujours spécifier `-t` pour nommer l’image (sinon elle sera anonyme).
#   • Utiliser un chemin relatif clair (`..`) plutôt qu’un absolu.
#
# 📘 EN SYNTHÈSE :
#   ➜ Construit l’image Docker à partir du Dockerfile racine
#   ➜ Associe le tag spécifié (ex. latest, test, v1.0)
#   ➜ Adapte le chemin du Dockerfile à la structure du projet
####################################################################################################

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed"
    exit $LASTEXITCODE
}

####################################################################################################
# 🔍 LIGNE SCRIPT
# if ($LASTEXITCODE -ne 0) {
#     Write-Error "Docker build failed"
#     exit $LASTEXITCODE
# }
#
# 🧠 RÔLE GÉNÉRAL :
# Vérifie si la commande précédente (`docker build`) s’est exécutée avec succès
# et interrompt le script en cas d’échec.
#
# ⚙️ DÉCOMPOSITION :
#   • `$LASTEXITCODE` → contient le code de retour de la dernière commande exécutée :
#         0  = succès
#         ≠0 = erreur
#   • `Write-Error` → affiche un message d’erreur formaté en rouge dans la console.
#   • `exit $LASTEXITCODE` → stoppe le script en renvoyant le même code d’erreur
#     (utile en CI/CD pour signaler un échec).
#
# 🧩 COMPORTEMENT :
#   Si le build échoue :
#       ❌ “Docker build failed”
#       Le script s’arrête immédiatement.
#
# 🧠 BONNES PRATIQUES :
#   • Toujours tester `$LASTEXITCODE` après une commande critique.
#   • Utiliser `exit $LASTEXITCODE` pour propager l’erreur au système appelant.
#
# 📘 EN SYNTHÈSE :
#   ➜ Vérifie le succès du build Docker
#   ➜ Affiche un message d’erreur clair en cas d’échec
#   ➜ Stoppe le script proprement avec le bon code de retour
####################################################################################################


Write-Host "Build completed: $ImageName"

####################################################################################################
# 🔍 LIGNE SCRIPT
# Write-Host "Build completed: $ImageName"
#
# 🧠 RÔLE GÉNÉRAL :
# Informe l’utilisateur que la construction de l’image s’est terminée avec succès.
#
# ⚙️ DÉCOMPOSITION :
#   • `Write-Host` → imprime un message simple dans la console.
#   • `$ImageName` → affiche le nom/tag de l’image finalisée.
#
# 🧩 UTILITÉ :
#   • Confirme visuellement que tout le processus s’est déroulé sans erreur.
#   • Fournit le nom exact de l’image, utile pour la commande suivante :
#         docker run -it --rm $ImageName
#
# 🧠 BONNES PRATIQUES :
#   • Ajouter un ✅ ou un message clair pour signaler la réussite.
#
# 📘 EN SYNTHÈSE :
#   ➜ Affiche un message de réussite de build
#   ➜ Fournit le tag exact de l’image prête à l’exécution
####################################################################################################
