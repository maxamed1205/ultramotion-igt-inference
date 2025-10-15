# syntax=docker/dockerfile:1.7
###########################################################
# 🧠 Service IA temps réel (D-FINE + SAM + IGTLink)
# ---------------------------------------------------------
#  • Gestionnaire : uv (lecture pyproject.toml)
#  • Base : CUDA 12.1 (Ubuntu 22.04)
#  • Objectif : image reproductible, sans requirements.txt
###########################################################

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base
####################################################################################################
# 🔍 LIGNE DOCKER
# FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base
#
# 🧠 RÔLE GÉNÉRAL :
# Cette ligne définit l’image de base à partir de laquelle toutes les couches suivantes du conteneur
# seront construites. Elle fournit le système d’exploitation et les bibliothèques sur lesquelles
# on empile toutes les dépendances (Python, librairies IA, code applicatif, etc.).
# Ici, l’image est spécialisée pour l’exécution GPU sous CUDA.
#
# ⚙️ COMPOSITION DE L’IMAGE :
#   Élément          | Description                                    | Rôle dans le conteneur
#   -----------------|------------------------------------------------|------------------------
#   nvidia/cuda      | Repository Docker officiel de NVIDIA            | Images préconfigurées CUDA optimisées GPU
#   12.1.1           | Version majeure du toolkit CUDA                | Compatibilité garantie avec PyTorch cu121
#   runtime          | Variante d’exécution (sans outils nvcc, etc.)  | Bibliothèques CUDA essentielles uniquement
#   ubuntu22.04      | Base Ubuntu LTS (Jammy Jellyfish)              | Environnement Linux stable et récent
#
# 🧰 POURQUOI « runtime » ET PAS « devel » :
#   - `runtime` contient uniquement les bibliothèques nécessaires à l’exécution GPU :
#     (libcudart.so, libcublas.so, libcudnn.so, etc.)
#   - `devel` ajoute les outils de développement (nvcc, headers, etc.), inutile pour un service
#     d’inférence et alourdit l’image (+3 Go environ).
#   ✅ On choisit donc `runtime` pour réduire la taille (≈1.2 Go) et accélérer les builds.
#
# 💡 COMPATIBILITÉ PYTORCH :
#   - Cette version CUDA (12.1.1) correspond aux builds officiels PyTorch :
#     https://download.pytorch.org/whl/cu121
#   - Cela évite les erreurs de type « CUDA driver mismatch » lors de l’exécution.
#
# 🧩 SUFFIXE « AS base » :
#   - Permet de nommer cette étape du build et de la réutiliser ensuite (multi-stage build).
#   - Exemple : « FROM base AS deps » crée une nouvelle étape héritant de celle-ci.
#   - Avantages :
#       • séparation entre installation lourde et image finale (runtime léger)
#       • réduction du poids final
#       • caching efficace pour accélérer les builds successifs
#
# 📘 EN SYNTHÈSE :
#   ➜ Définit l’image de base GPU du conteneur :
#       • `nvidia/cuda:12.1.1-runtime-ubuntu22.04` → environnement Ubuntu 22.04 optimisé pour CUDA 12.1  
#       • Variante `runtime` : légère, contient uniquement les bibliothèques nécessaires à l’exécution GPU  
#   ➜ Compatible avec PyTorch CUDA 12.1 (cu121) et idéale pour un service d’inférence sans outils de compilation.  
#   ➜ Le suffixe `AS base` permet la réutilisation dans les builds multi-étapes pour optimiser taille et cache.
####################################################################################################



ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8
####################################################################################################
# 🔍 LIGNE DOCKER
# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONUNBUFFERED=1 \
#     PYTHONIOENCODING=UTF-8
#
# 🧠 RÔLE GÉNÉRAL :
# Cette instruction définit trois variables d’environnement essentielles au bon fonctionnement
# du conteneur pendant le build et à l’exécution. Ces variables influencent :
#   • le comportement des commandes APT (installation sans blocage)
#   • la gestion des flux de logs Python (sortie immédiate)
#   • la cohérence de l’encodage des caractères (UTF-8 par défaut)
#
# ⚙️ DÉTAILS DES VARIABLES :
#
# 1️⃣ DEBIAN_FRONTEND=noninteractive
#     → Supprime toute interaction utilisateur lors des installations via APT.
#     → Empêche les invites de configuration (ex. fuseau horaire, services, clavier...).
#     → Indispensable dans un build Docker automatisé (sinon le build bloque).
#
# 2️⃣ PYTHONUNBUFFERED=1
#     → Désactive le buffering de la sortie standard de Python (stdout/stderr).
#     → Permet d’obtenir les logs en temps réel dans la console Docker (utile pour debug).
#     → Sans cette option, les sorties peuvent être retardées ou perdues dans les buffers.
#
# 3️⃣ PYTHONIOENCODING=UTF-8
#     → Force l’encodage d’entrée/sortie Python en UTF-8.
#     → Évite les erreurs liées aux caractères spéciaux (accents, symboles scientifiques, etc.).
#     → Garantit une compatibilité cohérente entre le système (Ubuntu) et Python.
#
# 🧩 BONNES PRATIQUES :
#   • Ces trois variables sont devenues un standard dans les conteneurs Python modernes.
#   • Elles assurent un comportement stable, prévisible et compatible avec les pipelines CI/CD.
#   • Elles facilitent également l’intégration de journaux structurés (ex. FastAPI, PyTorch logs).
#
# 📘 EN SYNTHÈSE :
#   ➜ Garantit un comportement stable et automatisé du conteneur :
#       • `DEBIAN_FRONTEND=noninteractive` : évite tout blocage APT durant le build
#       • `PYTHONUNBUFFERED=1` : assure des logs Python visibles en temps réel
#       • `PYTHONIOENCODING=UTF-8` : homogénéise l’encodage des caractères (prévention erreurs Unicode)
#   ➜ Ensemble, ces variables rendent le build non-interactif, les logs immédiats et les traitements texte sûrs.
####################################################################################################


WORKDIR /app
####################################################################################################
# 🔍 LIGNE DOCKER
# WORKDIR /app
#
# 🧠 RÔLE GÉNÉRAL :
# Cette instruction définit le **répertoire de travail par défaut** à l’intérieur du conteneur.
# Toutes les commandes suivantes (`RUN`, `COPY`, `CMD`, `ENTRYPOINT`, etc.) seront exécutées
# à partir de ce dossier, sauf indication contraire.
#
# ⚙️ DÉTAILS :
#   • `/app` devient la racine logique du projet dans le conteneur.
#   • Les chemins relatifs utilisés ensuite (ex. `COPY src/ ./src/`) seront interprétés
#     relativement à ce répertoire.
#   • Cela garantit une structure cohérente entre ton environnement local et le conteneur.
#
# 🧩 RAISON DU CHOIX « /app » :
#   - `/app` est devenu une convention dans les images Docker Python et ML modernes.
#   - Simple, explicite et compatible avec la plupart des orchestrateurs (Docker Compose, K8s).
#   - Évite les dossiers système (`/usr`, `/opt`) réservés aux binaires du système de base.
#
# 💡 BONNES PRATIQUES :
#   • Toujours définir explicitement `WORKDIR` plutôt que d’utiliser des chemins relatifs.
#   • Cela améliore la lisibilité du Dockerfile et la portabilité du projet.
#   • Si tu montes un volume (`-v $(pwd):/app`), ton code local apparaîtra directement ici.
#
# 🧱 STRUCTURE ATTENDUE DANS /app :
#   /app/
#     ├── src/               → code source Python (modules, services)
#     ├── pyproject.toml     → métadonnées du projet (gérées par uv)
#     ├── uv.lock            → dépendances figées (lockfile)
#     ├── logs/              → journaux d’exécution (montables en volume)
#     └── ...                → autres ressources nécessaires à l’inférence
#
# 📘 EN SYNTHÈSE :
#   ➜ Facilite les opérations de copie, de montage et de déploiement.
#   ➜ Garantit la cohérence des chemins dans tout le pipeline Docker/Python.
####################################################################################################



# 🧩 Install system dependencies
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential ffmpeg \
      libsm6 libxext6 libxrender1 && \
    rm -rf /var/lib/apt/lists/*
####################################################################################################
# 🔍 LIGNE DOCKER
# RUN --mount=type=cache,target=/var/cache/apt \
#     apt-get update && apt-get install -y --no-install-recommends \
#       ca-certificates curl git build-essential ffmpeg \
#       libsm6 libxext6 libxrender1 && \
#     rm -rf /var/lib/apt/lists/*
#
# 🧠 RÔLE GÉNÉRAL :
# Cette commande installe les dépendances système nécessaires au bon fonctionnement
# des bibliothèques Python utilisées dans le projet (notamment PyTorch, OpenCV, et pyigtl).
# Elle repose sur le gestionnaire de paquets APT fourni par Ubuntu.
#
# ⚙️ DÉCOMPOSITION DE LA SYNTAXE :
#
# 1️⃣ RUN
#     → Indique à Docker d’exécuter les commandes shell listées sur une seule couche du build.
#     → Chaque RUN crée une nouvelle couche dans l’image Docker finale.
#
# 2️⃣ --mount=type=cache,target=/var/cache/apt
#     → Syntaxe Docker BuildKit : crée un **cache persistant** pour le dossier `/var/cache/apt`.
#     → Cela permet de réutiliser les fichiers de paquets téléchargés entre différents builds,
#       accélérant grandement les reconstructions de l’image.
#     → Nécessite BuildKit activé (par défaut sur Docker ≥ 23).
#
# 3️⃣ apt-get update
#     → Met à jour la liste locale des paquets disponibles depuis les dépôts APT configurés.
#     → Indispensable avant toute installation pour garantir des versions récentes.
#
# 4️⃣ apt-get install -y --no-install-recommends [paquets...]
#     → Installe les paquets nécessaires.
#       - `-y` : approuve automatiquement l’installation (sinon Docker bloquerait en attente d’un “Yes/No”).
#       - `--no-install-recommends` : n’installe **que les dépendances strictement requises**, 
#         évitant les paquets supplémentaires facultatifs → gain de place.
#
# 5️⃣ rm -rf /var/lib/apt/lists/*
#     → Supprime la base locale des métadonnées APT une fois l’installation terminée.
#     → Cela nettoie plusieurs centaines de Mo et garde l’image plus légère.
#
# 📦 LISTE DÉTAILLÉE DES PAQUETS INSTALLÉS :
#
#   • ca-certificates : permet à Python, curl ou git de valider les connexions HTTPS (certificats SSL)
#   • curl            : outil de transfert HTTP(S), souvent utilisé pour télécharger des modèles
#   • git             : nécessaire si des modules Python sont installés depuis des dépôts Git
#   • build-essential : regroupe gcc, g++, make… requis par certaines wheels (ex. pyigtl, torchmetrics)
#   • ffmpeg          : indispensable pour la lecture/écriture vidéo (OpenCV, traitement d’images)
#   • libsm6, libxext6, libxrender1 : bibliothèques X11 minimales pour le rendu graphique OpenCV
#
# 💡 BONNES PRATIQUES :
#   • Grouper ces opérations sur une seule ligne RUN permet de réduire le nombre de couches Docker.
#   • Utiliser --mount=type=cache accélère les builds itératifs et économise la bande passante.
#   • Nettoyer /var/lib/apt/lists est essentiel pour minimiser la taille de l’image finale.
#
# 🧩 EN SYNTHÈSE :
#   ➜ Installe un socle Linux minimal compatible avec les bibliothèques IA (torch, opencv, pyigtl)
#   ➜ Utilise le cache APT pour optimiser la vitesse des builds
#   ➜ Nettoie immédiatement les fichiers temporaires pour garder une image légère et propre
####################################################################################################


# ⚙️ Copy uv executable
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
####################################################################################################
# 🔍 LIGNE DOCKER
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
#
# 🧠 RÔLE GÉNÉRAL :
# Cette instruction copie l’exécutable **uv** directement depuis l’image officielle distribuée
# par Astral (hébergée sur GitHub Container Registry). 
# Elle permet d’installer l’outil de gestion Python « uv » sans passer par pip ni curl,
# garantissant ainsi une installation propre, rapide et sécurisée.
#
# ⚙️ DÉCOMPOSITION DE LA SYNTAXE :
#
# 1️⃣ COPY
#     → Commande Docker qui copie des fichiers ou répertoires dans l’image.
#       Syntaxe générale : COPY [--from=<image_source>] <source> <destination>
#
# 2️⃣ --from=ghcr.io/astral-sh/uv:latest
#     → Indique que la source à copier provient **d’une autre image Docker**, et non du contexte local.
#     → Ici, la source est l’image officielle `uv` publiée sur GitHub Container Registry :
#         https://ghcr.io/astral-sh/uv
#     → `:latest` sélectionne la version la plus récente de l’exécutable précompilé.
#     → Cette approche “multi-stage copy” évite d’installer ou de compiler quoi que ce soit :
#         ✅ pas de `pip install uv`
#         ✅ pas de dépendances Python intermédiaires
#         ✅ gain de temps et de fiabilité
#
# 3️⃣ /uv
#     → Chemin du binaire `uv` à l’intérieur de l’image source (Astral).
#     → L’exécutable statique y est placé à la racine dans l’image `ghcr.io/astral-sh/uv`.
#
# 4️⃣ /bin/uv
#     → Destination dans le conteneur final.
#     → `/bin` fait partie du PATH par défaut sur Linux, donc la commande `uv` sera accessible
#       globalement dans tout le conteneur, sans avoir besoin de spécifier un chemin absolu.
#
# 🧩 AVANTAGES DE CETTE APPROCHE :
#   • **Rapidité** : l’image `uv` contient déjà le binaire compilé → aucune compilation ni dépendance.
#   • **Sécurité** : pas de téléchargement dynamique (ni `curl | bash`).
#   • **Reproductibilité** : Docker garantit que l’on copie exactement le même binaire à chaque build.
#   • **Légèreté** : aucune couche Python inutile (contrairement à `pip install uv` qui installe Rust).
#
# 💡 À SAVOIR :
#   • `uv` est un gestionnaire de dépendances Python moderne (par Astral), compatible avec `pyproject.toml`.
#   • Il remplace `pip`, `venv`, `pip-tools` et `virtualenv` en une seule commande.
#   • Sa vitesse repose sur un cœur Rust extrêmement optimisé.
#
# 🧩 EXEMPLES D’USAGE DANS LE CONTENEUR :
#     uv python install 3.11        → installe une version locale de Python
#     uv sync                       → installe les dépendances listées dans pyproject.toml
#     uv add fastapi                → ajoute un module et met à jour uv.lock
#
# 📘 EN SYNTHÈSE :
#   ➜ Copie un exécutable précompilé depuis l’image officielle Astral
#   ➜ Installe `uv` sans dépendances externes, de manière fiable et rapide
#   ➜ Prépare l’environnement Python pour les étapes suivantes du Dockerfile
####################################################################################################


# 🧠 Install Python 3.11 with uv
RUN uv python install 3.11
####################################################################################################
# 🔍 LIGNE DOCKER
# RUN uv python install 3.11
#
# 🧠 RÔLE GÉNÉRAL :
# Cette instruction installe une **version spécifique de Python (ici 3.11)** directement à l’intérieur
# du conteneur, en utilisant le gestionnaire `uv` (développé par Astral).  
# Elle remplace avantageusement les méthodes classiques comme `apt install python3` ou `pyenv install`.
#
# ⚙️ DÉCOMPOSITION DE LA SYNTAXE :
#
# 1️⃣ RUN
#     → Exécute la commande indiquée à l’intérieur d’une nouvelle couche Docker.
#     → Ici, on exécute la commande `uv python install 3.11` au moment du build.
#
# 2️⃣ uv python install 3.11
#     → Appelle le sous-module `python` de l’outil `uv`.
#     → Installe la version 3.11.x de Python (la plus récente compatible).
#     → Télécharge un **binaire Python précompilé** (build officiel CPython) directement depuis
#       les serveurs d’Astral, sans passer par APT ni compilation locale.
#
# 🧰 COMPORTEMENT TECHNIQUE :
#   - `uv` installe Python dans un dossier interne à l’utilisateur :
#       /root/.local/share/uv/python/cp311/
#   - Ce dossier contient un environnement Python complet et isolé :
#       ├── bin/python  → exécutable principal
#       ├── lib/        → bibliothèques standard
#       └── include/    → headers
#   - L’installation est **reproductible**, car `uv` conserve un cache de versions exactes.
#
# 💡 COMPARAISON AVEC D’AUTRES MÉTHODES :
#
#   | Méthode                    | Avantage                          | Inconvénient                        |
#   |-----------------------------|-----------------------------------|--------------------------------------|
#   | apt install python3         | Simple, natif Ubuntu              | Versions souvent anciennes (3.10)    |
#   | pyenv install 3.11          | Flexible                          | Lent, compile depuis les sources     |
#   | conda install python=3.11   | Multi-env pratique                | Lourde, dépendances Anaconda         |
#   | ✅ uv python install 3.11   | Rapide, léger, reproductible      | Nécessite l’outil uv (déjà inclus)   |
#
# 🧩 AVANTAGES CONCRETS :
#   • **Vitesse** : installation instantanée depuis des binaires Rust optimisés.
#   • **Isolation** : chaque version de Python est stockée indépendamment, sans polluer /usr/bin.
#   • **Reproductibilité** : même build Python sur tous les environnements (dev, CI, prod).
#   • **Compatibilité** : les chemins s’intègrent proprement dans Docker et PATH.
#
# 🧠 BONNES PRATIQUES :
#   • Toujours utiliser une version explicite (`3.11`, pas `latest`) pour garantir la stabilité.
#   • Cette commande s’exécute une seule fois par image (le cache Docker la réutilisera ensuite).
#   • En combinaison avec la ligne suivante :
#         ENV PATH="/root/.local/share/uv/python/cp311/bin:${PATH}"
#     → cela rend Python 3.11 disponible globalement dans le conteneur.
#
# 📘 EN SYNTHÈSE :
#   ➜ Installe Python 3.11 de manière propre, rapide et isolée via uv
#   ➜ Évite les limitations de apt et pyenv
#   ➜ Prépare un environnement reproductible pour la suite du build
####################################################################################################

ENV PATH="/root/.local/share/uv/python/cp311/bin:${PATH}"
####################################################################################################
# 🔍 LIGNE DOCKER
# ENV PATH="/root/.local/share/uv/python/cp311/bin:${PATH}"
#
# 🧠 RÔLE GÉNÉRAL :
# Cette instruction définit (ou modifie) la variable d’environnement **PATH** dans le conteneur.
# Elle ajoute au début du PATH le répertoire contenant l’exécutable Python installé via `uv`,
# afin que les commandes `python`, `pip`, et `uv` soient directement accessibles dans le terminal
# sans préciser leur chemin absolu.
#
# ⚙️ DÉCOMPOSITION DE LA SYNTAXE :
#
# 1️⃣ ENV
#     → Définit une variable d’environnement persistante à l’intérieur de l’image Docker.
#     → Ces variables sont automatiquement disponibles pour toutes les commandes `RUN`, `CMD`,
#       `ENTRYPOINT`, ainsi que dans le shell du conteneur une fois démarré.
#
# 2️⃣ PATH="/root/.local/share/uv/python/cp311/bin:${PATH}"
#     → Modifie la variable d’environnement PATH pour y **préfixer** le chemin du binaire Python
#       installé par `uv` :
#         /root/.local/share/uv/python/cp311/bin
#
#     → `${PATH}` à la fin permet de **conserver le PATH existant** (hérité du système Ubuntu),
#       en ajoutant simplement le dossier de Python en tête.  
#       Cela signifie que, lorsqu’on appelle une commande :
#         - Docker cherche d’abord dans `/root/.local/share/uv/python/cp311/bin`
#         - puis dans les autres répertoires standards comme `/usr/bin`, `/bin`, etc.
#
# 🧰 CHEMIN DÉTAILLÉ :
#   • `/root/.local/share/uv/python/cp311/bin/`
#       ├── python      → exécutable principal (interpréteur CPython 3.11)
#       ├── pip         → installateur de paquets lié à cette version
#       ├── uv          → gestionnaire global
#       └── autres outils dépendant de Python
#
# 💡 POURQUOI AJOUTER CE CHEMIN :
#   - Les exécutables installés par `uv` ne sont **pas placés dans `/usr/bin`** (réservé au système).
#   - En les ajoutant explicitement dans le PATH :
#       → `python` devient disponible globalement,
#       → les outils installés via `pip` ou `uv` seront également accessibles directement.
#   - Cela rend l’environnement **fonctionnel et cohérent** sans modifier les chemins système.
#
# 🧩 BONNES PRATIQUES :
#   • Toujours **préfixer** (et non remplacer) le PATH existant : `${PATH}` doit rester en fin.
#   • Éviter d’écrire des chemins absolus dans les commandes suivantes (grâce à ce ENV).
#   • Cette approche garantit la portabilité et la reproductibilité entre conteneurs et hôtes.
#
# 🧠 EXEMPLE CONCRET :
#   Après cette instruction, les commandes suivantes fonctionneront partout :
#       RUN python --version      → affiche "Python 3.11.x"
#       RUN pip install numpy     → installe les paquets dans l’env. uv
#       RUN uv --version          → exécute le gestionnaire uv global
#
# 📘 EN SYNTHÈSE :
#   ➜ Rend l’environnement Python 3.11 installé par uv globalement accessible
#   ➜ Préserve le PATH système d’origine
#   ➜ Évite toute confusion entre Python système (APT) et Python géré par uv
####################################################################################################

# 🔧 Configure uv for PyTorch index
ENV UV_LINK_MODE=copy \
    UV_NATIVE_TLS=true \
    UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121
####################################################################################################
# 🔍 LIGNE DOCKER
# ENV UV_LINK_MODE=copy \
#     UV_NATIVE_TLS=true \
#     UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121
#
# 🧠 RÔLE GÉNÉRAL :
# Ce bloc configure trois variables d’environnement propres à l’outil **uv** (Astral).
# Elles influencent :
#   • la manière dont uv gère les liens symboliques et les fichiers installés,
#   • la sécurité des connexions réseau (TLS natif),
#   • et les sources de téléchargement des paquets Python (index supplémentaires).
#
# ⚙️ DÉCOMPOSITION VARIABLE PAR VARIABLE :
#
# 1️⃣ UV_LINK_MODE=copy
#     → Définit le mode de lien utilisé par uv lors de l’installation des paquets.
#     → Par défaut, uv peut “lier” les fichiers installés via symlink ou hardlink pour économiser
#       de l’espace disque.  
#     → Ici, on force le mode `copy` afin :
#         - d’éviter tout comportement inattendu avec les couches Docker (les liens peuvent casser),
#         - d’assurer que chaque environnement possède sa propre copie physique des fichiers.
#     → Ce mode améliore la **reproductibilité** et la **stabilité** des builds.
#
# 2️⃣ UV_NATIVE_TLS=true
#     → Indique à uv d’utiliser la bibliothèque TLS native du système (libssl/libcrypto),
#       au lieu de la sienne embarquée.
#     → Avantages :
#         • meilleure compatibilité avec le certificat racine du système (ca-certificates)
#         • évite des erreurs SSL lors de connexions HTTPS (notamment vers PyPI ou GitHub)
#         • accélère les requêtes réseau (meilleure intégration OpenSSL)
#     → C’est une bonne pratique pour tous les environnements Docker sécurisés.
#
# 3️⃣ UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121
#     → Ajoute un index PyPI secondaire pour que uv/pip puisse y rechercher les paquets manquants.
#     → Ici, l’URL correspond au dépôt officiel des wheels PyTorch **CUDA 12.1 (cu121)** :
#           https://download.pytorch.org/whl/cu121
#     → Cela garantit que torch, torchvision, et torchaudio seront installés avec la bonne
#       compatibilité CUDA (correspondant à l’image `nvidia/cuda:12.1.1`).
#
# 🧩 COMPORTEMENT RÉEL :
#   • Lors d’un `uv sync` ou `uv pip install`, uv :
#       1. lit d’abord `pyproject.toml` et `uv.lock`,
#       2. cherche les paquets sur PyPI (index principal),
#       3. si besoin, bascule vers l’index supplémentaire (ici celui de PyTorch),
#       4. télécharge et installe les versions compatibles CUDA 12.1.
#
# 💡 BONNES PRATIQUES :
#   • Toujours définir ces variables dans le Dockerfile (et non via CLI) pour assurer la cohérence
#     entre builds et déploiements.
#   • `UV_LINK_MODE=copy` est recommandé dans tous les environnements Docker ou CI/CD.
#   • `UV_NATIVE_TLS=true` améliore la sécurité des builds en évitant des erreurs de certificats.
#   • L’index PyTorch doit correspondre strictement à la version CUDA de l’image de base.
#
# 📘 EN SYNTHÈSE :
#   ➜ Configure uv pour un comportement stable et compatible avec Docker
#   ➜ Active le TLS natif pour sécuriser les téléchargements de paquets
#   ➜ Ajoute l’index PyTorch officiel CUDA 12.1 pour installer torch/vision sans conflit
####################################################################################################


# 📄 Copy dependency manifests
COPY pyproject.toml uv.lock* /app/
####################################################################################################
# 🔍 LIGNE DOCKER
# COPY pyproject.toml uv.lock* /app/
#
# 🧠 RÔLE GÉNÉRAL :
# Cette commande copie dans l’image Docker les fichiers de configuration du projet Python
# nécessaires à la gestion des dépendances avec **uv** :
#   - `pyproject.toml` : décrit les métadonnées et dépendances du projet (standard PEP 621)
#   - `uv.lock`        : fige les versions exactes installées (lockfile de reproductibilité)
#
# ⚙️ DÉCOMPOSITION DE LA SYNTAXE :
#
# 1️⃣ COPY
#     → Directive Docker pour copier des fichiers depuis le contexte de build (dossier local)
#       vers le système de fichiers de l’image en cours de construction.
#     → Syntaxe : COPY <source> <destination>
#     → Ici :
#         • <source> = pyproject.toml et uv.lock* (le * permet de copier même si le fichier n’existe pas)
#         • <destination> = /app/ (le répertoire de travail défini précédemment)
#
# 2️⃣ pyproject.toml
#     → Fichier de configuration central des projets Python modernes (norme PEP 518/621).
#     → Rôle :
#         - Déclare le nom, la version, les dépendances et le backend de build du projet.
#         - Remplace les anciens fichiers séparés (setup.py, requirements.txt, etc.).
#         - Est lu par `uv`, `pip`, `poetry` ou tout autre gestionnaire conforme PEP.
#     → Exemple minimal :
#           [project]
#           name = "inference-service"
#           dependencies = ["torch", "opencv-python", "pyigtl"]
#
# 3️⃣ uv.lock*
#     → Fichier généré automatiquement par `uv` après un `uv sync` ou `uv add`.
#     → Contient la **résolution exacte des dépendances** (noms, versions, hashes SHA256).
#     → Le `*` à la fin signifie « copier si présent » — pratique lors du premier build,
#       avant qu’un lockfile n’existe encore.
#     → Avantage :
#         - permet à Docker de **mettre en cache l’étape d’installation** tant que ces fichiers
#           n’ont pas changé : si le code source évolue mais pas les dépendances, le build reste instantané.
#
# 🧩 BONNES PRATIQUES :
#   • Toujours copier d’abord `pyproject.toml` et `uv.lock` avant le reste du code source :
#       → cela maximise la réutilisation du cache Docker.
#       → l’étape suivante (`uv sync --frozen --system`) ne se relancera que si les dépendances changent.
#   • Éviter de copier tout le projet à ce stade (garder cette couche spécifique aux dépendances).
#
# 💡 COMPORTEMENT TYPIQUE DU CACHE :
#   1. Docker compare les fichiers copiés à la version précédente du cache.
#   2. Si `pyproject.toml` ou `uv.lock` n’a pas changé → l’étape d’installation sera sautée.
#   3. Si un de ces fichiers est modifié → uv réinstalle les paquets (étape suivante).
#
# 📘 EN SYNTHÈSE :
#   ➜ Copie les manifestes de dépendances Python dans /app/
#   ➜ Permet à Docker de détecter les changements de dépendances via le cache
#   ➜ Prépare l’étape suivante (installation via uv) sans encore copier le code source
####################################################################################################


# 📦 Install project dependencies directly from pyproject
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --system
####################################################################################################
# 🔍 LIGNE DOCKER
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen --system
#
# 🧠 RÔLE GÉNÉRAL :
# Cette commande installe toutes les dépendances Python du projet dans l’environnement global
# du conteneur, à partir des fichiers `pyproject.toml` et `uv.lock`.
# Elle est exécutée via `uv`, le gestionnaire moderne de dépendances, pour une installation :
#   • rapide  ⚡  (grâce à son cœur Rust)
#   • déterministe  🔒  (versions verrouillées dans uv.lock)
#   • optimisée  💾  (avec cache persistant partagé entre builds)
#
# ⚙️ DÉCOMPOSITION DE LA SYNTAXE :
#
# 1️⃣ RUN
#     → Crée une nouvelle couche Docker exécutant les commandes spécifiées.
#     → Chaque RUN s’exécute dans le contexte défini par les instructions précédentes (ici /app).
#
# 2️⃣ --mount=type=cache,target=/root/.cache/uv
#     → Directive Docker BuildKit permettant de **monter un cache persistant** entre builds successifs.
#     → Ici, le cache `/root/.cache/uv` sera conservé entre les reconstructions de l’image :
#         - Les fichiers de paquets téléchargés par uv seront réutilisés.
#         - Les installations ultérieures seront quasi instantanées si rien n’a changé.
#     → Avantage : gros gain de temps et réduction de la bande passante.
#     → Nécessite Docker BuildKit (activé par défaut depuis Docker 23+).
#
# 3️⃣ uv sync
#     → Commande principale de `uv` pour synchroniser les dépendances.
#     → Compare les paquets actuellement installés avec ceux listés dans `pyproject.toml`
#       (et verrouillés dans `uv.lock`), puis installe ou met à jour si nécessaire.
#     → Équivalent moderne de :
#           pip install -r requirements.txt
#       mais en beaucoup plus rapide et fiable.
#
# 4️⃣ --frozen
#     → Force uv à utiliser **uniquement** les versions exactes du `uv.lock`.
#     → Si une dépendance n’est pas résolue dans le lockfile, l’installation échoue.
#     → Cela garantit la **reproductibilité totale** du build, idéale pour la CI/CD ou les déploiements.
#
# 5️⃣ --system
#     → Installe les dépendances directement dans l’environnement système du conteneur,
#       au lieu d’un environnement virtuel local.
#     → Avantage : les paquets sont immédiatement accessibles via `python`, `pip`, etc.
#     → Dans un conteneur isolé, il n’y a pas de risque de “polluer” le système hôte.
#
# 🧩 COMPORTEMENT TECHNIQUE :
#   1. uv lit `pyproject.toml` et `uv.lock`
#   2. il résout les dépendances et vérifie les versions/hashes
#   3. les fichiers wheel sont téléchargés (cache sous /root/.cache/uv)
#   4. installation dans `/root/.local/share/uv/python/cp311/lib/...`
#   5. couche Docker figée — inchangée tant que pyproject.toml et uv.lock ne changent pas
#
# 💡 AVANTAGES DE CETTE APPROCHE :
#   • **Déterministe** : les versions figées dans `uv.lock` assurent le même environnement partout.
#   • **Performant** : le cache évite de retélécharger les wheels à chaque build.
#   • **Propre** : pas de virtualenv inutile, pas de `pip install` répétitif.
#   • **Sécurisé** : `--frozen` bloque tout changement non validé par un commit du lockfile.
#
# 🧠 BONNES PRATIQUES :
#   • Toujours exécuter `uv lock` localement avant un build Docker pour générer le fichier `uv.lock`.
#   • Utiliser `--system` uniquement dans les conteneurs (jamais sur l’hôte).
#   • Si tu veux forcer une réinstallation complète, supprime manuellement le cache :  
#         docker builder prune --filter type=cache
#
# 📘 EN SYNTHÈSE :
#   ➜ Installe les dépendances Python listées dans pyproject.toml / uv.lock
#   ➜ Exploite un cache partagé pour accélérer les builds
#   ➜ Assure une installation stable, déterministe et reproductible
####################################################################################################

# 📁 Copy your application
COPY src/ /app/src/
####################################################################################################
# 🔍 LIGNE DOCKER
# COPY src/ /app/src/
#
# 🧠 RÔLE GÉNÉRAL :
# Cette instruction copie le code source du projet (dossier `src/` situé sur la machine hôte)
# vers le répertoire `/app/src/` à l’intérieur du conteneur.  
# C’est ici que sont placés les modules Python, les scripts, et le service d’inférence
# exécuté par l’ENTRYPOINT final.
#
# ⚙️ DÉCOMPOSITION DE LA SYNTAXE :
#
# 1️⃣ COPY
#     → Directive Docker permettant de copier des fichiers ou répertoires depuis
#       le **contexte de build** (ton répertoire local où se trouve le Dockerfile)
#       vers le système de fichiers de l’image.
#     → Syntaxe : COPY <source> <destination>
#
# 2️⃣ src/
#     → Dossier source local contenant le code Python du projet :
#         src/
#           ├── service/
#           │    └── inference_service.py   → point d’entrée de ton service IA
#           ├── models/                     → éventuellement : SAM, D-FINE, etc.
#           ├── utils/                      → fonctions communes, IGTLink, etc.
#           └── __init__.py
#     → Ce dossier doit exister à la racine du projet, à côté du Dockerfile et du pyproject.toml.
#
# 3️⃣ /app/src/
#     → Destination dans l’image.  
#       On le place sous `/app` (défini comme WORKDIR plus haut) pour garder une arborescence claire :
#         /app/
#           ├── src/        → code source principal
#           ├── logs/       → journaux d’exécution
#           ├── pyproject.toml / uv.lock
#           └── ...
#
# 🧩 RAISON DE CET ORDRE :
#   • On copie le code source **après** avoir installé les dépendances (`uv sync`).
#     Cela permet de :
#       → tirer parti du cache Docker : si le code change mais pas les dépendances,
#         l’étape d’installation reste inchangée (gain de temps).
#       → isoler le code applicatif de l’environnement système.
#   • En pratique :
#       - Si tu modifies uniquement ton code Python → build quasi instantané ✅
#       - Si tu modifies `pyproject.toml` → rebuild complet des dépendances 🔁
#
# 🧠 BONNES PRATIQUES :
#   • Ne jamais copier tout le dossier du projet (`COPY . /app`) : cela inclurait les fichiers inutiles
#     (.git/, .venv/, datasets, etc.) et casserait la logique de cache.
#   • Toujours cibler explicitement le dossier `src/` et les fichiers de config nécessaires.
#   • Ajouter un `.dockerignore` pour exclure les gros fichiers temporaires :
#         .venv/
#         __pycache__/
#         *.npy
#         *.pt
#         *.log
#
# 💡 EXEMPLE DE STRUCTURE DE PROJET (résumée) :
#   /Ultramotion-Inference/
#     ├── Dockerfile
#     ├── pyproject.toml
#     ├── uv.lock
#     ├── src/
#     │    └── service/
#     │         └── inference_service.py
#     └── logs/
#
# 📘 EN SYNTHÈSE :
#   ➜ Copie le code source de ton application dans l’image
#   ➜ Respecte la séparation “environnement / code”
#   ➜ Optimise le cache Docker pour des rebuilds rapides
####################################################################################################

# 🌐 Expose port for IGTLink
EXPOSE 18944/tcp
####################################################################################################
# 🔍 LIGNE DOCKER
# EXPOSE 18945/tcp
#
# 🧠 RÔLE GÉNÉRAL :
# Cette instruction indique au moteur Docker et aux utilisateurs de l’image que le conteneur
# utilise le **port TCP 18945** pour communiquer avec l’extérieur.
# Cela ne “publie” pas le port à lui seul, mais le déclare comme **point d’entrée réseau officiel**
# de l’application à l’intérieur du conteneur.
#
# ⚙️ DÉCOMPOSITION DE LA SYNTAXE :
#
# 1️⃣ EXPOSE
#     → Directive déclarative (métadonnée) dans Dockerfile.
#     → Syntaxe : EXPOSE <port>[/<protocole>]
#     → Elle ne crée pas de règle de pare-feu : elle sert à documenter les ports utilisés
#       et à aider `docker run -P` ou `docker-compose` à mapper automatiquement les ports.
#
# 2️⃣ 18945
#     → Port réseau standard utilisé par **OpenIGTLink (IGTL)**, le protocole de communication
#       temps réel entre applications médicales comme 3D Slicer, PlusServer, et ton service IA.
#     → Par convention :
#         - PlusServer et Slicer se connectent via TCP sur 18944 ou 18945.
#         - Ici, on choisit 18945 pour différencier le flux de ton service IA
#           (segmentation / traitement) du flux principal d’acquisition.
#
# 3️⃣ /tcp
#     → Spécifie le protocole de transport utilisé.  
#       OpenIGTLink repose sur **TCP** pour garantir la fiabilité et l’ordre des paquets
#       (contrairement à UDP, qui privilégie la vitesse au détriment de la fiabilité).
#
# 🧩 COMPORTEMENT DANS DOCKER :
#   - À ce stade, le port est simplement “exposé” à l’intérieur de l’image.
#   - Pour qu’il soit accessible depuis l’hôte, il faut le **publier** au lancement du conteneur :
#         docker run -p 18945:18945 inference:latest
#   - Cette commande crée un tunnel entre le port 18945 du conteneur et le même port sur ta machine.
#
# 💡 UTILISATION TYPIQUE DANS TON PIPELINE :
#   • Slicer ou PlusServer se connectent via :
#         OpenIGTLink → IP du conteneur :18945
#   • Ton service IA (dans `inference_service.py`) écoute sur ce port via pyigtl :
#         server = pyigtl.OpenIGTLinkServer(port=18945)
#   • Les messages IGTL (image, tracking, segmentation) transitent alors via ce canal TCP.
#
# 🧠 BONNES PRATIQUES :
#   • Toujours documenter explicitement les ports exposés dans le Dockerfile.
#   • Conserver les numéros cohérents avec ceux utilisés dans Slicer / PlusServer
#     pour simplifier les connexions et scripts d’automatisation.
#   • Pour les environnements de production ou de cluster, déclarer aussi les ports
#     dans les fichiers docker-compose.yml ou Kubernetes manifest (service/ingress).
#
# 📘 EN SYNTHÈSE :
#   ➜ Déclare le port réseau utilisé par OpenIGTLink (18944/TCP)
#   ➜ Facilite la communication entre ton conteneur IA et Slicer / PlusServer
#   ➜ Documente la couche réseau de ton service d’inférence
####################################################################################################

# 🧪 Basic test (torch + numpy)
RUN python - <<'PY'
import torch, numpy, cv2
print("Torch:", torch.__version__, "CUDA OK:", torch.cuda.is_available())
print("NumPy:", numpy.__version__, "OpenCV:", cv2.__version__)
PY

# 🚀 Launch inference service
ENTRYPOINT ["python", "src/service/inference_service.py"]
####################################################################################################
# 🔍 LIGNE DOCKER
# ENTRYPOINT ["python", "src/service/inference_service.py"]
#
# 🧠 RÔLE GÉNÉRAL :
# Cette directive définit **le point d’entrée (entrypoint)** du conteneur — c’est-à-dire
# la commande principale qui sera exécutée automatiquement au démarrage.
# Dans ton cas, elle lance le service d’inférence Python basé sur ton module IA.
#
# ⚙️ DÉCOMPOSITION DE LA SYNTAXE :
#
# 1️⃣ ENTRYPOINT
#     → Spécifie la commande à exécuter par défaut lorsque le conteneur démarre.
#     → Contrairement à `CMD`, `ENTRYPOINT` ne peut pas être facilement remplacé au moment
#       du `docker run` (il fixe la “mission” du conteneur).
#     → Syntaxe JSON (recommandée) : chaque argument est séparé par une virgule, sans shell.
#
# 2️⃣ ["python", "src/service/inference_service.py"]
#     → Appelle l’interpréteur Python défini plus haut (installé via uv et ajouté au PATH).
#     → Exécute ton script principal d’inférence :
#         src/service/inference_service.py
#
# 🧩 COMPORTEMENT TECHNIQUE :
#   • Au démarrage du conteneur, Docker exécute :
#         python src/service/inference_service.py
#   • Si le service se bloque ou se termine, le conteneur s’arrête.
#   • Les logs produits par ton application (stdout/stderr) apparaissent directement
#     dans la console Docker — parfait pour le suivi en temps réel.
#
# 💡 POURQUOI UTILISER ENTRYPOINT (ET NON CMD) :
#   | Directive | Comportement typique |
#   |------------|----------------------|
#   | CMD        | Peut être remplacée par des arguments `docker run <cmd>` |
#   | ENTRYPOINT | Est exécutée systématiquement, même si des arguments sont ajoutés |
#
#   → Exemple :
#        docker run inference:latest
#        → lance automatiquement ton service d’inférence
#
#        docker run -it inference:latest bash
#        → ⚠️ ne fonctionnera pas directement (ENTRYPOINT fixe), sauf si tu l’overrides :
#             docker run -it --entrypoint bash inference:latest
#
# 🧠 BONNES PRATIQUES :
#   • Utiliser `ENTRYPOINT` pour les services “longue durée” (serveur, démon, API…).
#   • Utiliser `CMD` uniquement pour les conteneurs à usage ponctuel (scripts, outils).
#   • Conserver la syntaxe JSON (plutôt que la syntaxe shell) pour éviter les interprétations
#     inattendues et garantir la portabilité Linux/Windows.
#
# 🧩 COMPORTEMENT ATTENDU POUR TON PROJET :
#   • Au démarrage :
#         → Le service se connecte à OpenIGTLink (port 18945)
#         → Il écoute les flux d’images et de tracking envoyés depuis PlusServer/Slicer
#         → Il effectue les inférences (segmentation, traitement, etc.)
#         → Il renvoie les résultats (masques, volumes, scores) en temps réel
#   • Ce service agit comme un **micro-serveur IA** intégré dans la chaîne Ultramotion.
#
# 📘 EN SYNTHÈSE :
#   ➜ Définit le point d’entrée principal du conteneur
#   ➜ Démarre automatiquement le service d’inférence IA (pyigtl + torch)
#   ➜ Fournit une exécution stable et autonome du pipeline IA dans Docker
####################################################################################################
