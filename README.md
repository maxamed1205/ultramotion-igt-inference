# ultramotion-igt-inference

Prototype de service d’inférence OpenIGTLink déployable

## Objectif
Ce dépôt contient une structure minimale, compatible avec les conteneurs Docker, pour un **service d’inférence en temps réel**.  
Ce service s’abonne à un flux d’images OpenIGTLink (provenant de **PlusServer**),  
exécute une pipeline de segmentation (**D-FINE + MobileSAM**)  
et republie une **carte binaire de labels (`BoneMask`)** vers **3D Slicer** via OpenIGTLink.

---

## État actuel
Ce dépôt est une **ébauche initiale** (scaffold) :
- service squelette fonctionnel,  
- Dockerfile prêt pour le build,  
- workflow CI configuré.  

L’intégration des modèles réels et les tests GPU intensifs seront réalisés ultérieurement.

---

## Démarrage rapide (mode développement)
1. **Construire le conteneur**  
   (nécessite **NVIDIA Docker** sur Linux ou **WSL2** sous Windows) :

   ```bash
   # La commande de build est indiquée dans la section Dockerfile
