# 🚀 UltraMotion IGT - Dashboard Professionnel

## 📋 Vue d'ensemble

Dashboard de monitoring temps réel pour les systèmes UltraMotion IGT développé pour le **Laboratoire de Cinésiologie des Hôpitaux Universitaires de Genève (HUG)**. Cette interface professionnelle offre un monitoring complet des performances GPU, du pipeline d'inférence et des métriques système.

## ✨ Nouvelles Fonctionnalités

### 🎨 Design Professionnel

- **Interface moderne** inspirée des environnements médicaux professionnels
- **Thème sombre sophistiqué** avec palette de couleurs optimisée pour la lecture prolongée
- **Typographie professionnelle** avec Inter et JetBrains Mono
- **Animations subtiles** et micro-interactions pour une expérience fluide
- **Layout responsive** optimisé pour tous les écrans

### 📊 Visualisations Avancées

- **Cercle de progression GPU** avec animation et couleurs dynamiques
- **Pipeline flow visuel** avec indicateurs de latence inter-étapes
- **Barres de progression** pour les files d'attente
- **Cartes métriques interactives** avec hover effects
- **Indicateurs de tendances** et statistiques

### 🔄 Fonctionnalités Temps Réel

- **WebSocket intégré** avec reconnexion automatique
- **Mise à jour live** des métriques GPU et système
- **Indicateurs de connexion** avec statuts visuels
- **Horodatage synchronisé** avec mise à jour continue

### 🏥 Contexte Médical/Hospitalier

- **Branding HUG** avec logo et identité visuelle appropriée
- **Terminologie médicale** adaptée au contexte hospitalier
- **Couleurs inspirées** de l'environnement médical (bleu stérilisation, vert monitoring)
- **Interface professionnelle** digne d'un environnement de recherche

## 🗂️ Architecture des Fichiers

```
src/sandbox/web_monitor/
├── assets/
│   ├── style.css                 # Styles principaux (refactorisé)
│   ├── theme_dark.css           # Thème sombre professionnel
│   ├── dashboard-extensions.css # Extensions et composants avancés
│   └── [autres assets...]
├── templates/
│   ├── dashboard.html           # Interface principale (complètement redesignée)
│   └── dashboard_old.html       # Sauvegarde de l'ancienne version
├── javascript/
│   ├── main.js                  # Logique principale WebSocket + UI
│   └── [autres scripts...]
└── [autres dossiers...]
```

## 🎨 Système de Design

### Palette de Couleurs

- **Primaire** : `#4fc3f7` (Bleu médical/technologique)
- **Secondaire** : `#7c3aed` (Violet innovation)
- **Succès** : `#10b981` (Vert monitoring médical)
- **Attention** : `#f59e0b` (Ambre hospitalier)
- **Erreur** : `#ef4444` (Rouge urgence)

### Typographie

- **Titre/Interface** : Inter (Google Fonts)
- **Données/Code** : JetBrains Mono (Google Fonts)
- **Icônes** : Font Awesome 6.4.0

### Composants

- **Cartes modulaires** avec headers et contenus structurés
- **Navigation fixe** avec branding HUG
- **Sidebar résumé** avec métriques clés
- **Footer informatif** avec statuts de connexion

## 📱 Responsive Design

- **Desktop** : Layout 2 colonnes (sidebar + main)
- **Tablet** : Layout empilé avec sidebar réduite
- **Mobile** : Interface mobile-first avec navigation adaptée

## 🔧 Configuration JavaScript

### WebSocket

```javascript
const DASHBOARD_CONFIG = {
    websocket: {
        url: 'ws://localhost:8050/ws',
        reconnectInterval: 5000,
        maxReconnectAttempts: 10
    },
    // ... autres configs
};
```

### Seuils de Performance

- **GPU Warning** : 70%
- **GPU Critical** : 90%
- **Latence Warning** : 50ms
- **Latence Critical** : 100ms

## 🚀 Démarrage Rapide

1. **Servir les fichiers** via votre serveur web habituel
2. **WebSocket** doit être disponible sur `ws://localhost:8050/ws`
3. **Ouvrir** `dashboard.html` dans un navigateur moderne

## 💡 Améliorations Apportées

### Performance
- Code JavaScript optimisé avec classes ES6
- Animations GPU-accelerated avec CSS transforms
- Gestion intelligente de la reconnexion WebSocket

### UX/UI
- **10x plus professionnel** que la version précédente
- Interface digne d'un environnement hospitalier
- Feedback visuel constant pour l'utilisateur
- Accessibilité améliorée (ARIA, focus states)

### Maintenance
- Code structuré et documenté
- Système de design cohérent
- CSS modulaire et réutilisable

## 🔜 Fonctionnalités Futures

- [ ] Export de données en CSV/JSON
- [ ] Paramètres utilisateur persistants
- [ ] Graphiques historiques
- [ ] Alertes configurables
- [ ] Multi-language support
- [ ] Thème clair optionnel

## 🏥 Contexte HUG

Ce dashboard a été spécialement conçu pour le **Laboratoire de Cinésiologie des Hôpitaux Universitaires de Genève**, dans le cadre des travaux de master sur les systèmes de monitoring UltraMotion IGT. L'interface reflète les standards professionnels attendus dans un environnement de recherche médicale de pointe.

---

**Développé avec ❤️ pour le Laboratoire de Cinésiologie HUG**  
*Interface professionnelle pour systèmes de monitoring médical avancé*