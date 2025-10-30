# Frontend Style Audit - Module Graphiques UltraMotion IGT

## 🎨 Architecture CSS Existante

### Structure ITCSS (Inverted Triangle CSS)
L'architecture CSS suit le pattern ITCSS avec 7 couches :

```
01-settings/     → Variables globales
02-tools/        → Mixins et fonctions (vide)
03-generic/      → Reset CSS
04-elements/     → Éléments HTML nus (vide)
05-objects/      → Patterns de layout
06-components/   → Composants UI spécifiques
07-trumps/       → Utilitaires et overrides
```

## 🎯 Variables CSS Réutilisables

### Layout (à réutiliser)
```css
--sidebar-width: 320px;
--nav-height: 70px;
--max-width: 1600px;
```

### Couleurs Principales
```css
/* Backgrounds */
--bg-color: #0a0e1a;
--card-bg: #161b22;
--card-header-bg: #1c2128;

/* Texte */
--text-color: #e6edf3;
--text-muted: #8b949e;

/* Accents */
--accent-color: #4fc3f7;
--primary-color: #0ea5e9;
--success-color: #22c55e;
--warning-color: #f59e0b;
--danger-color: #ef4444;
```

### Couleurs Spécifiques GPU/Pipeline
```css
--gpu-utilization-high: #ef4444;
--gpu-utilization-medium: #f59e0b;
--gpu-utilization-low: #22c55e;

--pipeline-rx-gpu: #22c55e;
--pipeline-gpu-proc: #3b82f6;
--pipeline-proc-cpu: #eab308;
--pipeline-cpu-tx: #a855f7;
```

## 🧱 Classes de Layout à Réutiliser

### Conteneurs Principaux
- `.o-main-container` : Layout principal avec sidebar
- `.o-main-content` : Zone de contenu principale
- `.o-sidebar` : Sidebar fixe

### Headers de Contenu
- `.c-content-header` : En-tête de page avec titre et actions
- `.c-content-header__title` : Titre principal
- `.c-header-actions` : Zone d'actions (boutons)

## 🎴 Système de Cartes

### Classes de Base
```css
.c-card              → Carte principale
.c-card__header      → En-tête de carte
.c-card__title       → Titre de carte
.c-card__subtitle    → Sous-titre
.c-card__content     → Contenu principal
```

### Variantes de Cartes
- `.c-card--gpu` : Style spécifique GPU
- `.c-card--pipeline` : Style pipeline
- `.c-card--metric` : Métriques numériques

## 🎛️ Composants Boutons

### Classes Boutons
```css
.c-btn               → Bouton de base
.c-btn--primary      → Bouton principal (bleu)
.c-btn--secondary    → Bouton secondaire
.c-btn--success      → Bouton vert
.c-btn--warning      → Bouton orange
.c-btn--danger       → Bouton rouge
```

## 📊 Classes pour Graphiques (à créer)

### Recommandations pour compare.html
1. **Container principal** : `.c-chart-container`
2. **Zone graphique** : `.c-chart-area`
3. **Légende** : `.c-chart-legend`
4. **Contrôles** : `.c-chart-controls`

### Variables CSS à ajouter
```css
--chart-grid-color: rgba(255, 255, 255, 0.1);
--chart-axis-color: var(--text-muted);
--chart-tooltip-bg: var(--card-bg);
```

## 🎨 Typographie

### Fonts
- **Principale** : Inter (300, 400, 500, 600, 700)
- **Monospace** : JetBrains Mono (400, 500, 600)

### Classes Typographiques
- Variables de taille : `--font-size-sm`, `--font-size-xl`
- Poids : `--font-weight-semibold`

## 📱 Responsive

### Breakpoints à respecter
- Layout mobile : `max-width: 768px`
- Adaptation sidebar : collapse automatique
- Grilles adaptatives avec CSS Grid

## ✅ Recommandations pour compare.html

### Classes à Réutiliser Telles Quelles
- `.o-main-container` : Layout principal
- `.c-content-header` : En-tête de page
- `.c-btn` : Boutons de navigation
- Variables de couleurs GPU/Pipeline

### Classes à Étendre
- `.c-card` → `.c-chart-card` (cartes graphiques)
- `.c-card__content` → `.c-chart-content` (zone graphique)

### Nouvelles Classes à Créer
- `.c-compare-layout` : Layout spécifique à la page comparaison
- `.c-chart-toolbar` : Barre d'outils graphique
- `.c-chart-selector` : Sélecteur de métriques

## 🔗 Intégration avec WebSocket

### Classes d'État
- `.is-loading` : État de chargement
- `.is-error` : État d'erreur
- `.is-connected` : Connexion active

### Animations Existantes
- Transitions fluides : `var(--transition-smooth)`
- Effets hover : `transform: translateY(-4px)`

---

**Prêt pour l'implémentation** ✅
Toutes les variables et classes nécessaires sont identifiées et documentées.