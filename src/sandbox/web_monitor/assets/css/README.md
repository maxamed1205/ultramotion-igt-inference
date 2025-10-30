# Architecture CSS - UltraMotion IGT Dashboard

## Structure ITCSS (Inverted Triangle CSS)

```
assets/css/
├── 01-settings/          # Variables globales, configuration
│   ├── _variables.css
│   ├── _colors.css
│   ├── _typography.css
│   └── _spacing.css
│
├── 02-tools/             # Mixins et fonctions utilitaires
│   ├── _mixins.css
│   └── _utilities.css
│
├── 03-generic/           # Reset, normalize, box-sizing
│   ├── _reset.css
│   └── _base.css
│
├── 04-elements/          # Styles pour éléments HTML nus
│   ├── _typography.css
│   └── _forms.css
│
├── 05-objects/           # Patterns de layout réutilisables
│   ├── _grid.css
│   ├── _container.css
│   └── _layout.css
│
├── 06-components/        # Composants UI spécifiques
│   ├── _navigation.css
│   ├── _cards.css
│   ├── _buttons.css
│   ├── _pipeline.css
│   ├── _gpu-metrics.css
│   ├── _utilization.css
│   ├── _queues.css
│   └── _footer.css
│
├── 07-trumps/           # Utilitaires et overrides
│   ├── _responsive.css
│   ├── _animations.css
│   └── _utilities.css
│
└── main.css             # Point d'entrée principal
```

## Avantages de cette structure

1. **Séparation claire des responsabilités**
2. **Facilité de maintenance**
3. **Réutilisabilité des composants**
4. **Performance optimisée**
5. **Développement en équipe facilité**

## Convention de nommage BEM

- **Block** : `.card`
- **Element** : `.card__header`, `.card__content`
- **Modifier** : `.card--primary`, `.card--large`