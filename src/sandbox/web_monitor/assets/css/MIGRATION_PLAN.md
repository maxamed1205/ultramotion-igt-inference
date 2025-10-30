# Plan de Migration CSS - UltraMotion IGT

## Phase 1 : Préparation (Immédiate)
✅ Structure ITCSS créée
✅ Variables globales extraites
✅ Reset et layout de base migré
✅ Composants navigation, cartes, boutons créés

## Phase 2 : Migration des composants spécialisés (Prochaine étape)

### Composants à migrer depuis style.css :

1. **Pipeline (_pipeline.css)** - Lignes 590-720
   - `.pipeline-flow`
   - `.pipeline-stage`
   - `.pipeline-summary`
   - `.latency-bar-container`

2. **GPU Metrics (_gpu-metrics.css)** - Lignes 770-890
   - `.gpu-metrics`
   - `.metric-card`
   - `.metric-icon`

3. **Utilization (_utilization.css)** - Lignes 890-1000
   - `.gpu-utilization`
   - `.utilization-circle`
   - `.gpu-details`

4. **Queues (_queues.css)** - Lignes 1000-1100
   - `.queue-indicators`
   - `.queue-item`

5. **Footer (_footer.css)** - Lignes 1200-1300
   - `.footer`
   - `.footer-content`

## Phase 3 : Mise à jour des templates

### Templates à modifier :
- `layout_base.html` - Changer les imports CSS
- `dashboard.html` - Mettre à jour les classes BEM
- Composants HTML - Adopter la nomenclature BEM

### Exemple de changement de classes :
```html
<!-- Ancien -->
<div class="card card-primary">
  <div class="card-header">
    <h3>Titre</h3>
  </div>
</div>

<!-- Nouveau BEM -->
<div class="c-card c-card--primary o-grid-area--pipeline">
  <div class="c-card__header">
    <h3 class="c-card__title">Titre</h3>
  </div>
</div>
```

## Phase 4 : Nettoyage et optimisation

1. ⚠️ **IMPORTANT** : Garder `style.css` temporairement pour compatibilité
2. Migrer progressivement section par section
3. Tester chaque migration
4. Supprimer les sections migrées de `style.css`
5. Optimiser les imports CSS

## Avantages de cette architecture :

✅ **Maintenabilité** : Chaque composant dans son fichier
✅ **Réutilisabilité** : Classes BEM réutilisables
✅ **Performance** : Import sélectif possible
✅ **Équipe** : Plusieurs développeurs peuvent travailler simultanément
✅ **Debug** : Plus facile de localiser les problèmes CSS

## Prochaines étapes :

1. Migrer le composant Pipeline en premier (le plus complexe)
2. Tester sur le dashboard
3. Continuer avec GPU Metrics
4. Mettre à jour `layout_base.html` pour utiliser `main.css`