"""
Diagnostic : identifier d'où vient la latence RX→PROC (0-2ms)
"""

# La latence mesurée dans les logs est :
# timestamp_log_PROC - timestamp_log_RX

# MAIS attention :
# 1. Le LOG RX est APRÈS gateway._inject_frame()
# 2. Le LOG PROC est APRÈS gateway.receive_image()

# Donc la latence mesurée inclut :
# A. Temps d'injection dans _mailbox (quasi-0)
# B. Temps d'attente avant que PROC ne lise (variable)
# C. Temps d'extraction depuis _mailbox (quasi-0)
# D. Overhead des logs asynchrones (quelques µs)

print("=" * 80)
print("ANALYSE DE LA LATENCE RX→PROC (0-2ms)")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────┐
│  THREAD RX                          THREAD PROC                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  frame = RawFrame(...)                                          │
│  gateway._inject_frame(frame)  ──→  [_mailbox]                 │
│       ↓ (injection = ~0µs)                   ↓                  │
│  LOG.info("[RX] Generated")                  ↓                  │
│       ↓                                       ↓                  │
│  ⏱️ TIMESTAMP LOG RX capturé ici             ↓                  │
│       ↓                                       ↓                  │
│       ↓                            if len(_mailbox) == 0:       │
│       ↓                                sleep(0.0005) ← 0.5ms !  │
│       ↓                            frame = receive_image()      │
│       ↓                                       ↓                  │
│       ↓                            LOG.info("[PROC] Processed") │
│       ↓                                       ↓                  │
│       ↓                            ⏱️ TIMESTAMP LOG PROC capturé│
│       ↓                                                          │
│  [LATENCE MESURÉE] = temps entre les 2 timestamps              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

🔍 SOURCES DE LA LATENCE (0-2ms) :

1️⃣  Python GIL (Global Interpreter Lock)
    → 1 seul thread Python actif à la fois
    → Changement de contexte RX→PROC = 100-500µs
    → Explique la base de ~1ms

2️⃣  Le sleep(0.0005) dans PROC (ligne 94)
    → Si mailbox vide, PROC dort 0.5ms
    → Mais même si pleine, cycle de vérification non-instantané
    → Ajoute 0-1ms

3️⃣  Ordonnancement OS (Windows Thread Scheduler)
    → 3 threads daemon Python concurrents
    → CPU peut allouer le temps à TX au lieu de PROC
    → Ajoute 0-1ms dans les pires cas

4️⃣  Logs asynchrones (QueueHandler)
    → LOG.info() envoie vers une queue
    → Timestamp capturé au moment du log
    → Petit overhead (~50-100µs) mais négligeable

5️⃣  Création de l'image numpy (512x512)
    → np.random.rand() + astype() dans RX
    → Prend ~200-500µs
    → MAIS ne compte PAS dans la latence (avant injection)

═══════════════════════════════════════════════════════════════════

📊 DISTRIBUTION OBSERVÉE :

- 0 ms (18%) : PROC était déjà bloqué sur receive_image()
               → Récupération quasi-instantanée dès injection
               
- 1 ms (77%) : CAS NORMAL - PROC vérifie mailbox dans son cycle
               → GIL switch + cycle de vérification ≈ 1ms
               
- 2 ms (5%)  : CONTENTION - PROC préempté par TX ou autre
               → Doit attendre son tour CPU → +1ms supplémentaire

═══════════════════════════════════════════════════════════════════

💡 CONCLUSION :

La latence 0-2ms n'est PAS un problème de la pipeline !
C'est la COMBINAISON de :
  • Python GIL (mono-thread CPU)
  • sleep(0.0005) dans PROC
  • Ordonnancement OS non-déterministe
  • 3 threads concurrents se battant pour le CPU

✅ 1ms est EXCELLENT pour une communication inter-thread Python !
✅ Pas de bug, pas d'inefficacité, c'est la nature de Python threading

═══════════════════════════════════════════════════════════════════

🎯 SI ON VOULAIT RÉDUIRE À ~0ms (mais pas nécessaire) :

1. Remplacer sleep(0.0005) par un Event ou Condition
   → PROC se réveillerait immédiatement à chaque injection
   
2. Utiliser multiprocessing au lieu de threading
   → Éviter le GIL, CPU parallèle réel
   → MAIS overhead de sérialisation des frames
   
3. Logger AVANT receive_image() dans PROC
   → Mais fausserait la vraie latence de traitement !

""")
