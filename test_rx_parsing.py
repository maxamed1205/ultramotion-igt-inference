"""
Diagnostic : test du parsing des logs RX
"""
import re
import time

# Charger pipeline.log
with open("logs/pipeline.log", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()[-200:]  # Dernières 200 lignes

rx_t = {}
proc_t = {}
tx_t = {}

ts_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\]")

for line in lines:
    # Extraire timestamp
    t_match = ts_pattern.search(line)
    ts = None
    if t_match:
        ts = (
            time.mktime(time.strptime(f"{t_match.group(1)} {t_match.group(2)}", "%Y-%m-%d %H:%M:%S"))
            + int(t_match.group(3)) / 1000
        )
    
    # RX - chercher "[RX-SIM] Generated frame #XXX"
    if "[RX-SIM]" in line and "Generated frame" in line:
        m = re.search(r"frame [#—](\d+)", line)
        if m:
            fid = int(m.group(1))
            if ts:
                rx_t[fid] = ts
                print(f"✅ RX frame {fid}: timestamp={ts:.3f}")
    
    # PROC
    if "[PROC-SIM]" in line and "Processed frame" in line:
        m = re.search(r"frame [#—](\d+)", line)
        if m:
            fid = int(m.group(1))
            if ts:
                proc_t[fid] = ts
    
    # TX
    if "[TX-SIM]" in line and "Sent frame" in line:
        m = re.search(r"frame [#—](\d+)", line)
        if m:
            fid = int(m.group(1))
            if ts:
                tx_t[fid] = ts

print(f"\n📊 Résultat du parsing:")
print(f"   RX frames parsées: {len(rx_t)}")
print(f"   PROC frames parsées: {len(proc_t)}")
print(f"   TX frames parsées: {len(tx_t)}")

# Calculer latences RX→PROC pour les 5 premières frames communes
common = sorted(set(rx_t.keys()) & set(proc_t.keys()))[:5]
if common:
    print(f"\n⏱️  Latences RX→PROC (5 premières frames):")
    for fid in common:
        lat = (proc_t[fid] - rx_t[fid]) * 1000  # en ms
        print(f"   Frame {fid}: RX={rx_t[fid]:.3f}, PROC={proc_t[fid]:.3f}, Latence={lat:.2f} ms")
else:
    print("\n❌ Aucune frame commune entre RX et PROC !")
