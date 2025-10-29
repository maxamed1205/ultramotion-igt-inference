import sys
sys.path.insert(0, 'src')

# Test parsing timestamps
import re
import time

with open('logs/pipeline.log', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

ts_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),(\d{3})\]")

proc_t = {}
tx_t = {}

for line in lines:
    t_match = ts_pattern.search(line)
    ts = None
    if t_match:
        ts = (
            time.mktime(time.strptime(f"{t_match.group(1)} {t_match.group(2)}", "%Y-%m-%d %H:%M:%S"))
            + int(t_match.group(3)) / 1000
        )
    
    if "[PROC-SIM]" in line and "Processed frame" in line:
        m = re.search(r"frame [#—](\d+)", line)
        if m:
            fid = int(m.group(1))
            if ts:
                proc_t[fid] = ts
    
    elif "[TX-SIM]" in line and "Sent frame" in line:
        m = re.search(r"frame [#—](\d+)", line)
        if m:
            fid = int(m.group(1))
            if ts:
                tx_t[fid] = ts

print(f"PROC timestamps collectés: {len(proc_t)}")
print(f"TX timestamps collectés: {len(tx_t)}")
print(f"\nPremiers PROC timestamps: {dict(list(proc_t.items())[:3])}")
print(f"Premiers TX timestamps: {dict(list(tx_t.items())[:3])}")

# Calculer latences
if proc_t and tx_t:
    common = set(proc_t.keys()) & set(tx_t.keys())
    print(f"\nFrames communes PROC/TX: {len(common)}")
    if common:
        latencies = []
        for fid in sorted(common)[:5]:
            lat_ms = (tx_t[fid] - proc_t[fid]) * 1000.0
            latencies.append(lat_ms)
            print(f"Frame {fid}: PROC={proc_t[fid]:.3f}, TX={tx_t[fid]:.3f}, Latence={lat_ms:.2f} ms")
        
        if latencies:
            print(f"\nLatence moyenne PROC→TX: {sum(latencies)/len(latencies):.2f} ms")
