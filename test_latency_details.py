"""
Diagnostic: vérifier les données latency_details envoyées au dashboard
"""
import sys
sys.path.insert(0, "src")

from service.dashboard_service import MetricsCollector, DashboardConfig

config = DashboardConfig()
collector = MetricsCollector(config)
snapshot = collector.collect()

print("=== SNAPSHOT COMPLET ===")
print(f"latency_details présent: {'latency_details' in snapshot}")

if 'latency_details' in snapshot:
    details = snapshot['latency_details']
    print(f"\n=== LATENCY_DETAILS ===")
    print(f"Type: {type(details)}")
    print(f"Keys: {details.keys() if isinstance(details, dict) else 'N/A'}")
    
    if isinstance(details, dict):
        print(f"\nFrames: {details.get('frames', [])[:10]}... (total: {len(details.get('frames', []))})")
        print(f"RX→PROC: {details.get('rxproc', [])[:10]}... (total: {len(details.get('rxproc', []))})")
        print(f"PROC→TX: {details.get('proctx', [])[:10]}... (total: {len(details.get('proctx', []))})")
        print(f"RX→TX: {details.get('rxtx', [])[:10]}... (total: {len(details.get('rxtx', []))})")
        
        # Vérifier qu'on a bien des valeurs
        rxproc_vals = [v for v in details.get('rxproc', []) if v is not None]
        proctx_vals = [v for v in details.get('proctx', []) if v is not None]
        rxtx_vals = [v for v in details.get('rxtx', []) if v is not None]
        
        print(f"\n=== VALEURS NON-NULL ===")
        print(f"RX→PROC non-null: {len(rxproc_vals)}")
        print(f"PROC→TX non-null: {len(proctx_vals)}")
        print(f"RX→TX non-null: {len(rxtx_vals)}")
        
        if rxproc_vals:
            print(f"\nPremières valeurs RX→PROC: {rxproc_vals[:5]}")
        if proctx_vals:
            print(f"Premières valeurs PROC→TX: {proctx_vals[:5]}")
        if rxtx_vals:
            print(f"Premières valeurs RX→TX: {rxtx_vals[:5]}")
else:
    print("\n❌ latency_details ABSENT du snapshot !")
    print(f"\nClés présentes: {list(snapshot.keys())}")
