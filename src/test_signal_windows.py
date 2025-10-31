import signal
import threading
import time
import sys

print("DEBUG: Starting Windows signal test")

stop_event = threading.Event()

def shutdown_handler(signum, frame):
    print(f"DEBUG: Signal {signum} received")
    stop_event.set()

def keyboard_interrupt_handler():
    """Alternative handler using try/except"""
    try:
        while not stop_event.is_set():
            stop_event.wait(0.1)
    except KeyboardInterrupt:
        print("DEBUG: KeyboardInterrupt caught")
        stop_event.set()

# Test 1: Standard signal handler
print("DEBUG: Testing standard signal handler...")
signal.signal(signal.SIGINT, shutdown_handler)

print("Running... Press Ctrl+C to stop.")
print("If this doesn't work, we'll try another method...")

try:
    stop_event.wait(timeout=3)  # Wait 3 seconds for manual test
    if not stop_event.is_set():
        print("DEBUG: Timeout reached, trying alternative method")
        stop_event.clear()
        
        # Test 2: Using try/catch for KeyboardInterrupt
        print("DEBUG: Using KeyboardInterrupt method...")
        print("Running again... Press Ctrl+C to stop.")
        keyboard_interrupt_handler()
        
except KeyboardInterrupt:
    print("DEBUG: KeyboardInterrupt caught at top level")

print("DEBUG: Script finished")