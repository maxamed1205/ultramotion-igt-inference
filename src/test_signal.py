import signal
import threading
import time

# Create a global stop event
stop_event = threading.Event()

# Set up signal handler for graceful shutdown (e.g., Ctrl+C)
def shutdown_handler(signum, frame):
    """Handle shutdown signal gracefully"""
    print("DEBUG: Shutdown signal received!")
    stop_event.set()  # Set the event to stop the main loop

if __name__ == "__main__":
    print("DEBUG: Starting test script")
    
    # Set up signal handler for graceful shutdown (e.g., Ctrl+C)
    signal.signal(signal.SIGINT, shutdown_handler)
    
    print("DEBUG: Signal handler set up")
    print("Running... Press Ctrl+C to stop.")
    
    # Main loop to wait for shutdown signal
    try:
        stop_event.wait()  # Wait until the event is set (Ctrl+C will set it)
        print("DEBUG: Stop event received, exiting gracefully")
    except KeyboardInterrupt:
        print("DEBUG: KeyboardInterrupt caught")
    
    print("DEBUG: Script ended")