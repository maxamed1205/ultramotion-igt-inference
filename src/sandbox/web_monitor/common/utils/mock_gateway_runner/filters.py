import logging

class MockProcFilter(logging.Filter):
    def filter(self, record):
        # Filtre les logs liés au traitement PROC
        if "Processing frame" in record.getMessage():
            return False  # Ignore ces logs
        return True

class MockTxFilter(logging.Filter):
    def filter(self, record):
        # Filtre les logs liés à l'envoi des masques
        if "mark_tx" in record.getMessage():
            return False  # Ignore ces logs
        return True

class MockGpuFilter(logging.Filter):
    def filter(self, record):
        # Filtre les logs liés au GPU
        if "CUDA" in record.getMessage():
            return False  # Ignore ces logs
        return True
