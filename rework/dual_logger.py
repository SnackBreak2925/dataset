import sys
import os


class DualLogger:
    def __init__(self, model_name: str, timestamp: int, logs_dir="logs"):
        os.makedirs(logs_dir, exist_ok=True)
        self.log_file_path = os.path.join(
            logs_dir, f"stdout-{model_name}-{timestamp}.log"
        )
        self.terminal = sys.stdout
        self.log = open(self.log_file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
