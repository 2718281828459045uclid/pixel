#!/usr/bin/env python3
"""
Launch the live preview window and the control panel together.

    python3 run.py

Both are independent processes talking through `.live_settings.json`, so you can
also run them separately (python3 blob_window.py / python3 control_panel.py).
Closing either one leaves the other running; Ctrl-C here stops both.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    py = sys.executable
    window = subprocess.Popen([py, str(HERE / "blob_window.py")])
    panel  = subprocess.Popen([py, str(HERE / "control_panel.py")])
    procs = [window, panel]
    try:
        # Exit once either window is closed, then tear down the other.
        while True:
            for p in procs:
                if p.poll() is not None:
                    for other in procs:
                        if other.poll() is None:
                            other.terminate()
                    return
            try:
                window.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                pass
    except KeyboardInterrupt:
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
