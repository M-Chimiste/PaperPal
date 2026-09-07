"""Workers stop claiming work when their specific launching process disappears."""
import os
import threading
import time
import psutil


def worker_environment():
    parent = psutil.Process()
    return {**os.environ, 'THESEUS_PARENT_PID': str(parent.pid),
            'THESEUS_PARENT_STARTED': str(parent.create_time())}


def parent_alive(pid, started):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.create_time() == started
    except psutil.Error:
        return False


def watch_parent(worker):
    pid, started = os.getenv('THESEUS_PARENT_PID'), os.getenv('THESEUS_PARENT_STARTED')
    if not pid or not started:
        return  # Explicit standalone CLI workers have no parent owner.
    def watch():
        while worker.running:
            if not parent_alive(int(pid), float(started)):
                worker.running = False
                return
            time.sleep(2)
    threading.Thread(target=watch, daemon=True, name='worker-owner-watch').start()
