"""
NIFTY Options Backtester - Python Launcher
===========================================
Cross-platform launcher with GUI popup
"""

import subprocess
import webbrowser
import time
import os
import sys
import socket

def is_port_in_use(port):
    """Check if port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """Kill any process on the specified port (Windows)"""
    try:
        result = subprocess.run(
            f'netstat -aon | findstr :{port} | findstr LISTENING',
            shell=True, capture_output=True, text=True
        )
        for line in result.stdout.strip().split('\n'):
            if line:
                pid = line.strip().split()[-1]
                subprocess.run(f'taskkill /F /PID {pid}', shell=True, capture_output=True)
    except:
        pass

def main():
    # Configuration
    PORT = 8504
    APP_FILE = "nifty_options_backtest_v3.py"
    APP_DIR = r"C:\Users\rohit\OneDrive\Desktop\TradersCafe"
    URL = f"http://localhost:{PORT}"
    
    print("\n" + "="*60)
    print("   NIFTY OPTIONS BACKTESTER V3.1 - ULTIMATE EDITION")
    print("="*60 + "\n")
    
    # Change to app directory
    os.chdir(APP_DIR)
    
    # Kill existing process on port
    if is_port_in_use(PORT):
        print(f"Port {PORT} is in use. Cleaning up...")
        kill_process_on_port(PORT)
        time.sleep(1)
    
    # Start Streamlit
    print(f"Starting dashboard on {URL}...")
    
    process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", APP_FILE, 
         "--server.headless", "true", 
         "--server.port", str(PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    )
    
    # Wait for server to start
    print("Waiting for server...")
    for _ in range(10):
        time.sleep(1)
        if is_port_in_use(PORT):
            break
    
    # Open browser
    print(f"Opening browser: {URL}")
    webbrowser.open(URL)
    
    print("\n" + "-"*60)
    print(f"Dashboard is running at: {URL}")
    print("Press Ctrl+C to stop the server")
    print("-"*60 + "\n")
    
    # Keep running
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        process.terminate()

if __name__ == "__main__":
    main()
