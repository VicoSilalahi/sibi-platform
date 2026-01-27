import os
import sys
import argparse

def create_systemd_service(install=False):
    """Generate and optionally install a systemd service file for the current user."""
    
    python_path = sys.executable
    script_path = os.path.abspath("sibi.py")
    working_dir = os.path.dirname(script_path)
    
    service_content = f"""[Unit]
Description=SIBI Platform Gesture Recognition Service
After=network.target

[Service]
Type=simple
WorkingDirectory={working_dir}
ExecStart={python_path} {script_path} run --modality camera --headless
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""
    
    print("--- SIBI Service Configuration ---")
    print(service_content)
    
    if install:
        if os.name != 'posix':
            print("Error: Service installation is only supported on Linux/POSIX systems.")
            return

        service_dir = os.path.expanduser("~/.config/systemd/user")
        service_path = os.path.join(service_dir, "sibi.service")
        
        os.makedirs(service_dir, exist_ok=True)
        
        with open(service_path, "w") as f:
            f.write(service_content)
            
        print(f"\n[SUCCESS] Service file created at: {service_path}")
        print("\nTo enable and start the service, run:")
        print("  systemctl --user daemon-reload")
        print("  systemctl --user enable sibi.service")
        print("  systemctl --user start sibi.service")
        print("\nTo check status:")
        print("  systemctl --user status sibi.service")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="Install the service file to ~/.config/systemd/user")
    args = parser.parse_args()
    create_systemd_service(args.install)
