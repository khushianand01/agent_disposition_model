import os
import sys
import subprocess

# This is a wrapper to start the API from the root directory
# The actual implementation is in api/app.py

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api", "app.py")
    print(f"Starting API from {script_path}...")
    try:
        # Run the API script with the same arguments passed to this wrapper
        result = subprocess.run([sys.executable, script_path] + sys.argv[1:])
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nAPI stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting API: {e}")
        sys.exit(1)
