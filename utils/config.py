import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")

def load_config(path=None):
    """
    Load configuration from YAML file.
    """
    target_path = path if path else CONFIG_PATH
    try:
        with open(target_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"[Config] Warning: Config file not found at {target_path}. Using defaults or failing.")
        return {}
    except Exception as e:
        print(f"[Config] Error loading config: {e}")
        return {}

# Singleton-like access if needed
_CONFIG = None

def get_config():
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG
