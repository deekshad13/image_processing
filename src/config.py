import yaml
import os

_CONFIG = None

def load_config(config_path=None):
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config", "config.yaml"
        )
    with open(config_path) as f:
        _CONFIG = yaml.safe_load(f)
    return _CONFIG

def get_config():
    if _CONFIG is None:
        return load_config()
    return _CONFIG
