import yaml

def load_config(config_path=r"D:\Documents\GitHub\credit_line_eligibility\src\config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)