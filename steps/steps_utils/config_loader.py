import yaml

def load_config(path=r"D:\Documents\GitHub\credit_line_eligibility\steps\config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config