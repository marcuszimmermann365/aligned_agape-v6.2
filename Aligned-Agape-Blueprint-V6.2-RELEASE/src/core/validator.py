
import yaml

class ValidationError(Exception):
    pass

def validate_yaml_file(path):
    try:
        data = yaml.safe_load(open(path, encoding="utf-8").read())
    except Exception as e:
        raise ValidationError(f"YAML error: {e}")
    if "actors" not in data or not isinstance(data["actors"], list):
        raise ValidationError("No actors[] in scenario.")
    return True
