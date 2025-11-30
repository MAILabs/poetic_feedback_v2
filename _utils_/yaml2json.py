import json
import yaml

with open("phrases.yaml", "r", encoding="utf-8") as f:
    yaml_data = yaml.safe_load(f)

with open("phrases.json", "w", encoding="utf-8") as f:
    json.dump(yaml_data, f, ensure_ascii=False, indent=2)
