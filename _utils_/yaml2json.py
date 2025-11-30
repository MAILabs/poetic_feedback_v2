import json
import yaml

with open("phrases.yaml", "r", encoding="utf-8") as f:
    yaml_data = yaml.safe_load(f)

with open("phrases.js", "w", encoding="utf-8") as f:
    f.write("const PHRASES_DATA = ")
    json.dump(yaml_data, f, ensure_ascii=False, indent=2)
    f.write(";")