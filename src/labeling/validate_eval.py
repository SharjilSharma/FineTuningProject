import json
from pathlib import Path

VALID_GUIDANCE = {"raised", "lowered", "maintained", "not_discussed"}
VALID_TONES = {"confident", "cautious", "evasive", "neutral"}

def validate():
    eval_file = Path("data/labels/eval_verified.jsonl")
    if not eval_file.exists():
        print(f"Error: {eval_file} does not exist.")
        return

    errors_found = False
    with open(eval_file, "r") as f:
        for i, line in enumerate(f, 1):
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Row {i}: Invalid JSON line.")
                errors_found = True
                continue
                
            status = data.get("status")
            if status == "skipped":
                continue # Skipped chunks don't matter
                
            label = data.get("label", {})
            if not label:
                print(f"Row {i} (Chunk: {data.get('chunk_id')}): Missing 'label' object.")
                errors_found = True
                continue

            gd = str(label.get("guidance_direction"))
            tn = str(label.get("tone"))

            if gd not in VALID_GUIDANCE:
                print(f"Row {i} (Chunk: {data.get('chunk_id')}): Invalid guidance_direction -> '{gd}'")
                errors_found = True
            
            if tn not in VALID_TONES:
                print(f"Row {i} (Chunk: {data.get('chunk_id')}): Invalid tone -> '{tn}'")
                errors_found = True

    if not errors_found:
        print(f"Validation passed! All verified labels in {eval_file} match the strict schema.")

if __name__ == "__main__":
    validate()
