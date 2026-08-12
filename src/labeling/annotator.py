"""
annotator.py
------------
CLI tool to manually review the evaluation set (call_date >= 2024-01-01).
Pulls 80 random chunks, fetches a bootstrapped suggestion on-the-fly,
and records the user's manual review to eval_verified.jsonl.
"""
import os
import json
import time
from pathlib import Path
import pandas as pd
from loguru import logger

# We import the gemini fetcher from bootstrap to get on-the-fly suggestions
# for the eval set, so we don't need a pre-computed eval bootstrap file.
from src.labeling.bootstrap import fetch_gemini_label

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_verified_ids(out_file: Path) -> set:
    if not out_file.exists(): return set()
    with open(out_file, "r") as f:
        return {json.loads(line)["chunk_id"] for line in f}

def parse_flags(val: str, default: list) -> list:
    if not val.strip(): return default
    return [x.strip() for x in val.split(",") if x.strip()]

def prompt_with_default(prompt_text: str, default_val: str) -> str:
    ans = input(f"  {prompt_text} [{default_val}]: ").strip()
    return ans if ans else default_val

def run_annotator():
    df_path = Path("data/processed/phase1_it_transcripts.parquet")
    if not df_path.exists():
        print("Missing parquet file.")
        return

    df = pd.read_parquet(df_path)
    # Eval set is 2024+
    eval_df = df[df["call_date"] >= "2024-01-01"].copy()

    out_dir = Path("data/labels")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "eval_verified.jsonl"

    done_ids = load_verified_ids(out_file)
    pending_df = eval_df[~eval_df["chunk_id"].isin(done_ids)]

    # We want ~80 chunks. We shuffle them.
    TARGET = 80
    remaining_needed = max(0, TARGET - len(done_ids))
    if remaining_needed == 0:
        print("Target of 80 verified eval chunks reached!")
        return

    sample_df = pending_df.sample(frac=1, random_state=42).head(remaining_needed)

    session_approved = 0
    session_corrected = 0
    session_skipped = 0
    truncation_count = 0

    total_done_start = len(done_ids)

    with open(out_file, "a") as f:
        for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
            clear_screen()
            cid = row["chunk_id"]

            # Note truncation
            is_truncated = row.get("truncation_warning", False)
            if is_truncated:
                truncation_count += 1

            print("=" * 70)
            print(f"CHUNK {total_done_start + idx} / 80  |  {cid}")
            print(f"ticker: {row['ticker']}  |  call_date: {row['call_date']}  |  speaker_role: {row['speaker_role']}")
            print("─" * 70)

            # Print wrapped text
            import textwrap
            text = str(row["chunk_text"])
            print(textwrap.fill(text, width=70))
            if is_truncated:
                print("\n[TRUNCATION WARNING: chunk exceeded token limit and was split]")

            print("\n" + "─" * 70)
            print("Fetching suggestion via Gemini...")
            try:
                sug = fetch_gemini_label(text)
            except Exception:
                print("[!] Gemini API quota exceeded. Falling back to blank template.")
                sug = {
                    "guidance_direction": "not_discussed",
                    "tone": "neutral",
                    "hedging_score": 0.5,
                    "key_flags": []
                }

            print("SUGGESTED LABEL (bootstrapped):")
            print(f"  guidance_direction : {sug.get('guidance_direction')}")
            print(f"  tone               : {sug.get('tone')}")
            print(f"  hedging_score      : {sug.get('hedging_score')}")
            print(f"  key_flags          : {sug.get('key_flags')}")
            print("─" * 70)

            while True:
                choice = input("  [A] approve all   [C] correct field(s)   [S] skip chunk\n> ").strip().lower()
                if choice in ('a', 'c', 's', 'q'):
                    break

            if choice == 'q':
                break

            out_label = {
                "chunk_id": cid,
                "ticker": row["ticker"],
                "call_date": str(row["call_date"]),
                "status": "skipped",
                "label": None,
                "skip_reason": None
            }

            if choice == 'a':
                out_label["status"] = "approved"
                out_label["label"] = sug
                print("✓ Saved as APPROVED.")
                session_approved += 1

            elif choice == 'c':
                print("\nFields: guidance_direction / tone / hedging_score / key_flags")
                print("  Leave blank to keep the suggestion. Press Enter to accept.\n")

                gd = prompt_with_default("guidance_direction", str(sug.get('guidance_direction')))
                tn = prompt_with_default("tone", str(sug.get('tone')))
                hs = prompt_with_default("hedging_score", str(sug.get('hedging_score')))
                try: hs = float(hs)
                except: hs = sug.get('hedging_score')

                # key_flags is special - we show the list but accept comma separated
                kf_default_str = ", ".join(sug.get('key_flags', []))
                kf_ans = input(f"  key_flags          [{kf_default_str}]: ").strip()
                kf = parse_flags(kf_ans, sug.get('key_flags', [])) if kf_ans else sug.get('key_flags', [])

                final_label = {
                    "guidance_direction": gd,
                    "tone": tn,
                    "hedging_score": hs,
                    "key_flags": kf
                }

                changes = sum(1 for k in final_label if final_label[k] != sug.get(k))
                out_label["status"] = "corrected"
                out_label["label"] = final_label
                print(f"✓ Saved as CORRECTED ({changes} fields changed).")
                session_corrected += 1

            elif choice == 's':
                reason = input("  Reason (optional, press Enter to leave blank): ").strip()
                out_label["skip_reason"] = reason
                print("✗ Skipped.")
                session_skipped += 1

            f.write(json.dumps(out_label) + "\n")
            f.flush()

            print("─" * 70)
            print(f"Progress: {total_done_start + idx}/80 chunks | Approved: {session_approved} | Corrected: {session_corrected} | Skipped: {session_skipped}")
            time.sleep(1.5)

    print("\nReview session ended.")
    print(f"Total chunks with TRUNCATION WARNING seen this session: {truncation_count}")
    print("If many were truncated, consider updating the chunking logic or explicitly skipping them.")

if __name__ == "__main__":
    run_annotator()
