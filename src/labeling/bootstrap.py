"""
bootstrap.py
------------
Auto-labels the training set using Gemini API (gemini-2.5-flash).
This generates 'suggestions' that avoid circularity with the Groq 
base-model baseline in Phase 2.

Output:
data/labels/train_bootstrap.jsonl
"""
import os
import json
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("google-genai not installed. Run: pip install google-genai")

load_dotenv()

# Constants
MODEL_ID = "gemini-flash-lite-latest"
SYSTEM_PROMPT = """
You are a financial analyst assistant. Your task is to analyze earnings call transcript excerpts and extract structured signals.
Respond with a valid JSON object matching this schema:
{
  "guidance_direction": "raised" | "lowered" | "maintained" | "not_discussed",
  "tone": "confident" | "cautious" | "evasive" | "neutral",
  "hedging_score": float between 0.0 and 1.0,
  "key_flags": [list of up to 5 short phrases that stand out]
}
"""

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        _client = genai.Client(api_key=api_key)
    return _client

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(5)
)
def fetch_gemini_label(chunk_text: str) -> dict:
    client = get_client()
    prompt = f"Analyze this excerpt and return ONLY the JSON signal:\n\n--- EXCERPT ---\n{chunk_text[:6000]}\n--- END ---"

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[
            types.Content(role="user", parts=[
                types.Part.from_text(text=SYSTEM_PROMPT),
                types.Part.from_text(text=prompt)
            ])
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
    )

    try:
        return json.loads(response.text)
    except Exception:
        return {
            "guidance_direction": "not_discussed",
            "tone": "neutral",
            "hedging_score": 0.5,
            "key_flags": []
        }

def run_bootstrap(max_chunks=None):
    df_path = Path("data/processed/phase1_it_transcripts.parquet")
    if not df_path.exists():
        logger.error(f"{df_path} not found. Run Phase 1 first.")
        return

    df = pd.read_parquet(df_path)
    # Split: Train is before 2024
    train_df = df[df["call_date"] < "2024-01-01"].copy()

    if max_chunks:
        train_df = train_df.head(max_chunks)

    logger.info(f"Bootstrapping {len(train_df)} chunks using {MODEL_ID}...")

    out_dir = Path("data/labels")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "train_bootstrap.jsonl"

    # Check what's already done
    done_ids = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            for line in f:
                done_ids.add(json.loads(line).get("chunk_id"))

    with open(out_file, "a") as f:
        for i, row in train_df.iterrows():
            cid = row["chunk_id"]
            if cid in done_ids:
                continue

            logger.info(f"Labeling {cid}")
            label = fetch_gemini_label(str(row["chunk_text"]))
            label["chunk_id"] = cid

            f.write(json.dumps(label) + "\n")
            f.flush()
            time.sleep(4) # Pacing for free tier

    logger.info("Bootstrap complete!")

if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_bootstrap(limit)
