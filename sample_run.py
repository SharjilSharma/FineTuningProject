"""
sample_run.py — 8-chunk Groq sample run to verify output format.
Run manually, then delete.
"""
import pandas as pd
from src.baselines.base_model import score_dataframe
from src.eval.extraction_quality import validate_schema

sample_chunks = [
    ("NVDA_20231025__prep__p001", "NVDA",
     "Revenue for Q3 was 18.1 billion dollars, up 206 percent from a year ago. "
     "We are raising our Q4 revenue guidance to 20 billion dollars plus or minus two percent. "
     "Data center demand remains exceptionally strong and we beat consensus estimates significantly."),

    ("MSFT_20231025__qa__q001__p001", "MSFT",
     "[Analyst, Goldman Sachs]: Can you speak to the Azure growth trajectory and when we should expect acceleration? "
     "[CFO]: Azure grew 29 percent in constant currency. We expect that to reaccelerate as AI capacity comes online, "
     "though we may see some near-term constraints related to supply chain. "
     "We are cautious about providing a specific timeline given macro uncertainty."),

    ("AAPL_20231102__prep__p001", "AAPL",
     "iPhone revenue was 43.8 billion dollars, slightly below our guidance range. "
     "Services revenue reached an all-time high of 22.3 billion. "
     "We are maintaining our outlook for the December quarter, though foreign exchange headwinds "
     "may create approximately 200 basis points of revenue pressure."),

    ("INTC_20231026__qa__q002__p001", "INTC",
     "[Analyst, Morgan Stanley]: What is your confidence in the 2025 volume ramp for Intel Foundry? "
     "[CEO]: We remain on track with our roadmap commitments. I want to be clear: "
     "we acknowledge the competitive environment is challenging and we have significant execution risk ahead. "
     "That said we are reaffirming our target to regain process leadership by 2025."),

    ("ACN_20231214__prep__p001", "ACN",
     "Accenture delivered first quarter revenues of 16.2 billion dollars, up 3 percent in local currency. "
     "We are raising our full-year revenue growth outlook to 2 to 5 percent in local currency. "
     "Bookings were a record 18.4 billion dollars. Our generative AI business is ramping faster than expected."),

    ("CRM_20231129__qa__q003__p001", "CRM",
     "[Analyst, UBS]: How should we think about the Data Cloud attach rate as customers face budget pressure? "
     "[CEO]: Frankly customers are being very deliberate about spend. We are seeing elongated sales cycles "
     "and some deals slipping into next quarter. We are not changing guidance but want to flag that pipeline "
     "conversion is uncertain given the current environment."),

    ("TXN_20231024__prep__p001", "TXN",
     "Revenue for Q3 was 4.53 billion dollars, below our guidance midpoint. "
     "Inventory correction in industrial and automotive end markets is taking longer than we anticipated. "
     "We are lowering our Q4 guidance to 3.93 to 4.27 billion dollars, reflecting continued weakness. "
     "We may see some improvement in the second half of 2024 but visibility remains limited."),

    ("IBM_20231025__qa__q001__p001", "IBM",
     "[Analyst, Citi]: Can you walk through the software segment margin outlook? "
     "[CFO]: Software margins expanded 200 basis points year on year. "
     "We are on track for our full-year free cash flow target of 10.5 billion dollars. "
     "The Red Hat integration continues to perform in line with our expectations and we reaffirm all guidance."),
]

df = pd.DataFrame(sample_chunks, columns=["chunk_id", "ticker", "chunk_text"])

print("Running 8-chunk sample against Groq (llama-3.1-8b-instant)...")
print("Estimated time: ~64 seconds (8s pacing per request)")
print()

out = score_dataframe(df, max_chunks=8, request_interval=8.0)

print()
print("=" * 70)
print("SAMPLE OUTPUT — llama-3.1-8b-instant zero-shot baseline")
print("=" * 70)
for _, row in out.iterrows():
    is_valid, errors = validate_schema(row.to_dict())
    valid_str = "OK" if is_valid else f"INVALID: {errors}"
    print(f"\nchunk_id            : {row['chunk_id']}")
    print(f"  guidance_direction : {row['guidance_direction']}")
    print(f"  tone               : {row['tone']}")
    print(f"  hedging_score      : {row['hedging_score']}")
    print(f"  key_flags          : {row['key_flags']}")
    print(f"  schema             : {valid_str}")

n_valid = sum(validate_schema(row.to_dict())[0] for _, row in out.iterrows())
print(f"\n{'='*70}")
print(f"Schema valid: {n_valid}/{len(out)} chunks")
print(f"Parse errors: {out['_parse_error'].sum()}/{len(out)} chunks")
