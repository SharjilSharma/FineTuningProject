# Runbook

This is the "how do I actually run this" document. PROJECT.md /
project-spec.md covers *what* to build. ROADMAP.md covers *what order*
to build it in. This covers *where and how each piece actually runs* —
commands, accounts needed, and machine (local vs Colab).

---

## 1. Accounts you need, set up once, before Phase 0

| Account | Needed for | Cost |
|---|---|---|
| GitHub | Pushing code so Colab can pull it | Free |
| Hugging Face | Downloading the transcripts dataset | Free |
| Groq Console | Zero-shot baseline API calls | Free tier |
| Google account | Running Colab notebooks | Free |

Get API keys/tokens for HF and Groq now, keep them handy for your local
`.env` (never commit this file — confirm `.gitignore` covers it).

---

## 2. Local environment setup (do this once)

```bash
# From inside your project folder
python3 --version        # confirm 3.10+ 
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# now open .env and paste your real HF_TOKEN and GROQ_API_KEY
```

Run this venv activation every time you open a new terminal to work on
the project. Everything in Phases 0–3 and 5–6 runs inside this venv, on
your own machine — no GPU needed.

---

## 3. GitHub — set up before Phase 4, useful from Phase 0 onward

```bash
git init                      # if not already done
git add .
git commit -m "initial commit"
# create an empty repo on github.com first, then:
git remote add origin <your-repo-url>
git push -u origin main
```

From here, **commit and push after every phase is verified working
locally** — not just once at the end. This is what lets Colab pull a
working version of your code later.

Before every push, double-check: `git status` should never show `.env`
as a tracked/staged file. If it does, your `.gitignore` isn't catching
it — fix that before pushing, not after.

---

## 4. Running each phase — actual commands

### Phase 0 — Scaffolding
Nothing to "run" — just confirm the install worked:
```bash
pip install -r requirements.txt   # should complete with no errors
pytest tests/ -v                  # should collect tests (may be empty/stubs)
```

### Phase 1 — Data pipeline
Run locally, in order:
```bash
python -m src.data.load_transcripts
python -m src.data.price_data
python -m src.data.preprocessing
pytest tests/test_data.py -v
```
Output: a parquet file (confirm the path the agent used, e.g.
`data/processed/transcripts_prices.parquet` — check this is also in
`.gitignore`, since data files shouldn't be committed to GitHub).

### Phase 2 — Baselines
```bash
python -m src.baselines.lexicon
python -m src.baselines.base_model     # calls Groq API — watch rate limits
pytest tests/test_baselines.py -v
```

### Phase 3 — Labeling
```bash
python -m src.labeling.annotator       # your manual labeling tool
python -m src.labeling.bootstrap       # optional LLM-assisted pre-labeling
```
This phase is mostly you, manually, not a script you run once — budget
real days/weeks here, not minutes.

### Phase 4 — Fine-tuning (the one Colab phase)
Local, before touching Colab:
```bash
pytest tests/test_finetune.py -v      # smoke test only, must pass first
```
Then, in a browser (not your terminal):
1. Push your latest commit to GitHub
2. Go to colab.research.google.com
3. File → Open notebook → GitHub tab → paste your repo URL → open
   `notebooks/finetune_colab.ipynb`
4. Runtime → Change runtime type → T4 GPU
5. Add secrets via the key icon in the left sidebar (HF_TOKEN, and
   GROQ_API_KEY if the notebook needs it) — toggle notebook access on
6. Run all cells — the notebook should just `git clone` your repo,
   `pip install -r requirements.txt`, and call `src/finetune/train.py`
7. Download the resulting adapter weights (or push them to a HF model
   repo — simpler than downloading large files through the browser)
8. Back on your local machine: quantize to GGUF, verify it runs via
   Ollama/llama.cpp

### Phase 5 — Evaluation
Local, CPU only (the fine-tuned model is quantized by now):
```bash
python -m src.eval.extraction_quality
python -m src.eval.backtest
```

### Phase 6 — Agent layer
Local. Postgres and Redis need to actually be running — easiest via
Docker:
```bash
docker compose up -d      # spins up local postgres + redis (add a
                           # docker-compose.yml if one doesn't exist yet)
uvicorn src.api.main:app --reload
```

---

## 5. When to use Kaggle instead of Colab

Kaggle Notebooks are a fine alternative to Colab for Phase 4 if Colab's
free GPU quota runs out on a given day — same general idea (T4 GPU,
free tier), but:
- You'd upload/link your GitHub repo differently (Kaggle supports
  attaching a GitHub repo as a "dataset" input, or you `git clone` with
  a token inside the notebook the same way)
- Kaggle's free GPU quota is weekly (hours/week), Colab's resets more
  often — useful to have both as backup if one runs dry mid-training

Not required to set this up now — just know it's there as a fallback
if Colab throttles you during Phase 4.

---

## 6. If something in the spec still isn't covered here

Ask before assuming — for anything the agent (or you) hits that isn't
spelled out in project-spec.md, roadmap.md, or this file, stop and ask
rather than guessing an approach and building on top of it.
