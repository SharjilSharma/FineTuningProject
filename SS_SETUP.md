# Environment Setup — Claude Code + OmniRoute + Headroom

Follow in order. Each step has a way to confirm it actually worked before
moving on — don't skip the checks, a silently broken routing setup just
looks like "the agent works" until your budget disappears faster than
expected.

## 1. Anthropic Console account (pay-as-you-go, not a subscription)

1. Go to `console.anthropic.com`, sign up, add a payment method.
2. Load credits (e.g. $10 to start — leaves room to test before committing
   the full $24).
3. **Set a hard spending limit** equal to what you loaded, under the
   Limits section. Set a notification threshold at 50%.
4. Create an API key under API Keys. Save it somewhere safe — you'll need
   it once, for OmniRoute, not for Claude Code directly.

**Check:** the Console dashboard shows your credit balance and the limit
you set, not "no limit configured."

## 2. Install OmniRoute

```bash
npm install -g omniroute
omniroute        # starts gateway + dashboard on port 20128
```

Open `http://localhost:20128/dashboard` in a browser — this is your
control panel for everything below, no YAML editing required.

**Check:** the dashboard loads and shows "serving on :20128."

## 3. Connect your providers in the dashboard

- Under Providers: add your Anthropic API key (from step 1) as a paid
  provider.
- Under Providers: connect free-tier providers — the dashboard lists 90+
  with sign-in, no card required (this is where GLM, DeepSeek, Groq,
  Google AI Studio's free Gemini tier, etc. get added).
- Under Combos: build (or use a default) combo that puts free providers
  first and Anthropic Sonnet as the fallback/escalation tier.

**Check:** run `omniroute providers list` — every provider you connected
should show as configured. `omniroute providers test <id>` live-tests one.

## 4. Point Claude Code at OmniRoute

You don't need to install Claude Code separately or hand-edit config —
OmniRoute does it for you:

```bash
omniroute setup-claude
```

This writes `~/.claude/profiles/<name>/settings.json` pointing Claude
Code's API base URL and key at your local OmniRoute gateway instead of
Anthropic directly. Then either:

```bash
omniroute launch          # spawns Claude Code with the right env, no config needed
```

or just run `claude` normally if `setup-claude` already wrote the config.

**Check:** run `omniroute doctor` — it diagnoses providers, ports, and
whether Claude Code is correctly wired to the gateway.

## 5. VS Code extension

Install the official "Claude Code" extension from the VS Code
Marketplace. It bundles the CLI and reads the same `~/.claude` config
you just set up in step 4 — nothing extra to configure. Open your project
folder, and the chat panel (the one you saw in your screenshot) is now
also running through OmniRoute.

**Check:** open the extension panel, send a trivial prompt ("say hi"),
and confirm in the OmniRoute dashboard's Analytics tab that a request
was logged and which provider served it (should be a free one for a
trivial prompt, assuming your combo is set to free-first).

## 6. Enable Headroom compression inside OmniRoute

OmniRoute ships Headroom's compaction engine as one of its built-in
pipeline steps — you don't need to run a separate Headroom proxy. In the
dashboard, under Compression / Compression Studios, enable the Headroom
engine on your active combo.

**Check:** send a prompt with a large tool output (e.g. ask it to read a
big file), then check the dashboard's cost/analytics view for a
token-savings figure on that request — confirms compression is actually
firing, not just toggled on.

## 7. Drop project files into your repo

- `CLAUDE.md` → repo root (routing rules, testing discipline, repo
  structure).
- `docs/project-spec.md` → your original project spec.
- `docs/roadmap.md` → the phase-by-phase roadmap.

Claude Code reads `CLAUDE.md` automatically at the start of every
session in that folder — nothing to load manually.

**Check:** open Claude Code in the project folder and ask "what are the
routing rules for this project?" — it should answer from CLAUDE.md
without you pasting anything.

## 8. Full end-to-end verification, before Phase 0 work starts

Run these three test prompts and confirm the expected behavior in the
OmniRoute dashboard each time:

| Prompt | Expected route | How to confirm |
|---|---|---|
| "Format this file's docstrings" (boilerplate) | Free-tier provider | Dashboard Analytics shows a free provider, $0 cost |
| "Design the LoRA config for this fine-tune" (per CLAUDE.md routing) | Anthropic Sonnet | Dashboard shows Anthropic as provider; Console usage ticks up by a small, expected amount |
| Large file read/tool output | Any provider, but compressed | Analytics shows a token-savings number > 0 |

If all three land where expected, the whole chain — Console limit →
OmniRoute routing → Headroom compression → Claude Code / VS Code — is
verified working, and you're clear to start Phase 0 of the roadmap.

## Ongoing habit

Check the OmniRoute dashboard and the Anthropic Console usage page once
a week (not "eventually"). If Sonnet spend is climbing faster than
CLAUDE.md's routing rules would predict, that's your signal to tighten
the combo rules, not to lower the spending cap.
