# Labelling guide (Zindi BNBR Tech4MentalHealth → Sikiza labels)

Every message gets two labels, decided from **the text alone**. The deployed system sees only
the text, so the labels must not rely on knowing which Zindi survey topic a message came from.

## crisis (0/1)

`1` when the writer expresses, about **themselves**:

- suicidal thoughts or a wish to die, active or passive ("I feel like committing suicide",
  "the only way out is death", "I don't deserve to be alive", "no reason to continue living",
  "why am I alive?", "is life even worth it", "giving up **in/on/with** life");
- a plan, method question or past attempt ("how does someone commit suicide?",
  "will people judge me for trying to take my life", "I wanted to jump off a building");
- whether they should end their life ("Is suicide the solution?", "Should I commit suicide").

`0` for:

- informational or third-person questions ("causes of suicide by youths", "how to convince someone
  who is about to commit suicide", "why do people commit suicide", "what causes suicidal thoughts?");
- despair without reference to dying or to life itself ("I feel like giving up", "I lost hope",
  "life is so unfair").

**When unsure, label 1.** A missed crisis costs far more than a false alarm (cost ratio 100:1).

## category (one of six)

| category | use for |
|---|---|
| `academic_stress` | exams, results, retakes, coursework, concentration, deferring or dropping classes for academic reasons |
| `financial_strain` | fees, rent, debt, money for needs, gambling losses, unemployment, deferring because of fees |
| `social_anxiety` | anxiety, nervousness or fear, especially of people, judgement, rejection by peers, fitting in, trust |
| `low_mood` | sadness, depression, hopelessness, loneliness, grief, low self-esteem, heartbreak, general stress; **also every crisis message unless another cause clearly dominates** |
| `substance_use` | alcohol, bhang/weed, drugs, smoking, addiction, rehab, peer pressure to drink or smoke |
| `other` | general information questions (incl. about suicide), relationship or family advice, religion, physical health, empty or meaningless answers ("Yes", "Not experienced") |

If the text alone is too short to tell (e.g. "How can I stop?"), use the Zindi topic as a tie-breaker:
Alcohol/Drugs → `substance_use`, Depression → `low_mood`, Suicide → `other`.

`substance_use` was **added to the five categories in the proposal** because about 30% of the real
messages are about alcohol or drugs. Putting them in `other` would have given those students a generic reply.

## Reviewer decisions (human review, 20 November 2026)

The human review of every crisis label changed the rules above in two places. The final labels follow these:

- **Questions about why people take their own lives are crisis = 1** ("Causes of suicide", "Why do people commit
  suicide?", "What causes suicidal thoughts?"). A student asking this may be asking about themselves, and the
  cost of a miss is high. Questions about **helping someone else** stay 0 ("How to convince someone against
  committing suicide").
- **Existential despair can be crisis = 1** ("Why is life important?", "What is the meaning of life?",
  "I hate my life", "I feel like giving up") when it reads as questioning whether life is worth living.
- A few messages with an unclear time frame or meaning were set to 0 by the reviewer (e.g. "I was once depressed
  and I feel like giving up with my life").

Result: 94 crisis rows (91 unique messages) vs 69 in the AI first pass; Cohen's κ between the AI first pass and
the human review = 0.74 (32 crises added, 7 removed). The reviewer saw the AI suggestions, so this measures
review agreement, not independent inter-annotator reliability.

## Provenance and how to review

`zindi_labels.csv` columns: `ID, review_priority, text, zindi_label, category, crisis, crisis_ai, ai_note, labelled_by, reviewed`.
`crisis` is the final, human-reviewed label; `crisis_ai` keeps the AI first pass for comparison.
Open it in Excel or Numbers; only `ID`, `category` and `crisis` are used by `scripts/prepare_zindi.py`.

- `labelled_by = ai_suggested+ai_checked`: an AI assistant labelled every row from this guide, then did a
  second consistency pass (2 crisis labels changed; borderline calls explained in `ai_note`).
- `reviewed` is for **human** review only. It stays 0 until a person has checked the row.

To review: filter `review_priority = 1` first (120 rows: every crisis label, every borderline call, and every
message Zindi tagged "Suicide" that is labelled non-crisis here). Correct `crisis` / `category` where you
disagree, set `reviewed = 1`, then run `make real`. For inter-annotator agreement, a second person labels a
random sample of at least 100 rows **without looking at these labels**, in `category_2` / `crisis_2`;
the evaluation then reports Cohen's κ automatically.
