# Sikiza: Sheng-Aware Distress Triage and Coping Support

**DSA 8401 Applied Machine Learning · Capstone report · Lousa Sheenaz Yogo · November 2026**

> **Summary.** Sikiza decides whether a student's check-in signals a crisis (sent to a human counsellor)
> and otherwise answers with a counsellor-approved coping strategy for its category. It is built end to end:
> data pipeline, four models, calibration and a cost-derived threshold, evaluation with honest confidence
> intervals, a FastAPI service in Docker, and a React Native app. The models are trained and tested on
> **884 real messages written by students at Kenyan universities** (Zindi BNBR Tech4MentalHealth), relabelled
> for this task with human-reviewed crisis labels. The real messages are in English; Swahili/Sheng handling is
> only exercised on synthetic data so far (Section 7).

---

## 1. Problem framing

**What is predicted.** For each short student message:

1. **Crisis vs non-crisis**: the safety decision. Crisis means signs of suicidal thinking, self-harm or
   immediate danger.
2. **One of six categories** for the response: academic stress, financial strain, social anxiety, low mood,
   substance use, other. (*Substance use* was added after the proposal, because about 30% of the real
   messages are about alcohol or drugs.)

A crisis goes to a human counsellor's escalation queue, and the student immediately sees a crisis message
with hotline numbers. Any other message gets a coping strategy **retrieved** from a counsellor-approved
library for its category. Nothing is written freely by a model.

**Why it matters.** Kenya has fewer than 100 practising psychiatrists for over 50 million people, and a
student can wait about three weeks for a counselling appointment. Global models misread code-mixed text
("nime-hang", "kujitoa"), so distress is missed exactly when students express it most naturally. Sikiza is
digital first aid in front of counselling, not a diagnostic device.

**The costs are asymmetric, and that drives the design.** A missed crisis is catastrophic; a false alarm
costs a counsellor a few minutes. I set the cost of a false negative to 100 and of a false positive to 1. The
decision threshold is derived from these costs instead of 0.5, and the success criterion is **crisis recall
≥ 0.95, judged on the lower bound of a 95% confidence interval**, reported together with the escalation rate.

## 2. Data

### 2.1 Sources

| Source | Size | Role |
|---|---|---|
| **Zindi: Basic Needs Basic Rights Kenya – Tech4MentalHealth** (2020): answers by students at several Kenyan universities to "What is on your mind?", originally tagged Depression / Alcohol / Suicide / Drugs | 925 messages (616 tagged + 309 untagged) → **884 distinct** | Seed training data and the **only** test data |
| Augmentation of the training seeds (Sheng word swaps, SMS-style noise, optional back-translation) | ~8,000 rows | Training volume only |
| Synthetic English/Swahili/Sheng check-ins | 224 | Code-mixed tokenizer and drift checks; pipeline smoke tests |
| GoEmotions, AfriSenti-swa, tabularisai Swahili sentiment | ~58k / ~3k / ~48k | Optional stage-1 fine-tuning (not used for the reported models) |
| Sheng lexicon; coping library | 33 rules; 16 entries + crisis response | Safety valve; retrieval corpus |

The consented Strathmore check-ins in my proposal were not available in time. Zindi is the closest real
substitute: the same population (Kenyan university students), the same kind of short, unedited text, and it
contains real crisis disclosures.

### 2.2 Real-world messiness I had to handle

- **Labels that do not fit the task.** Zindi's tags are survey topics, not risk labels. Comparing them with
  text-based crisis labels gives Cohen's κ = 0.45: of the 66 messages tagged "Suicide", 34 are not crises
  ("Should I go back to ex girlfriend?", "How to convince someone against committing suicide"), and 29 crisis messages were tagged something else, mostly
  "Depression" ("I feel like killing myself…", "Can I kill myself?"). Training on Zindi's tags would have
  taught the model the wrong thing, so **every message was relabelled from its text alone**
  (`data/labels/LABELLING_GUIDE.md`).
- **Labelling process.** An AI assistant produced a first pass from the guide; **I then reviewed every crisis
  label**, changing 39 (32 crises added, 7 removed) and setting my own rules for suicide-related questions.
  Agreement between the first pass and my review is κ = 0.74. I saw the suggestions while reviewing, so this is
  review agreement, not independent inter-annotator reliability; a second independent coder is still needed.
  Category labels are the first pass. Final data: **91 crisis messages (10.3%)**.
- **Duplicates and train/test overlap.** 925 rows contain 884 distinct texts, and 13 texts appear in both
  Zindi's train and test files. Identical texts are merged into one seed (crisis if any copy is a crisis), so
  the same message can never be on both sides of my split.
- **Spelling, run-on words and very short messages.** "I feel hopelessFor now I am just fine", "Dissappointed",
  "Yes". Augmentation originally produced empty variants from one-word messages ("Yes" → ""); variants that
  delete the message are now rejected.
- **Class imbalance**: 10% crisis; categories range from 483 low-mood to 16 social-anxiety messages.
- **Leakage control.** Augmented variants keep their seed's ID; splits are **grouped by seed** and augmented
  rows join **train only**; the evaluator asserts the test set holds real messages only.
- **Privacy.** Pseudonymisation runs first (names, phone numbers, emails, M-Pesa codes, student IDs).

**Splits** (fixed once, never re-drawn after seeing results): train 530 messages (51 crisis) + augmentation,
calibration 133 (14 crisis), test **221 (26 crisis)**.

## 3. Techniques and why I chose them

### Architecture

```
Student app ─▶ FastAPI ─▶ pseudonymise ─▶ safety valve ─┐
 (Expo RN)                                              ├─ OR ─▶ crisis? ─yes─▶ escalation queue + crisis response
                         classifier (int8 ONNX) ────────┘            └─no──▶ retrieve approved strategy
```

### What I used

| Technique | Why it fits this problem |
|---|---|
| **TF-IDF (word + char 2–5-grams) + logistic regression** | A cheap, explainable floor that every other model must beat. Character n-grams cope with Sheng spelling variation without a tokenizer. |
| **AfroXLM-R (Davlan/afro-xlmr-base) with LoRA**, two heads (crisis, 6 categories), mean pooling | XLM-R further pre-trained on 17 African languages including Swahili. LoRA (r = 16, α = 32) trains about 1% of the weights, which suits a ~1,000-message dataset (full fine-tuning of 270M parameters would overfit) and trains on a laptop CPU. |
| **DistilBERT (multilingual) as comparison** | Smaller and faster; tests whether the African-language pre-training actually matters. |
| **Tokenizer fertility measured first** | Encoder choice should follow from how badly each tokenizer fragments Sheng, not from reputation. |
| **Two-stage training (optional stage 1)** | Sentiment pre-training on GoEmotions + AfriSenti + tabularisai teaches affect in English and Swahili before the small Strathmore set. I mapped all three to 3-way sentiment, the only label they share, instead of forcing e.g. GoEmotions "nervousness" onto "social anxiety". |
| **Trained without class weights + temperature scaling** on the calibration split | A cost-based threshold only works on real probabilities. Class weights shift the logits, and calibration is the fix. |
| **Cost-derived threshold** t = C_fp / (C_fp + C_fn) = 1/101 ≈ 0.0099, lowered further if needed to reach recall 0.95 on calibration | Bayes-optimal decision for the stated costs; 0.5 would assume equal costs. |
| **Safety valve: keyword rules OR classifier** | Either signal escalates. Rules catch explicit phrases with certainty and work offline on the phone; the classifier catches implicit warning signs. An AND-gate at a high threshold would miss crises. |
| **Retrieval, not generation**, for responses | Advice is grounded, auditable and pre-approved by counsellors. The same input always gives the same answer. |
| **ONNX export + dynamic int8 quantisation** (per channel), then re-calibration | Serves on cheap CPU VMs and is small enough for weak connections. Quantisation shifts the logits, so the int8 model gets its own calibration and threshold. |
| **Drift: novelty (nearest-neighbour distance) + unseen-word rate + KMeans clusters**, both cut-offs calibrated on held-out training messages | Sheng changes within months. The monthly report surfaces clusters of new words for the lexicon review. |
| **Evaluation:** bootstrap and exact (Clopper–Pearson) CIs, McNemar's exact test, Cohen's κ, ECE, ROC-AUC / average precision | The success criterion is about the CI lower bound, so honest intervals are essential with few crises. |

### What I did not use, and why

| Alternative | Why I ruled it out |
|---|---|
| **Oversampling / SMOTE / class weights** | SMOTE interpolates feature vectors, which has no meaning for text. Weights and oversampling distort the probabilities the cost threshold relies on. The threshold handles the imbalance directly. |
| **A 0.5 threshold, or an AND-gate between rules and model** | Assumes equal costs. An AND-gate means a crisis is missed whenever either signal misses it. |
| **Generating responses with an LLM (or RAG with free generation)** | A model could produce harmful or wrong advice to a student in crisis, and every answer would need checking. I did not build the optional "rephrase at τ ≈ 0" step from the proposal: verbatim approved text is safer and easier to audit. |
| **A large hosted LLM as the classifier** | Sends health data to servers outside Kenya (DPA 2019), costs money per message, and is slower and non-deterministic. |
| **Zero-shot NLI as the main classifier** | It has no Sheng training. I kept it only as a reference check (Section 5). |
| **Full fine-tuning** | Far more parameters than ~1,000 messages can support; overfits, and needs a GPU. |
| **Random forest / XGBoost on TF-IDF** | No advantage over logistic regression on sparse text, and their probabilities are poorly calibrated. |
| **Evidently AI for drift** | Its text-drift reports detect a distribution shift but don't tell me *which new words* to add to the lexicon. My report does, and it keeps message text on the server. |
| **MLflow / a model registry** | Each model bundle carries a manifest with its model, data fingerprint, hyperparameters, calibration, threshold and lexicon version, and the API logs these with every prediction. That was enough for one person and one model family. |
| **GPU serving / TorchServe** | The int8 model runs in about 5 ms on a CPU. A GPU would add cost and no useful speed. |

## 4. Evaluation design

- **The test set holds real seed messages only** (no augmented variants). It is used once, at the end.
- **Three gates are reported for every model:** valve only, classifier only, and the deployed OR-gate.
- **Crisis recall with two CIs.** A percentile bootstrap collapses to [1, 1] when every crisis is caught, so
  I also report the exact Clopper–Pearson interval and judge the target on the more conservative lower bound.
- **Escalation rate is always shown next to recall.** Recall at a 100% escalation rate means nothing, and the
  evaluator warns when the rate exceeds 50%.
- **Threshold-free ranking metrics** (ROC-AUC, average precision, recall if the top 20% are escalated) separate
  "the model cannot tell crises apart" from "the threshold is wrong".
- **Category quality:** macro-F1 with bootstrap CI and a confusion matrix per model.
- **Subgroups:** English-only vs code-mixed messages.
- **Paired comparison:** McNemar's exact test on per-message correctness, baseline vs each model.

## 5. Results (real data: 221 test messages, 26 crises)

Full output: `artifacts/reports/eval.md` and `eval.json`. All four models were trained on the same 530
training messages (+ augmentation), calibrated and thresholded on the same 133, and tested once on 221.

### 5.1 Crisis decision (the safety metric)

| Model | Recall | Exact 95% CI | Precision | Escalation rate | Category macro-F1 [95% CI] |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 1.000 (26/26) | [0.868, 1.000] | 0.164 | **0.719** | 0.465 [0.389, 0.526] |
| AfroXLM-R + LoRA (PyTorch) | 0.962 (25/26) | [0.804, 0.999] | 0.207 | 0.548 | **0.655** [0.556, 0.722] |
| DistilBERT + LoRA | 0.962 (25/26) | [0.804, 0.999] | 0.236 | 0.480 | 0.552 [0.444, 0.642] |
| **AfroXLM-R int8 ONNX (deployed)** | **0.923 (24/26)** | [0.749, 0.991] | **0.286** | **0.380** | **0.656** [0.563, 0.721] |
| Keyword valve alone | 0.308 (8/26) | [0.143, 0.518] | 0.889 | 0.041 | – |

(Model rows are the deployed OR-gate; on this data it equals the classifier alone, because every crisis the
valve caught, the classifier also caught.)

**Interpretation.**

- **No model meets the success criterion**, and with 26 test crises none could: even 26/26 gives an exact
  lower bound of 0.868. The deployed model catches 24 of 26 crises with a lower bound of 0.75.
- **Recall must be read with the escalation rate.** The baseline's perfect recall comes from escalating 72% of
  all messages, which no counselling team could absorb. The transformers reach similar recall while escalating
  far less: the deployed int8 model sends 38% of messages to a counsellor, about half the baseline's load.
  McNemar's test on per-message crisis decisions favours every transformer over the baseline (p < 0.0001;
  e.g. int8 correct on 80 messages where the baseline was wrong, the reverse on 9), almost entirely through
  fewer false alarms.
- **The cost threshold is still very low** (1/101 ≈ 0.0099, with only 14 calibration crises), so even the best
  model escalates over a third of messages. At a cost ratio of 100:1 that is the "right" answer, but it should be
  agreed with counsellors against their real capacity rather than adjusted silently.
- **The keyword valve alone catches only 8 of 26 real crises** (31%), though with few false alarms. The lexicon
  was written for Sheng/Swahili and explicit phrases; real students phrase crises in many indirect ways
  ("is life even worth it", "I feel there was no reason to continue living", "I don't see the need to live in the
  world"), and a missing space defeats exact matching ("lack of motivationsuicidal ideas"). This is the strongest evidence for the
  OR-gate design: the classifier is not optional.

### 5.2 Can the models separate crises at all? (threshold-free)

| Model | ROC-AUC [95% CI] | Average precision [95% CI] | Recall if top 20% escalated | Test ECE |
|---|---|---|---|---|
| TF-IDF + LR | 0.929 [0.876, 0.969] | 0.664 [0.481, 0.844] | 0.808 | 0.041 |
| AfroXLM-R + LoRA | 0.926 [0.862, 0.971] | 0.654 [0.478, 0.841] | 0.769 | 0.038 |
| DistilBERT + LoRA | 0.932 [0.866, 0.976] | 0.722 [0.535, 0.868] | 0.846 | 0.051 |
| AfroXLM-R int8 | 0.917 [0.858, 0.965] | 0.625 [0.448, 0.820] | 0.808 | 0.065 |

Chance average precision = prevalence = 0.118.

**Interpretation.** All four models rank crises well above chance (ROC-AUC ≈ 0.92–0.93), and their intervals
overlap almost completely: **on English text, the encoders do not rank crises measurably better than TF-IDF.**
Their advantage in 5.1 comes from producing better-calibrated scores at the low end, so fewer non-crises clear
the very low threshold. Calibration is good for all (expected calibration error 0.04–0.07), which is what makes
a cost-derived threshold meaningful.

### 5.3 Categories

- **AfroXLM-R is the best category model**: macro-F1 0.655 vs 0.465 for the baseline, significant by McNemar's
  exact test (p = 0.013; AfroXLM-R right and baseline wrong on 21 messages, the reverse on 7). The int8 model
  keeps this (0.656, p = 0.050).
- **DistilBERT is not significantly better than the baseline** on categories (0.552, p = 0.86).
- The rare categories drive macro-F1 down: social anxiety (16 messages in total) and academic stress (21)
  have very few test examples. The confusion matrices are in `artifacts/reports/confusion_*.png`.

### 5.4 English-only vs code-mixed

All 221 real test messages are English, so the comparison the proposal asked for cannot be made on real data.
The proxy evidence is the tokenizer fertility on code-mixed (synthetic) text: AfroXLM-R/XLM-R use 1.57 tokens
per word vs 1.93 for multilingual DistilBERT, with 12.5% vs 23.5% of words split into three or more pieces. On
the real English text all three tokenizers are about equal (≈ 1.24). So on this data the African-language
pre-training has nothing to show on crisis ranking; where it helps (categories) the effect is modest. Its real
test is Sheng and Swahili check-ins.

### 5.5 Quantisation and latency

| | Value |
|---|---|
| Model size, fp32 → int8 | 1,110 MB → 279 MB |
| Max crisis-logit change after int8 | 2.70 |
| Crisis decisions unchanged on calibration set | 88% |
| Category predictions unchanged | 96% |
| Model-only CPU time (M1) | ~5 ms per message |
| HTTP round trip through Docker | p50 25 ms, p95 37 ms |

Quantisation moved the crisis logits far more on the real model than on the demo one, which is why the int8
bundle is re-calibrated and re-thresholded on its own outputs rather than reusing the PyTorch threshold. After
that, it escalates less (38% vs 55%) at the cost of one more missed crisis (24 vs 25 of 26). That trade-off is
worth stating to counsellors; an fp32 ONNX export (1.1 GB, also supported) avoids it. Latency is far inside
the 400 ms budget, so the budget is network-bound, as the proposal predicted.

### 5.6 Drift monitoring

| Scenario | Messages flagged as novel | Alert |
|---|---|---|
| Control: held-out real messages (no drift) | 2.3% (designed ≈ 5%) | no |
| Students switch to Sheng/Swahili (synthetic code-mixed set) | 73% | **yes**, with new-word clusters such as "naogopa", "inanisumbua", "nimehang" |

My first version used a fixed rule (a message is novel if ≥ 25% of its words are unseen). On real, typo-heavy
student text that flagged 16% of ordinary messages and raised a false alarm. Calibrating the rule by
cross-validation on the training messages fixed it without losing sensitivity to a real language shift.

### 5.7 Zero-shot NLI reference

An off-the-shelf multilingual NLI model (mDeBERTa-v3-base, XNLI) with no training on this task
(`scripts/zero_shot.py`), thresholded to reach recall 0.95 on the calibration split:

| | Value |
|---|---|
| Crisis ROC-AUC [95% CI] | 0.56 [0.42, 0.70], not distinguishable from chance |
| Crisis recall on test | 0.69 (18/26) at an escalation rate of 76% |
| Category macro-F1 [95% CI] | 0.29 [0.20, 0.39] |

Without task data it is close to useless here, which confirms why I ruled it out as the main classifier.
Every trained model, including the simple baseline, is far better.


## 6. Deployment

- **API:** FastAPI (`serving/app.py`). `POST /v1/checkin` returns the crisis flag, category, response text and
  hotlines. Counsellor endpoints list and resolve escalations behind an API key. `/v1/lexicon` feeds the app's
  offline valve, and `/v1/version` reports the model, threshold, lexicon and library versions.
- **Container:** `serving/Dockerfile` + `docker-compose.yml`. Python 3.11-slim with onnxruntime, no PyTorch;
  447 MB compressed. It runs as a non-root user with a read-only model and a health check. The prediction log
  and escalation database live on a volume.
- **Safety by default:** the container refuses to start unless every library entry is counsellor-approved
  (overridable only for development). Placeholder hotline numbers are never sent to the app.
- **Privacy (DPA 2019):** the prediction log stores versions, scores and rule IDs, never the message. The
  escalation queue stores pseudonymised text only, because a counsellor needs it to act. New-word statistics
  are shown only after a word appears in at least 5 requests. Hosting is intended to be in Kenya.
- **App:** Expo React Native (`mobile/`), English/Swahili. If the phone is offline, the keyword valve runs on
  the device and shows the crisis response and hotlines. Crisis check-ins are queued and sent on reconnect.
  The TypeScript valve gives identical results to the Python valve on 234 test cases, and CI checks this.
- **CI:** GitHub Actions runs the Python tests, the app typecheck, the valve parity check, and a Docker build
  with a smoke test.

## 7. Limitations

1. **English only on real data.** The Zindi messages are English (Kenyan usage, with typos). The proposal's
   central claim, that an African-language encoder handles Sheng and Swahili better, is supported only
   indirectly: by tokenizer fertility and by the drift test on synthetic code-mixed text. It needs real
   Sheng/Swahili check-ins.
2. **The recall target cannot be proven with 26 test crises.** A 95% lower bound ≥ 0.95 needs at least 72 test
   crises with no misses (exact Clopper–Pearson: 0.025^(1/n) ≥ 0.95). With 26/26 caught the lower bound
   would be 0.87. More crisis examples (or pooled cross-validation) are needed before any claim of 0.95.
3. **Labels.** The crisis labels are my review of an AI first pass; there is no independent second coder yet,
   and the category labels were not reviewed. My reviewer rules (Section 2.2) deliberately err towards crisis.
4. **Zindi is not Strathmore.** The prompt ("What is on your mind?"), the year (2020) and the universities
   differ from the intended deployment, and some messages describe the past.
5. **The stage-1 intermediate fine-tuning is implemented but was not used** for the reported models, to keep
   CPU training time short. Its benefit is untested.
6. **The lexicon and coping library are unvalidated drafts** and must be reviewed by counsellors and
   Sheng-speaking students. The API refuses to run in production until the library is approved.
7. **Only the keyword valve runs on the phone**, and latency on a mid-range Android phone over campus Wi-Fi has
   not been measured.

## 8. Real-world relevance and next steps

Sikiza addresses a real gap: counselling capacity is scarce, and students describe distress in their own
words. The design choices matter as much as the scores. The OR-gate and offline valve mean a crisis is not
missed silently. Retrieval of approved text means no generated advice reaches a student. Cost-based thresholds
with honest intervals and escalation rates show counsellors the real trade-off. Privacy-preserving logging
means messages are not kept unnecessarily.

Before a pilot: ethics approval and consented Strathmore data including Sheng/Swahili messages with enough crisis
examples; an independent second coder; counsellor approval of the library and student review of the lexicon;
the cost ratio agreed with counsellors against their capacity; Kenya-hosted deployment with HTTPS, an on-call
rota, and monthly drift reviews.

## Appendix: reproducing this report

```bash
make setup
make real     # prepare Zindi → augment → baseline → AfroXLM-R + DistilBERT → ONNX int8 → evaluate,
              # zero-shot, fertility, drift  (about 30 minutes on a laptop CPU)
```

Outputs: `artifacts/reports/` (`eval.md`, `eval.json`, confusion matrices, `zindi_data.json`, drift and fertility
reports). AI assistance is disclosed in [`AI_USE.md`](AI_USE.md).
