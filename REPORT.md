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
| **Positive / neutral check-ins**: GoEmotions comments with one positive emotion (joy, gratitude, excitement, optimism, pride, relief), filtered hard, + 51 everyday Kenyan student check-ins in English, Swahili and Sheng written for this project (`scripts/prepare_positive.py`) | 351 | 282 training (×3) and 69 calibration rows, **never test**. Added after deployment, see Section 5.8 |
| Synthetic English/Swahili/Sheng check-ins | 224 | Code-mixed tokenizer and drift checks; pipeline smoke tests |
| GoEmotions, AfriSenti-swa, tabularisai Swahili sentiment | ~58k / ~3k / ~48k | Optional stage-1 fine-tuning (not used for the reported models) |
| Sheng lexicon; coping library | 33 rules; 17 entries + crisis response | Safety valve; retrieval corpus |

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
calibration 133 (14 crisis), test **221 (26 crisis)**. The positive/neutral rows (Section 5.8) are added to
training and calibration only; the test set is unchanged.

## 3. Techniques and why I chose them

### Architecture

```
Student app ─▶ FastAPI ─▶ pseudonymise ─▶ safety valve ─┐
 (Expo RN)                                              ├─ OR ─▶ crisis? ─yes─▶ escalation queue + crisis response
                         classifier (fp32 ONNX) ────────┘            └─no──▶ retrieve approved strategy
```

### What I used

| Technique | Why it fits this problem |
|---|---|
| **TF-IDF (word + char 2–5-grams) + logistic regression** | A cheap, explainable floor that every other model must beat. Character n-grams cope with Sheng spelling variation without a tokenizer. |
| **AfroXLM-R (Davlan/afro-xlmr-base) with LoRA**, two heads (crisis, 7 categories incl. `positive`), mean pooling | XLM-R further pre-trained on 17 African languages including Swahili. LoRA (r = 16, α = 32) trains about 1% of the weights, which suits a ~1,000-message dataset (full fine-tuning of 270M parameters would overfit) and trains on a laptop CPU. |
| **DistilBERT (multilingual) as comparison** | Smaller and faster; tests whether the African-language pre-training actually matters. |
| **Tokenizer fertility measured first** | Encoder choice should follow from how badly each tokenizer fragments Sheng, not from reputation. |
| **Two-stage training (optional stage 1)** | Sentiment pre-training on GoEmotions + AfriSenti + tabularisai teaches affect in English and Swahili before the small Strathmore set. I mapped all three to 3-way sentiment, the only label they share, instead of forcing e.g. GoEmotions "nervousness" onto "social anxiety". |
| **Trained without class weights + temperature scaling** on the calibration split | A cost-based threshold only works on real probabilities. Class weights shift the logits, and calibration is the fix. |
| **Cost-derived threshold** t = C_fp / (C_fp + C_fn) = 1/101 ≈ 0.0099, lowered further if needed to reach recall 0.95 on calibration | Bayes-optimal decision for the stated costs; 0.5 would assume equal costs. |
| **Safety valve: keyword rules OR classifier** | Either signal escalates. Rules catch explicit phrases with certainty and work offline on the phone; the classifier catches implicit warning signs. An AND-gate at a high threshold would miss crises. |
| **Retrieval, not generation**, for responses | Advice is grounded, auditable and pre-approved by counsellors. The same input always gives the same answer. |
| **ONNX export, served at full precision (fp32)** | Serves on CPU VMs without PyTorch. I first deployed a dynamic int8 version (4× smaller), but on the final model it shifted the small crisis probabilities enough to escalate ordinary happy messages (Section 5.5), so the deployed model is fp32. int8 remains available (`--quantization int8`). |
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
| **GPU serving / TorchServe** | The fp32 ONNX model runs in about 10 ms per message on a CPU. A GPU would add cost and no useful speed. |

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
training messages (+ augmentation + 282 positive/neutral check-ins), calibrated and thresholded on the same 133
(+ 69 positive/neutral), and tested once on the same 221 real messages. These are the final models, after the
fix in Section 5.8; the first deployed version's results are given there for comparison.

### 5.1 Crisis decision (the safety metric)

| Model | Recall | Exact 95% CI | Precision | Escalation rate | Category macro-F1 [95% CI] |
|---|---|---|---|---|---|
| TF-IDF + logistic regression | 1.000 (26/26) | [0.868, 1.000] | 0.197 | **0.597** | 0.461 [0.385, 0.523] |
| DistilBERT + LoRA | 0.923 (24/26) | [0.749, 0.991] | 0.240 | 0.452 | 0.521 [0.423, 0.614] |
| **AfroXLM-R + LoRA, fp32 ONNX (deployed)** | **0.923 (24/26)** | [0.749, 0.991] | **0.316** | **0.344** | **0.639** [0.523, 0.717] |
| Keyword valve alone | 0.308 (8/26) | [0.143, 0.518] | 0.889 | 0.041 | – |

(Model rows are the deployed OR-gate; on this data it equals the classifier alone, because every crisis the
valve caught, the classifier also caught. The fp32 ONNX model gives exactly the PyTorch model's decisions.)

**Interpretation.**

- **No model meets the success criterion**, and with 26 test crises none could: even 26/26 gives an exact
  lower bound of 0.868. The deployed model catches 24 of 26 crises with a lower bound of 0.75.
- **Recall must be read with the escalation rate.** The baseline's perfect recall comes from escalating 60% of
  all messages, which no counselling team could absorb. The transformers reach similar recall while escalating
  far less: the deployed model sends 34% of messages to a counsellor, a little over half the baseline's load.
  McNemar's test on per-message crisis decisions favours both transformers over the baseline (AfroXLM-R:
  p < 0.0001, correct on 60 messages where the baseline was wrong, the reverse on 8), almost entirely through
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
| TF-IDF + LR | 0.928 [0.871, 0.970] | 0.680 [0.492, 0.853] | 0.808 | 0.051 |
| DistilBERT + LoRA | 0.896 [0.796, 0.968] | 0.727 [0.558, 0.859] | 0.808 | 0.051 |
| AfroXLM-R + LoRA (deployed, fp32) | 0.915 [0.822, 0.980] | 0.751 [0.582, 0.892] | 0.846 | 0.044 |

Chance average precision = prevalence = 0.118.

**Interpretation.** All models rank crises well above chance (ROC-AUC ≈ 0.90–0.93), and their intervals
overlap almost completely: **on English text, the encoders do not rank crises measurably better than TF-IDF.**
AfroXLM-R has the highest average precision (0.75 vs 0.68), but the intervals overlap.
Their advantage in 5.1 comes from producing better-calibrated scores at the low end, so fewer non-crises clear
the very low threshold. Calibration is good for all (expected calibration error 0.04–0.05), which is what makes
a cost-derived threshold meaningful.

### 5.3 Categories

- **AfroXLM-R is the best category model**: macro-F1 0.639 vs 0.461 for the baseline (AfroXLM-R right and
  baseline wrong on 23 messages, the reverse on 11; McNemar exact p = 0.058). The first deployed version had a
  similar gap with p = 0.013, so the advantage is consistent but, with 221 messages, only borderline significant.
- **DistilBERT is not better than the baseline** on categories (0.521, p = 1.0).
- The seventh category, `positive`, has no test messages (the Zindi survey contains none), so macro-F1 is
  averaged over the six categories present in the test set. No real test message was classified `positive`.
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

| | int8 (first deployed) | **fp32 (deployed now)** |
|---|---|---|
| Model size | 279 MB | 1,110 MB |
| Crisis caught / escalation rate (final model, test) | 24/26 / 48% | **24/26 / 34%** |
| Category macro-F1 (final model, test) | 0.584 | **0.639** |
| "I am feeling happy today…" | escalated (p = 0.010) | **not escalated** (p = 0.004) |
| Max crisis-logit change vs PyTorch | up to 2.7 | 0.00002 |
| Model-only CPU time (M1) | ~5 ms per message | ~10 ms per message |
| Container memory (2 workers) | ~1.4 GB | ~2.9 GB |
| HTTP round trip, Docker on M1 (p50 / p95) | 25 / 37 ms | 55 / 106 ms |

I first deployed the int8 model: quantisation shifts the logits, so it was re-calibrated and re-thresholded on
its own outputs. On the final model that was not enough. int8 moved the small crisis probabilities by 2–3×,
which matters when the threshold is 1%: it escalated 48% of test messages instead of 34% and pushed ordinary
happy messages back over the threshold. **The fp32 export reproduces the PyTorch model exactly, so I deploy it**
and accept a 4× larger model. Latency is still far inside the 400 ms budget (a full HTTPS round trip to the
Azure deployment takes 0.3–0.5 s), so the budget is network-bound, as the proposal predicted. int8 would only be
needed for on-device inference, and would then need a better quantisation scheme (e.g. static, with
calibration data) and re-validation.

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
| Category macro-F1 [95% CI] | 0.24 [0.15, 0.34] |

(The first version of this script mapped the NLI model's "other" answers onto `substance_use` by list position,
because the zero-shot label set has no substance hypothesis; it reported 0.29. Fixed and re-run.)

Without task data it is close to useless here, which confirms why I ruled it out as the main classifier.
Every trained model, including the simple baseline, is far better.


### 5.8 Ordinary messages: a problem found after deployment

**What I saw.** Testing the deployed chat page, I typed "I am feeling happy today, it is a very sunny day!" and
Sikiza treated it as a crisis. So did "I passed my exams, so excited" and "Leo niko poa sana".

**Why.** Every Zindi message comes from a mental-health survey about depression, suicide, alcohol and drugs, so
the model had **never seen an ordinary or happy message**: this input was out of distribution. Its crisis
probability for these messages was about 1–12%, and the cost threshold is 1%, so any uncertainty escalated them.
A keyword rule ("happy → not a crisis") was not acceptable: "I'm happy now that I've decided to end it" would
pass it.

**Fix: add the missing kind of data.** I built 351 positive/neutral check-ins (Section 2.1): GoEmotions comments
with a single positive emotion, filtered to first-person statements of a good feeling with no distress, risk or
substance words and no safety-valve hit, plus 51 everyday Kenyan student check-ins in English, Swahili and Sheng.
They go to training (×3, about 9% of training rows) and calibration (20%), never to the test set.

**What the experiments showed** (same 221 test messages throughout):

| Version | Crises caught | Escalation | Category macro-F1 | Happy messages |
|---|---|---|---|---|
| First deployed (no positive data, int8) | 24/26 | 38% | 0.656 | escalated |
| A: positive rows labelled `other` | 25/26 | 32% | 0.521 | not escalated |
| B: new `positive` category | 26/26 | 53% | 0.607 | escalated again |
| C: B + positive rows in calibration + fixed seed + no exam words, int8 | 24/26 | 48% | 0.584 | escalated |
| **C, fp32 (deployed now)** | **24/26** | **34%** | **0.639** | **not escalated** |

- **A** fixed the happy messages but taught the model that `other` means "not clearly negative": low-mood
  recall fell from 0.93 to 0.68, so sad students would get a generic reply. Positive messages need their own
  category.
- **B** gave the new category but came out worse on escalation. The model's initial weights were not seeded,
  so runs differed a lot; I fixed the seed for C. Some written positive examples mentioned exams, and the model
  started calling "I am so stressed about my exams" `positive`; I removed those 9 examples for C.
- **C** included positive messages in calibration, because real check-in traffic includes them. The trained
  model was good, but the int8 copy was not (Section 5.5). Serving it at fp32 gave the final result.

**Safeguard.** A `positive` reply ("That's really good to hear…") said to a sad student would be harmful, so the
API uses `positive` only when the model is at least 60% sure and no distress word from the lexicon is present;
otherwise it falls back to the next most likely category (`triage/pipeline.py`, `pick_category`).

**What is still imperfect.** "I passed my exams, so excited" and "I have no money for fees this semester" are
still escalated (probability about 1–2%): exam and money words are strongly tied to distress in the training
data. At a 1% threshold the model stays on the cautious side, which is the intended direction of error. The
lesson for the report: **test sets drawn from the same survey as the training data cannot reveal this**; it
showed up only when I used the deployed app like a real student would.

## 6. Deployment

- **API:** FastAPI (`serving/app.py`). `POST /v1/checkin` returns the crisis flag, category, response text and
  hotlines. Counsellor endpoints list and resolve escalations behind an API key. `/v1/lexicon` feeds the app's
  offline valve, and `/v1/version` reports the model, threshold, lexicon and library versions.
- **Container:** `serving/Dockerfile` + `docker-compose.yml`. Python 3.11-slim with onnxruntime and the fp32
  model, no PyTorch. It runs as a non-root user with a read-only model and a health check. The prediction log
  and escalation database live on a volume.
- **Safety by default:** the container refuses to start unless every library entry is counsellor-approved
  (overridable only for development). Placeholder hotline numbers are never sent to the app.
- **Privacy (DPA 2019):** the prediction log stores versions, scores and rule IDs, never the message. The
  escalation queue stores pseudonymised text only, because a counsellor needs it to act. New-word statistics
  are shown only after a word appears in at least 5 requests. Hosting is intended to be in Kenya.
- **App:** Expo React Native (`mobile/`), English/Swahili. If the phone is offline, the keyword valve runs on
  the device and shows the crisis response and hotlines. Crisis check-ins are queued and sent on reconnect.
  The TypeScript valve gives identical results to the Python valve on 234 test cases, and CI checks this.
- **Live demo:** Azure Container Apps (South Africa North, the closest region my student subscription allows;
  Azure has no Kenyan region), 2 vCPU / 4 GB, scales to zero. The same container serves a chat page at `/`
  where a student types a message and sees the reply; crisis replies show call buttons for the hotlines.
  `deploy/azure/deploy.sh` reproduces it. It is a demo, not the production site (outside Kenya, draft library).
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
7. **The positive/neutral data is partly from another domain.** Most of it is English Reddit comments
   (GoEmotions); the 51 Kenyan check-ins were AI-drafted for this project, not written by students. It
   fixes the obvious failure, but how often real Strathmore students send ordinary messages, and how the
   model handles them, can only be measured on real check-ins.
8. **Only the keyword valve runs on the phone**, and latency on a mid-range Android phone over campus Wi-Fi has
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
make real     # prepare Zindi + positive check-ins → augment → baseline → AfroXLM-R + DistilBERT → ONNX (fp32)
              # → evaluate, zero-shot, fertility, drift  (about 30 minutes on a laptop CPU)
```

Outputs: `artifacts/reports/` (`eval.md`, `eval.json`, confusion matrices, `zindi_data.json`, drift and fertility
reports). AI assistance is disclosed in [`AI_USE.md`](AI_USE.md).
