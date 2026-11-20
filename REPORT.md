# Sikiza: Sheng-Aware Distress Triage and Coping Support

**DSA 8401 Applied Machine Learning · Capstone report · Lousa Sheenaz Yogo · November 2026**

> **Read this first.** The system is built end to end, but every number in this report comes from a
> **synthetic demo dataset** (224 template-built check-ins), because the consented Strathmore
> check-ins were not available by the deadline. The numbers show that the pipeline and the
> evaluation work; they are **not** evidence of how the models would perform on real students.
> Section 7 explains what that means for each result.

---

## 1. Problem framing

**What is predicted.** For each short student check-in written in English, Swahili, Sheng or a mix:

1. **Crisis vs non-crisis**: the safety decision. Crisis means signs of suicidal thinking, self-harm or
   immediate danger.
2. **One of five categories** for non-crisis messages: academic stress, financial strain, social anxiety,
   low mood, other.

The system then responds. A crisis goes to a human counsellor's escalation queue, and the student
immediately sees a crisis message with hotline numbers. Any other message gets a coping strategy
**retrieved** from a counsellor-approved library for its category. Nothing is written freely by a model.

**Why it matters.** Kenya has fewer than 100 practising psychiatrists for over 50 million people, and a
student can wait about three weeks for a counselling appointment. Global models misread code-mixed text
("nime-hang", "kujitoa", "nimeboeka"), so distress is missed exactly when students express it most
naturally. Sikiza is digital first aid in front of counselling, not a diagnostic device.

**The costs are asymmetric, and that drives the design.** A missed crisis is catastrophic; a false alarm
costs a counsellor a few minutes. I set the cost of a false negative to 100 and of a false positive to 1.
The decision threshold is derived from these costs instead of using 0.5, and the main success
criterion is **crisis recall ≥ 0.95, with the lower bound of a 95% confidence interval**, not just the point
estimate.

**Success criteria (from the proposal):** crisis recall ≥ 0.95 with a bootstrap 95% CI; macro-F1 and
confusion matrix for the categories; results split by English-only vs code-mixed; McNemar's test against
the baseline; Cohen's κ for label agreement; under 400 ms round trip on a mid-range Android phone.

## 2. Data

| Source | Size | Role | Known issues |
|---|---|---|---|
| Strathmore check-ins (consented, de-identified, human-labelled) | ~1,000 planned; **not yet available** | Seed training data and the **only** test data | Small; crisis class rare; single annotator |
| Synthetic demo seed (`scripts/make_demo_data.py`) | 224 rows, 24 crisis (10.7%); 80 sw / 77 en / 67 sheng | Stand-in for the seed set so every stage could be built and tested | Template-built, so far more regular than real text |
| Augmentation (`triage/augment.py`) | 2,010 rows from the 134 training seeds | Training volume only | Near-duplicates of seed rows |
| GoEmotions, AfriSenti-swa, tabularisai Swahili sentiment | ~58k / ~3k / ~48k | Optional stage-1 intermediate fine-tuning (3-way sentiment) | Not mental-health or Sheng; tabularisai is LLM-generated |
| Sheng lexicon | 33 rules | Safety-valve keywords | My draft; not yet validated by Sheng speakers |
| Coping library | 14 entries + crisis response | Retrieval corpus | Drafts; `approved: false` until counsellors sign off |

**Realistic complications built into the pipeline** (the brief requires them):

- **Class imbalance.** Crisis is about 5–11% of messages. I handle it with a cost-derived threshold, not
  resampling (Section 3).
- **Code-mixing and spelling noise.** Sheng, SMS spellings ("nmechoka", "coz") and English–Swahili
  switching inside one sentence. Normalisation (`triage/text.py`) lower-cases, folds repeated letters and
  splits on hyphens ("nime-hang"). Augmentation adds Sheng swaps and SMS noise to the training data.
- **Augmentation leakage.** Every augmented variant keeps its seed's `seed_id`. Splits are **grouped by
  seed_id** and augmented rows join **train only**, so no variant of a test message is ever trained on.
  `scripts/evaluate.py` asserts that the test set contains real seed messages only. This matters because
  the original proposal's F1 of 0.91 was measured with augmented variants on both sides of the split.
- **Label noise.** A double-coded sample (`category_2`, `crisis_2`) gives Cohen's κ.
- **Privacy.** Pseudonymisation runs before anything else, both at data load and at serving time. It removes
  names (from introductions and a gazetteer), phone numbers, emails, URLs, M-Pesa codes, student IDs and
  social handles.

**Splits** (fixed once, never re-drawn after seeing test results): train 134 seeds (+ augmentation),
calibration 34 (4 crisis), test 56 (6 crisis), stratified by crisis and grouped by seed.

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
| **AfroXLM-R (Davlan/afro-xlmr-base) with LoRA**, two heads (crisis, 5 categories), mean pooling | XLM-R further pre-trained on 17 African languages including Swahili. LoRA (r = 16, α = 32) trains about 1% of the weights, which suits a ~1,000-message dataset (full fine-tuning of 270M parameters would overfit) and trains on a laptop CPU. |
| **DistilBERT (multilingual) as comparison** | Smaller and faster; tests whether the African-language pre-training actually matters. |
| **Tokenizer fertility measured first** | Encoder choice should follow from how badly each tokenizer fragments Sheng, not from reputation. |
| **Two-stage training (optional stage 1)** | Sentiment pre-training on GoEmotions + AfriSenti + tabularisai teaches affect in English and Swahili before the small Strathmore set. I mapped all three to 3-way sentiment, the only label they share, instead of forcing e.g. GoEmotions "nervousness" onto "social anxiety". |
| **Trained without class weights + temperature scaling** on the calibration split | A cost-based threshold only works on real probabilities. Class weights shift the logits, and calibration is the fix. |
| **Cost-derived threshold** t = C_fp / (C_fp + C_fn) = 1/101 ≈ 0.0099, lowered further if needed to reach recall 0.95 on calibration | Bayes-optimal decision for the stated costs; 0.5 would assume equal costs. |
| **Safety valve: keyword rules OR classifier** | Either signal escalates. Rules catch explicit phrases with certainty and work offline on the phone; the classifier catches implicit warning signs. An AND-gate at a high threshold would miss crises. |
| **Retrieval, not generation**, for responses | Advice is grounded, auditable and pre-approved by counsellors. The same input always gives the same answer. |
| **ONNX export + dynamic int8 quantisation** (per channel), then re-calibration | Serves on cheap CPU VMs and is small enough for weak connections. Quantisation shifts the logits, so the int8 model gets its own calibration and threshold. |
| **Drift: novelty (nearest-neighbour distance) + unseen-word rate + KMeans clusters** | Sheng changes within months. The monthly report surfaces clusters of new words for the lexicon review. |
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

## 5. Results (synthetic demo data)

Test set: 56 messages, 6 crisis; 14 English-only, 42 code-mixed. Full output: `artifacts/reports/eval.md`.

### 5.1 Crisis decision

| Model | Gate | Recall | Exact 95% CI | Precision | Escalation rate | Macro-F1 [95% CI] |
|---|---|---|---|---|---|---|
| TF-IDF + LR | OR-gate | 1.000 | [0.541, 1.000] | 0.107 | **1.000** | 0.864 [0.737, 0.949] |
| AfroXLM-R + LoRA (PyTorch) | OR-gate | 1.000 | [0.541, 1.000] | 0.107 | **1.000** | 0.669 [0.524, 0.805] |
| DistilBERT + LoRA | OR-gate | 1.000 | [0.541, 1.000] | 0.107 | **1.000** | 0.757 [0.614, 0.867] |
| AfroXLM-R int8 ONNX (deployed) | OR-gate | 1.000 | [0.541, 1.000] | 0.107 | **1.000** | 0.644 [0.456, 0.767] |
| Keyword valve alone | – | 0.667 (4/6) | [0.223, 0.957] | 1.000 | 0.071 | – |

**Interpretation.** Every model "achieves" recall 1.0 only because it **escalates every message**. With only
four crisis examples in the calibration split, the threshold fell to the cost value 1/101, and after
calibration every test message scores above it. That is safe but useless: counsellors would receive all
traffic. None of the models meets the target, because the exact lower bound is 0.54. Six test crises
cannot prove 0.95 even if all are caught.

### 5.2 Can the models separate crises at all? (threshold-free)

| Model | ROC-AUC [95% CI] | Average precision [95% CI] | Recall if top 20% escalated | Test ECE |
|---|---|---|---|---|
| TF-IDF + LR | 0.937 [0.848, 1.000] | 0.672 [0.297, 1.000] | 0.667 | 0.053 |
| AfroXLM-R + LoRA | 0.957 [0.890, 1.000] | 0.721 [0.333, 1.000] | **1.000** | 0.103 |
| DistilBERT + LoRA | 0.853 [0.707, 0.971] | 0.465 [0.148, 0.832] | 0.667 | 0.093 |
| AfroXLM-R int8 | 0.960 [0.885, 1.000] | 0.797 [0.433, 1.000] | 0.833 | 0.118 |

Chance average precision is the prevalence, 0.107.

**Interpretation.** The scores carry real signal. AfroXLM-R ranks crises well above chance, and escalating
only its top 20% would have caught all six test crises. So the failure in 5.1 is in **setting the
threshold from four calibration crises**, not in the model. The fix is more crisis examples in calibration,
not a different model. AfroXLM-R ranks better than DistilBERT, which matches the tokenizer result in 5.4,
but the confidence intervals overlap heavily with n = 6.

### 5.3 Categories, subgroups and significance

- **Macro-F1:** the baseline is highest (0.86), then DistilBERT (0.76), AfroXLM-R (0.67), int8 (0.64).
  The synthetic messages are built from templates with fixed key phrases, which favours a bag-of-n-grams model,
  and the transformers had 2 epochs on 134 seed messages. I don't expect this ordering to survive real data,
  but I won't claim the reverse without measuring it.
- **McNemar (category correctness, baseline vs model):** AfroXLM-R p = 0.077 (12 vs 4 discordant), DistilBERT
  p = 0.18, int8 p = 0.019. Only the int8 difference is significant at 0.05, in the baseline's favour. Crisis
  decisions are identical (p = 1.0), because every model escalates everything.
- **English-only vs code-mixed macro-F1:** baseline 0.62 vs 0.93; AfroXLM-R 0.71 vs 0.65; DistilBERT 0.69 vs
  0.76. With only 14 English-only messages, these differences are within noise.
- **Cohen's κ:** 0.87 (category) and 1.0 (crisis) on 66 double-coded rows. The second coding is also
  synthetic, so this only shows the computation works.

### 5.4 Tokenizer fertility (measured before choosing the encoder)

| Tokenizer | Tokens per word, English | Tokens per word, code-mixed | Words split into ≥ 3 pieces (code-mixed) |
|---|---|---|---|
| AfroXLM-R / XLM-R (shared) | 1.18 | **1.57** | 12.5% |
| Multilingual DistilBERT | 1.26 | **1.93** | 23.5% |

The DistilBERT tokenizer splits Sheng and Swahili words into many more pieces ("chapaa", "kujinyonga"), which
supports the proposal's choice of an XLM-R-family encoder. This is a property of the tokenizers, so it should
largely hold on real text, but I'll re-measure it there.

### 5.5 Quantisation and latency

| | Value |
|---|---|
| Model size, fp32 → int8 | 1,110 MB → 279 MB |
| Max crisis-logit change after int8 | 1.34 (fp32 ONNX export itself matched PyTorch exactly) |
| Crisis decisions unchanged on calibration set | 100% |
| Category predictions unchanged | 71% |
| Model-only CPU time (M1) | ~5.7 ms per message |
| HTTP round trip, local | ~7.5 ms |
| HTTP round trip through Docker | p50 25 ms, p95 37 ms |

Quantisation shifted the logits enough that reusing the fp32 calibration would have been wrong, so the int8
bundle is re-calibrated and re-thresholded on its own outputs. The cost shows in category agreement (71%) and
macro-F1 (0.67 → 0.64), while crisis ranking held. Latency is far inside the 400 ms budget, so the budget is
network-bound, as the proposal predicted. It still has to be measured on a mid-range Android phone.

### 5.6 Safety valve and drift

- The keyword valve alone caught 4 of 6 test crises with **no false alarms** (escalation rate 7%). The two it
  missed were implicit ("I've written goodbye letters…"), and only the classifier catches those. This is
  the reason for the OR-gate.
- **Drift check:** on a simulated month of 162 new messages with injected new slang ("wueh", "mafala"), 12.3%
  were flagged as novel (vs 5% expected with no drift), the unseen-word rate was 12.8%, and the new terms came
  out as the top candidates in the novelty clusters, ready for the lexicon review.

### 5.7 Zero-shot NLI reference

An off-the-shelf multilingual NLI model (mDeBERTa-v3-base, XNLI) with no Strathmore training
(`scripts/zero_shot.py`):

| | Value |
|---|---|
| Crisis ROC-AUC [95% CI] | 0.69 [0.46, 0.89] |
| Crisis average precision | 0.34 (chance 0.107) |
| Escalation rate at the threshold reaching recall 0.95 on calibration | 0.82 |
| Category macro-F1 [95% CI] | 0.55 [0.41, 0.67] |

Without training it ranks crises far worse than any trained model (its ROC-AUC interval reaches chance), and
to reach the recall target it must escalate 82% of messages. That confirms why I ruled it out as the main
classifier, and it is the floor the fine-tuned models clearly beat.

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

1. **No real data yet. This is the main limitation.** All results come from template-built synthetic
   messages. They show that the pipeline, the evaluation and the deployment work. They say nothing about
   real-world accuracy, the ranking between models, or the English vs code-mixed gap. The same commands
   produce the real results once `data/raw/strathmore_checkins.csv` exists (README).
2. **The recall target needs more crisis examples than a random split gives.** Proving a 95% lower bound
   ≥ 0.95 requires **at least 72 test crises with zero misses** (exact Clopper–Pearson: 0.025^(1/n) ≥ 0.95).
   About 1,000 messages at about 5% prevalence gives roughly 12 test crises, so a lower bound near 0.74 at
   best. Options: enrich the test set with additional consented crisis messages, pool predictions with
   grouped k-fold cross-validation, or report the honest interval.
3. **Threshold setting is fragile with few calibration crises** (4 here), as Section 5.1 shows. The
   evaluator now reports the escalation rate and ranking metrics so this can't hide behind a recall of 1.0.
   If counsellor capacity is the constraint, the cost ratio should be revisited with them rather than the
   threshold being raised quietly.
4. **The stage-1 intermediate fine-tuning is implemented but was not used for the reported models**, to keep
   CPU training time short (`stage1: null` in the manifests). Its benefit is untested.
5. **The lexicon and coping library are unvalidated drafts.** Sheng speakers and counsellors must review them.
6. **Pseudonymisation misses names that are neither introduced nor in the gazetteer.** It is defence in
   depth, not a guarantee.
7. **Only the keyword valve runs on the phone.** The proposal's on-device int8 model needs an Expo development
   build and an XLM-R tokenizer in JavaScript.
8. **Latency on a real mid-range Android phone over campus Wi-Fi has not been measured.**

## 8. Real-world relevance and next steps

Sikiza addresses a real gap at Strathmore and other Kenyan campuses. Counselling capacity is scarce, and
students describe distress in the language they actually use. The design choices matter more than the
model scores:

- the OR-gate and offline valve, so a crisis is never missed silently;
- retrieval of approved text, so no generated advice reaches students;
- calibrated, cost-based thresholds with honest intervals, so stakeholders can see real performance;
- privacy built into logging, so student messages aren't kept unnecessarily.

These make it something a counselling department could adopt responsibly. The next steps before any pilot:

1. ethics approval and the consented Strathmore dataset, with enough crisis examples (Limitation 2);
2. counsellor approval of the library and student-panel review of the lexicon;
3. retraining with stage 1, then re-evaluation, with the cost ratio agreed with counsellors;
4. Kenya-hosted deployment with HTTPS, an on-call rota for escalations, and monthly drift reviews.

## Appendix: reproducing this report

```bash
make setup && make demo                                    # data → augment → baseline → AfroXLM-R → ONNX → eval
make distilbert DEMO=--demo && make evaluate DEMO=--demo   # adds DistilBERT; writes artifacts/reports/eval.md
make fertility DEMO=--demo && make zeroshot DEMO=--demo
```

AI assistance is disclosed in [`AI_USE.md`](AI_USE.md).
