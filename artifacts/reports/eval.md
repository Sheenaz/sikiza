Test set: 221 real seed messages, 26 crisis; groups {'english_only': 221}. Cohen's κ: None

| model | gate | crisis recall | bootstrap 95% CI | exact 95% CI | precision | escalation rate | macro-F1 [95% CI] | recall ≥ 0.95 (lower bound) |
|---|---|---|---|---|---|---|---|---|
| baseline-f49c2ebd | classifier_only | 1.000 | [1.000, 1.000] | [0.868, 1.000] | 0.164 | 0.719 | 0.465 [0.389, 0.526] | no |
| baseline-f49c2ebd | combined_or_gate | 1.000 | [1.000, 1.000] | [0.868, 1.000] | 0.164 | 0.719 | 0.465 [0.389, 0.526] | no |
| baseline-f49c2ebd | valve_only | 0.308 | [0.130, 0.500] | [0.143, 0.518] | 0.889 | 0.041 | – | no |
| afroxlmr-f49c2ebd | classifier_only | 0.962 | [0.870, 1.000] | [0.804, 0.999] | 0.207 | 0.548 | 0.655 [0.556, 0.722] | no |
| afroxlmr-f49c2ebd | combined_or_gate | 0.962 | [0.870, 1.000] | [0.804, 0.999] | 0.207 | 0.548 | 0.655 [0.556, 0.722] | no |
| afroxlmr-f49c2ebd | valve_only | 0.308 | [0.130, 0.500] | [0.143, 0.518] | 0.889 | 0.041 | – | no |
| distilbert-f49c2ebd | classifier_only | 0.962 | [0.870, 1.000] | [0.804, 0.999] | 0.236 | 0.480 | 0.552 [0.444, 0.642] | no |
| distilbert-f49c2ebd | combined_or_gate | 0.962 | [0.870, 1.000] | [0.804, 0.999] | 0.236 | 0.480 | 0.552 [0.444, 0.642] | no |
| distilbert-f49c2ebd | valve_only | 0.308 | [0.130, 0.500] | [0.143, 0.518] | 0.889 | 0.041 | – | no |
| afroxlmr-f49c2ebd-onnx-int8 | classifier_only | 0.923 | [0.800, 1.000] | [0.749, 0.991] | 0.286 | 0.380 | 0.656 [0.563, 0.721] | no |
| afroxlmr-f49c2ebd-onnx-int8 | combined_or_gate | 0.923 | [0.800, 1.000] | [0.749, 0.991] | 0.286 | 0.380 | 0.656 [0.563, 0.721] | no |
| afroxlmr-f49c2ebd-onnx-int8 | valve_only | 0.308 | [0.130, 0.500] | [0.143, 0.518] | 0.889 | 0.041 | – | no |

- ⚠ baseline-f49c2ebd / classifier_only: escalates more than half of all messages — recall is not informative
- ⚠ baseline-f49c2ebd / combined_or_gate: escalates more than half of all messages — recall is not informative
- ⚠ afroxlmr-f49c2ebd / classifier_only: escalates more than half of all messages — recall is not informative
- ⚠ afroxlmr-f49c2ebd / combined_or_gate: escalates more than half of all messages — recall is not informative

**Crisis score ranking (threshold-free)**

| model | ROC-AUC [95% CI] | average precision [95% CI] | chance AP | recall if top 20% escalated | test ECE |
|---|---|---|---|---|---|
| baseline-f49c2ebd | 0.929 [0.876, 0.969] | 0.664 [0.481, 0.844] | 0.118 | 0.808 | 0.041 |
| afroxlmr-f49c2ebd | 0.926 [0.862, 0.971] | 0.654 [0.478, 0.841] | 0.118 | 0.769 | 0.038 |
| distilbert-f49c2ebd | 0.932 [0.866, 0.976] | 0.722 [0.535, 0.868] | 0.118 | 0.846 | 0.051 |
| afroxlmr-f49c2ebd-onnx-int8 | 0.917 [0.858, 0.965] | 0.625 [0.448, 0.820] | 0.118 | 0.808 | 0.065 |

**English-only vs code-mixed (combined OR-gate)**

| model | group | n | crisis recall [95% CI] | macro-F1 |
|---|---|---|---|---|
| baseline-f49c2ebd | english_only | 221 | 1.000 [1.000, 1.000] | 0.465 |
| afroxlmr-f49c2ebd | english_only | 221 | 0.962 [0.870, 1.000] | 0.655 |
| distilbert-f49c2ebd | english_only | 221 | 0.962 [0.870, 1.000] | 0.552 |
| afroxlmr-f49c2ebd-onnx-int8 | english_only | 221 | 0.923 [0.800, 1.000] | 0.656 |

**McNemar (exact) vs reference**

- baseline-f49c2ebd vs afroxlmr-f49c2ebd: crisis p=0.0000 (14/50), category p=0.0125 (7/21)
- baseline-f49c2ebd vs distilbert-f49c2ebd: crisis p=0.0000 (6/57), category p=0.8555 (14/16)
- baseline-f49c2ebd vs afroxlmr-f49c2ebd-onnx-int8: crisis p=0.0000 (9/80), category p=0.0501 (10/22)
