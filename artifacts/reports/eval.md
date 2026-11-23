Test set: 221 real seed messages, 26 crisis; groups {'english_only': 221}. Cohen's κ: None

| model | gate | crisis recall | bootstrap 95% CI | exact 95% CI | precision | escalation rate | macro-F1 [95% CI] | recall ≥ 0.95 (lower bound) |
|---|---|---|---|---|---|---|---|---|
| baseline-634682de | classifier_only | 1.000 | [1.000, 1.000] | [0.868, 1.000] | 0.197 | 0.597 | 0.461 [0.385, 0.523] | no |
| baseline-634682de | combined_or_gate | 1.000 | [1.000, 1.000] | [0.868, 1.000] | 0.197 | 0.597 | 0.461 [0.385, 0.523] | no |
| baseline-634682de | valve_only | 0.308 | [0.130, 0.500] | [0.143, 0.518] | 0.889 | 0.041 | – | no |
| afroxlmr-634682de | classifier_only | 0.923 | [0.800, 1.000] | [0.749, 0.991] | 0.316 | 0.344 | 0.639 [0.523, 0.717] | no |
| afroxlmr-634682de | combined_or_gate | 0.923 | [0.800, 1.000] | [0.749, 0.991] | 0.316 | 0.344 | 0.639 [0.523, 0.717] | no |
| afroxlmr-634682de | valve_only | 0.308 | [0.130, 0.500] | [0.143, 0.518] | 0.889 | 0.041 | – | no |
| distilbert-634682de | classifier_only | 0.923 | [0.808, 1.000] | [0.749, 0.991] | 0.240 | 0.452 | 0.521 [0.423, 0.614] | no |
| distilbert-634682de | combined_or_gate | 0.923 | [0.808, 1.000] | [0.749, 0.991] | 0.240 | 0.452 | 0.521 [0.423, 0.614] | no |
| distilbert-634682de | valve_only | 0.308 | [0.130, 0.500] | [0.143, 0.518] | 0.889 | 0.041 | – | no |
| afroxlmr-634682de-onnx-fp32 | classifier_only | 0.923 | [0.800, 1.000] | [0.749, 0.991] | 0.316 | 0.344 | 0.639 [0.523, 0.717] | no |
| afroxlmr-634682de-onnx-fp32 | combined_or_gate | 0.923 | [0.800, 1.000] | [0.749, 0.991] | 0.316 | 0.344 | 0.639 [0.523, 0.717] | no |
| afroxlmr-634682de-onnx-fp32 | valve_only | 0.308 | [0.130, 0.500] | [0.143, 0.518] | 0.889 | 0.041 | – | no |

- ⚠ baseline-634682de / classifier_only: escalates more than half of all messages — recall is not informative
- ⚠ baseline-634682de / combined_or_gate: escalates more than half of all messages — recall is not informative

**Crisis score ranking (threshold-free)**

| model | ROC-AUC [95% CI] | average precision [95% CI] | chance AP | recall if top 20% escalated | test ECE |
|---|---|---|---|---|---|
| baseline-634682de | 0.928 [0.871, 0.970] | 0.680 [0.492, 0.853] | 0.118 | 0.808 | 0.051 |
| afroxlmr-634682de | 0.915 [0.822, 0.980] | 0.751 [0.582, 0.892] | 0.118 | 0.846 | 0.044 |
| distilbert-634682de | 0.896 [0.796, 0.968] | 0.727 [0.558, 0.859] | 0.118 | 0.808 | 0.051 |
| afroxlmr-634682de-onnx-fp32 | 0.915 [0.822, 0.980] | 0.751 [0.582, 0.892] | 0.118 | 0.846 | 0.044 |

**English-only vs code-mixed (combined OR-gate)**

| model | group | n | crisis recall [95% CI] | macro-F1 |
|---|---|---|---|---|
| baseline-634682de | english_only | 221 | 1.000 [1.000, 1.000] | 0.461 |
| afroxlmr-634682de | english_only | 221 | 0.923 [0.800, 1.000] | 0.639 |
| distilbert-634682de | english_only | 221 | 0.923 [0.808, 1.000] | 0.521 |
| afroxlmr-634682de-onnx-fp32 | english_only | 221 | 0.923 [0.800, 1.000] | 0.639 |

**McNemar (exact) vs reference**

- baseline-634682de vs afroxlmr-634682de: crisis p=0.0000 (8/60), category p=0.0576 (11/23)
- baseline-634682de vs distilbert-634682de: crisis p=0.0015 (23/51), category p=1.0000 (18/19)
- baseline-634682de vs afroxlmr-634682de-onnx-fp32: crisis p=0.0000 (8/60), category p=0.0576 (11/23)
