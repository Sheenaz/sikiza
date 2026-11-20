> **DEMO DATA — synthetic rows; numbers are meaningless.**

Test set: 56 real seed messages, 6 crisis; groups {'code_mixed': 42, 'english_only': 14}. Cohen's κ: {'category_kappa': 0.8662807525325615, 'category_n': 66, 'crisis_kappa': 1.0, 'crisis_n': 66}

| model | gate | crisis recall | bootstrap 95% CI | exact 95% CI | precision | escalation rate | macro-F1 [95% CI] | recall ≥ 0.95 (lower bound) |
|---|---|---|---|---|---|---|---|---|
| baseline-46e01191 | classifier_only | 1.000 | [1.000, 1.000] | [0.541, 1.000] | 0.107 | 1.000 | 0.864 [0.737, 0.949] | no |
| baseline-46e01191 | combined_or_gate | 1.000 | [1.000, 1.000] | [0.541, 1.000] | 0.107 | 1.000 | 0.864 [0.737, 0.949] | no |
| baseline-46e01191 | valve_only | 0.667 | [0.200, 1.000] | [0.223, 0.957] | 1.000 | 0.071 | – | no |
| afroxlmr-46e01191 | classifier_only | 1.000 | [1.000, 1.000] | [0.541, 1.000] | 0.107 | 1.000 | 0.669 [0.524, 0.805] | no |
| afroxlmr-46e01191 | combined_or_gate | 1.000 | [1.000, 1.000] | [0.541, 1.000] | 0.107 | 1.000 | 0.669 [0.524, 0.805] | no |
| afroxlmr-46e01191 | valve_only | 0.667 | [0.200, 1.000] | [0.223, 0.957] | 1.000 | 0.071 | – | no |
| distilbert-46e01191 | classifier_only | 1.000 | [1.000, 1.000] | [0.541, 1.000] | 0.107 | 1.000 | 0.757 [0.614, 0.867] | no |
| distilbert-46e01191 | combined_or_gate | 1.000 | [1.000, 1.000] | [0.541, 1.000] | 0.107 | 1.000 | 0.757 [0.614, 0.867] | no |
| distilbert-46e01191 | valve_only | 0.667 | [0.200, 1.000] | [0.223, 0.957] | 1.000 | 0.071 | – | no |
| afroxlmr-46e01191-onnx-int8 | classifier_only | 1.000 | [1.000, 1.000] | [0.541, 1.000] | 0.107 | 1.000 | 0.644 [0.456, 0.767] | no |
| afroxlmr-46e01191-onnx-int8 | combined_or_gate | 1.000 | [1.000, 1.000] | [0.541, 1.000] | 0.107 | 1.000 | 0.644 [0.456, 0.767] | no |
| afroxlmr-46e01191-onnx-int8 | valve_only | 0.667 | [0.200, 1.000] | [0.223, 0.957] | 1.000 | 0.071 | – | no |

- ⚠ baseline-46e01191 / classifier_only: escalates more than half of all messages — recall is not informative
- ⚠ baseline-46e01191 / combined_or_gate: escalates more than half of all messages — recall is not informative
- ⚠ afroxlmr-46e01191 / classifier_only: escalates more than half of all messages — recall is not informative
- ⚠ afroxlmr-46e01191 / combined_or_gate: escalates more than half of all messages — recall is not informative
- ⚠ distilbert-46e01191 / classifier_only: escalates more than half of all messages — recall is not informative
- ⚠ distilbert-46e01191 / combined_or_gate: escalates more than half of all messages — recall is not informative
- ⚠ afroxlmr-46e01191-onnx-int8 / classifier_only: escalates more than half of all messages — recall is not informative
- ⚠ afroxlmr-46e01191-onnx-int8 / combined_or_gate: escalates more than half of all messages — recall is not informative

**Crisis score ranking (threshold-free)**

| model | ROC-AUC [95% CI] | average precision [95% CI] | chance AP | recall if top 20% escalated | test ECE |
|---|---|---|---|---|---|
| baseline-46e01191 | 0.937 [0.848, 1.000] | 0.672 [0.297, 1.000] | 0.107 | 0.667 | 0.053 |
| afroxlmr-46e01191 | 0.957 [0.890, 1.000] | 0.721 [0.333, 1.000] | 0.107 | 1.000 | 0.103 |
| distilbert-46e01191 | 0.853 [0.707, 0.971] | 0.465 [0.148, 0.832] | 0.107 | 0.667 | 0.093 |
| afroxlmr-46e01191-onnx-int8 | 0.960 [0.885, 1.000] | 0.797 [0.433, 1.000] | 0.107 | 0.833 | 0.118 |

**English-only vs code-mixed (combined OR-gate)**

| model | group | n | crisis recall [95% CI] | macro-F1 |
|---|---|---|---|---|
| baseline-46e01191 | english_only | 14 | 1.000 [1.000, 1.000] | 0.622 |
| baseline-46e01191 | code_mixed | 42 | 1.000 [1.000, 1.000] | 0.925 |
| afroxlmr-46e01191 | english_only | 14 | 1.000 [1.000, 1.000] | 0.710 |
| afroxlmr-46e01191 | code_mixed | 42 | 1.000 [1.000, 1.000] | 0.647 |
| distilbert-46e01191 | english_only | 14 | 1.000 [1.000, 1.000] | 0.689 |
| distilbert-46e01191 | code_mixed | 42 | 1.000 [1.000, 1.000] | 0.760 |
| afroxlmr-46e01191-onnx-int8 | english_only | 14 | 1.000 [1.000, 1.000] | 0.590 |
| afroxlmr-46e01191-onnx-int8 | code_mixed | 42 | 1.000 [1.000, 1.000] | 0.641 |

**McNemar (exact) vs reference**

- baseline-46e01191 vs afroxlmr-46e01191: crisis p=1.0000 (0/0), category p=0.0768 (12/4)
- baseline-46e01191 vs distilbert-46e01191: crisis p=1.0000 (0/0), category p=0.1796 (10/4)
- baseline-46e01191 vs afroxlmr-46e01191-onnx-int8: crisis p=1.0000 (0/0), category p=0.0192 (15/4)
