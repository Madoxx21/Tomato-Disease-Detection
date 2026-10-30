# Project decisions

## Lab 03 — Acceptance

### Decision after the baseline

Decision: Narrow

Reason: Based on **62.5% accuracy (50 of 80 held-out images correct)** in [the Lab 02 baseline](../reports/baseline_metrics.json) and the important limitation that these are controlled PlantVillage photographs, we decided to limit the current prototype's evaluation claims to clear, individual tomato-leaf color photos under comparable conditions and the eight supported categories. Grower phone-photo use remains unvalidated. Target Spot and mosaic virus remain outside the supported categories because usable leaf mappings were unavailable; we are not selecting a new project or changing the model today.

The measured result is a first course baseline, not an acceptance threshold or evidence of readiness. The small, equally sampled evaluation set does not represent field conditions or natural class frequencies. The [Lab 02 split strategy](split_strategy.md) remains in effect; future tuning needs separate validation data.

### Required behavior

- **Normal input:** For a clear, readable color photo of one tomato leaf within the supported scope, return a suggested supported category, clearly marked preliminary. A healthy suggestion must not imply that disease has been ruled out. The user can inspect the plant, retake the photo, or discuss the suggestion with an expert before deciding on treatment.
- **Invalid input:** For an unreadable file, a non-image, or a visibly unusable photo (for example, severe blur or no identifiable leaf), return a specific reason and photo guidance without a category. The user can submit a clearer single-leaf color photo. Recognizing unsuitable content is required behavior to implement and evaluate later, not a demonstrated baseline capability.
- **Valid input with an uncertain result:** Return “Uncertain — needs review” when a usable photo cannot support a reliable category; explain that unsupported conditions, including the two excluded diseases, may be present. The user can retake the photo or seek expert review. The uncertainty rule and any model-score cutoff are not set yet; they need separate validation, including unsupported-condition examples. This fallback is planned, not implemented or validated by Lab 02.

### One quality that matters

- **Response time:** The user should receive either a suggestion or an actionable fallback promptly. No time target is set yet. First measure elapsed time from submitting a photo to displaying the complete result or error, including upload and processing, on an agreed representative grower phone and network connection. Record the device, connection, photo sizes, and timing distribution before proposing a target.

### One acceptance check

- **Important failure:** A diseased leaf receives a healthy suggestion, giving the grower false reassurance and potentially delaying closer inspection or expert review. This connects to the normal-input requirement that a healthy suggestion must not rule out disease and to the uncertain-result fallback.
- **Metric that would show this failure:** Diseased-to-healthy error rate = number of expert-labeled diseased input photos returned as healthy / total expert-labeled diseased input photos in the evaluation sample. Include all submitted diseased photos in the denominator and record uncertain/rejected outcomes separately. Report fallback coverage and errors among category outputs alongside this rate so rejecting every photo cannot count as successful useful behavior. Lab 02 overall accuracy does not directly measure this failure; later evaluation needs labels and per-image outputs to calculate it.
- **Preliminary acceptance threshold:** not set yet.
- **Where will we check it?** In a later evaluation on a new, labeled sample held separate from training and validation, initially within the narrowed controlled-photo scope. Report counts and uncertainty with the rate. Before any grower-use claim, repeat the check on permitted, expert-labeled grower phone photos, including unsupported conditions, and report those results separately. Do not tune on the existing Lab 02 held-out sample or the new evaluation sample.
- **Why this number for now?** There is no proposed number yet. The measured 62.5% baseline accuracy cannot establish a tolerable diseased-to-healthy error rate. We need grower/adviser input on the consequences of false reassurance and enough labeled examples to estimate that specific error before choosing a limit.
- **Evidence still missing:** A baseline confusion matrix or per-image predictions; a larger independent diseased sample across supported categories; expert-labeled examples of unsupported conditions; permitted realistic phone photos; grower/adviser feedback on acceptable misses and useful fallback frequency; and validation evidence for an uncertainty rule. Set the threshold before running the later acceptance evaluation. A future model-score cutoff would govern individual predictions; the acceptance threshold would judge the error rate across the evaluation sample.

### Failure cases

| Area | What could go wrong and why it matters | How would we notice it? | Planned response |
| --- | --- | --- | --- |
| Data or input | Blurry phone photo hides leaf symptoms; category may mislead | Labeled unusable-photo checks show a category instead of a rejection | Explain blur; request a clear, single-leaf color photo |
| Model behavior | Diseased leaf returned as healthy; grower may delay inspection | Diseased-to-healthy error rate in the acceptance check; inspect error counts and examples | Mark suggestions preliminary; route uncertain results to review; investigate healthy errors before acceptance |
| Integration or workflow | Upload or processing stalls; user receives no next step | End-to-end checks show submission with no result or error; record elapsed time | Show an actionable timeout/error and retry option; provide retake or expert-review guidance |

### Review before presentation

Trace the connected chain: usable diseased input → healthy suggestion or uncertain fallback → model-behavior failure case → diseased-to-healthy error rate → threshold **not set yet** → missing labeled predictions, independent examples, and grower/adviser evidence. Review this chain with a human teammate before presenting; no human teammate sign-off is recorded here.

Lab 03 records required behavior and future evaluation decisions only. It does not change the model or choose an architecture.
