# Pothole Classification Project Summary

## 1. Project Goal Refinement

The project did not remain a generic road-damage classification task for long. After reviewing the data, it became clear that the label **"pothole"** was not defined consistently across datasets. As a result, the working objective gradually became more precise:

- train a classifier that reliably detects **clear, visually obvious potholes**
- avoid over-predicting potholes on confusing but non-pothole structures such as:
  - severe cracks
  - alligator cracks
  - manholes
  - roadside holes or water-edge pits

In practice, this became a **strict pothole recognition** problem rather than a broad road-damage recognition problem.

## 2. Initial Dataset Setup and the First Structural Problem

The first dataset was an RDD-style road-damage detection dataset in YOLO format with four classes:

- longitudinal
- transverse
- alligator
- pothole

The original idea was to convert it into a CNN-style binary classification dataset:

- pothole
- non_pothole

The first major issue appeared during conversion: the original test split could not be used as a supervised binary test set. After inspection, this was not because the test split contained no potholes, but because the original detection benchmark **did not publish test labels**. It was a prediction-only split.

### Fix

Only the labeled train and validation data were used, and a new stratified binary split was created:

- train
- val
- test

This was the first step toward making the task experimentally valid.

## 3. Label Semantics and Boundary Ambiguity

Once the four classes were visually inspected, the class definitions turned out to be uneven:

- **longitudinal** and **transverse** were relatively clean and visually consistent
- **alligator** was highly mixed
- **pothole** itself was not always pure; some samples looked more like severe surface failure than textbook potholes

The main ambiguity was concentrated in **alligator**:

- some samples were shallow, web-like cracks
- some were severe fractured regions that looked visually close to potholes

This created a boundary-definition problem:

- treating alligator as an ordinary negative class would contaminate the negative set
- treating alligator as a positive class would contaminate the positive set

### Fix

Alligator was removed from the main training negatives and placed into a separate challenge set. The main negative class was restricted to the cleaner classes:

- longitudinal
- transverse

This helped stabilize the core decision boundary before any harder negatives were introduced.

## 4. External Dataset Integration and Domain Shift

To strengthen the positive class, a Sovit pothole dataset was added. Its structure was straightforward:

- `Train data / Positive data`
- `Train data / Negative data`

### First mistake

At first, only **Sovit positive** was added to `train/pothole`, while Sovit negative was not added to training.

This caused a strong failure mode:

- the model predicted almost all `external_sovit_negative` images as pothole
- increasing the threshold did not fix the problem
- average pothole probability was close to 1

### Interpretation

This was not a threshold-selection problem. It was a classic **domain shortcut**:

> the model learned that "images with the Sovit visual style" were correlated with pothole.

### Fix

The Sovit dataset was then integrated properly:

- Sovit positive -> added to `train/pothole`
- Sovit negative -> split into:
  - training negatives
  - external negative holdout

After this change, false positives on Sovit negative dropped sharply, confirming that the earlier issue was domain leakage rather than simple overconfidence.

## 5. Alligator Challenge Set and Hard Negative Mining

Once a more stable v2 model was obtained, performance was measured separately on:

- `challenge_alligator`
- `exclusive_alligator`

`exclusive_alligator` was defined as alligator samples that did **not** co-occur with pothole labels. This gave a cleaner way to test whether the model was confusing severe cracking with potholes.

### Observation

The model still produced a high false-positive rate on exclusive alligator, suggesting that it was confusing potholes with:

- severe cracks
- severe alligator patterns
- deep cross-cracks
- some manhole-like structures

### Manual review

Images predicted as pothole from the exclusive alligator set were exported and reviewed manually. Most were not true potholes. They were visually confusing but semantically different.

### Experiment: hard negative mining

A subset of those high-confidence alligator false positives was added to `train/non_pothole`, and the model was fine-tuned again.

### Result

This reduced alligator false positives, but it introduced a serious tradeoff:

- the model became more conservative
- recall on true potholes dropped
- miss rate increased substantially on an external positive-only pothole dataset

### Conclusion

Hard negative mining was useful, but the boundary tightened too aggressively when too many alligator hard negatives were added.

## 6. External Positive-Only Testing

An additional pothole segmentation dataset was later used as an external positive-only test set.

After verification:

- total images: 780
- all 780 had non-empty labels
- therefore the set could be treated as an **all-pothole external recall benchmark**

This dataset was important because it measured whether the model still recognized true potholes under a new visual domain.

### Key result

The version trained with heavy alligator hard negatives became overly conservative and missed many obvious potholes. The earlier v2 model without that aggressive hard-negative step performed better.

### Conclusion

This showed that reducing false positives on hard negatives could damage generalization on real potholes if done too aggressively.

## 7. Threshold Tuning as a Multi-Dataset Decision

The final decision was not based on raw argmax output. Threshold sweeps were performed across multiple evaluation sets:

- RDD internal test precision / recall / F1
- exclusive alligator false-positive rate
- external Sovit negative false-positive rate
- external positive-only pothole recall

This made the tradeoff explicit:

- higher thresholds reduced false positives
- but higher thresholds also reduced pothole recall
- the best threshold depended on which failure mode mattered most

Threshold selection therefore became part of model design, not a last-minute adjustment.

## 8. Key Conclusions Before the Latest Round of Work

Several conclusions were already clear before the latest experiments:

1. External positives must not be added alone. If positive samples are added without matching negatives from the same domain, the model learns source shortcuts.
2. Alligator should not be treated as an ordinary negative by default. Its visual boundary with pothole is too ambiguous.
3. Hard negative mining is helpful, but only when used carefully and in limited quantity.
4. A single internal test split is not enough. Proper evaluation needs:
   - internal test
   - external negative set
   - external positive set
   - challenge set
5. The core problem is a boundary-definition problem:
   - what counts as pothole
   - what does not
   - what counts as severe but non-pothole road damage

## 9. Main Failure Categories Encountered

The main problems encountered can be grouped into four categories:

### Data structure issues

- original detection test split had no labels
- folder structures differed across datasets and had to be adapted

### Label semantics issues

- pothole class was not fully pure
- alligator was visually ambiguous
- different datasets used different pothole definitions

### Domain shift issues

- adding external positive samples without external negative samples created source shortcuts

### Training boundary issues

- too many hard negatives reduced recall dramatically

### Evaluation issues

- internal test alone underestimated crack/alligator confusion and external pothole miss behavior

## 10. Improvements Completed Before the Current Round

Before the latest collaboration round, the following major improvements had already been completed:

1. converted the YOLO road-damage dataset into a binary CNN dataset
2. rebuilt a valid stratified train/val/test split
3. removed alligator from the main training negative pool
4. integrated external Sovit data
5. identified the external-positive-only shortcut problem
6. added Sovit negative correctly into training and retained an external holdout
7. created an exclusive alligator challenge set
8. ran a hard negative mining experiment
9. evaluated on an external positive-only pothole dataset
10. performed threshold sweeps across multiple datasets

## 11. Additional Work Completed in the Latest Round

After that earlier work, the latest round focused on testing whether the remaining high false-negative rate was caused by noisy labels or by domain shift.

### 11.1 Class-weight experiment

A controlled test was run by enabling class weights while keeping the rest of the setup unchanged.

Result:

- only very small recall gains on the internal RDD test
- precision dropped
- external false-positive behavior worsened
- external pothole miss rate barely changed

Interpretation:

> class imbalance was not the main problem.

This strongly suggested that the remaining external miss behavior was driven by **domain shift**, not mainly by label noise.

### 11.2 YOLOv8-domain adaptation experiment

A new v3 model was trained by adding positive pothole samples from the YOLOv8 pothole dataset to the training set.

At first, a random split inside the same YOLOv8 dataset produced extremely low miss rates on a holdout subset, but that was not considered conclusive because the holdout was still too close to the training distribution.

### 11.3 Independent external validation on the Neha dataset

To obtain a cleaner answer, the model was tested on the independent **Normal-Pothole-dataset** from Zenodo (balanced pothole vs normal road).

This became the most important external evaluation in the latest round.

## 12. Best Comparative Result So Far

At threshold `0.50`, the comparison on the independent Neha external test was:

### v2 (before YOLOv8 domain adaptation)

- accuracy: **0.8062**
- pothole precision: **0.9293**
- pothole recall: **0.6628**
- pothole F1: **0.7738**
- miss rate (FN): **0.3372**
- false-positive rate (FP): **0.0504**

### v3 (with YOLOv8 positive samples added during training)

- accuracy: **0.8824**
- pothole precision: **0.8649**
- pothole recall: **0.9064**
- pothole F1: **0.8852**
- miss rate (FN): **0.0936**
- false-positive rate (FP): **0.1416**

## 13. Interpretation of the Best Result

This was a decisive result.

The v3 model reduced external pothole miss rate dramatically on a fully independent binary pothole dataset:

- miss rate dropped from **33.7%** to **9.4%**
- recall increased from **66.3%** to **90.6%**
- overall accuracy increased from **80.6%** to **88.2%**

This strongly supports the following interpretation:

> the main reason for the earlier high false-negative rate was **domain shift**, not simply noisy labels in the original training data.

The tradeoff is also clear:

- v3 is much better at detecting real potholes
- but it is more willing to predict pothole, so false positives rise on some normal-road images

In other words, the project has moved from an under-sensitive model toward a much more recall-capable model, with a manageable precision tradeoff.

## 14. Current Best Model Positioning

The current best model is the **v3 classifier**:

- ResNet18 backbone
- strict binary pothole classification
- trained on:
  - cleaned RDD-derived pothole vs non_pothole split
  - Sovit positive and negative samples
  - additional YOLOv8 pothole positives
- evaluated on:
  - internal RDD test
  - external Sovit negative
  - exclusive alligator challenge
  - external positive-only pothole set
  - independent Neha pothole/normal dataset

This model is currently the most deployment-ready version for the next pipeline stage.

## 15. Deployment Status

The model has been prepared for downstream pipeline integration through:

- `weights.pth`
- `config.json`
- optional `model_v3_traced.pt`
- reusable `pothole_classifier.py`

This means the project is no longer only in an experimentation phase; it now has a reproducible inference package for downstream use.

## 16. Team-Level Summary

In one sentence:

> the project evolved from a simple pothole classifier into a carefully defined strict-pothole recognition system, where the main challenge was not only class imbalance or noise, but the combination of semantic ambiguity, domain shift, and boundary management between potholes and severe non-pothole road damage.

The most important strategic lesson is:

> better pothole recognition did not come from a single trick, but from combining cleaner task definition, challenge-set design, external-domain testing, and carefully targeted data integration.

## 17. Recommended Next Steps

1. keep v3 as the current deployment candidate
2. review high-confidence false positives on the Neha normal set to separate true model errors from label-definition mismatch
3. add one more fully independent external test dataset if a stronger generalization claim is needed
4. keep hard negative mining selective and human-reviewed rather than large-scale and automatic
5. document the task definition clearly for the team:
   - what is counted as pothole
   - what is excluded
   - what should be considered ambiguous
