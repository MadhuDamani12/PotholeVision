
raw
Pothole readme · MD
# PotholeVision: Automated Pothole Detection & Geo-Mapping
 
## Overview
 
Road inspection is a slow, expensive, and largely manual process for city governments. Municipalities struggle to identify road damage at scale and often lack a systematic way to prioritize where repairs are most urgently needed.
 
**PotholeVision** is an end-to-end big data pipeline that automates pothole detection from video and image data, and surfaces findings through an interactive geospatial dashboard — enabling Public Works planners to make faster, data-driven repair decisions.
 
---
 
## The Problem
 
- Manual road inspections are resource-intensive and inconsistent
- Cities have no scalable way to monitor road damage across large networks
- Without location and traffic context, repair prioritization is largely guesswork
---
 
## Solution & Pipeline
 
We built a production-style big data pipeline with three stages:
 
**Ingest → Classify → Visualize**
 
1. **Bronze layer** — Raw video and image data ingested in batches; frames extracted
2. **Silver layer** — Frames evaluated using a fine-tuned ResNet18 CNN for pothole classification
3. **Gold layer** — Positive detections enriched with GPS coordinates and pushed to an ArcGIS dashboard
The dashboard gives city planners a real-time view of pothole locations, counts, and street traffic volumes — so the highest-impact repairs get prioritized first.
 
---
 
## Model Performance
 
| Version | Key Change | Recall | Miss Rate |
|---------|-----------|--------|-----------|
| V1 | Baseline ResNet18 | — | — |
| V2 | Hard negative mining + domain adaptation | — | — |
| **V3** | Strict boundary + external benchmark | **90.6%** | **9.4%** |
 
The final model was evaluated on an independent external benchmark (Neha dataset), confirming its generalizability beyond training data.
 
**Key insight:** High false-negative rates in early versions were driven by domain shift — visual style differences across datasets — not label noise. Removing ambiguous classes (e.g. alligator cracks) from the negative training pool significantly stabilized the decision boundary.
 
---
 
## Tech Stack
 
- **Modeling:** Python, PyTorch, ResNet18 (CNN)
- **Pipeline:** Databricks (Bronze/Silver/Gold architecture)
- **Visualization:** ArcGIS Dashboard
- **Data:** RDD 2020, Sovit Pothole Dataset, YOLOv8 Pothole Segmentation, Neha Dataset
---
 
## Repo Structure
 
```
├── notebooks/
│   ├── Pothole_CNN.ipynb               # Baseline training experiments
│   ├── Pothole_version2.ipynb          # Fine-tuning & domain adaptation
│   └── pothole_pipeline_final.ipynb    # Full Databricks inference pipeline
├── model_artifacts/                    # Trained weights & config
├── src/
│   └── pothole_classifier.py           # Reusable inference wrapper
├── reports/
│   └── team_pothole_summary_report.pdf # Full project report
└── requirements.txt
```
 
---
 
## Quick Start
 
```python
from src.pothole_classifier import PotholeClassifier
 
clf = PotholeClassifier("./model_artifacts")
result = clf.predict("path/to/image.jpg")
print(result)
```
 
---
 
## Business Impact
 
- Reduces reliance on manual inspections
- Enables data-driven repair prioritization based on pothole density and traffic volume
- Scalable to any city with dashcam or drone footage infrastructure
- Directly reduces vehicle accidents and maintenance costs over time
---
 
## Team
 
Chunfang Wang, James Pashek, Joseph Sheehan, Moses Akoto, Madhu Damani, Tao Fang
 
*Developed as part of the Big Data Analytics course — MS Business Analytics, Carlson School of Management, University of Minnesota.*
 
