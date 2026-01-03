# 🔬 Technical Methodology

## 1. Data Processing
The raw dataset contained high missingness in `weight` (97%), `payer_code` (40%), and `medical_specialty` (47%). 
* **Decision:** `weight` was dropped. `medical_specialty` was imputed with a "Missing" category as it often indicates general admissions.
* **Feature Engineering:** The most complex task was mapping ICD-9 codes. Raw codes (e.g., `428`, `250.01`) were grouped into 9 categories using a custom mapping function to reduce dimensionality while preserving clinical meaning.

## 2. Model Selection
We compared Random Forest and LightGBM.
* **Why LightGBM?** It natively handles categorical splits better and trains faster.
* **Imbalance Handling:** The dataset has an 11% readmission rate. We used the `scale_pos_weight` hyperparameter (calculated as `count(neg)/count(pos)`) to penalize false negatives more heavily. This boosted Recall from ~15% to 60%.

## 3. Business Logic Layer
A pure probability score is insufficient for clinical workflows. We implemented a rule-based engine on top of the model:
* **High Risk (>60%):** Triggers "Social Worker Consult."
* **Polypharmacy Rule:** IF `medications > 20`, triggers "Pharmacist Review."
* **History Rule:** IF `number_inpatient > 1`, triggers "Chronic Care Management."