# 📖 User Documentation

## Dashboard Features

### 1. Patient Intake Form (Sidebar)
Nurses can input patient vitals including:
* **History:** Number of inpatient visits in the last year.
* **Current Stay:** Length of stay, lab procedures, medication count.
* **Demographics:** Age group, insulin status.

### 2. Risk Gauge
A real-time probability meter that changes color based on risk thresholds:
* 🟢 **Green (<30%):** Standard Discharge.
* 🟠 **Orange (30-60%):** Watchlist.
* 🔴 **Red (>60%):** Urgent Intervention.

### 3. Explainability (SHAP)
The "Why this score?" section displays a Force Plot.
* **Red Bars:** Factors pushing risk HIGHER.
* **Blue Bars:** Factors pushing risk LOWER.

### 4. Recommendation Engine
The right-hand panel dynamically updates to show specific clinical actions required based on the input data.