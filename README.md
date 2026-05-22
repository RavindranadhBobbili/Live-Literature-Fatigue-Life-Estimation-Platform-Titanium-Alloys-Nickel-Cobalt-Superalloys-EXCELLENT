# Live-Literature-Fatigue-Life-Estimation-Platform-Titanium-Alloys-Nickel-Cobalt-Superalloys-EXCELLENT
A production-style AI/ML platform for fatigue-life prediction using literature-derived fatigue data, physics-informed feature engineering, live literature extraction, and deployable machine learning workflows.  

# Literature-to-Model Fatigue Life Intelligence Platform for Titanium Alloys and Superalloys

A production-style AI/ML platform for fatigue-life prediction using literature-derived fatigue data, physics-informed feature engineering, live literature extraction, and deployable machine learning workflows.

This platform integrates:

* Live literature mining from scientific databases
* Physics-informed fatigue modeling
* Titanium alloy and nickel-based superalloy fatigue prediction
* Machine learning model comparison and validation
* Interactive Flask dashboard deployment
* REST API endpoints
* CSV/PDF ingestion and dataset generation
* Prediction uncertainty estimation

---

## Key Features

### Live Literature Intelligence

* Queries scientific literature sources such as Europe PMC
* Extracts fatigue-related parameters from abstracts and uploaded PDFs
* Supports literature-to-dataset workflows

### Supported Materials

* Titanium alloys

  * Ti-6Al-4V
  * Ti64
  * Ti-6242
* Nickel-based superalloys

  * Inconel 718
  * GH4169
  * Waspaloy
  * Rene alloys

### Physics-Informed Fatigue Features

The platform incorporates engineering fatigue descriptors including:

* Stress amplitude
* Stress/UTS ratio
* Strain amplitude
* Temperature effects
* Frequency effects
* R-ratio
* Thermal interaction terms
* Logarithmic fatigue scaling

### Machine Learning Models

Includes multiple validated ML regressors:

* ExtraTrees
* RandomForest
* GradientBoosting
* HistGradientBoosting

Automatic model selection is performed using:

* R² score
* RMSE
* MAE
* Cross-validation metrics

### Dashboard Features

* Build literature datasets
* Upload PDF fatigue papers
* Upload CSV datasets
* Train/retrain ML models
* Predict fatigue life
* Download processed datasets
* Visualize model metrics

### REST API

Prediction endpoint:

```bash
POST /api/predict
```

Example JSON:

```json
{
  "alloy_family": "Titanium alloys",
  "alloy_name": "Ti-6Al-4V",
  "stress_mpa": 550,
  "uts_mpa": 950,
  "strain_amp": 0.006,
  "temperature_c": 25,
  "frequency_hz": 10,
  "r_ratio": -1
}
```

---

## Workflow

```text
Literature Search
        ↓
Fatigue Data Extraction
        ↓
Dataset Cleaning & Validation
        ↓
Physics Feature Engineering
        ↓
ML Model Training
        ↓
Cross-Validation & Metrics
        ↓
Fatigue Life Prediction
        ↓
Dashboard + REST API Deployment
```

---

## Applications

* Aerospace alloy fatigue assessment
* Turbine blade life estimation
* Titanium alloy durability prediction
* High-temperature superalloy design
* Materials informatics research
* AI-assisted fatigue engineering
* Literature-driven fatigue analytics

---

## Technologies Used

### Backend

* Python
* Flask
* SQLite
* REST API

### Machine Learning

* Scikit-learn
* RandomForest
* ExtraTrees
* Gradient Boosting

### Data Processing

* Pandas
* NumPy
* Regex-based literature extraction
* PDF parsing using pypdf

### Deployment

* Flask Web Dashboard
* REST API endpoints
* CSV export
* GitHub-ready structure

---

## Future Improvements

* Direct PDF table extraction using Camelot/Tabula
* SHAP explainability
* Deep learning fatigue models
* Transformer-based literature extraction
* Bayesian uncertainty estimation
* Automated retraining pipelines
* Cloud deployment (AWS/Azure/GCP)
* Docker/Kubernetes support

---

## Author

Ravindranadh B
Materials Informatics | AI/ML for Materials Engineering | Fatigue Modeling | Physics-Informed ML

---

## License

MIT License
