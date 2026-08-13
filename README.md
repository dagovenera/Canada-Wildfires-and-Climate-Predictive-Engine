# Canada-Wildfires-and-Climate-Predictive-Engine
A fully deployed XGBoost Machine Learning Engine that predicts Wildfires in Canada, using environmental data

# 🔥 Canadian Wildfire & Climate Predictive Engine

[![Streamlit App](https://streamlit.io)](https://canada-wildfires-and-climate-predictive-engine.streamlit.app/)
[![Python](https://shields.io)](https://python.org)
[![XGBoost](https://shields.io)](https://readthedocs.io)

An end-to-end data engineering and predictive machine learning architecture designed to forecast localized, compounding wildfire risks across high-density Canadian forest structures. The system ingests, cleans, and merges multi-layered geospatial and meteorological data streams to expose a real-time risk inference dashboard.

---

## 🏗️ System Architecture & Data Pipeline

The project handles extreme class imbalance and spatial-temporal alignment across two primary federal tracking networks over a 7-year historical horizon (2017–2023):

1. **Data Ingestion & Engineering (`/pipeline/stack_weather_data.py`):** Automatically extracts a comprehensive 7-year historical weather matrix directly from Environment and Climate Change Canada (ECCC) station records, previously downloaded as yearly CSV files. Custom modules select and stack dataframes from target year range and parse text lines to extract key variables/features and geographic coordinates onto every entry vector.
2. **Geospatial & Temporal Join Engine (`/pipeline/build_mvp_pipeline.py`):** Ingests point-source wildfire shapefiles, previously downloaded from the Canadian National Fire Database (CNFDB). Coordinates are programmatically reprojected via GeoPandas to a standardized `EPSG:4326` coordinate system. The script computes a vector proximity distance matrix, matching weather stations to active ignitions within a strict 50km radius.
3. **Feature Engineering & ML Training (`/pipeline/train_mvp_model.py`):** Employs **Inferential Statistics** trained a XGBClassifier to predict the risk of wildfire occurrence from average temperatures, wind gust speed, precipitation amount, and two engineered features that capture the compounding drying effects that primes a forest for wildfires: 7-day rolling temperature averages and 7-day cumulative precipitation.
4. **Interactive Inference Deployment (`/pipeline/app.py`):** A fully responsive web interface built with Streamlit, allowing stakeholders to dynamically manipulate weather variables via slider to compute real-time risk probabilities.

---

## 🛠️ Technical Toolkit

- **Languages:** Python (Pandas, GeoPandas, NumPy, Scikit-Learn)
- **Machine Learning:** XGBoost (Gradient Boosting Classifier), Stratified Cross-Validation, Imbalance Scaling, Precision-Recall Optimization
- **Geospatial Analysis:** Shapely (Vector Points/Polygons), Coordinate Reference System (CRS) Reprojection
- **Deployment:** Streamlit Community Cloud, Git/GitHub Version Control

---

## 🧪 Robust Testing & Simulation Framework

To guarantee continuous development velocity, this project features a dual-mode engineering architecture that allowed to create all the pipeline while data were being obtained from Canada government open portals.

A native statistical simulation engine (`/tests/mock_weather_data.py` and `/tests/mock_fire_data.py`) was custom-built using Numpy and GeoPandas to generate synthetic weather matrices and spatial vectors. This decoupled testing framework successfully unblocked local pipeline construction and model threshold calibration before scaling up to real historical data streams.

---

## 📊 Machine Learning Optimization

Wildfire occurrences represent an extreme minority class relative to baseline tracking days. To protect the model from default classification bias:
- Implemented an **XGBoost Classifier** configured with dynamic `scale_pos_weight` ratios to penalize minority misclasssifications natively.
- Replaced chronological data partitioning with a **Stratified Train-Test Split**, ensuring equal density distribution of sparse fire vectors across both training and validation layers.
- Model performance is driven by **Recall** and **ROC-AUC Score (0.80+)**, maximizing the system's capacity to intercept high-risk ignition anomalies.

---

## 🚀 Local Installation & Execution

### 1. Clone the Architecture and Initialize the Virtual Environment
```bash
git clone https://github.com
cd Canada-Wildfires-and-Climate-Predictive-Engine
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Execute the Test Simulation Framework (Optional)
```bash
python tests/mock_weather_data.py
python tests/mock_fire_data.py
```

### 3. Run the Production Pipeline & Launch the App
```bash
python pipeline/download_weather.py
python pipeline/build_mvp_pipeline.py
python pipeline/train_mvp_model.py
streamlit run pipeline/app.py
```

---

## 👨‍💻 Developer Profile

**Dagoberto E. Venera-Ponton, PhD**  
*Bioinformatics, Ecology, Advanced Statistical Pipelines & Machine Learning*  
- **LinkedIn:** [https://linkedin.com/in/dagoberto-venera-ponton-phd]
- **Email:** [dagovenera@gmail.com]