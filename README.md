# Wind Power Generation Forecasting
![Project Banner]
<img width="1863" height="730" alt="image" src="https://github.com/user-attachments/assets/4a1512f1-d77d-4659-96fb-293278390d08" />


## 🌟 Project Overview

Developed a robust **wind power forecasting system** using real-world meteorological and time-series data from a wind turbine SCADA system. The project focuses on accurate short-term power generation prediction to support renewable energy grid integration and climate risk management.

This project closely aligns with real industry challenges in **renewable energy forecasting**, demonstrating strong skills in time-series modeling, feature engineering, and model robustness — directly relevant to EARTHQUANT's mission of building AI-driven climate intelligence.

## 🎯 Key Highlights

- **Best Model Performance**: Random Forest achieved **R² = 0.9899**, MAE = **67.66 kW**
- **Strong Feature Engineering**: Physics-based features (Wind Speed³), cyclical time encoding, lags & rolling statistics
- **Proper Time-Series Validation**: Chronological train/test split (first 80% train, last 20% test)
- **Interactive Dashboard**: Built with Streamlit for real-time forecasting and what-if analysis
- **Domain Understanding**: Incorporated turbine physics and operational regions

## 🛠 Tech Stack

- **Core**: Python, Pandas, NumPy, Scikit-learn
- **Advanced ML**: XGBoost, Random Forest
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **Others**: Pickle (model persistence)

## 📊 Dataset

- **Source**: Wind Turbine SCADA Dataset (T1.csv)
- **Time Granularity**: 10-minute intervals for full year 2018 (~50,530 records)
- **Features**: Wind Speed, Wind Direction, Theoretical Power Curve, Active Power (target)
- **Real-world Characteristics**: Non-linear relationship, seasonal patterns, turbine cut-in/cut-out behavior

## 📋 Project Structure
WIND_POWER_FORECASTING/
│
├── data/
│   └── T1.csv
│
├── notebooks/
│   ├── .ipynb_checkpoints/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Modeling.ipynb
│
├── models/
│   └── random_forest.pkl
│
├── src/
│   └── (source code modules)
│
├── streamlit_app/
│   └── streamlit_app.py
│
├── utils/
│   └── (helper functions & utilities)




## 🔬 Methodology

### 1. Data Preprocessing
- Proper datetime parsing and indexing
- Handled negative power values
- Outlier analysis on wind speed

### 2. Feature Engineering (Key Strength)
- **Time Features**: Hour, dayofweek, month, weekend flag
- **Cyclical Encoding**: Sine/Cosine for hour and month
- **Physics-based**: WindSpeed², WindSpeed³, WindSpeed⁴
- **Wind Direction**: Sin/Cos components
- **Lag & Rolling**: Multiple lags + rolling mean/std (1h to 6h)
- **Turbine Metrics**: Power Coefficient and Efficiency

### 3. Modeling
- **Time-based Train/Test Split** (80/20 chronological)
- Models: Linear Regression (baseline), Random Forest, XGBoost
- Evaluation: MAE, RMSE, R² with proper time-series validation

### 4. Results

| Model               | MAE (kW) | RMSE (kW) | R² Score |
|---------------------|----------|-----------|----------|
| Linear Regression   | 111.37   | 160.40    | 0.9857   |
| **Random Forest**   | **67.66**    | **135.14**    | **0.9899**   |
| XGBoost             | 71.21    | 147.46    | 0.9879   |

## 🚀 Interactive Dashboard

Built a **Streamlit Dashboard** with:
- Exploratory Data Analysis
- Model Performance Comparison
- What-If Simulator (change wind speed and see predicted power)
- Real-time forecasting interface

**Run locally**: `streamlit run streamlit_app.py`

## 📈 Business & Environmental Impact

- Accurate forecasting helps optimize wind turbine operation and reduce energy curtailment
- Supports better grid integration of renewable energy
- Contributes to climate change mitigation through improved renewable energy utilization

## 💡 Learnings & Future Scope

- Importance of domain knowledge (wind physics) in feature engineering
- Value of proper time-series validation over random split
- Future work: Add LSTM/Transformer models, deploy on cloud, integrate weather API

## 🧑‍💻 How to Run

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order
4. Launch dashboard: `streamlit run streamlit_app.py`
