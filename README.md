# National Water Quality Intelligence Dashboard (1994–2024)
*A Unified Multi-Decadal Environmental Analytics and Decision-Support System for India*

This project consolidates fragmented water-quality records from CPCB, CGWB, and State Pollution Control Boards into a unified, standardized, and analytics-ready dataset spanning **30 years (1994–2024)**. The dashboard provides national-scale intelligence through **geospatial contamination hotspot mapping**, **seasonal and decadal trend analysis**, **regulatory compliance evaluation**, and **parameter interaction modeling**. A machine learning pipeline is also developed to classify water potability and enable predictive early-warning capabilities.

---

##  Key Features

| Area | Capabilities |
|------|-------------|
| **Data Integration** | Standardized multi-source datasets (CSV, PDF, Excel, Field Reports) into structured format |
| **Geospatial Analytics** | GIS-based hotspot identification & contamination pattern visualization |
| **Trend Analysis** | Seasonal cycles, monsoon dilution, decadal shifts in water chemistry indicators |
| **Compliance Monitoring** | Automated scoring against **BIS IS 10500** and **WHO** drinking water standards |
| **Correlation Modeling** | Interaction insights between pH, DO, Temperature, Conductivity & Heavy Metals |
| **Machine Learning (Potability Classification)** | Random Forest, XGBoost, LightGBM, SVM tested (~72% accuracy achieved) |
| **Future Forecasting Pipeline** | ARIMA & LSTM time-series predictive modeling planned |

---

##  Analytical Insights Generated

- **Industrial corridors** exhibit persistent heavy metal concentrations → hotspot clusters identified.
- **Monsoon cycle** significantly influences turbidity dilution and DO improvement patterns.
- **Agricultural runoff** correlates strongly with nitrate & hardness patterns in groundwater belts.
- **Temperature and DO show inverse interaction**, confirming thermodynamic oxygen solubility behavior.
- **Several states maintain regulatory compliance**, yet localized sub-district pockets show chronic exceedances.

---

##  System Architecture

**Core Tools Used:**
- **Power BI** (Visual Analytics, DAX, GIS Mapping)
- **Python** (Data Cleaning, Feature Engineering, ML Models)
- **Pandas, NumPy, Scikit-learn, XGBoost**
- **Matplotlib / Seaborn** (Visualization)


---

## 📊 Dashboard Modules

| Dashboard Page | Purpose |
|---|---|
| **National Overview** | High-level environmental and habitation status |
| **Contamination Hotspot Map (GIS)** | Geographic risk clusters & priority zones |
| **Seasonal & Decadal Trend Analysis** | Long-term water quality evolution |
| **Parameter Interaction Explorer** | Scientific and chemical behavior mapping |
| **Compliance Scorecard** | Regulatory adherence assessment |
| **Water Testing Infrastructure Map** | Lab coverage & capacity assessment |
| **Social & Population Context Layer** | Links environmental impact to human geography |

---
<img width="1169" height="656" alt="image" src="https://github.com/user-attachments/assets/bf79f639-945a-4c80-8825-bd187cdcb370" />
<img width="1162" height="656" alt="image" src="https://github.com/user-attachments/assets/0c00d230-c6bf-4451-82c8-53090774240b" />
<img width="1166" height="651" alt="image" src="https://github.com/user-attachments/assets/9955291c-03d5-4d03-8b3a-2e85835f3837" />
<img width="1168" height="656" alt="image" src="https://github.com/user-attachments/assets/568f30af-ede5-483f-8e97-a8736c2de09d" />
<img width="1137" height="632" alt="image" src="https://github.com/user-attachments/assets/3ea7cb7a-add2-4a4d-af79-f28cabd44374" />
<img width="1171" height="652" alt="image" src="https://github.com/user-attachments/assets/9d2ffd52-e0ac-41e3-b51b-bae7e15bfc49" />




---

##  Machine Learning Pipeline Overview

| Model | Purpose | Performance (Approx.) |
|------|---------|----------------------|
| Random Forest | Baseline potability classification | **~72% accuracy** |
| XGBoost | Enhanced feature-weighted model | Stable & interpretable |
| LightGBM | Fast boosting-based learner | Used for comparison |
| LSTM (Planned) | Temporal pattern forecasting | For long-term anomaly prediction |
| ARIMA (Planned) | Univariate seasonal forecasting | For monsoon-adjusted prediction |

**Top Influential Features Identified:**
- pH  
- Chloramines  
- Sulfate  
- Hardness  
- Conductivity  

---

##  Business, Research & Policy Value

| Stakeholder | Value Delivered |
|------------|----------------|
| Government & Jal Shakti | Prioritization of treatment & infrastructure funding |
| Environmental Scientists | Multi-decadal pattern discovery & scientific validation |
| Urban Planners & Water Boards | Early-warning contamination detection |
| NGOs & Public Health Institutions | Community water-risk awareness insights |

This project provides a **scalable foundation** for **real-time environmental monitoring, predictive pollution alerts, and sustainable water governance**.



---

##  Contributors
Project completed under the Department of Computer Science & Engineering (Data Science), Vishwakarma Institute of Technology, Pune.

---

##  Contact
If you'd like to discuss environmental analytics, GIS-based decision systems, or sustainability-focused ML:

**Email:** sumitrg0007@gmail.com
**LinkedIn:** https://www.linkedin.com/in/sumit-gavali-99bbb7337/

---

### ⭐ If you found this project meaningful, consider starring the repository!
