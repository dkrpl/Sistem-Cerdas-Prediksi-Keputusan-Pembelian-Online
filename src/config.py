"""
Configuration, Constants, and Research Metadata.
Based on CRISP-DM Methodology for Online Purchase Decision Prediction.

Title: Sistem Cerdas Prediksi Keputusan Pembelian Online dengan Explainable AI dan Actionable Counterfactual Explanations
Domain: E-Commerce / Consumer Behavior / Intelligent Decision Support System
Target: Revenue (True = Sesi Berakhir Pembelian / Purchase, False = Sesi Tidak Berakhir Pembelian / Non-Purchase)
"""

from pathlib import Path

# Random Seed for Complete Research Reproducibility
RANDOM_STATE = 42

# Google Drive Dataset Folder ID from User
GDRIVE_FOLDER_ID = "1YoffIdW61xTVfH1Yr0IcOz42cTcQNPAA"
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}?usp=drive_link"
DATASET_FILENAME = "online_shoppers_intention.csv"

# Target Definition (Outcome Transaksi Keputusan Pembelian, bukan niat psikologis)
TARGET_COLUMN = "Revenue"

# Feature Categorization (17 Predictors)
NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

ALL_PREDICTORS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Actionability and Constraint Definitions (Research Novelty Core)
# 1. Immutable / Non-Actionable Features (Locked during counterfactual generation)
IMMUTABLE_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "SpecialDay"
]

# 2. Candidate Actionable / Behavioral Features (Permitted to be changed)
ACTIONABLE_FEATURES = [
    "ProductRelated",
    "ProductRelated_Duration",
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration"
]

# 3. Derived / Analytics Features (Handled cautiously as model scenarios)
ANALYTICAL_FEATURES = [
    "BounceRates",
    "ExitRates",
    "PageValues"
]

# Feature Ranges and Physical Constraints for Plausibility
FEATURE_BOUNDS = {
    "Administrative": (0, 30),
    "Administrative_Duration": (0.0, 3500.0),
    "Informational": (0, 25),
    "Informational_Duration": (0.0, 3000.0),
    "ProductRelated": (0, 700),
    "ProductRelated_Duration": (0.0, 65000.0),
    "BounceRates": (0.0, 1.0),
    "ExitRates": (0.0, 1.0),
    "PageValues": (0.0, 400.0),
    "SpecialDay": (0.0, 1.0)
}

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Research Questions Reference (RQ1 - RQ5)
RESEARCH_QUESTIONS = {
    "RQ1": "Model machine learning mana yang memberikan performa paling sesuai untuk memprediksi keputusan pembelian online pada dataset yang digunakan?",
    "RQ2": "Faktor apa yang paling memengaruhi prediksi keputusan pembelian secara global dan individual berdasarkan SHAP?",
    "RQ3": "Apakah penerapan actionability constraints dapat menghasilkan counterfactual yang lebih feasible, plausible, sparse, dan actionable dibandingkan unconstrained counterfactual? (Novelty Utama)",
    "RQ4": "Bagaimana penghilangan PageValues memengaruhi performa prediksi, pola explanation SHAP, dan kualitas counterfactual?",
    "RQ5": "Bagaimana prediction, explanation, constrained counterfactual, dan counterfactual evaluation dapat diintegrasikan ke dalam sistem cerdas berbasis Streamlit?"
}

# Key Literature Citations (R1 - R11)
PRIMARY_REFERENCES = [
    {"code": "R1", "citation": "Sakar, C. O., et al. (2019). Real-time prediction of online shoppers purchasing intention using MLP and LSTM. Neural Computing and Applications, 31, 6893-6908."},
    {"code": "R2", "citation": "Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository."},
    {"code": "R3", "citation": "Bastos, J. A., & Bernardes, M. I. (2024). Understanding Online Purchases with Explainable Machine Learning. Information, 15(10), 587."},
    {"code": "R4", "citation": "Liu, J., & Liu, C. (2026). Prediction of Online Shoppers Purchasing Intention Based on SHAP-IGWO-EM Method. IJSSIR, 17(1), 1-28."},
    {"code": "R5", "citation": "Cui, N. (2026). Explaining and predicting consumer decision-making in E-commerce using an intelligent hybrid framework. Scientific Reports, 16, 17040."},
    {"code": "R6", "citation": "Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017."},
    {"code": "R7", "citation": "Wachter, S., et al. (2017/2018). Counterfactual Explanations without Opening the Black Box. Harvard JL & Tech."},
    {"code": "R8", "citation": "Mothilal, R. K., et al. (2020). Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations. ACM FAccT, 607-617."},
    {"code": "R9", "citation": "Ustun, B., et al. (2019). Actionable Recourse in Linear Classification. ACM FAccT, 10-19."},
    {"code": "R10", "citation": "Guidotti, R. (2024). Counterfactual explanations and how to find them: literature review and benchmarking. Data Mining & Knowledge Discovery, 38, 2770-2824."},
    {"code": "R11", "citation": "Rasouli, P., & Yu, I. C. (2024). CARE: coherent actionable recourse based on sound counterfactual explanations. Int. J. Data Sci. Anal., 17, 13-38."}
]
