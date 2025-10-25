# Crop Yield Prediction App 🌾

A **Streamlit-based web application** that predicts crop yield based on various agricultural and environmental factors using a trained **LightGBM machine learning model**. This app aims to help farmers, agronomists, and researchers make data-driven decisions to optimize crop production.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Folder Structure](#folder-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

Crop yield depends on a variety of factors such as soil nutrients, environmental conditions, and pest incidence. This project leverages historical agricultural data to predict crop yields efficiently. Using a **LightGBM regression model**, the app provides accurate yield predictions given user inputs.

The app is deployed on **Streamlit Community Cloud**, allowing anyone to access and use it via a browser.

---

## Features

- Predict crop yield based on:
  - Soil nutrients (Nitrogen, Phosphorus, Potassium)
  - Fertilizer application (Urea, DAP, Potash)
  - Area under cultivation
  - Pest incidence score
  - Year and other relevant factors
- Interactive web interface for easy input
- Visual display of predicted crop yield
- Supports multiple crops (if dataset includes them)
- **(From App Code):** Visual display of model **Feature Importances** for transparency.

---

## Dataset

The app uses a curated agricultural dataset containing:

| Column Name | Description |
| :--- | :--- |
| `Year` | Year of cultivation |
| `Area_Hectares` | Area of cultivation in hectares |
| `Nitrogen_KgPerHa` | Nitrogen content per hectare |
| `Phosphorus_KgPerHa` | Phosphorus content per hectare |
| `Potassium_KgPerHa` | Potassium content per hectare |
| `Pest_Incidence_Score` | Pest incidence score (0–10) |
| `Urea_Applied_KgPerHa` | Urea fertilizer applied per hectare |
| `DAP_Applied_KgPerHa` | DAP fertilizer applied per hectare |
| `Potash_Applied_KgPerHa` | Potash fertilizer applied per hectare |
| `Crop_Yield` | Target variable (Yield in Kg/Ha) |

> The dataset should be placed locally and used to train the model if you want to retrain it.

---

## Model

- **Algorithm**: LightGBM Regressor
- **Purpose**: Predict crop yield based on multiple features
- **Artifacts Required**:
  - `lgbm_model.joblib` → Trained model
  - `label_encoders.joblib` → Encoders for categorical variables
  - `features_list.joblib` → List of features used in the model

The model is pre-trained and loaded when the app starts.

---

## Installation

Clone the repository:

```bash
git clone [https://github.com/Ketchupncurd/Crop_Yield_Predict.git](https://github.com/Ketchupncurd/Crop_Yield_Predict.git)
cd Crop_Yield_Predict
