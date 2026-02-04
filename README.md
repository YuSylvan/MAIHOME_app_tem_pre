# MAI Home: Smart Heating Digital Twin 🌡️

This repository contains the **Digital Twin** engine for the MAI Home project, focusing on high-resolution indoor temperature forecasting. The system utilizes real-time sensor data to predict thermal dynamics across multiple rooms.

---

## 🚀 Key Features

### 🟢 Module 1: Intelligent Data Pre-processing
* **Multi-Room Synchronization:** Aligns asynchronous data from various sensors into a unified 10-minute "heartbeat" using linear interpolation.
* **Feature Filtering:** Automatically identifies and extracts `Temperature`, `Setpoints`, and `PIR (Occupancy)` data while excluding irrelevant noise (e.g., watermeter data).
* **Cyclical Time Encoding:** Uses Sine/Cosine transformations for Hours and Days to help the AI perceive temporal continuity.

### 🔵 Module 2: AI Forecasting Engine
* **Architecture:** Multi-Output Multi-Layer Perceptron (MLP).
* **High Resolution:** Provides a **3-hour forecast** with **10-minute granularity** (18 prediction points per room).
* **Cross-Room Intelligence:** The model learns the thermal interdependency between different living spaces for higher accuracy.

---

## 📂 Project Structure

```text
MAIHOME_app_tem_pre/
├── src/
│   ├── api_call.py        # API communication & data extraction
│   └── temp_pre.py        # DigitalTwinModel & processing logic
├── data/                  # Local database and raw samples (ignored by git)
├── main_demo.ipynb        # End-to-end demonstration notebook
├── requirements.txt       # Project dependencies
└── .gitignore             # Prevents bloat by ignoring .db and .pth files
