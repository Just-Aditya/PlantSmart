# PlantSmart - Intelligent Soil Monitoring System

## Team Members

**Name** | **Roll Number**
--- | ---
Aditya Kumar | 22051222
Affan Ahmed | 22051567
Atul Kumar | 22053933
Abhinandan Maji | 2205784

## Overview

PlantSmart is an advanced soil monitoring system designed to provide real-time insights into soil conditions, enabling farmers and agricultural experts to make informed decisions for optimal crop cultivation. By integrating sensor data with machine learning models, PlantSmart predicts the best crops to cultivate, determines optimal planting times, and recommends necessary soil treatments to maximize yield and minimize costs.

## Features

- **Real-time Soil Monitoring:** Continuously measures key soil parameters such as pH, nutrient content (NPK), humidity, temperature, and rainfall.
- **Intelligent Crop Prediction:** Uses machine learning algorithms to suggest the most suitable crops based on soil and weather conditions.
- **Data-Driven Insights:** Provides actionable recommendations on water requirements and necessary soil additives.
- **Weather Integration:** Incorporates external climate and rainfall data to refine predictions and improve accuracy.
- **User-Friendly Web Interface:** A planned web-based dashboard for visualizing data and receiving recommendations.
- **Hardware Integration:** Utilizes soil sensors to collect real-time pH, NPK, and humidity data, ensuring precise monitoring and prediction.

## Core System Framework

- **Backend:** Python (FastAPI)
- **Machine Learning:** Scikit-Learn, TensorFlow
- **Database:** MongoDB
- **Frontend:** HTML, CSS, JavaScript, Bootstrap
- **Hardware:** Soil sensors for pH, NPK, and humidity
- **Data Sources:** Indiastats and Kaggle datasets for crop prediction

## Hardware Integration

The PlantSmart system integrates hardware components to collect real-time soil data, which is then processed by the machine learning models. The hardware setup includes:

- **pH Sensor:** WKM pH Electrode Sensor Probe with BNC Connector
- **NPK Sensor:**
  - Modbus Module: RS485/MAX485
- **Microcontroller:** ESP32 for data collection and transmission.
- **Wireless Connectivity:** nRF24L01 PA+LNA.

## Installation & Setup

Currently, the project is in the early development stages and runs on a local environment. To set up the backend:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Just-Aditya/PlantSmart.git
   cd PlantSmart
   ```
2. **Set up a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Start the backend server:**
   ```bash
   python app.py
   ```

## Future Enhancements

- **Frontend Integration:** Develop a React-based web dashboard for real-time monitoring and recommendations.
- **Cloud Deployment:** Deploy the system on a cloud platform for remote accessibility.
- **Automated Irrigation System:** Implement smart irrigation control based on real-time soil moisture levels.
- **Multi-Region Support:** Expand the model to accommodate diverse soil and climate conditions.
- **Web Dev:** Develop a web-friendly interface for farmers.

## License

This project is proprietary and is not open-source. Unauthorized use, modification, or distribution is strictly prohibited without explicit permission from the author.

For inquiries, please contact the author.

© 2025 Aditya Kumar. All rights reserved.

