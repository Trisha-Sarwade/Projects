# Projects

## Projects Overview

Welcome to the repository showcasing some of my key projects. This collection reflects my work across various domains including finance, machine learning, supply chain management, and blockchain technology. Each project demonstrates my skills in implementing advanced techniques to solve real-world problems and develop innovative solutions.

## Project Highlights

1. **[Portfolio Optimization | Optimization Methods in Finance (Course Project)](#portfolio-optimization--optimization-methods-in-finance-course-project)**
   - Optimized investment portfolios using linear and quadratic programming techniques.
   - Applied Markowitz and Sharpe Ratio models for efficient portfolio construction and risk management.
   - Performed dynamic portfolio rebalancing and risk estimation using Monte Carlo simulations.

2. **[Federated Learning for Late Payment Risk Prediction | M-Tech Term Project](#federated-learning-for-late-payment-risk-prediction--m-tech-term-project)**
   - Implemented a federated learning model to predict late payment risks in supply chain financing.
   - Integrated features such as Late Delivery Risk and Payment Amount to enhance risk assessment while ensuring data privacy.

3. **[Web-based Prediction of Supply Base](#web-based-prediction-of-supply-base--IIT-Kharagpur)**
   - Developed a UI/UX website to predict the supply base using data extracted from ProwessIQ.
   - Achieved high accuracy with XGBoost and Random Forest models, and analyzed the impact of supply chain complexity on financial metrics.

4. **[Restaurant Review Analysis on Yelp Data](#restaurant-review-analysis-on-yelp-data--IIM-Ahmedabad)**
   - Analyzed Yelp reviews to enhance review rating systems and understand customer sentiments.
   - Employed sentiment analysis and topic modeling techniques using advanced machine learning models.

5. **[Blockchain & IoT Integration for Cold Chain Monitoring](#blockchain--iot-integration-for-cold-chain-monitoring)**
   - Created a blockchain-integrated IoT system for monitoring cold chain temperatures in mango containers.
   - Implemented secure data logging with Hyperledger Fabric and developed a web application for real-time data visualization.

Each project in this repository represents a significant effort to apply cutting-edge technologies and methodologies to complex problems. I invite you to explore these projects to gain insights into the solutions and techniques used.

---

## [Portfolio Optimization | Optimization Methods in Finance (Course Project)](https://github.com/Trisha-Sarwade/Portfolio-Optimization)

### Overview
This project focuses on optimizing investment portfolios using linear and quadratic programming techniques applied to Nifty datasets. The key optimization models employed include the **Markowitz Model** and **Sharpe Ratio Model**. Additionally, portfolio rebalancing was performed using **integer and stochastic programming**, and **Monte Carlo simulations** were used to estimate **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)**.

### Project Details
- **Course**: Optimization Methods in Finance
- **Institution**: IIT Kharagpur
- **Duration**: August 2023 - November 2023

### Key Features
- **Portfolio Optimization**:
  - Used **Markowitz Model** for mean-variance optimization.
  - Applied **Sharpe Ratio** for risk-adjusted return analysis.
  - Implemented **linear and quadratic programming** for efficient portfolio construction.

- **Portfolio Rebalancing**:
  - Applied **integer programming** to optimize discrete asset allocations.
  - Employed **stochastic programming** to account for uncertainties in market returns.

- **Risk Estimation**:
  - Utilized **Monte Carlo methods** to simulate asset returns and estimate risk metrics.
  - Calculated **VaR** and **CVaR** for risk management.

### Tools and Technologies
- **Programming Languages**: Python, R
- **Libraries**: NumPy, Pandas, SciPy, PyPortfolioOpt, cvxpy, Matplotlib
- **Optimization**: Linear Programming, Quadratic Programming, Integer Programming
- **Risk Analysis**: Monte Carlo Simulation, VaR, CVaR

### Results
- Constructed efficient portfolios with optimal risk-return trade-offs.
- Rebalanced portfolios dynamically to maintain optimal allocation.
- Estimated risk metrics for various portfolios and demonstrated risk management techniques.

## [Federated Learning for Late Payment Risk Prediction | M-Tech Term Project](https://github.com/Trisha-Sarwade/Federated-Learning)

### Overview
This project aims to predict late payment risks in supply chain financing using **federated learning** techniques. The model leverages features such as **Late Delivery Risk**, **Payment Amount**, and others to improve risk assessment. The federated learning approach enables distributed model training across multiple clients without sharing sensitive data, addressing privacy concerns in supply chain operations.

### Project Details
- **Course**: M-Tech Term Project
- **Institution**: IIT Kharagpur
- **Duration**: August 2024 - November 2024

### Key Features
- **Federated Learning Implementation**:
  - Built a federated learning model to predict **buyer’s late payment risk** using distributed client data.
  - Integrated features such as **Late Delivery Risk**, **Payment Amount**, and other payment-related attributes.

- **Risk Assessment**:
  - Enhanced late payment risk prediction by incorporating advanced algorithms to analyze key risk factors.
  - Addressed challenges in **supply chain financing** by improving the detection of late payments.

- **Privacy-Preserving Machine Learning**:
  - Ensured data privacy by using a federated approach that trains models locally on each client’s data without sharing it.
  - Reduced data leakage risks while maintaining model performance.

### Tools and Technologies
- **Programming Languages**: Python
- **Libraries**: TensorFlow Federated, PyTorch, NumPy, Pandas, Scikit-learn
- **Federated Learning Framework**: TensorFlow Federated (TFF)
- **Machine Learning Models**: Logistic Regression, XGBoost, Neural Networks
- **Risk Prediction**: Late Payment Risk, Late Delivery Risk, Payment Amount

### Results
- Successfully built a federated learning model that predicts late payment risk with high accuracy.
- Demonstrated improved risk assessment in supply chain financing scenarios by integrating advanced features.
- Reduced data sharing risks while achieving comparable results to traditional centralized machine learning models.

## [Web-based Prediction of Supply Base | IIT Kharagpur](https://github.com/Trisha-Sarwade/Supply-Base-Prediction)

### Overview
This project focuses on predicting the supply base for companies using advanced machine learning techniques. The project involved data extraction from **ProwessIQ**, correlation computation, anomaly detection, and the development of a **UI/UX website** to predict the supply base with high accuracy. The project also analyzed the impact of **supply chain complexity** on financial metrics such as **ROA** and **Tobin's Q**.

### Project Details
- **Supervisor**: Prof. Sarada Prasad Sarmah
- **Institution**: IIT Kharagpur
- **Duration**: December 2022 - April 2023

### Key Features
- **Data Extraction and Processing**:
  - Extracted data for **780+ companies** from the **ProwessIQ** database.
  - Computed correlations between various features and identified important variables for supply base prediction.
  - Applied **Isolation Forest** for anomaly detection in the dataset.

- **Supply Base Prediction**:
  - Developed a **UI/UX website** that predicts the supply base with an accuracy of **98.95%**.
  - Utilized **XGBoost** and **Random Forest** models for prediction, using key features identified through the **Boruta feature selection** algorithm.

- **Supply Chain Complexity**:
  - Calculated supply chain complexity using **quintile functions** to categorize and assess companies' supply chains.
  - Analyzed the impact of supply chain complexity on financial metrics such as **Return on Assets (ROA)** and **Tobin's Q**.

### Tools and Technologies
- **Programming Languages**: Python, HTML, JavaScript, CSS
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, Random Forest, Boruta, Flask
- **Web Development**: Flask for the backend, HTML/CSS for UI/UX design
- **Machine Learning Models**: XGBoost, Random Forest, Isolation Forest (for anomaly detection)

### Results
- Achieved **98.95% accuracy** in predicting the supply base using machine learning models.
- Successfully implemented **supply chain complexity calculations** and demonstrated its correlation with key financial metrics.
- Developed a fully functional **UI/UX website** for real-time supply base predictions.

## [Restaurant Review Analysis on Yelp Data | IIM Ahmedabad](https://github.com/Trisha-Sarwade/Restaurant-Review-Analysis)

### Overview
This project focuses on analyzing restaurant reviews from Yelp to provide a robust review rating system and identify customer sentiments. The analysis involved **sentiment analysis**, **topic modeling** using **NMF** and **LDA**, and the development of machine learning models using **TF-IDF vectors** and **deep learning techniques** like **LSTM, GRU, CNN, DNN**, and **BERT**.

### Project Details
- **Supervisor**: Prof. Adrija Majumdar
- **Institution**: IIM Ahmedabad
- **Duration**: December 2021 - January 2022

### Key Features
- **Sentiment Analysis**:
  - Performed sentiment analysis on restaurant reviews to classify sentiments as positive, negative, or neutral.
  - Utilized **TextBlob** and **VADER** for basic sentiment analysis and **BERT** for more advanced sentiment understanding.

- **Topic Modeling**:
  - Applied **Non-Negative Matrix Factorization (NMF)** and **Latent Dirichlet Allocation (LDA)** for identifying topics within reviews.
  - Analyzed customer expectations and preferences based on identified topics.

- **Machine Learning Models**:
  - Developed models using **TF-IDF vectors** and applied deep learning techniques such as **LSTM**, **GRU**, **CNN**, and **DNN** to improve review rating systems.
  - Used **BERT** for advanced text representation and sentiment classification.

### Tools and Technologies
- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow, Keras, NLTK, SpaCy, Gensim, TextBlob
- **Machine Learning Models**: LSTM, GRU, CNN, DNN, BERT
- **Text Processing**: TF-IDF, NMF, LDA

### Results
- Enhanced the review rating system with improved accuracy and robustness.
- Successfully identified and analyzed customer sentiments and expectations.
- Developed advanced machine learning models that outperformed traditional sentiment analysis methods.

## [Blockchain & IoT Integration for Cold Chain Monitoring](https://github.com/Trisha-Sarwade/Blockchain-IoT-Integration)

### Overview
This project focuses on integrating blockchain technology with IoT systems to monitor and control cold chain temperatures for mango containers. The solution involved real-time temperature monitoring using **Raspberry Pi** and secure data logging with **Hyperledger Fabric**. A **web application** was developed to present the temperature data through interactive graphs and detailed tables.

### Project Details
- **Course**: Information Systems
- **Institution**: IIT Kharagpur
- **Duration**: January 2023 - March 2023

### Key Features
- **Blockchain Integration**:
  - Implemented **Hyperledger Fabric** for secure and tamper-proof data logging.
  - Used **Raspberry Pi** to relay temperature data in real-time via POST requests.

- **IoT System**:
  - Developed a real-time monitoring system for cold chain temperature control.
  - Enabled automated alerts and notifications for temperature deviations.

- **Web Application**:
  - Created a web application to visualize temperature data with **interactive graphs** and **detailed data tables**.
  - Implemented a user-friendly interface for monitoring and analysis.

### Tools and Technologies
- **Programming Languages**: Python, JavaScript, HTML, CSS
- **Libraries**: Flask, Hyperledger Fabric, Raspberry Pi libraries
- **Web Development**: HTML, CSS, JavaScript for frontend development
- **Blockchain**: Hyperledger Fabric for secure data logging

### Results
- Successfully implemented a blockchain-integrated IoT system for cold chain monitoring.
- Achieved real-time data monitoring and secure data logging with Hyperledger Fabric.
- Developed a comprehensive web application for data visualization and analysis.

---

