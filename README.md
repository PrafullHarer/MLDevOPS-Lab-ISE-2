# 🌸 Iris Flower Classifier — ML DevOps Lab ISE 2

A machine learning web application that classifies Iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal measurements. Built with **Flask** and **scikit-learn**, containerized with **Docker**, and automated with **Jenkins CI/CD**.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.11** | Programming Language |
| **Flask** | Web Application Framework |
| **scikit-learn** | Machine Learning Library |
| **Git** | Version Control |
| **GitHub** | Source Code Hosting |
| **Docker** | Application Containerization |
| **Jenkins** | CI/CD Pipeline Automation |

## 📂 Project Structure

```
MLDevOPS-Lab-ISE-2/
├── app.py              # Flask web application
├── train_model.py      # ML model training script
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker containerization
├── Jenkinsfile         # Jenkins CI/CD pipeline
├── .gitignore          # Git ignored files
├── .dockerignore       # Docker ignored files
├── templates/
│   ├── index.html      # Home page (prediction form)
│   └── result.html     # Prediction results page
├── tests/
│   └── test_app.py     # Unit tests
└── model/
    └── iris_model.pkl  # Trained model (generated)
```

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/PrafullHarer/MLDevOPS-Lab-ISE-2.git
cd MLDevOPS-Lab-ISE-2
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run the Application
```bash
python app.py
```
Visit **http://localhost:5000** in your browser.

### 4. Run Tests
```bash
python -m pytest tests/ -v
```

### 5. Docker Deployment
```bash
docker build -t iris-classifier .
docker run -p 5000:5000 iris-classifier
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI — Prediction form |
| `/predict` | POST | Classify an iris flower |
| `/health` | GET | Health check for monitoring |

### Example API Call
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## 📜 License

This project is for educational purposes — ISE 2 Lab, A.Y. 2026-27.
