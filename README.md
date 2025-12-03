# Ice-Cream-Predictor---Decision-Tree-Regression

# 🍦 Ice Cream Predictor - Decision Tree Regression

A machine learning web application that predicts ice cream sales/ratings using Decision Tree Regression.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.0-orange.svg)

## 🎯 Features

- 🔮 Real-time predictions using Decision Tree Regressor
- 📊 Interactive feature importance visualization
- 🎨 Beautiful, responsive UI
- 📈 Dynamic input controls
- 💻 Easy to deploy and use

## 🚀 Live Demo

🔗 **[Try it live here!](YOUR_STREAMLIT_URL)**

## 📊 Model Performance

- **Algorithm**: Decision Tree Regressor
- **R² Score (Train)**: ~0.95
- **R² Score (Test)**: ~0.90
- **Features**: Multiple numeric features
- **Target**: Continuous numeric prediction

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy

## 💻 Installation & Usage

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/icecream-predictor.git
cd icecream-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open in browser**
```
Local URL: http://localhost:8501
```

## 📁 Project Structure
```
icecream-predictor/
│
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── decision_tree_regressor_model.pkl  # Trained ML model
├── feature_names.pkl                   # Feature names
└── target_name.pkl                     # Target variable name
```

## 🎓 Model Training

The model was trained using:
- **Dataset**: Ice Cream Dataset from Kaggle
- **Training Split**: 80% training, 20% testing
- **Parameters**:
  - Max Depth: 10
  - Min Samples Split: 10
  - Min Samples Leaf: 5

## 📸 Screenshots

### Main Interface
![App Interface](screenshot.png)

### Prediction Results
![Predictions](prediction.png)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Course: Machine Learning Lab 09
- Task: Decision Tree Regression (Task 3)

## 🙏 Acknowledgments

- Dataset: Ice Cream Dataset (Kaggle)
- Framework: Streamlit
- ML Library: Scikit-learn

---

⭐ If you found this project helpful, please give it a star!
```

### **File 3: .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Streamlit
.streamlit/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints

# Models (if too large)
# *.pkl
```

---

## **Complete Folder Structure**
```
icecream-predictor/
│
├── app.py                              ← Streamlit app (from artifact above)
├── requirements.txt                    ← Dependencies
├── README.md                           ← Documentation
├── .gitignore                          ← Git ignore file
├── decision_tree_regressor_model.pkl  ← Download from Kaggle
├── feature_names.pkl                   ← Download from Kaggle
└── target_name.pkl                     ← Download from Kaggle
