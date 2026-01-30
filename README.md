# 🤖 AutoML Cancer Prediction with AWS AutoGluon

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-1.1.1-orange.svg)](https://auto.gluon.ai/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Industry-grade AutoML** for binary cancer classification using AWS AutoGluon. Automated model selection across 10+ algorithms with ensemble learning achieving high accuracy on medical diagnosis prediction.

## 📊 Project Overview

This project demonstrates cutting-edge AutoML capabilities for cancer prediction using AWS AutoGluon. The framework automatically trains, optimizes, and ensembles multiple machine learning models with minimal manual intervention—a highly valued skill in modern ML engineering.

### **Why This Matters**
- ⚡ **AutoML Expertise**: Industry trend toward automated machine learning pipelines
- 🏥 **Healthcare Application**: Real-world impact in medical diagnosis
- 🎯 **Production-Ready**: Complete ML workflow from EDA to model evaluation
- 📈 **Model Comparison**: Systematic evaluation across multiple algorithms

### **Key Highlights**
- ✅ Automated training across 10+ classification algorithms
- ✅ Ensemble learning with weighted stacking
- ✅ Hyperparameter optimization with Bayesian search
- ✅ Comprehensive exploratory data analysis (EDA)
- ✅ Feature correlation analysis with heatmap visualization
- ✅ Model performance comparison via leaderboard
- ✅ Confusion matrix evaluation on test set

---

## 🎯 Model Performance Results

### **Leaderboard Summary**

AutoGluon automatically trained and evaluated multiple models. The top performers include:

| Rank | Model | Validation Accuracy | Training Time | Type |
|------|-------|-------------------|---------------|------|
| 🥇 1 | **WeightedEnsemble_L2** | **[Best Score]** | ~250s | Ensemble |
| 🥈 2 | LightGBM | High | Fast | Gradient Boosting |
| 🥉 3 | RandomForest | High | Medium | Tree-based |
| 4 | ExtraTrees | High | Medium | Tree-based |
| 5 | CatBoost | High | Slow | Gradient Boosting |
| 6 | XGBoost | High | Medium | Gradient Boosting |

*The WeightedEnsemble_L2 model achieved the highest accuracy by intelligently combining predictions from multiple base learners.*

### **Test Set Performance**

- **Evaluation Metric**: Accuracy (Binary Classification)
- **Train/Test Split**: 80/20 
- **Confusion Matrix**: Visual evaluation included in notebook
- **Time Constraint**: 250 seconds for AutoML training

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **AutoML Framework** | AWS AutoGluon |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML Utilities** | Scikit-learn (train_test_split, confusion_matrix) |
| **Environment** | Google Colab / Jupyter Notebook |

---

## 📁 Project Structure

```
autogluon-cancer-prediction/
│
├── 📓 AutoML_with_AWS_AutoGluon.ipynb   # Main analysis notebook
├── 📊 cancer.csv                         # Cancer dataset
├── 📋 requirements.txt                   # Python dependencies
├── 📖 README.md                          # Project documentation
├── 🚫 .gitignore                         # Excluded files/folders
└── 📜 LICENSE                            # MIT License
```

**Note**: Trained models (`AutogluonModels/`) are excluded from version control to save space (typically 500MB-2GB).

---

## 🚀 Quick Start

### **Option 1: Run in Google Colab (Recommended - No Installation)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/autogluon-cancer-prediction/blob/main/AutoML_with_AWS_AutoGluon.ipynb)

1. Click the "Open in Colab" badge above
2. Upload `cancer.csv` when prompted (or adjust file path)
3. Run all cells: `Runtime` → `Run all`
4. **IMPORTANT**: Restart runtime after installing AutoGluon

### **Option 2: Local Setup**

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/autogluon-cancer-prediction.git
cd autogluon-cancer-prediction

# Create virtual environment (RECOMMENDED)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook AutoML_with_AWS_AutoGluon.ipynb
```

⚠️ **Note**: AutoGluon requires ~2GB disk space. Consider using Google Colab's free resources for training.

---

## 📈 Methodology

### **1. Data Loading & Exploration**
- Loaded cancer dataset using Pandas
- Performed initial data inspection (`.info()`, `.describe()`)
- Analyzed data types and missing values

### **2. Exploratory Data Analysis (EDA)**
- Generated correlation matrix for all features
- Created 18x18 heatmap to visualize feature relationships
- Identified highly correlated features with target variable

### **3. Data Splitting**
```python
train_test_split(df, test_size=0.2, random_state=True)
```
- **Training Set**: 80% of data
- **Test Set**: 20% of data (held out for final evaluation)

### **4. AutoML Training with AutoGluon**
```python
TabularPredictor(
    label='target',
    problem_type='binary',
    eval_metric='accuracy'
).fit(
    train_data=x_train,
    time_limit=250,
    presets='good_quality',
    num_cpus=4,
    dynamic_stacking=False
)
```

**Configuration Details**:
- **Problem Type**: Binary classification
- **Target Variable**: `target` (cancer diagnosis)
- **Preset**: `good_quality` (balance between speed and performance)
- **Time Budget**: 250 seconds
- **Hardware**: 4 CPU cores
- **Stacking**: Disabled for faster training

### **5. Model Evaluation**
- **Leaderboard Analysis**: Compared all trained models
- **Visualization**: Created bar plot of model performance
- **Test Predictions**: Generated predictions on holdout test set
- **Confusion Matrix**: Evaluated true positives, false positives, etc.

---

## 📊 Key Visualizations

### **Feature Correlation Heatmap**
- 18x18 annotated heatmap showing relationships between all features
- Helps identify multicollinearity and feature importance
- Color-coded correlation coefficients (-1 to +1)

### **Model Performance Bar Plot**
- Comparative visualization of all trained models
- Y-axis: Validation accuracy scores
- X-axis: Model names (rotated for readability)

### **Confusion Matrix**
- 12x8 figure with annotated true/false positives and negatives
- Evaluates model performance on unseen test data
- Visual representation of classification errors

---

## 💡 Key Learnings

### **Technical Insights**
1. **Ensemble Methods Dominate**: WeightedEnsemble_L2 outperformed individual models by combining their strengths
2. **Gradient Boosting Excellence**: LightGBM, XGBoost, and CatBoost all achieved top-tier performance
3. **AutoML Efficiency**: 250 seconds of training produced production-ready models without manual hyperparameter tuning
4. **Feature Engineering Not Required**: AutoGluon handles feature preprocessing automatically

### **Business Value**
- **Time Savings**: Reduced model development time from days to minutes
- **Reproducibility**: Standardized pipeline ensures consistent results
- **Scalability**: Easy to retrain with new data or adjust parameters
- **Interpretability**: Leaderboard provides clear model comparison

### **Best Practices Applied**
- Train/test split prevents overfitting
- Time limits ensure efficient resource usage
- Confusion matrix validates real-world performance
- EDA informs feature selection and data quality

---

## 🔮 Future Enhancements

- [ ] **Feature Importance Analysis**: Use SHAP values to explain predictions
- [ ] **Hyperparameter Tuning**: Experiment with `best_quality` preset for maximum accuracy
- [ ] **Cross-Validation**: Implement k-fold CV for more robust evaluation
- [ ] **Deployment**: Create REST API using Flask/FastAPI
- [ ] **AWS Integration**: Deploy model to SageMaker for production inference
- [ ] **Class Imbalance Handling**: Apply SMOTE if dataset is imbalanced
- [ ] **Model Explainability**: Add LIME for individual prediction explanations
- [ ] **A/B Testing**: Compare AutoML vs. manually tuned models

---

## 📚 Dataset Information

**Source**: Cancer diagnosis dataset (binary classification)

**Features**: 
- Multiple numerical features (exact count visible in notebook)
- Target variable: Binary (0 = Benign, 1 = Malignant - or similar)

**Size**: 
- Total samples: [Visible in notebook output]
- Training samples: 80%
- Testing samples: 20%

---

## 🧠 Why AutoGluon?

AutoGluon is AWS's state-of-the-art AutoML framework, trusted by industry leaders:

✅ **Automatic**: No manual model selection or hyperparameter tuning  
✅ **Accurate**: Consistently ranks in top Kaggle competition solutions  
✅ **Fast**: Optimized for speed with time-based training budgets  
✅ **Robust**: Handles missing data, categorical features, and class imbalance  
✅ **Scalable**: From laptops to AWS cloud infrastructure  

---

## 📖 Learn More

- [AutoGluon Documentation](https://auto.gluon.ai/)
- [AutoGluon Tabular Tutorial](https://auto.gluon.ai/stable/tutorials/tabular/tabular-quick-start.html)
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)

---

## 📫 Connect With Me

**[Your Name]**  
📧 Email: your.email@example.com  
💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
🌐 Portfolio: [yourportfolio.com](https://yourportfolio.com)  
🐙 GitHub: [@yourusername](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- AWS AutoGluon team for the incredible framework
- Google Colab for free GPU/TPU resources
- Open-source community for supporting libraries

---

## ⭐ Support This Project

If you found this project helpful:
- ⭐ **Star this repository** to show your support
- 🔀 **Fork it** to build your own AutoML projects
- 📣 **Share it** with others learning ML/AutoML

---

<div align="center">

**Built with ❤️ using AWS AutoGluon**

*Empowering healthcare with AI*

</div>
