# Lab 1: Deep Learning Unleashed 🚀

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Neural%20Nets%20%26%20Beyond-blue?style=for-the-badge&logo=python)  
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)  
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensource)  
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0.1-orange?style=for-the-badge&logo=pytorch)

## 🔥 Project Overview
Welcome to **Lab 1: Deep Learning Unleashed**, my submission for the Deep Learning course! This project showcases two killer applications of neural networks:
- **Regression**: Predicting NYSE stock closing prices using a DNN, with regularization to keep overfitting in check.
- **Classification**: Detecting machine failures with a DNN classifier, optimized via grid search and beefed up with dropout.

Built with **PyTorch** on Google Colab, this repo is a testament to data wrangling, model crafting, and performance tuning—all in one slick Jupyter Notebook.

---

## 🛠️ Features
- 📈 **NYSE Regression**: Predicts stock prices using a 3-layer DNN (3-64-32-1) with ReLU and dropout.
- ⚙️ **Maintenance Classification**: Classifies machine failures (binary) with a tuned DNN (7-64-32-2).
- 📊 **Data Viz**: Histograms, scatter plots, loss/MAE/accuracy curves—visuals that pop!
- 🧠 **Regularization**: Dropout (0.2) and weight decay (0.01) to fight overfitting like a boss.
- ⚡ **GPU Power**: Leverages CUDA for lightning-fast training.
- 🔧 **Hyperparameter Tuning**: Grid search over learning rates and optimizers for peak performance.

---

## 📋 Table of Contents
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Shoutouts](#-shoutouts)

---

## 📦 Requirements
To run this beast, you’ll need:
- **Python 3.8+**
- **Jupyter Notebook** (or Colab)
- **Dependencies**:
  ```plaintext
  torch==2.0.1
  pandas==2.0.3
  numpy==1.24.3
  matplotlib==3.7.1
  seaborn==0.12.2
  scikit-learn==1.3.0
  imblearn==0.11.0
  kagglehub
  ```
- **Optional**: GPU with CUDA for max speed.

---

## ⚙️ Installation
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/zachary013/lab1-deep-learning.git
   cd lab1-deep-learning
   ```

2. **Set Up a Virtual Env**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install the Goods**:
   ```bash
   pip install -r requirements.txt
   ```
   *No `requirements.txt` yet? Copy the list above and run `pip install <package>` for each.*

4. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

---

## 🚀 Usage
1. **Open the Notebook**:
   - Fire up `lab1_deep_learning.ipynb` in Jupyter or upload it to [Google Colab](https://colab.research.google.com/).

2. **Run It**:
   - Hit **Run All** (or `Shift + Enter` cell-by-cell) to:
     - Download datasets via `kagglehub`.
     - Preprocess, train, and evaluate both models.
     - Generate dope plots.

3. **Outputs**:
   - Check inline plots (loss, MAE, accuracy) and printed metrics (accuracy, sensitivity, F1).

*Pro Tip*: Got a GPU? Colab’s free CUDA support will make this fly!

---

## 📂 Project Structure
```
lab1-deep-learning/
├── lab1_deep_learning.ipynb  # The main event: code, viz, and results
├── requirements.txt          # Dependency list (create it if missing)
└── README.md                 # This badass guide
```
*Datasets are fetched dynamically via `kagglehub`—no local storage needed!*

---

## 🎯 Results
### NYSE Regression
- **Task**: Predict closing prices from `open`, `high`, and `low`.
- **Model**: DNN (3-64-32-1) + ReLU + Dropout (0.2).
- **Loss**: MSE dropped from 12,062 to 16.55 (regularized) over 100 epochs.
- **MAE**: Slashed to ~4.0 on test set with regularization.
- **Takeaway**: Dropout and weight decay crushed overfitting—test loss stayed tight!

*Loss Plot (Regularized)*:  
![image](https://github.com/user-attachments/assets/58380f26-d16f-4df6-b5b8-250d250d7462)
![image](https://github.com/user-attachments/assets/14292a67-2ac8-4183-945c-e22577db8145)

### Maintenance Classification
- **Task**: Predict machine failure (`Target`: 0 or 1).
- **Model**: DNN (7-64-32-2) + ReLU + Dropout (0.2).
- **Best Params**: `lr=0.01`, `optimizer=adam` (via grid search).
- **Metrics**:
  - Accuracy: **95.47%**
  - Sensitivity: **96.84%**
  - F1 Score: **95.53%**
- **Takeaway**: SMOTE balanced the classes, and regularization kept generalization solid.

*Accuracy Plot (Regularized)*:  
![image](https://github.com/user-attachments/assets/5a0fc26f-b15e-4dae-b9b9-8f4b8c9d74f5)
![image](https://github.com/user-attachments/assets/94d1f3b9-2c6a-40de-8dfe-526b39dc9450)



---

## 🤝 Contributing
This is my homework, so I’m not looking for pull requests—but feel free to fork it and tweak it for your own deep learning adventures! Got feedback? Hit me up via GitHub Issues.

---

## 📜 License
Licensed under the [MIT License](LICENSE)—use it, share it, just don’t blame me if your GPU melts! 🔥

---

## 🙌 Shoutouts
- **My Prof & TAs**: For dropping this dope assignment.
- **PyTorch Crew**: For the slick framework.
- **Kaggle**: For the datasets that made this possible.
- Submitted by **Zachary** on **March 9, 2025**.

---

### How to Add This to Your Repo
1. **Create/Update `README.md`**:
   - Copy the text above into a file named `README.md` in your repo root.
   - Replace placeholder images (e.g., `https://via.placeholder.com/...`) with actual plot URLs from your notebook:
     - In Colab, right-click a plot > “Copy image address” > paste it in.
     - Or download plots, upload to GitHub, and link them (e.g., `![Loss Plot](plots/loss_reg.png)`).

2. **Push It**:
   ```bash
   git add README.md
   git commit -m "Add badass README for Lab 1"
   git push origin main
   ```

3. **Optional: Add `requirements.txt`**:
   - Create a `requirements.txt` with the dependency list above and push it too.

---

### Final Touches
- **Plots**: Your notebook generates loss/MAE/accuracy plots—export them from Colab and add them to the README for that extra wow factor.
- **Polish**: If your teacher loves detail, add a section like “Lessons Learned” (e.g., “Overfitting sucks, regularization rules!”).
- **Test It**: View it on GitHub to ensure the icons and formatting look sharp.

## Prepared by :

| Avatar                                                                                                  | Name | GitHub |
|---------------------------------------------------------------------------------------------------------|------|--------|
| <img src="https://github.com/zachary013.png" width="50" height="50" style="border-radius: 50%"/>        | Zakariae Azarkan | [@zachary013](https://github.com/zachary013) |
