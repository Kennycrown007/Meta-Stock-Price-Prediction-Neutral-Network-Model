# 📈 Neural Network Prediction: Meta Stock Price

This project uses a feedforward neural network to predict Meta’s closing stock price based on historical daily data.

---

## 📊 Dataset

- Source: `META.csv`
- Features: `Open`, `High`, `Low`, `Volume`
- Target: `Close`

---

## ⚙️ Model Structure

- Input: 4 financial features
- Hidden Layers: 64 and 32 neurons (ReLU)
- Output: Single value (closing price)

---

## 📉 Training Process

- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Epochs: 100
- Batch size: 16



## 📈 Evaluation

- Visual comparison of actual vs predicted values

  
- Training vs validation loss plot

  ![Picture1](https://github.com/user-attachments/assets/408c3b5e-2ee8-4849-b6b7-9eedc3a00f28)

- Final MSE printed on test data

---

## 🔍 Key Insights

- Predictions track actual prices in trend
- MSE provides quantitative model performance
- Highlights limits of feedforward NNs on stock price without temporal modeling

## 🧰 Requirements

- Python 3.8+
- Jupyter Notebook
- Required Libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - scikit-learn
  - TensorFlow (for NN parts)
## 🚀 Usage

1. Clone the repository
2. Install the required packages (e.g., via `pip install -r requirements.txt`)
3. Open the `.ipynb` notebook in Jupyter and run the cells in order

## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://www.datascienceportfol.io/KehindeAromona)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kehinde-gabriel-aromona-808578119/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/kennycrown7)


## Badges
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

