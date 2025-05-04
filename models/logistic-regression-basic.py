# Re-import after code execution environment reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Input data
temperatures = np.array([[10], [15], [18], [22], [25], [28]])
bought_ice_cream = np.array([0, 0, 0, 1, 1, 1])  # 0 = No, 1 = Yes

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(temperatures, bought_ice_cream)

# Predict probabilities over a range of temperatures
x_range = np.linspace(8, 30, 100).reshape(-1, 1)
y_probs = log_reg.predict_proba(x_range)[:, 1]  # Probability of class 1 (buying)

# Plotting
plt.figure(figsize=(8, 5))
plt.scatter(temperatures, bought_ice_cream, color='blue', label='Actual Data')
plt.plot(x_range, y_probs, color='red', label='Logistic Curve (Probability)')
plt.title("Probability of Buying Ice Cream vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Probability of Buying Ice Cream")
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Prediction Test
log_reg.predict([[19.6]])[0]
