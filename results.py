import matplotlib.pyplot as plt

from helper_functions import plot_decision_boundary
from main import model, X_blob_train, y_blob_train, X_blob_test, y_blob_test

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_blob_train, y_blob_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_blob_test, y_blob_test)
