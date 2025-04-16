from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

digits = load_digits()

# Features: images, Labels: targets
X = digits.data      # shape: (1797, 64)
y = digits.target    # shape: (1797,)

plt.gray()
for i in range(5):
    plt.matshow(digits.images[i])
    plt.title(f"Label: {digits.target[i]}")
    plt.show()

plt.figure(figsize=(8, 4))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"Pred: {y_pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
