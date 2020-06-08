import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 0, 1, 1]
y_true = [1, 1, 0, 1]
print accuracy_score(y_true, y_pred)