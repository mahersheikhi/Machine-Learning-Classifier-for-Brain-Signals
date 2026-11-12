#Lab 5 – Train a Machine-Learning Classifier for Brain Signals

from ast import For

from ast import For
from scipy.signal import welch
from lab4 import epochs
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

#print(epochs.event_id)

#Extract NumPy arrays:

X = epochs.get_data()     # shape: (n_trials, n_channels, n_samples)
y = epochs.events[:, -1]  # extract event codes

fs = 250
# print(X.shape, y.shape)



### Option 1 — Flatten Time-Series (Simple & Works for ERPs)
n_trials, n_channels, n_samples = X.shape
X_flat = X.reshape(n_trials, n_channels * n_samples)
#print("Feature shape:", X_flat.shape)

#Option 2 — Extract Bandpower Features (Good for eyes-open/closed)
 
def compute_bandpower_features(epoch, fs):
    f, psd = welch(epoch, fs=fs, nperseg=fs)
    theta = psd[(f >= 4) & (f <= 8)].mean()
    alpha = psd[(f >= 8) & (f <= 12)].mean()
    beta  = psd[(f >= 13) & (f <= 30)].mean()
    return [theta, alpha, beta]
 
X_bp = np.array([
    sum([compute_bandpower_features(epoch[ch], fs) for ch in range(n_channels)], [])
    for epoch in X
])
print(X_bp.shape)



#Option 3 — PCA for Dimensionality Reduction
from sklearn.decomposition import PCA

n_components = min(50, n_trials, X_flat.shape[1]) 
pca = PCA(n_components=50)  # adjust based on dataset
X_pca = pca.fit_transform(X_flat)
print(X_pca.shape)


#6. Step 3 — Train/Test Split
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y)


#7. Step 4 — Train a Classifier
#We will test a few classifiers:

#Option A – Logistic Regression
from sklearn.linear_model import LogisticRegression
 
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)


#####Option B – Linear SVM
from sklearn.svm import SVC
 
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)


######Option C – Random Forest
from sklearn.ensemble import RandomForestClassifier
 
clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_train, y_train)


#######8. Step 5 — Evaluate the Model
##Accuracy Score
from sklearn.metrics import accuracy_score
 
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)



######Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



##########ROC Curve
from sklearn.metrics import roc_curve, auc
 
positive_label = epochs.event_id["target"]
positive_class_index = list(clf.classes_).index(positive_label)
y_prob = clf.predict_proba(X_test)[:, positive_class_index]
fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=positive_label)
roc_auc = auc(fpr, tpr)
 
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.show()



####9. Step 6 — k-Fold Cross-Validation
from sklearn.model_selection import cross_val_score
 
scores = cross_val_score(clf, X_flat, y, cv=5)
print("Cross-val accuracy:", scores)
print("Mean accuracy:", scores.mean())

##10. Step 7 — Feature Importance (Optional)
####Random Forest Feature Importance
import numpy as np
 
importances = clf.feature_importances_
idx = np.argsort(importances)[-20:]  # top 20 features
 
plt.figure(figsize=(10,6))
plt.barh(range(20), importances[idx])
plt.yticks(range(20), idx)
plt.title("Top 20 Most Important Features")
plt.show()


###For PCA → Visualize Explained Variance
plt.plot(pca.explained_variance_ratio_)
plt.title("PCA Explained Variance")
plt.show()



####11. Step 8 — Export the Model


import joblib
joblib.dump(clf, "p300_classifier.pkl")



#####Load in a deployment script:

clf = joblib.load("p300_classifier.pkl")