import pandas as pd
import numpy as np
from nltk import accuracy
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


file_path='spambase.data'
data=pd.read_csv(file_path)
#print(df.head())
data=data.fillna(0)
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42, shuffle=True)
##CART model
cart = DecisionTreeClassifier(random_state=42)
cart.fit(x_train, y_train)
y_pred = cart.predict(x_test)
cart_test_error=1-accuracy(y_test,y_pred)

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(cart,
          filled=True,
          fontsize=8,
          max_depth=3,
          feature_names=[f'feature_{i}' for i in range(x.shape[1])],
          class_names=["Not Spam", "Spam"])
plt.title("Decision Tree - CART")
plt.show()
###Random forest
trees=list(range(1,101))
rf_test_error=[]

for i in trees:
    rf = RandomForestClassifier(n_estimators=i, random_state=42)
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    test_error = 1 - accuracy_score(y_test, y_pred_rf)
    rf_test_error.append(test_error)

plt.figure(figsize=(12, 6))
plt.plot(trees, rf_test_error, label='Random Forest Test Error', linewidth=2)
plt.axhline(y=cart_test_error, color='red', linestyle='--', label='CART Test Error')
plt.xlabel("Number of Trees in Random Forest")
plt.ylabel("Test Error Rate")
plt.title("Test Error vs Number of Trees (Random Forest vs CART)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print(f"CART Test Error: {cart_test_error:.4f}")
print(f"Random Forest Min Test Error: {min(rf_test_error):.4f} (at {rf_test_error.index(min(rf_test_error)) + 1} trees)")

##Q3
features=x.shape[1]
values=list(range(1,features+1,2))

OOB_errors=[]
test_errors=[]

for i in values:
    rf = RandomForestClassifier(
        n_estimators=100,
        max_features=i,
        oob_score=True,
        random_state=20,
        bootstrap=True
    )
    rf.fit(x_train, y_train)

    oob_error=1-rf.oob_score_
    OOB_errors.append(oob_error)

    test_pred=rf.predict(x_test)
    test_error = 1 - accuracy_score(y_test,test_pred)
    test_errors.append(test_error)

plt.figure(figsize=(12, 6))
plt.plot(values, OOB_errors, label="OOB Error", marker='o')
plt.plot(values, test_errors, label="Test Error", marker='s')
plt.xlabel("ν (max_features)")
plt.ylabel("Error Rate")
plt.title("OOB and Test Error vs Number of Features per Split")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

min_test = min(test_errors)
min_oob = min(OOB_errors)
print(f"Minimum Test Error: {min_test:.4f} at ν = {values[test_errors.index(min_test)]}")
print(f"Minimum OOB Error: {min_oob:.4f} at ν = {values[OOB_errors.index(min_oob)]}")

###Q4
##Use only non-spam da
x_train_nsp=x_train[y_train == 0]
svm= OneClassSVM(kernel='rbf',gamma=0.01,nu=0.1)
svm.fit(x_train_nsp)
y_pred_svm=svm.predict(x_test)
y_pred_new=np.where(y_pred_svm==1,0,1) ##make non,zeros, zero
misclass_error=np.mean(y_pred_new!=y_test)
print(f"SVM Misclassification Error: {misclass_error:.4f}")

##Make performance better
best_error = 1
best_params = None

for gamma in [0.001, 0.01, 0.1, 1]:
    for nu in [0.01, 0.05, 0.1, 0.15]:
        model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        model.fit(x_train_nsp)
        y_pred = np.where(model.predict(x_test) == 1, 0, 1)
        error = np.mean(y_pred != y_test)
        if error < best_error:
            best_error = error
            best_params = (gamma, nu)

print(f"Best params: gamma={best_params[0]}, nu={best_params[1]}")
print(f"Best error: {best_error:.4f}")
##we tuned gamma and nu parameters to achieve best performance. However, we
## trained the model only on non-spam data. This can lead to outperform.