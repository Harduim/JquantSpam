import re

import dtale
import pandas as pd
from numpy.random import seed
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# constantes
PATH_DATA = "dataset_emails.csv"
N_SPLITS = 10
SCORING = "accuracy"
TEST_SIZE = 0.3
seed(42)

# http://letmegooglethat.com/?q=most+frequently+used+words+in+spam
smelly_words = ["won", "free", "FreeMsg", "sex", "urgent", "discount", "txt"]
smelly_capture_group = f'({".?|".join(smelly_words)}.?)'

# ver encoding
emails = pd.read_csv(PATH_DATA, encoding="latin", usecols=["v1", "v2"])
emails = emails.rename(columns={"v1": "target", "v2": "email"})
emails["email"] = emails.email.str.lower()
emails["has_urls"] = emails.email.str.extract("(www|http)").notna().astype(int)
emails["has_telefone_numbers"] = emails.email.str.contains(r"\d{6,8}").astype(int)
emails["smelly_words"] = emails.email.str.count(smelly_capture_group)
emails["target_num"] = emails["target"].apply(lambda w: 0 if w == "spam" else 1)

################ Baseline ###################
svc = SVC()
log_reg = LogisticRegression()
grad_boost = GradientBoostingClassifier()
rnd_forest = RandomForestClassifier()
gauss = GaussianProcessClassifier()
perceptron = Perceptron()
ridge = RidgeClassifier()
gaussiannb = GaussianNB()
mlp = MLPClassifier()
dec_tree = DecisionTreeClassifier()

X = emails.loc[:, ["has_urls", "has_telefone_numbers", "smelly_words"]]
y = emails.loc[:, "target_num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)


kfold = KFold(n_splits=N_SPLITS, shuffle=True)

list_models = [
    ("SVC", svc),
    ("Log_Reg", log_reg),
    ("Grad_Boost", grad_boost),
    ("RandomForestReg", rnd_forest),
    ("Perceptron", perceptron),
    ("Ridge", ridge),
    ("DecisionTreeClassifier", dec_tree),
]

df_scores = pd.DataFrame()
# cross-val, treino e avalicao tudo num for
for name, model in list_models:
    scores = cross_val_score(model, X_train, y_train, scoring=SCORING, cv=kfold)
    df_scores.loc[:, name] = scores
    print('------------------------------------------------------------------')
    print(f'{name} Cross-val:{N_SPLITS} folds')
    print(f'Mean: {scores.mean()*100:.2f}%')
    print(f'Std : {scores.std()*100:.2f}%')  
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Acc teste: {accuracy_score(y_test, y_pred)*100:.2f}%')
    print('------------------------------------------------------------------')

############# baseline Mean: 96.38% ####################

dtale.show(emails.drop(columns="email"), ignore_duplicate=True)
