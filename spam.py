import dtale
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from numpy.random import seed
import re


# constantes
PATH_DATA = "dataset_emails.csv"
seed(42)

# http://letmegooglethat.com/?q=most+frequently+used+words+in+spam
smelly_words = ["won", "free", "FreeMsg", "sex", "urgent", "discount"]
smelly_capture_group = f'({".?|".join(smelly_words)}.?)'

# ver encoding
emails = pd.read_csv(PATH_DATA, encoding="latin", usecols=["v1", "v2"])
emails = emails.rename(columns={"v1": "target", "v2": "email"})
emails["email"] = emails.email.str.lower()
emails["has_urls"] = emails.email.str.extract("(www|http)").notna().astype(int)
emails["has_telefone_numbers"] = emails.email.str.contains(r"\d{6,8}").astype(int)
emails["smelly_words"] = emails.email.str.count(smelly_capture_group)


dtale.show(emails.drop(columns="email"), ignore_duplicate=True)
