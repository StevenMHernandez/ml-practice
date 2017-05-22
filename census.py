import os
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
rnd.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix


DATA_URI = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
STORAGE_PATH = "datasets/census"

def fetch_data(uri, storage_path, filename):
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    data_path = os.path.join(storage_path, filename)
    urllib.request.urlretrieve(uri, data_path)

fetch_data(DATA_URI + "/adult.data", STORAGE_PATH, "adult.data")
fetch_data(DATA_URI + "/adult.test", STORAGE_PATH, "adult.test")

def load_data(filename):
    csv_path = os.path.join(STORAGE_PATH, filename)
    return pd.read_csv(csv_path, names=[
        "age", "wordclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ])

data = load_data("adult.data")
data.head()

data.info()

data["native-country"].value_counts()

print(data.describe())

data.hist(bins=10, figsize=(12,9))
plt.show()

# We have already have a predefined test set (defined above)
# but just for practice, we are splitting here again
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
print("train:", len(train_set), "test:", len(test_set))
test_set.head()

train_set["age"].hist()
plt.show()

# Probably not useful, just following along with the example
# creating a new attribute category
data["hours_category"] = np.ceil(data["hours-per-week"] / 10)
data["hours_category"].value_counts()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split.split(data, data["hours_category"])
for train_index, test_index in split.split(data, data["hours_category"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

print("strat_train:", len(strat_train_set), "strat_test:", len(strat_test_set), "\n")


strat_train_set.head()


train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

def hours_cat_proportions(data):
    return data["hours_category"].value_counts() / len(data)

data.head(2)
strat_test_set.head(2)
test_set.head(2)

compare_props = pd.DataFrame({
    "Overall": hours_cat_proportions(data),
    "Stratified": hours_cat_proportions(strat_test_set),
    "Random": hours_cat_proportions(test_set),
})
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split.split(data, data["education-num"])
for train_index, test_index in split.split(data, data["education-num"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# compare stratified to random
def education_cat_proportions(data):
    return data["education-num"].value_counts() / len(data)

compare_props = pd.DataFrame({
    "Overall": education_cat_proportions(data),
    "Stratified": education_cat_proportions(strat_test_set),
    "Random": education_cat_proportions(test_set),
})
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props



# Correlation matrices

coor_matrix = data.corr()

data.info()

coor_matrix["income_category"].sort_values(ascending=False)

attributes = ["education-num", "age", "hours-per-week", "capital-gain", "capital-loss"]

scatter_matrix(data[attributes], figsize=(20,10))
plt.show()




































# I need lots of extra space to jump around.
# Thanks for making me have to do this Atom.
