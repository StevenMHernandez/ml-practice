import os
import zipfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
rnd.seed(42)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


DATA_URI = "http://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
STORAGE_PATH = "datasets/student-performance"

def fetch_zip_data(uri, storage_path):
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    data_path = os.path.join(storage_path, "data.zip")
    urllib.request.urlretrieve(uri, data_path)
    zip_ref = zipfile.ZipFile(data_path, 'r')
    zip_ref.extractall(storage_path)
    zip_ref.close()

fetch_zip_data(DATA_URI, STORAGE_PATH)

def load_data(filename):
    csv_path = os.path.join(STORAGE_PATH, filename)
    return pd.read_csv(csv_path, delimiter=";")

data = load_data("student-mat.csv")
data.head()

data.info()

data["studytime"].value_counts()

print(data.describe())

data.hist(bins=10, figsize=(12,9))
plt.show()

# We have already have a predefined test set (defined above)
# but just for practice, we are splitting here again
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
print("train:", len(train_set), "test:", len(test_set))
test_set.head()





split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
split.split(data, data["studytime"])
for train_index, test_index in split.split(data, data["studytime"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# compare stratified to random
def studytime_cat_proportions(data):
    return data["studytime"].value_counts() / len(data)

compare_props = pd.DataFrame({
    "Overall": studytime_cat_proportions(data),
    "Stratified": studytime_cat_proportions(strat_test_set),
    "Random": studytime_cat_proportions(test_set),
})
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props



# Correlation matrices

coor_matrix = data.corr()

coor_matrix["G3"].sort_values(ascending=False)

attributes = ["G2", "G1",  "failures"]

scatter_matrix(data[attributes], figsize=(20,10))
plt.show()

data["studytime_to_absences"] = data["studytime"] / data["absences"]
data["freetime_to_absences"] = data["freetime"] / data["absences"]
data["traveltime_to_absences"] = data["traveltime"] / data["absences"]
data["studytime_to_freetime"] = data["studytime"] / data["freetime"]


coor_matrix = data.corr()
coor_matrix["G3"].sort_values(ascending=False)


data = strat_train_set.drop("G3", axis=1)
data_labels = strat_train_set["G3"].copy()

data.count()
data_labels.count()

data.head()

data.info()

string_attributes = ["address", "school", "sex", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian", "schoolsup",
    "famsup", "paid", "activities", "nursery", "higher",
    "internet", "romantic"]

data_copy = data[string_attributes].copy()

data

data_copy

encoder = LabelEncoder()
data_cat = data["school"]
data_cat_encoded = encoder.fit_transform(data_cat)
data_cat_encoded

encoder.classes_



# copied from the book's example jupyter notebook
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

data

data_num = data.drop(string_attributes, axis=1)

data_num.info()


list(data_num)

corr

data.info()

data


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        studytime_to_absences = xxx["studytime"] / X["absences"]
        freetime_to_absences = X["freetime"] / X["absences"]
        traveltime_to_absences = X["traveltime"] / X["absences"]
        studytime_to_freetime = X["studytime"] / X["freetime"]
        return np.c_[X, studytime_to_absences, freetime_to_absences, traveltime_to_absences, studytime_to_freetime]


num_attribs = list(data_num.drop(['G3'], axis=1))
# num_attribs = ['age']
cat_attribs = string_attributes
cat_attribs = ['sex']

num_attribs

# le = LabelEncoder

cat_attribs
num_attribs

data[num_attribs].count()

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        # ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])

preparation_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


data_prepared = preparation_pipeline.fit_transform(data)
data_prepared

data_prepared.shape

data_labels

lin_reg = LinearRegression()
lin_reg.fit(data_prepared, data_labels)

some_data = data.iloc[:5]

some_label = data_labels.iloc[:5]

some_data_prepared = preparation_pipeline.transform(some_data)

print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_label))

# Linear Regression
data_predictions = lin_reg.predict(data_prepared)
lin_mse = mean_squared_error(data_labels, data_predictions)
lin_rmse = np.sqrt(lin_mse)

lin_mse
lin_rmse

lin_mae = mean_absolute_error(data_labels, data_predictions)
lin_mae




tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_prepared, data_labels)
data_predictions = tree_reg.predict(data_prepared)

tree_mse = mean_squared_error(data_labels, data_predictions)
tree_rmse = np.sqrt(tree_mse)

tree_mse
tree_rmse

tree_mae = mean_absolute_error(data_labels, data_predictions)
tree_mae





def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

tree_scores = cross_val_score(tree_reg, data_prepared, data_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, data_prepared, data_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)



forest_reg = RandomForestRegressor()
forest_reg.fit(data_prepared, data_labels)
data_predictions = forest_reg.predict(data_prepared)
forest_mse = mean_squared_error(data_labels, data_predictions)
forest_rmse = forest_mse
forest_rmse


forest_scores = cross_val_score(forest_reg, data_prepared, data_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)



svm_reg = SVR(kernel="linear")
svm_reg.fit(data_prepared, data_labels)

data_predictions = svm_reg.predict(data_prepared)
svm_mse = mean_squared_error(data_labels, data_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse

data_prepared.shape

param_grid = [
    {'n_estimators': [25, 30, 35, 40, 45], 'max_features': [8,9,10,11,12]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(data_prepared, data_labels)


grid_search.best_params_

grid_search.best_estimator_



cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


pd.DataFrame(grid_search.cv_results_)


param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}

forest_reg = RandomForestRegressor()
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_iter=10,cv=5, scoring='neg_mean_squared_error')
rnd_search.fit(data_prepared, data_labels)

cvres=rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

attributes = num_attribs + cat_attribs
sorted(zip(feature_importances, attributes), reverse=True)


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop('G3', axis=1)
y_test = strat_test_set['G3'].copy()

X_test_transformed = preparation_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_transformed)


final_mse = mean_squared_error(y_test, final_predictions)
final_mse = np.sqrt(final_mse)
final_mse







# I need lots of extra space to jump around.
# Thanks for making me have to do this Atom.
