#!/usr/bin/env python
# coding: utf-8

# # Frame the Problem and Look at the Big Picture

# ## Define the objective in business terms.

# Predict the median housing in any disctrict, given all the other metrics.

# ## How will your solution be used?

# 

# ## What are the current solutions/workarounds (if any)?

# 

# ## How should you frame this problem (supervised/unsupervised, online/offline, etc.)?

# 

# ## How should performance be measured?

# 

# ## Is the performance measure aligned with the business objective?

# 

# ## What would be the minimum performance needed to reach the business objective?

# 

# ## What are comparable problems? Can you reuse experience or tools?

# 

# ## Is human expertise available?

# 

# ## How would you solve the problem manually?

# 

# ## List the assumptions you (or others) have made so far.

# 

# ## Very assumptions if possible.

# 

# # Get the Data

# ## List the data you need and how much you need.

# In[ ]:





# ## Find and document where you can get that data.

# https://raw.githubusercontent.com/ageron/handson-ml/master/

# ## Check how much space it will take.

# In[ ]:





# ## Check legal obligations, and get authorization if necessary.

# 

# ## Get access authorizations.

# 

# ## Create a workspace (with enough storage space).

# 

# ## Get the data

# In[1]:


import os
import tarfile
import urllib

# URL del file da scaricare
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Percorso dove i dati saranno salvati
HOUSING_PATH = os.path.join("../Data", "housing")

# Funzione per scaricare il file
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.exists(housing_path):  # Se la cartella non esiste, creala
        os.makedirs(housing_path)

    # Scarica il file
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    print(f"File scaricato in: {tgz_path}")

    # Estrai il file .tgz senza filtro o con un filtro che non influisce sull'estrazione
    with tarfile.open(tgz_path) as housing_tgz:
        # Rimuoviamo il filtro, estraiamo tutti i file
        housing_tgz.extractall(path=housing_path)
    print(f"File estratti in: {housing_path}")

# Esegui la funzione per scaricare e estrarre il dataset
fetch_housing_data()


# ## Convert the data to a format you can easily manipulate (without changing the data itself).

# In[2]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()


# ## Ensure sensitive information is deleted or protected (eg. anonymized).

# In[ ]:





# ## Check the size and type of data (time series, sample, geographical, etc.).

# In[3]:


housing.info()


# In[4]:


housing.describe()


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show


# ## Sample a test set, put it aside, and never look at it (no data snooping!).

# #### Random Sampling

# In[10]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# #### Stratfield sampling

# In[11]:


import numpy as np
housing['income_cat'] = pd.cut(housing['median_income'],
                              bins = [0, 1.5, 3, 4.5, 6, np.inf],
                              labels = [1, 2, 3, 4, 5])
housing[['median_income','income_cat']]


# In[12]:


import numpy as np
housing['income_cat'] = pd.cut(housing['median_income'],
                              bins = [0, 1.5, 3, 4.5, 6, np.inf],
                              labels = [1, 2, 3, 4, 5])
housing['income_cat'].hist()
plt.show()


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print("Category proportion for overall dataset:\n", housing['income_cat'].value_counts() / len(housing))
print("Category proportion for stratified sampling:\n", strat_train_set['income_cat'].value_counts() / len(strat_train_set))


# In[80]:


#for set_ in (strat_train_set, strat_test_set):
#    set_.drop("income_cat", axis=1, inplace=True)


# # Explore the Data

# ## Create a copy of the data for exploration (sampling it down to a manageable size if necessary).

# In[ ]:


housing = strat_train_set.copy()


# ## Study each attribute and its characteristics.
# - Name
# - Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
# - % of missing values
# - Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
# - Usefulness for the task
# - Type of distribution (Gaussian, uniform, logarithmic, etc.)

# In[16]:


#READ ME > Dependences problem with Profile Report and pandas_profiling
#from pandas_profiling import ProfileReport
#profile = ProfileReport(housing, title="Pandas Profiling Report")
#profile.to_notebook_iframe()


# ## For supervised learning tasks, identify the target attribute(s).

# The target varable is housing_median_value column 

# ## Visualize the data

# In[17]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.cm.jet, colorbar=True)
plt.legend()
plt.show()


# ### Study the correlations between attributes.

# In[18]:


#corr column
corr_matrix = housing.select_dtypes(include=[float, int]).corr()
median_house_value_corr = corr_matrix["median_house_value"].sort_values(ascending=False)
print(median_house_value_corr)


# In[19]:


from pandas.plotting import scatter_matrix

attributes = median_house_value_corr.index[0:4]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()


# ## Study how you would solve the problem manually.

# In[ ]:





# ## Identify the promising transformations you may want to apply.

# In[20]:


housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_household"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["populations_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.select_dtypes(include=['float64', 'int64']).corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# ## Identify extra data that would be useful.

# In[ ]:





# ## Document what you have learned.

# 

# # Prepare the Data
# - Work on copies of the data (keep the original dataset intact).
# - Write functions for all data transformations you apply.
#     - So you can easily prepare the data the next time you get a fresh dataset
#     - So you can apply these transformations in future projects
#     - To clean and prepare the test set
#     - To clean and prepare new data instances once your solution is live
#     - To make it easy to treat your preparation choices as hyperparameters

# ## Data cleaning.
# - Fix or remove outliers (optional)
# - Fill in missing values (e.g., with zeros, mean, median...) or drop their rows (or columns) 

# In[21]:


housing = strat_train_set.drop ("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[22]:


#prepare data for missing data with the median of the column
#calculate median of each column
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
print(imputer.statistics_)


# ## Feature selection (optional).
# - Drop the attributes that provide no useful information for the task
# - Use some dimensionality reduction technique if necessary (PCA, KernelPCA, LLE...)

# In[ ]:





# ## Feature engineering, where appropriate.
# - Discretize continuous features
# - Decompose features (e.g., categorical, date/time, etc.)
# - Add promising transformations of features (e.g., log(x), sqrt(x), x^2, etc.)
# - Aggregate features into promising new features

# #### Handling Text and Categorical Attributes (added after processing 4.3)

# In[23]:


#manipulate columns for manage categorical var (like string)
housing_cat = housing[['ocean_proximity']].reset_index()
housing_cat.head()


# In[24]:


#convert column in a float
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot.toarray())
print(cat_encoder.categories_)


# ### Aggregate features into promising new features (after 4.3)

# In[25]:


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedAttributeAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Supponendo che X sia un array numpy o DataFrame
        rooms_ix = housing.columns.get_loc("total_rooms")
        bedrooms_ix = housing.columns.get_loc("total_bedrooms")
        population_ix = housing.columns.get_loc("population")
        households_ix = housing.columns.get_loc("households")

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Uso corretto
att_adder = CombinedAttributeAdder(add_bedrooms_per_room=False)
housing_extra_attribs = att_adder.transform(housing_num.values)


# In[26]:


housing_extra_attribs.shape


# ## Feauture scaling
# - Standardize or normalize features

# In[27]:


#scale features > set the same order of magnitude of the values
from sklearn.preprocessing import StandardScaler #we take the min e max of each column and we divide each n of the column for the range
#keep attention to the ouliers
scaler = StandardScaler()
housing_extra_attribs_scaled = scaler.fit_transform(housing_extra_attribs)
housing_extra_attribs_scaled


# #### Reconstruct data
# - use pipelines to automate all steps if possible

# In[28]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

#merge vars num and cat

num_attribs = housing_num.columns.tolist()
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributeAdder()),
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# # Shortlisting Promising Models
# - If the data is huge, you may want to sample smaller training sets so you can train many different models in a reasonable time (be aware that this penalizes complex models such as large neural nets or Random Forests).
# - Once again, try to automate these steps as much as possible.

# ## Train many quick-and-dirty models from different categories (e.g., linear, naive Bayes, SVM, Random Forest, neural net, etc.) using standard parameters.

# In[29]:


#test linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[30]:


from sklearn.metrics import mean_squared_error #evaluate our model trough MSE
#MSE = index of how much the predicted values ​​differ from those actually measured by the labels

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

#result 68376.51$ > it is a high value, it indicates that the typical price of a house within a district has an error of that value, because the model is linear and the features are not distributed in a linear way 


# #### Decision Tree

# In[31]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[32]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# #### Random Forest

# In[52]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)


# In[55]:


#the result performe better the the other 2 as the error result 1860$
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# ## Measure and compare their performance.
# - For each model, use N-fold cross-validation and compute the mean and standard deviation of the performance measure on the N folds.

# ### use cross validation, train the model severl time in the same training set

# In[ ]:





# In[43]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# ### Liner Regression

# In[44]:


from sklearn.model_selection import cross_val_score

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse = np.sqrt(-lin_scores)
display_scores(lin_rmse)

#we can understand which is the avr error if we use a linear model


# ### Decision Tree

# In[45]:


tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse = np.sqrt(-tree_scores)
display_scores(lin_rmse)

#values are highe than LR. Worst than the LR, we discover it trought the cross validation Model


# ### Random Forest

# In[62]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=5)
forest_rmse = np.sqrt(-forest_scores)
display_scores(forest_rmse)


# ## Analyze the most significant variables for each algorithm.

# In[64]:


extra_attribs = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
cat_one_hot_attribs = list(full_pipeline.named_transformers_["cat"].categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
def feature_importance(features, attributes):
    for f, a in sorted(zip(features, attributes), reverse=True):
        print(f"{a}\t{f}")
# This function will take the feature importance values
#(the model coefficients, e.g. from a regression or classification model) 
#and their attribute names, sort them and print them.


# ### Linear regression

# In[66]:


feature_importance(lin_reg.coef_, attributes)


# ### Decision Tree

# In[68]:


feature_importance(tree_reg.feature_importances_, attributes)


# ### Random Forest

# In[69]:


feature_importance(forest_reg.feature_importances_, attributes)


# ## Analyze the types of errors the models make.
# - What data would a human have used to avoid these errors?

# In[ ]:





# ## Perform a quick round of feature selection and engineering.

# In[ ]:





# ## Perform one or two more quick iterations of the five previous steps.

# In[ ]:





# ## Shortlist the top three to five most promising models, preferring models that make different types of errors.

# In[ ]:





# # Fine-Tune the System
# - You will want to use as much data as possible for this step, especially as you move towards the end of fine-tuning.
# - As always automate what you can.

# ## Fine-tune the hyperparameters using cross-validation.
# - Treat your data transformation choices as hyperparameters, especially when you are not sure about them (e.g., if you are not sure whether to replace missing values with zeros or with the median value, or to just drop the rows).
# - Unless there are very few hyperparameter values to explore, prefer random search over grid search. If training is very long, you may prefer a Bayesian optimization approach (e.g., using Gaussian process priors).

# In[71]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 5, 8]},  # Search combinations of n_estimators and max_features
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}  # Alternative search for bootstrap=False
]

forest_reg = RandomForestRegressor()  # Initialize the RandomForestRegressor model

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,  # Perform grid search with 5-fold cross-validation
                          scoring="neg_mean_squared_error",  # Use negative mean squared error as the scoring metric
                          return_train_score=True)  # Return training scores in addition to validation scores

grid_search.fit(housing_prepared, housing_labels)  # Fit the grid search model to the data


# In[ ]:





# ## Try Ensemble methods. Combining your best models will often produce better performance than running them individually.

# In[74]:


#see the best value where the model performed better
print(grid_search.best_params_)
print(grid_search.best_estimator_)


# In[75]:


cv_result = grid_search.cv_results_  # Get the results of the grid search
for mean_score, param in zip(cv_result["mean_test_score"], cv_result["params"]):  # Loop through mean scores and corresponding parameters
    print(np.sqrt(-mean_score), param)  # Print the root mean square error and the parameters


# ## Once you are confident about your final model, measure its performance on the test set to estimate the generalization error.
# - Don't tweak your model after measuring the generalization error: you would just start overfitting the test set.

# In[76]:


final_model = grid_search.best_estimator_  # Get the best model from the grid search

# Fix typo in variable names
X_test = strat_test_set.drop("median_house_value", axis=1)  # Drop the target column from the test set
y_test = strat_test_set["median_house_value"].copy()  # Corrected typo here: 'start_test_set_set' -> 'strat_test_set'

# Prepare the test data using the same pipeline
X_test_prepared = full_pipeline.transform(X_test)

# Make predictions using the final model
final_predictions = final_model.predict(X_test_prepared)

# Calculate the Mean Squared Error and then the Root Mean Squared Error
final_mse = mean_squared_error(y_test, final_predictions)
final_mse = np.sqrt(final_mse)

# Print the RMSE
print(final_mse)


# ### CONCLUSION
# ## Our odel perform with an error of 47572 $. It perform not so good, but we can try other model and combine or add another features or paraeters. 

# # Present Your Solution

# ## Document what you have down.

# 

# ## Create a nice presentation.
# - Make sure you highlight the big picture first.

# 

# ## Explain why your solution achieves the business objective.

# 

# ## Don't forget to present interesting points you noticed along the way.
# - Describe what worked and what did not.
# - List your assumptions and your system's limitations.

# 

# ## Ensure your key findings are communicated through beautiful visalizations or easy-to-remember statements.

# 

# # Launch!

# ## Get your solution ready for production (plug into production data inputs, write unit tests, etc.).

# 

# ## Write monitoring code to check your system's live performance at regular intervals and trigger alerts when it drops.
# - Beware of slow degradation: models tend to "rot" as data evolves.
# - Measuring performance may require a human pipeline (e.g., via a crowdsourcing service).
# - Also monitor your inputs' quality (e.g., a malfunctioning sensor sending random values, or another team's output becoming stale). This is particularly important for online learning systems.

# 

# ## Retrain your models on regular basis on fresh data (automate as much as possible).

# 
