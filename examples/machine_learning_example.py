from neurocombat_sklearn import CombatModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Loading data
data = np.load('data/bladder-expr.npy')
covars = pd.read_csv('data/bladder-pheno.txt', delimiter='\t')

# Split data between training and test set
X_train, X_test, batch_train, batch_test, age_train, \
age_test, cancer_train, cancer_test = train_test_split(data,
                                                       covars['batch'],
                                                       covars['age'],
                                                       covars['cancer'],
                                                       test_size=0.33,
                                                       random_state=32)

# Creating model
model = CombatModel()

# Fitting the model and transforming the training set
X_train_harmonized = model.fit_transform(X_train,
                                         batch_train[:, np.newaxis],
                                         cancer_train[:, np.newaxis],
                                         age_train[:, np.newaxis])

# Harmonize test set using training set fitted parameters
X_test_harmonized = model.transform(X_test,
                                    batch_test[:, np.newaxis],
                                    cancer_test[:, np.newaxis],
                                    age_test[:, np.newaxis])
