import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize

from prior_data_analys import Prior_Data_Analys as pda
from distribution import Distribution as dist

class Data_Preprocessing:
  def __init__(self,df,target_feature = None):
    self.df = df
    self.target_feature = target_feature

  def _feature_target_split(self):
    if self.target_feature != None:
      X = self.df[[feature for feature in self.df.columns if feature != self.target_feature]]
      y = self.df[self.target_feature]
      return X,y

    else:
      X = self.df
      return X

  def _one_hot(self,X):
    ohe = OneHotEncoder(handle_unknown = 'ignore',sparse_output = False).set_output(transform = 'pandas')
    
    return ohe.fit_transform(X)

  def _normalize(self,X):
    X = normalize(X)
    return X
    

  def transform(self):
    if self.target_feature != None:
      eda = pda(self.df)
      numerical_features = eda.numerical_features()
      numerical_features = list(numerical_features)
      numerical_features.remove(self.target_feature)
      numerical_features = np.array(numerical_features)

      categorical_features = eda.categorical_features()

      X,y = self._feature_target_split()
      X,y = X.fillna(value = 0),y.fillna(value = 'Nan')

      X_num = self._normalize(X[numerical_features].astype(np.float64))
      X_cat = self._one_hot(X[categorical_features].astype('str'))

      X_num = pd.DataFrame(X_num,columns = numerical_features)
      X_cat = pd.DataFrame(X_cat)

      X = pd.concat([X_num,X_cat],axis = 1)

      return X,y

    else:
      numerical_features = [column for column in self.df.columns if self.df[column].dtype != 'O']
      categorical_features = eda.categorical_features()

      X = self._feature_target_split()
      X = X.fillna(value = 0)

      X_num = self._normalize(X[numerical_features].astype(np.float64))
      X_cat = self._one_hot(X[categorical_features].astype('str'))

      X_num = pd.DataFrame(X_num,columns = numerical_features)
      X_cat = pd.DataFrame(X_cat)

      X = pd.concat([X_num,X_cat],axis = 1)

      return X