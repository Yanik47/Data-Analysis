import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prior_data_analys import Prior_Data_Analys

colors = sns.color_palette("rocket")

class Distribution:
  def __init__(self,df,target_feature):
    self.df = df
    self.target_feature = target_feature
    self._eda = Prior_Data_Analys(df)


  def distribution_of_target_feature(self):
    plt.figure(figsize = (10,10))
    for visualization in range(2):
      plt.subplot(2,1,visualization + 1)
      if visualization == 0:
        sns.histplot(data = self.df,x = self.target_feature,color = colors[3])
        plt.title(f'Distribution of Target feature: {self.target_feature}')
        plt.show()
        
      else:
        sns.boxplot(x = self.df[self.target_feature],color = colors[0])
        plt.title(f'Distribution of Target feature: {self.target_feature}')
        plt.show()


  def distribution_of_numerical_features(self):
      numerical_features = self._eda.numerical_features()
      
      num_features = len(numerical_features)
      cols = 3  
      rows = (num_features // cols) + (num_features % cols > 0)  

      fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))  
      axes = axes.flatten() 

      for feature_idx, feature in enumerate(numerical_features):
          sns.histplot(data=self.df, x=feature, color=colors[1], ax=axes[feature_idx])  
          axes[feature_idx].set_title(f'Distribution of {feature}')

      for idx in range(len(numerical_features), len(axes)):
          fig.delaxes(axes[idx])

      plt.tight_layout()
      plt.show()


  def distribution_of_categorical_features(self,top_k = 5):
    categorical_features = self._eda.categorical_features()
    plt.figure(figsize = (20,20))

    for feature_idx,feature in enumerate(categorical_features):

      values = self._eda.group_by(feature)[self.target_feature].sort_values(ascending = False).values[:top_k]
      labels = self._eda.group_by(feature)[self.target_feature].sort_values(ascending = False).index[:top_k]

      data_dict = {'Labels': labels, 'Values': values}
      data = pd.DataFrame(data_dict)

      plt.subplot(len(categorical_features) // 2 + 1 , 2 , feature_idx + 1)
      sns.barplot(data = data,y = 'Labels',x = 'Values',color = colors[-1])
      plt.title(f'Top {top_k} {feature} by count')

    plt.tight_layout()
    plt.show()

  def relation_numerical_feats_to_target_feat(self, df, eda):
    numerical_features = eda.numerical_features()

    data = df[numerical_features]
    corr = data.corr()

    return sns.heatmap(corr,annot = True, cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))

  def relation_categorical_feats_to_target_feat(self,top_k = 3):
    categorical_features = self._eda.categorical_features()
    plt.figure(figsize = (20,20))

    for feature_idx,feature in enumerate(categorical_features):

      feats = self._eda.group_by(feature)[self.target_feature].sort_values(ascending = False)[:top_k].index

      data = self.df
      data = data[data[feature].isin(feats)]

      plt.subplot(len(categorical_features) // 2 + 1 , 2 , feature_idx + 1)
      sns.stripplot(data = data, x = feature, y = self.target_feature,hue = feature)
      plt.title(f'Relation between {feature} (top {top_k}) & {self.target_feature}')

    plt.tight_layout()
    plt.show() 