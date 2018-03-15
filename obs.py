#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from data_preprocessing import preprocess_data

#%%
seasons = np.arange(1998,2018,1)
tourney, regseason, sub = preprocess_data(seasons)

# tourney columns:
# - unusable:
#   'Season', 'DayNum', 'T1ID', 'T2ID', 'T1PtsF', 'T2PtsF', 'T1PtsA', 'T2PtsA', 'T2Result', 'WDeltaRatio'
# - usable but meh: 
#   'NumOT', 'T1Games', 'T2Games', 'GamesDiff', 'GamesRatio', 'RatingRatio'
# - usable:
#   'T1Seed', 'T2Seed', 'SeedDiff', 'SeedRatio', 'T1Loc',
#   'T1Rating', 'T1PtsFor', 'T1PtsAgainst', 'T1PtsDelta', 'T1WRatio', 'T1WDelta', 'T1WPyt', 'T1WWRatio', 
#   'T2Rating', 'T2PtsFor', 'T2PtsAgainst', 'T2PtsDelta', 'T2WRatio', 'T2WDelta', 'T2WPyt', 'T2WWRatio', 
#   'RatingDiff', 'PtsForDiff','PtsForRatio', 'PtsAgainstDiff', 'PtsAgainstRatio', 'PtsDeltaDiff', 'PtsDeltaRatio',
#   'WRatioDiff', 'WRatioRatio', 'WDeltaDiff', 'WPytDiff', 'WPytRatio',
#   'WWRatioDiff', 'WWRatioRatio'
# - label:
#   'T1Result'


#%%
tourney.plot(x='SeedDiff', y='Result', kind='scatter')
plt.show()

#%% --------------------------------- Obs -------------------------------------

pca = PCA(n_components=2)
features = ['T1Rating','T2Rating']
X = tourney.loc[:,features]
y = tourney['T1Result']
X.loc[:,features] = StandardScaler().fit_transform(X[features])
X_r = pca.fit_transform(X)
for color, i, target_name in zip(['green','red'], [0, 1], ['win','loss']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
plt.show()


