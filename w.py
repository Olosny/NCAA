import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import preprocess_data, make_sub
from models import log_loss, better_conf, build_train_test, cross_train_predict, stacking_train_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

seasons = np.arange(1985,2019,1)
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

def predict(tourney, train_seasons, test_seasons, sub=None):
    train_preds = pd.DataFrame()
    test_preds = pd.DataFrame()

    # Logistic Regression
    features = ['RatingDiff', 'SeedDiff', 'PtsAgainstDiff', 'WRatioDiff', 'WDeltaDiff', 'WWRatioDiff']
    X_train, X_test, y_train, y_test = build_train_test(tourney, train_seasons, test_seasons, features, sub)
    logReg = LogisticRegression(C=0.55)
    train_pred, test_pred = cross_train_predict(logReg, 2, X_train, y_train, X_test)
    train_preds['logReg'] = train_pred
    test_preds['logReg'] = test_pred
    
    # GBR
    features = ['RatingDiff', 'SeedDiff']
    X_train, X_test, y_train, y_test = build_train_test(tourney, train_seasons, test_seasons, features, sub)
    gbr = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_features=0.8, subsample=1)
    train_pred, test_pred = cross_train_predict(gbr, 20, X_train, y_train, X_test)
    train_preds['gbr'] = train_pred
    test_preds['gbr'] = test_pred

    # NN1
    features = ['RatingDiff', 'SeedDiff', 'PtsAgainstDiff', 'WRatioDiff', 'WDeltaDiff', 'WWRatioDiff']
    X_train, X_test, y_train, y_test = build_train_test(tourney, train_seasons, test_seasons, features, sub)
    train_pred, test_pred = cross_train_predict('nn1', 20, X_train, y_train, X_test)
    train_preds['nn1'] = train_pred
    test_preds['nn1'] = test_pred

    # NN2
    #features = ['RatingDiff', 'SeedDiff', 'PtsAgainstDiff', 'WRatioDiff', 'WDeltaDiff', 'WWRatioDiff']
    #X_train, X_test, y_train, y_test = build_train_test(tourney, train_seasons, test_seasons, features, sub)
    #train_pred, test_pred = cross_train_predict('nn2', 20, X_train, y_train, X_test)
    #train_preds['nn2'] = train_pred
    #test_preds['nn2'] = test_pred

    # Stacking
    y_pred = stacking_train_predict(train_preds, y_train, test_preds, y_test, 'nn')

    y_pred = pd.Series(y_pred).apply(lambda x: better_conf(x, 1.2))

    y_pred = y_pred.values

    if sub is None:
        print('logReg | train:', log_loss(y_train, train_preds['logReg']),'| test:', log_loss(y_test, test_preds['logReg'].values))
        print('gbr | train:', log_loss(y_train, train_preds['gbr']),'| test:', log_loss(y_test, test_preds['gbr'].values))
        print('nn1 | train:', log_loss(y_train, train_preds['nn1']),'| test:', log_loss(y_test, test_preds['nn1'].values))
    #    print('nn2 | train:', log_loss(y_train, train_preds['nn2']),'| test:', log_loss(y_test, test_preds['nn2'].values))
    else:
        print('logReg | train:', log_loss(y_train, train_preds['logReg']))
        print('gbr | train:', log_loss(y_train, train_preds['gbr']))
        print('nn1 | train:', log_loss(y_train, train_preds['nn1']))
    #    print('nn2 | train:', log_loss(y_train, train_preds['nn2']))

    if sub is not None:
        make_sub(sub, y_pred)
        return 'done'

    else:
        return log_loss(y_test, y_pred)
    

test_seasons = [2018]
#train_seasons = list(set(seasons)-set(test_seasons))
train_seasons = [1989,1992,1993,1994,1995,1996,1999,2000,2004,2005,2007,2008,2009,2015,2017]
print(predict(tourney, train_seasons, test_seasons, sub))




def sub_lists(l, min_s=1, max_s=None):
    subs = []
    max_s = len(l) if max_s is None else min(max_s, len(l))
    min_s = max(min_s, 1)
    for size in np.arange(min_s,max_s+1,1):
        subs = subs + [list(x) for x in set(itertools.combinations(l, size))]
    return subs

def find_best_features(data, train_seasons, test_seasons, features):
    best_fs = None
    best_val = np.float('inf')
    for fs in sub_lists(features):
        X_train, X_test, y_train, y_test = build_train_test(data, train_seasons, test_seasons, fs)
        train_pred, test_pred = cross_train_predict('nn1', 10, X_train, y_train, X_test, y_test)
        val = log_loss(y_train, train_pred.values)
        if val < best_val:
            best_val = val
            best_fs = fs
    return best_fs, best_val


#test_seasons = [2014,2015,2016,2017]
#train_seasons = seasons
#features = ['RatingDiff', 'SeedDiff', 'PtsAgainstDiff', 'PtsDeltaDiff', 'WRatioDiff', 'WDeltaDiff', 'WPytDiff', 'WWRatioDiff', 'FHRatingDiff']
#best_fs, best_val = find_best_features(tourney, train_seasons, test_seasons, features)
#print(best_fs)
#print(best_val)
#print(predict(tourney, train_seasons, test_seasons, features))

