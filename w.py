# todo : add sort of season performance history per team instead of mean 
#        combine model trained on tourney and model trained on regseason
#        ensemble, blend, stack...

# Remarks : trees works ok for 2014 2015 but bad for 2016 2017
#           SVM also...


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# --------------------------------- Load data ---------------------------------
cities = pd.read_csv('./data_w/WCities.csv')
gamecities = pd.read_csv('./data_w/WGameCities.csv')
tourney = pd.read_csv('./data_w/WNCAATourneyCompactResults.csv')
seeds = pd.read_csv('./data_w/WNCAATourneySeeds.csv')
slots = pd.read_csv('./data_w/WNCAATourneySlots.csv')
regseason = pd.read_csv('./data_w/WRegularSeasonCompactResults.csv')
seasons = pd.read_csv('./data_w/WSeasons.csv')
teamspellings = pd.read_csv('./data_w/WTeamSpellings.csv', engine='python')
teams = pd.read_csv('./data_w/WTeams.csv')
sub = pd.read_csv('./data_w/WSampleSubmissionStage1.csv')


# ----------------------------- Validation metric -----------------------------
def val_score(y_true, y_pred):
    return -(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)).mean()


# ----------------------------------- Ballzy ----------------------------------
def better_conf(x, c=1):
    ''' c > 1 -> increase confidence of result
        c < 1 -> decrease ...
    '''
    if x == 0:
        return 0
    return 1 / (1 + (-1 + 1 / x)**c)
    #return 1/2 + np.arctan(c*np.tan(np.pi*(x-1/2)))/np.pi


# --------------------------------- Formating ---------------------------------
# format tourney and regseason results
# switch from WTeam / LTeam to sub format T1ID < T2ID
def format_results(df):
    df['Result'] = 1
    for i, row in df.iterrows():
        if row['WTeamID'] > row['LTeamID']:
            df.set_value(i, 'WTeamID', row['LTeamID'])
            df.set_value(i, 'LTeamID', row['WTeamID'])
            df.set_value(i, 'Result', 0)
            df.set_value(i, 'WScore', row['LScore'])
            df.set_value(i, 'LScore', row['WScore'])
            if row['WLoc'] == 'H':
                df.set_value(i, 'WLoc', 'A')
            if row['WLoc'] == 'A':
                df.set_value(i, 'WLoc', 'H')
    df.rename(columns={'WTeamID':'T1ID','LTeamID':'T2ID'}, inplace=True)
    df.rename(columns={'WScore':'T1Score','LScore':'T2Score'}, inplace=True)
    df.rename(columns={'WLoc':'T1Loc'}, inplace=True)

format_results(tourney)
format_results(regseason)

# merge seeds & tourney results
def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

seeds['Seed'] = seeds['Seed'].apply(seed_to_int)
t1seeds = seeds.rename(columns={'TeamID':'T1ID', 'Seed':'T1Seed'})
t2seeds = seeds.rename(columns={'TeamID':'T2ID', 'Seed':'T2Seed'})
tourney = pd.merge(left=tourney, right=t1seeds, how='left', on=['Season', 'T1ID'])
tourney = pd.merge(left=tourney, right=t2seeds, how='left', on=['Season', 'T2ID'])
tourney['SeedDiff'] = -(tourney['T1Seed'] - tourney['T2Seed'])

# extract stats from reg season
# pts scored/taken ; w/l ...
t1stats = regseason.groupby(['Season','T1ID'])['T1Score'].sum().reset_index()
t1stats.rename(columns={'T1ID':'TeamID','T1Score':'T1PtsSc'}, inplace = True)
t1stats['T1PtsTk'] = regseason.groupby(['Season','T1ID'])['T2Score'].sum().reset_index()['T2Score']
t1stats['T1ct'] = regseason.groupby(['Season','T1ID'])['T1Score'].count().reset_index()['T1Score']
t1stats['T1Wins'] = regseason.groupby(['Season','T1ID'])['Result'].sum().reset_index()['Result']

t2stats = regseason.groupby(['Season','T2ID'])['T2Score'].sum().reset_index()
t2stats.rename(columns={'T2ID':'TeamID','T2Score':'T2PtsSc'}, inplace = True)
t2stats['T2PtsTk'] = regseason.groupby(['Season','T2ID'])['T1Score'].sum().reset_index()['T1Score']
t2stats['T2ct'] = regseason.groupby(['Season','T2ID'])['T2Score'].count().reset_index()['T2Score']
t2stats['T2Wins'] = regseason.groupby(['Season','T2ID'])['Result'].sum().reset_index()['Result']
t2stats['T2Wins'] = t2stats['T2ct'] - t2stats['T2Wins']

stats = pd.merge(left=t1stats, right=t2stats, how='outer', on=['Season', 'TeamID'])
stats.fillna(0, inplace=True)

stats['MPtsSc'] = (stats['T1PtsSc'] + stats['T2PtsSc']) / (stats['T1ct'] + stats['T2ct'])
stats['MPtsTk'] = (stats['T1PtsTk'] + stats['T2PtsTk']) / (stats['T1ct'] + stats['T2ct'])
stats['MdeltaPts'] = stats['MPtsSc'] - stats['MPtsTk']
stats['Wratio'] = (stats['T1Wins'] + stats['T2Wins']) / (stats['T1ct'] + stats['T2ct'])
stats['Wdelta'] = 2*(stats['T1Wins'] + stats['T2Wins']) - (stats['T1ct'] + stats['T2ct'])
stats.drop(['T1PtsSc','T1PtsTk','T2PtsSc','T2PtsTk','T1ct','T2ct','T1Wins','T2Wins'], axis=1, inplace=True)

# merge team stats with tourney & regseason
stats_list = ['MPtsSc','MPtsTk','MdeltaPts','Wratio','Wdelta']
t1stats = stats.rename(columns=dict({'TeamID':'T1ID'}, **{stat:'T1'+stat for stat in stats_list}))
t2stats = stats.rename(columns=dict({'TeamID':'T2ID'}, **{stat:'T2'+stat for stat in stats_list}))

tourney = pd.merge(left=tourney, right=t1stats, how='left', on=['Season', 'T1ID'])
tourney = pd.merge(left=tourney, right=t2stats, how='left', on=['Season', 'T2ID'])

for stat in stats_list:
    tourney[stat+'Diff'] = tourney['T1'+stat] - tourney['T2'+stat]

regseason = pd.merge(left=regseason, right=t1stats, how='left', on=['Season', 'T1ID'])
regseason = pd.merge(left=regseason, right=t2stats, how='left', on=['Season', 'T2ID'])

for stat in stats_list:
    regseason[stat+'Diff'] = regseason['T1'+stat] - regseason['T2'+stat]

# Extract weighted win ratio and merge
regseason['T1WResult'] = regseason['Result'] * ((1+regseason['T2Wratio']) / (1+regseason['T1Wratio'])) + (1 - regseason['Result']) * ((1+regseason['T2Wratio']) / (1+regseason['T1Wratio']) - 1)
regseason['T2WResult'] = (1 - regseason['Result']) * ((1+regseason['T1Wratio']) / (1+regseason['T2Wratio'])) + regseason['Result'] * ((1+regseason['T1Wratio']) / (1+regseason['T2Wratio']) - 1)

t1stats = regseason.groupby(['Season','T1ID'])['T1WResult'].sum().reset_index()
t1stats.rename(columns={'T1ID':'TeamID'}, inplace = True)
t1stats['T1ct'] = regseason.groupby(['Season','T1ID'])['T1WResult'].count().reset_index()['T1WResult']

t2stats = regseason.groupby(['Season','T2ID'])['T2WResult'].sum().reset_index()
t2stats.rename(columns={'T2ID':'TeamID'}, inplace = True)
t2stats['T2ct'] = regseason.groupby(['Season','T2ID'])['T2WResult'].count().reset_index()['T2WResult']

stats = pd.merge(left=t1stats, right=t2stats, how='outer', on=['Season', 'TeamID'])
stats.fillna(0, inplace=True)

stats['WWratio'] = (stats['T1WResult'] + stats['T2WResult']) / (stats['T1ct'] + stats['T2ct'])
stats.drop(['T1WResult','T2WResult','T1ct','T2ct'], axis=1, inplace=True)

stats_list = ['WWratio']
t1stats = stats.rename(columns=dict({'TeamID':'T1ID'}, **{stat:'T1'+stat for stat in stats_list}))
t2stats = stats.rename(columns=dict({'TeamID':'T2ID'}, **{stat:'T2'+stat for stat in stats_list}))

tourney = pd.merge(left=tourney, right=t1stats, how='left', on=['Season', 'T1ID'])
tourney = pd.merge(left=tourney, right=t2stats, how='left', on=['Season', 'T2ID'])

for stat in stats_list:
    tourney[stat+'Diff'] = tourney['T1'+stat] - tourney['T2'+stat]

regseason = pd.merge(left=regseason, right=t1stats, how='left', on=['Season', 'T1ID'])
regseason = pd.merge(left=regseason, right=t2stats, how='left', on=['Season', 'T2ID'])

for stat in stats_list:
    regseason[stat+'Diff'] = regseason['T1'+stat] - regseason['T2'+stat]

#%%
print(regseason.count())

#%% --------------------------------- Obs -------------------------------------
#tourney.plot(x='SeedDiff', y='Result', kind='scatter')
#plt.show()
pca = PCA(n_components=2)
features = ['MPtsScDiff','MPtsTkDiff','MdeltaPtsDiff','WratioDiff','WdeltaDiff','SeedDiff','WWratioDiff']
X = tourney.loc[:,features]
y = tourney['Result']
X.loc[:,features] = StandardScaler().fit_transform(X[features])
X_r = pca.fit_transform(X)
for color, i, target_name in zip(['green','red'], [0, 1], ['win','loss']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
plt.figure()

#%%
pca = PCA(n_components=2)
features = ['MPtsScDiff','MPtsTkDiff','MdeltaPtsDiff','WratioDiff','WdeltaDiff','SeedDiff','WWratioDiff']
data = tourney[tourney['Season']==1999]
X = data.loc[:,features]
y = data['Result']
X.loc[:,features] = StandardScaler().fit_transform(X[features])
X_r = pca.fit_transform(X)
for color, i, target_name in zip(['green','red'], [0, 1], ['win','loss']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
plt.figure()

#%%
pca = PCA(n_components=2)
features = ['MPtsScDiff','MPtsTkDiff','MdeltaPtsDiff','WratioDiff','WdeltaDiff','WWratioDiff']
X = regseason.loc[:,features]
y = regseason['Result']
X.loc[:,features] = StandardScaler().fit_transform(X[features])
#X = MinMaxScaler().fit_transform(X)
#X = MaxAbsScaler().fit_transform(X)
X_r = pca.fit_transform(X)
for color, i, target_name in zip(['green','red'], [0, 1], ['win','loss']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
plt.figure()

#%%
pca = PCA(n_components=2)
features = ['MPtsScDiff','MPtsTkDiff','MdeltaPtsDiff','WratioDiff','WdeltaDiff','WWratioDiff']
data = regseason[regseason['Season']==2017]
X = data.loc[:,features]
y = data['Result']
X.loc[:,features] = StandardScaler().fit_transform(X[features])
#X = MinMaxScaler().fit_transform(X)
#X = MaxAbsScaler().fit_transform(X)
X_r = pca.fit_transform(X)
for color, i, target_name in zip(['green','red'], [0, 1], ['win','loss']):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
plt.figure()


# ---------------------------------- Model 1 ----------------------------------
#%% trained on tourney results 
# based on prev regular season stats + seed diff.
print("trained on tourney")

features = ['MPtsScDiff','MPtsTkDiff','MdeltaPtsDiff','WratioDiff','WdeltaDiff','SeedDiff','WWratioDiff']

def get_train_test_tourney(features, year):
    data = tourney.loc[:,['Season','Result']+features]
    sym_data = data.copy()
    for feature in features:
        sym_data[feature] *= -1
    sym_data['Result'] = 1 - sym_data['Result']
    train = pd.concat([data, sym_data]).reset_index(drop=True)
    train = train[train['Season']!=year]
    test = data.copy()
    test = test[test['Season']==year]
    scaler = StandardScaler()
    train.loc[:,features] = scaler.fit_transform(train[features])
    test.loc[:,features] = scaler.transform(test[features])

    X_train = train.drop(['Result'], axis=1)
    y_train = train['Result']
    X_test = test.drop(['Result'], axis=1)
    y_test = test['Result']
    return X_train, X_test, y_train, y_test

vals = []
years = [2014,2015,2016,2017]
for year in years:
    X_train, X_test, y_train, y_test = get_train_test_tourney(features, year)
    
    estimator = LogisticRegression()
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(estimator, params, cv=10, scoring='neg_log_loss', refit=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    
    #estimator = Ridge(alpha=0)
    #estimator.fit(X_train, y_train)
    #y_pred = estimator.predict(X_test)
    #y_pred = np.clip(y_pred,0.001,0.999)
    
    #estimator = GradientBoostingClassifier(n_estimators=50)
    #params = {'max_depth':np.arange(1,4,1), 'min_samples_split':np.arange(2,12,2), 'learning_rate':np.arange(0.16,0.24,0.02)}
    #clf = GridSearchCV(estimator, params, cv=10, scoring='neg_log_loss', refit=True)
    #clf.fit(X_train, y_train)
    #print(clf.best_params_)
    #y_pred = clf.predict_proba(X_test)[:,1]
    #y_pred = np.clip(y_pred,0.01,0.99)

    #estimator = SVC(C=0.01, kernel='sigmoid', gamma='auto', shrinking=True, probability=True, cache_size=1000)
    #params = {'gamma':np.logspace(start=-2, stop=2, num=5),'coef0':np.logspace(start=-2, stop=2, num=5)}
    #clf = GridSearchCV(estimator, params, cv=10, scoring='neg_log_loss', refit=True)
    #clf.fit(X_train, y_train)
    #print(clf.best_params_)
    #y_pred = clf.predict_proba(X_test)[:,1]
    #y_pred = np.clip(y_pred,0.01,0.99)

    print("["+str(year)+"] score:", val_score(y_test,y_pred))

    min_val = 1
    best_c = 0
    for c in np.arange(0.5,2,0.01):
        f = np.vectorize(lambda x: better_conf(x,c))
        y = f(y_pred)
        if val_score(y_test,y) < min_val:
            min_val = val_score(y_test,y)
            best_c = c
    print("["+str(year)+"] c:", best_c, "|", "score:", min_val)
    
    vals.append(val_score(y_test,y_pred))
print("[All] score:", 1/4*sum(vals))


# ---------------------------------- Model 2 ----------------------------------
#%% trained on regseason results 
# based on regular season stats
print("trained on regseason")

features = ['MPtsScDiff','MPtsTkDiff','MdeltaPtsDiff','WratioDiff','WdeltaDiff','WWratioDiff']

def get_train_test_regseason(features, year):
    data = regseason.loc[:,['Season','Result','T1Loc']+features]
    sym_data = data.copy()
    for feature in features:
        sym_data[feature] *= -1
    sym_data['T1Loc'].apply(lambda x: 'H' if x=='A' else 'A' if x=='H' else x)
    sym_data['Result'] = 1 - sym_data['Result']
    train = pd.concat([data, sym_data]).reset_index(drop=True)
    train = train[train['Season']!=year]
    train = pd.get_dummies(train, columns=['T1Loc'], prefix='', prefix_sep='')
    test = tourney.loc[:,['Season','Result','T1Loc']+features]
    test = pd.get_dummies(test, columns=['T1Loc'], prefix='', prefix_sep='')
    test = test[test['Season']==year]
    scaler = StandardScaler()
    train.loc[:,features] = scaler.fit_transform(train[features])
    test.loc[:,features] = scaler.transform(test[features])

    X_train = train.drop(['Result'], axis=1)
    y_train = train['Result']
    X_test = test.drop(['Result'], axis=1)
    y_test = test['Result']
    return X_train, X_test, y_train, y_test

vals = []
years = [2014,2015,2016,2017]
for year in years:
    X_train, X_test, y_train, y_test = get_train_test_regseason(features, year)
    
    estimator = LogisticRegression()
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(estimator, params, cv=10, scoring='neg_log_loss', refit=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    
    #estimator = Ridge(alpha=0)
    #estimator.fit(X_train, y_train)
    #y_pred = estimator.predict(X_test)
    #y_pred = np.clip(y_pred,0.001,0.999)
    
    #estimator = GradientBoostingClassifier(n_estimators=50)
    #params = {'max_depth':np.arange(1,4,1), 'min_samples_split':np.arange(2,12,2), 'learning_rate':np.arange(0.16,0.24,0.02)}
    #clf = GridSearchCV(estimator, params, cv=10, scoring='neg_log_loss', refit=True)
    #clf.fit(X_train, y_train)
    #print(clf.best_params_)
    #y_pred = clf.predict_proba(X_test)[:,1]
    #y_pred = np.clip(y_pred,0.01,0.99)

    #estimator = SVC(C=0.01, kernel='sigmoid', gamma='auto', shrinking=True, probability=True, cache_size=1000)
    #params = {'gamma':np.logspace(start=-2, stop=2, num=5),'coef0':np.logspace(start=-2, stop=2, num=5)}
    #clf = GridSearchCV(estimator, params, cv=10, scoring='neg_log_loss', refit=True)
    #clf.fit(X_train, y_train)
    #print(clf.best_params_)
    #y_pred = clf.predict_proba(X_test)[:,1]
    #y_pred = np.clip(y_pred,0.01,0.99)

    print("["+str(year)+"] score:", val_score(y_test,y_pred))

    min_val = 1
    best_c = 0
    for c in np.arange(0.5,2,0.01):
        f = np.vectorize(lambda x: better_conf(x,c))
        y = f(y_pred)
        if val_score(y_test,y) < min_val:
            min_val = val_score(y_test,y)
            best_c = c
    print("["+str(year)+"] c:", best_c, "|", "score:", min_val)
    
    vals.append(val_score(y_test,y_pred))
print("[All] score:", 1/4*sum(vals))