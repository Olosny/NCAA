#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

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


# --------------------------------- Formating ---------------------------------
# format tourney and regseason results
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
tourney['SeedDiff'] = tourney['T1Seed'] - tourney['T2Seed']

# extract stats from reg season
# pts scored/taken ; w/l ...
t1pts = regseason.groupby(['Season','T1ID'])['T1Score'].sum().reset_index()
t1pts.rename(columns={'T1ID':'TeamID','T1Score':'T1PtsSc'}, inplace = True)
t1pts['T1PtsTk'] = regseason.groupby(['Season','T1ID'])['T2Score'].sum().reset_index()['T2Score']
t1pts['T1ct'] = regseason.groupby(['Season','T1ID'])['T1Score'].count().reset_index()['T1Score']
t1pts['T1Wins'] = regseason.groupby(['Season','T1ID'])['Result'].sum().reset_index()['Result']

t2pts = regseason.groupby(['Season','T2ID'])['T2Score'].sum().reset_index()
t2pts.rename(columns={'T2ID':'TeamID','T2Score':'T2PtsSc'}, inplace = True)
t2pts['T2PtsTk'] = regseason.groupby(['Season','T2ID'])['T1Score'].sum().reset_index()['T1Score']
t2pts['T2ct'] = regseason.groupby(['Season','T2ID'])['T2Score'].count().reset_index()['T2Score']
t2pts['T2Wins'] = regseason.groupby(['Season','T2ID'])['Result'].sum().reset_index()['Result']
t2pts['T2Wins'] = t2pts['T2ct'] - t2pts['T2Wins']

pts = pd.merge(left=t1pts, right=t2pts, how='outer', on=['Season', 'TeamID'])
pts.fillna(0, inplace=True)

pts['MPtsSc'] = (pts['T1PtsSc'] + pts['T2PtsSc']) / (pts['T1ct'] + pts['T2ct'])
pts['MPtsTk'] = (pts['T1PtsTk'] + pts['T2PtsTk']) / (pts['T1ct'] + pts['T2ct'])
pts['Wratio'] = (pts['T1Wins'] + pts['T2Wins']) / (pts['T1ct'] + pts['T2ct'])
pts['Wdelta'] = 2*(pts['T1Wins'] + pts['T2Wins']) - (pts['T1ct'] + pts['T2ct'])
pts.drop(['T1PtsSc','T1PtsTk','T2PtsSc','T2PtsTk','T1ct','T2ct','T1Wins','T2Wins'], axis=1, inplace=True)

t1pts = pts.rename(columns={'TeamID':'T1ID', 'MPtsSc':'T1MPtsSc', 'MPtsTk':'T1MPtsTk', 'Wratio':'T1Wratio', 'Wdelta':'T1Wdelta'})
t2pts = pts.rename(columns={'TeamID':'T2ID', 'MPtsSc':'T2MPtsSc', 'MPtsTk':'T2MPtsTk', 'Wratio':'T2Wratio', 'Wdelta':'T2Wdelta'})
regseason = pd.merge(left=regseason, right=t1pts, how='left', on=['Season', 'T1ID'])
regseason = pd.merge(left=regseason, right=t2pts, how='left', on=['Season', 'T2ID'])

regseason['MPtsScDiff'] = regseason['T1MPtsSc'] - regseason['T2MPtsSc']
regseason['MPtsTkDiff'] = regseason['T1MPtsTk'] - regseason['T2MPtsTk']
regseason['WratioDiff'] = regseason['T1Wratio'] - regseason['T2Wratio']
regseason['WdeltaDiff'] = regseason['T1Wdelta'] - regseason['T2Wdelta']

tourney = pd.merge(left=tourney, right=t1pts, how='left', on=['Season', 'T1ID'])
tourney = pd.merge(left=tourney, right=t2pts, how='left', on=['Season', 'T2ID'])

tourney['MPtsScDiff'] = tourney['T1MPtsSc'] - tourney['T2MPtsSc']
tourney['MPtsTkDiff'] = tourney['T1MPtsTk'] - tourney['T2MPtsTk']
tourney['WratioDiff'] = tourney['T1Wratio'] - tourney['T2Wratio']
tourney['WdeltaDiff'] = tourney['T1Wdelta'] - tourney['T2Wdelta']

print(tourney.describe())

# ----------------------------- Validation metric -----------------------------
def val_score(y_pred, y_true):
    return -(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)).mean()


# --------------------------------- 1st model ---------------------------------
# tourney based only. based on seed diff
print("Tourney seeds based model")

data = tourney.loc[:,['Season','T1ID','T2ID','SeedDiff','Result']]
train1 = data.loc[:,['Season','SeedDiff','Result']]
train2 = data.loc[:,['Season','SeedDiff','Result']]
train2['SeedDiff'] *= -1
train2['Result'] = 1 - train2['Result']
full_train = pd.concat([train1, train2]).reset_index(drop=True)
full_test = data[(data['Season']>=2014) & (data['Season']<=2017)]

# test each season based on other seasons
vals = []
years = [2014,2015,2016,2017]
for year in years:
    train = full_train[full_train['Season']!=year]
    
    X_train = train[['SeedDiff']]
    y_train = train['Result']
    X_train, y_train = shuffle(X_train, y_train) 
    
    test = full_test[full_test['Season']==year]
    X_test = test[['SeedDiff']]
    y_test = test['Result']

    logreg = LogisticRegression()
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(logreg, params, cv=10, scoring='neg_log_loss', refit=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    print("["+str(year)+"] score:", val_score(y_pred,y_test))
    vals.append(val_score(y_pred,y_test))
print("[All] score:", 1/4*sum(vals))


# --------------------------------- 2nd model ---------------------------------
# based on prev regular season. Pts scored/taken diff, winrate diff
print("Reg season stats based model")

data = tourney.loc[:,['Season','T1ID','T2ID','MPtsScDiff','MPtsTkDiff','WratioDiff','WdeltaDiff','Result']]
train1 = data.loc[:,['Season','MPtsScDiff','MPtsTkDiff','WratioDiff','WdeltaDiff','Result']]
train2 = data.loc[:,['Season','MPtsScDiff','MPtsTkDiff','WratioDiff','WdeltaDiff','Result']]
train2['MPtsScDiff'] *= -1
train2['MPtsTkDiff'] *= -1
train2['WratioDiff'] *= -1
train2['WdeltaDiff'] *= -1
train2['Result'] = 1 - train2['Result']
full_train = pd.concat([train1, train2]).reset_index(drop=True)
full_test = data[(data['Season']>=2014) & (data['Season']<=2017)]

#features = ['MPtsScDiff','MPtsTkDiff','WratioDiff']
features = ['MPtsScDiff','MPtsTkDiff','WdeltaDiff']
#features = ['MPtsScDiff','MPtsTkDiff']
#features = ['WratioDiff']
#features = ['WdeltaDiff']

# test each season based on other seasons
vals = []
years = [2014,2015,2016,2017]
for year in years:
    train = full_train[full_train['Season']!=year]
    
    X_train = train[features]
    y_train = train['Result']
    X_train, y_train = shuffle(X_train, y_train) 
    
    test = full_test[full_test['Season']==year]
    X_test = test[features]
    y_test = test['Result']

    logreg = LogisticRegression()
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(logreg, params, cv=10, scoring='neg_log_loss', refit=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:,1]
    print("["+str(year)+"] score:", val_score(y_pred,y_test))
    vals.append(val_score(y_pred,y_test))
print("[All] score:", 1/4*sum(vals))