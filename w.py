#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# load data
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

#%% 1st model : tourney based only
# based on seed diff
def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

seeds['Seed'] = seeds['Seed'].apply(seed_to_int)

wseeds = seeds.rename(columns={'TeamID':'WTeamID', 'Seed':'WSeed'})
lseeds = seeds.rename(columns={'TeamID':'LTeamID', 'Seed':'LSeed'})
seeds = pd.merge(left=tourney, right=wseeds, how='left', on=['Season', 'WTeamID'])
seeds = pd.merge(left=seeds, right=lseeds, how='left', on=['Season', 'LTeamID'])
seeds['SeedDiff'] = seeds['WSeed'] - seeds['LSeed']
seeds['Result'] = 1
seeds.drop(['DayNum','WScore','LScore','WLoc','NumOT','WSeed','LSeed'], axis=1, inplace=True)

for i, row in seeds.iterrows():
    if row['WTeamID'] > row['LTeamID']:
        wtid = row['WTeamID']
        row['WTeamID'] = row['LTeamID']
        row['LTeamID'] = wtid
        row['SeedDiff'] *= -1
        row['Result'] = 0

print(seeds)

wins = seeds.loc[:,['Season','SeedDiff']]
wins['Result'] = 1
losses = seeds.loc[:,['Season','SeedDiff']]
losses['SeedDiff'] *= -1
losses['Result'] = 0
data = pd.concat([wins, losses]).reset_index(drop=True)


    


def get_before(df, year):
    return df[df['Season']<year]

#%%
years = [2014,2015,2016,2017]
for year in years:
    train = get_before(data, year)
    X_train = train.loc[:,['SeedDiff']]
    y_train = train['Result']
    X_train, y_train = shuffle(X_train, y_train)

    print("training on seeds before", year)
    print("training on", X_train.shape[0], "samples")
    logreg = LogisticRegression()
    params = {'C': np.logspace(start=-5, stop=3, num=9)}
    clf = GridSearchCV(logreg, params, cv=5, scoring='neg_log_loss', refit=True)
    clf.fit(X_train, y_train)
    print('Best log_loss:', clf.best_score_)
    print('best C:', clf.best_params_['C'])

    







#%% number of w/l and pts scored/taken per season
pts = regseason.groupby(['Season','WTeamID'])['WScore'].sum().reset_index()
pts.rename(columns={'WTeamID':'TeamID', 'WScore':'WPtsSc'}, inplace = True)
pts['LPtsSc'] = regseason.groupby(['Season','LTeamID'])['LScore'].sum().reset_index()['LScore']
pts['WPtsTk'] = regseason.groupby(['Season','WTeamID'])['LScore'].sum().reset_index()['LScore']
pts['LPtsTk'] = regseason.groupby(['Season','LTeamID'])['WScore'].sum().reset_index()['WScore']
pts['Wct'] = regseason.groupby(['Season','WTeamID'])['WScore'].count().reset_index()['WScore']
pts['Lct'] = regseason.groupby(['Season','LTeamID'])['LScore'].count().reset_index()['LScore']

#%% mean pts scored and win ratio per season
pts['MPtsSc'] = (pts['WPtsSc'] + pts['LPtsSc']) / (pts['Wct'] + pts['Lct'])
pts['MPtsTk'] = (pts['WPtsTk'] + pts['LPtsTk']) / (pts['Wct'] + pts['Lct'])
pts['Wratio'] = pts['Wct'] / (pts['Wct'] + pts['Lct'])
pts.drop(['WPtsSc','WPtsTk','LPtsSc','LPtsTk','Wct','Lct'], axis=1, inplace=True)
print(pts.head())

#%% merge pts infos with regseason
wpts = pts.rename(columns={'TeamID':'WTeamID', 'MPtsSc':'WMPtsSc', 'MPtsTk':'WMPtsTk', 'Wratio':'WWratio'})
lpts = pts.rename(columns={'TeamID':'LTeamID', 'MPtsSc':'LMPtsSc', 'MPtsTk':'LMPtsTk', 'Wratio':'LWratio'})
regseason = pd.merge(left=regseason, right=wpts, how='left', on=['Season', 'WTeamID'])
regseason = pd.merge(left=regseason, right=lpts, how='left', on=['Season', 'LTeamID'])
print(regseason.head())



#lmscores = regseason.groupby(['Season','LTeamID'])['LScore'].mean().reset_index().rename(columns={'LScore':'LMScore'})
#regseason = pd.merge(left=regseason, right=wmscores, how='left', on=['Season', 'WTeamID'])
#regseason = pd.merge(left=regseason, right=lmscores, how='left', on=['Season', 'LTeamID'])
#regseason['MScoreDiff'] = regseason['WMScore'] - regseason['LMScore']

#print(regseason[regseason['MScoreDiff']<-16].head(10))
