import numpy as np
import pandas as pd


# --------------------------------- Load data ---------------------------------
def load_data():
    #cities = pd.read_csv('./data_w/WCities.csv')
    #gamecities = pd.read_csv('./data_w/WGameCities.csv')
    tourney = pd.read_csv('./data_m/NCAATourneyCompactResults.csv')
    seeds = pd.read_csv('./data_m/NCAATourneySeeds.csv')
    #slots = pd.read_csv('./data_w/WNCAATourneySlots.csv')
    regseason = pd.read_csv('./data_m/RegularSeasonCompactResults.csv')
    #seasons = pd.read_csv('./data_w/WSeasons.csv')
    #teamspellings = pd.read_csv('./data_w/WTeamSpellings.csv', engine='python')
    #teams = pd.read_csv('./data_w/WTeams.csv')
    sub = pd.read_csv('./data_m/SampleSubmissionStage2.csv')
    return tourney, regseason, seeds, sub


# --------------------------------- Formating ---------------------------------
# switch from WTeam / LTeam to sub format T1ID < T2ID
def format_results(df):
    df.rename(columns={'WTeamID':'WID','LTeamID':'LID'}, inplace=True)
    df['WResult'] = 1
    df['LResult'] = 0
    df.rename(columns={'WScore':'WPtsF','LScore':'LPtsF'}, inplace=True)
    df['WPtsA'] = df['LPtsF']
    df['LPtsA'] = df['WPtsF']
    df['LLoc'] = df['WLoc'].apply(lambda x: 'H' if x == 'A' else 'A' if x == 'H' else 'N')
    to_switch = ['ID', 'Result', 'PtsA', 'PtsF', 'Loc']
    for i, row in df.iterrows():
        if row['WID'] > row['LID']:
            for x in to_switch:
                df.set_value(i, 'W'+x, row['L'+x])
                df.set_value(i, 'L'+x, row['W'+x])
    new_cols = {'W'+x: 'T1'+x for x in to_switch}
    new_cols.update({'L'+x: 'T2'+x for x in to_switch})
    df.rename(columns=new_cols, inplace=True)
    return df

#
def sub_to_test(sub):
    sub['Season'], sub['T1ID'], sub['T2ID'] = sub['ID'].str.split('_').str
    sub['Season'] = sub['Season'].astype('int64')
    sub['T1ID'] = sub['T1ID'].astype('int64')
    sub['T2ID'] = sub['T2ID'].astype('int64')
    return sub

def make_sub(sub, pred):
    sub['Pred'] = pred
    sub[['ID','Pred']].to_csv('sub_m_b.csv', sep=',', index=False)    

# switch from T1 / T2 to Home / Away
def convert_home_away(df):
    new_df = df.copy()
    for i, row in new_df.iterrows():
        to_switch = ['ID', 'Result', 'PtsA', 'PtsF', 'Loc']
        if row['T1Loc'] == 'A':
            for x in to_switch:
                new_df.set_value(i, 'T1'+x, row['T2'+x])
                new_df.set_value(i, 'T2'+x, row['T1'+x])
    new_cols = {'T1'+x: 'HT'+x for x in to_switch}
    new_cols.update({'T2'+x: 'AT'+x for x in to_switch})   
    new_df.rename(columns=new_cols, inplace=True)
    return new_df

# double dataset and switch T1 <-> T2
def symmetrize(df):
    sym = df.copy()
    to_switch = ['ID', 'Result', 'PtsA', 'PtsF', 'Loc']
    for x in to_switch:
        sym['T1'+x] = df['T2'+x]
        sym['T2'+x] = df['T1'+x]
    return pd.concat([df, sym]).reset_index(drop=True)

# seeds to int (don't use region)
def seed_to_int(seed):
    s_int = int(seed[1:3])
    return s_int

# ------------------------------ Stats extraction -----------------------------
# Rating : if t1 has r1 and t2 has r2, expected score will be r1 - r2
def calc_season_ratings(games):
    games = games.reset_index(drop=True)
    x = games['HTLoc'].apply(lambda x: 0 if x == 'N' else 1)   # = 0 if game was played on neutral court, 1 otherwise
    y = games['HTPtsF'] - games['ATPtsF']   # score diff between hometeam & awayteam
    U = pd.get_dummies(games['HTID'])
    V = pd.get_dummies(games['ATID']) * (-1)
    W = U.add(V, fill_value=0)   # W(game,team) = 1 if team is hometeam of game, -1 if team is awayteam, 0 otherwise

    ratings = pd.DataFrame(W.columns.values, columns=['TeamID'])

    W.insert(0, 'H', x)   
    W.loc['c',:] = 1
    y.loc['c'] = 0               # sum of ratings

    A = W.T.dot(W)               # W.r = y -> W.T.W.r = W.T.y
    b = W.T.dot(y)               #
    r = np.linalg.inv(A).dot(b)  # A.r = b -> r = A^-1.b

    r = np.delete(r,0)       # r[0] = season homecourt advantage
        
    ratings['Rating'] = r.reshape(-1,1)
    return ratings

def calc_ratings(games, seasons):
    ratings = pd.DataFrame()
    for season in seasons:
        r = calc_season_ratings(convert_home_away(games[games['Season']==season]))
        r['Season'] = season
        ratings = ratings.append(r, ignore_index=True)
    return ratings

# pts scored/taken ; w/l ...
def build_agg(stats, idx):
    agg, dic = {}, {}
    for stat in stats:
        agg.update({idx+stat[0]: stat[1]})
        for i, name in enumerate(stat[2]):
            dic.update({idx+stat[0]+'_'+stat[1][i]: idx+name})
    return agg, dic

def group_stats(from_df, stats, idx):
    agg, dic = build_agg(stats, idx)
    team_stats = from_df.groupby(['Season',idx+'ID']).agg(agg)
    team_stats.columns = ["_".join(x) for x in team_stats.columns.ravel()]
    team_stats.reset_index(inplace=True)
    team_stats.rename(columns={idx+'ID':'TeamID'}, inplace=True)
    team_stats.rename(columns=dic, inplace=True)
    return team_stats

EXPONENT = 1.5
def compute_stat(df, operation, cols):
    if operation == 'mean':
        return (df['T1'+cols[0]] + df['T2'+cols[0]]) / (df['T1Games'] + df['T2Games'])
    if operation == 'mean_delta':
        return compute_stat(df, 'mean', [cols[0]]) - compute_stat(df, 'mean', [cols[1]])
    if operation == 'delta':
        return 2*(df['T1'+cols[0]] + df['T2'+cols[0]]) - (df['T1Games'] + df['T2Games'])
    if operation == 'mean_pyt':
        return 1 / (1 + ((df['T1'+cols[1]] + df['T2'+cols[1]]) / (df['T1'+cols[0]] + df['T2'+cols[0]]))**EXPONENT)
    if operation == 'sum':
        return (df['T1'+cols[0]] + df['T2'+cols[0]])

def combine_stats(of_df, stats, combined_stats):
    t1stats = group_stats(of_df, stats, 'T1')
    t2stats = group_stats(of_df, stats, 'T2')
    team_stats = pd.merge(left=t1stats, right=t2stats, how='outer', on=['Season', 'TeamID'])
    team_stats.fillna(0, inplace=True)
    for stat in combined_stats:
        team_stats[stat[0]] = compute_stat(team_stats, stat[1], stat[2])
    team_stats.drop(['T1'+stat for val in stats for stat in val[2]]+['T2'+stat for val in stats for stat in val[2]], axis=1, inplace=True)
    return team_stats

# merge df with team stats on team1 and team2
def merge_team_stats(df, stats_df, stats):
    if isinstance(stats, str):
        stats = [stats]
    t1stats =  stats_df.rename(columns=dict({'TeamID':'T1ID'}, **{stat:'T1'+stat for stat in stats}))
    t2stats =  stats_df.rename(columns=dict({'TeamID':'T2ID'}, **{stat:'T2'+stat for stat in stats}))
    df = pd.merge(left=df, right=t1stats, how='left', on=['Season', 'T1ID'])
    df = pd.merge(left=df, right=t2stats, how='left', on=['Season', 'T2ID'])
    for stat in stats:
        df[stat+'Diff'] = df['T1'+stat] - df['T2'+stat]
        df[stat+'Ratio'] = df['T1'+stat] / df['T2'+stat]
    return df

# does all the stuff
def preprocess_data(seasons):
    tourney, regseason, seeds, sub = load_data()

    tourney = format_results(tourney)
    regseason = format_results(regseason)
    sub = sub_to_test(sub)

    tourney = symmetrize(tourney)

    seeds['Seed'] = seeds['Seed'].apply(seed_to_int)
    tourney = merge_team_stats(tourney, seeds, 'Seed')
    sub = merge_team_stats(sub, seeds, 'Seed')

    team_ratings = calc_ratings(regseason, seasons)
    tourney = merge_team_stats(tourney, team_ratings, 'Rating')
    sub = merge_team_stats(sub, team_ratings, 'Rating')

    first_half = regseason.loc[regseason['DayNum']<=93]
    second_half = regseason.loc[regseason['DayNum']>53]
    fh_ratings = calc_ratings(first_half, seasons)
    fh_ratings.rename(columns={'Rating':'FHRating'}, inplace=True)
    tourney = merge_team_stats(tourney, fh_ratings, 'FHRating')
    sub = merge_team_stats(sub, fh_ratings, 'FHRating')
    sh_ratings = calc_ratings(second_half, seasons)
    sh_ratings.rename(columns={'Rating':'SHRating'}, inplace=True)
    tourney = merge_team_stats(tourney, sh_ratings, 'SHRating')
    sub = merge_team_stats(sub, sh_ratings, 'SHRating')


    stats = [
        ('PtsF', ['sum'], ['PtsFor']),          
        ('PtsA', ['sum'], ['PtsAgainst']),
        ('Result', ['count','sum'], ['Games','Wins'])
    ]

    combined_stats = [
        ('PtsFor', 'mean', ['PtsFor']),
        ('PtsAgainst', 'mean', ['PtsAgainst']),
        ('PtsDelta', 'mean_delta', ['PtsFor', 'PtsAgainst']),
        ('WRatio', 'mean', ['Wins']),
        ('WDelta', 'delta', ['Wins']),
        ('WPyt', 'mean_pyt', ['PtsFor', 'PtsAgainst']),
        ('Games', 'sum', ['Games'])
    ]

    team_stats = combine_stats(regseason, stats, combined_stats)
    tourney = merge_team_stats(tourney, team_stats, [x[0] for x in combined_stats])
    regseason = merge_team_stats(regseason, team_stats, [x[0] for x in combined_stats])
    sub = merge_team_stats(sub, team_stats, [x[0] for x in combined_stats])

    def weighted_result(r, wr, wrOpp):
        return r*((1+wrOpp)/(1+wr)) + (1-r)*((1+wrOpp)/(1+wr) - 1)

    regseason['T1WResult'] = weighted_result(regseason['T1Result'], regseason['T1WRatio'], regseason['T2WRatio'])
    regseason['T2WResult'] = weighted_result(regseason['T2Result'], regseason['T2WRatio'], regseason['T1WRatio'])

    stats = [
        ('WResult', ['count','sum'], ['Games','WWins'])
    ]

    combined_stats = [
        ('WWRatio', 'mean', ['WWins'])
    ]

    team_stats_2 = combine_stats(regseason, stats, combined_stats)
    tourney = merge_team_stats(tourney, team_stats_2, [x[0] for x in combined_stats])
    regseason = merge_team_stats(regseason, team_stats_2, [x[0] for x in combined_stats])
    sub = merge_team_stats(sub, team_stats_2, [x[0] for x in combined_stats])

    return tourney, regseason, sub



