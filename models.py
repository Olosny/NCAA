import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.constraints import maxnorm


# ----------------------------- Validation metric -----------------------------
def log_loss(y_true, y_pred):
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


# ------------------------------------ nn -------------------------------------
def nn1(X_train, y_train, X_test, y_test=None):
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    opti = Adam(lr=0.01, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])
    validation_data = (X_test, y_test) if y_test is not None else None
    model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=0, validation_data=validation_data)
    y_pred = model.predict(X_test).transpose()[0]
    y_pred = np.clip(y_pred, 0.0001, 0.9999)
    return y_pred

def nn2(X_train, y_train, X_test, y_test=None):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='hard_sigmoid'))
    opti = Adam(lr=0.01, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])
    validation_data = (X_test, y_test) if y_test is not None else None
    model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=0, validation_data=validation_data)
    y_pred = model.predict(X_test).transpose()[0]
    y_pred = np.clip(y_pred, 0.0001, 0.9999)
    return y_pred


# --------------------------------- Train / Test ------------------------------
# split data into train test
def build_train_test(tourney, train_seasons, test_seasons, features, sub=None):
    train = tourney.loc[:,['Season','T1Result']+features][tourney['Season'].isin(train_seasons)]
    if sub is not None:
        test = sub.loc[:,['Season','T1Result']+features]
    else:
        test = tourney.loc[:,['Season','T1Result']+features][tourney['Season'].isin(test_seasons)]
    scaler = StandardScaler()
    train.loc[:,features] = scaler.fit_transform(train[features])
    test.loc[:,features] = scaler.transform(test[features])
    X_train = train.drop(['Season','T1Result'], axis=1)
    y_train = train['T1Result']
    X_test = test.drop(['Season','T1Result'], axis=1)
    y_test = test['T1Result']
    return X_train, X_test, y_train, y_test

def train_predict(estimator, X_train, y_train, X_test, y_test=None):
    if estimator == 'nn1':
        pred = nn1(X_train, y_train, X_test, y_test)
    elif estimator == 'nn2':
        pred = nn2(X_train, y_train, X_test, y_test)
    else:
        estimator.fit(X_train, y_train)
        pred = estimator.predict_proba(X_test)[:,1]
    return pred

def cross_train_predict(estimator, folds, X_train, y_train, X_test, y_test=None):
    train_pred = pd.Series(index=X_train.index)

    kf = KFold(n_splits = folds, shuffle=True)
    for train_index, val_index in kf.split(X_train):
        X_train_f = X_train.iloc[train_index,:]
        X_val_f = X_train.iloc[val_index,:]
        y_train_f = y_train.iloc[train_index]
        y_val_f = y_train.iloc[val_index]

        fold_pred = train_predict(estimator, X_train_f, y_train_f, X_val_f, y_val_f)
        train_pred.iloc[val_index] = fold_pred

    test_pred = train_predict(estimator, X_train, y_train, X_test, y_test)
    return train_pred, test_pred

def stacking_train_predict(X_train, y_train, X_test, y_test=None, estimator='average'):
    if estimator == 'average':
        y_pred = X_test.apply('mean', axis=1)
    
    if estimator == 'logreg':
        model = LogisticRegression()
        params = {'C': np.arange(0.05,5,0.05)}
        clf = GridSearchCV(model, params, cv=10, scoring='neg_log_loss', refit=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:,1]

    if estimator == 'nn':
        model = Sequential()
        model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        opti = Adam(lr=0.01, decay=0.01)
        model.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])
        validation_data = (X_test, y_test) if y_test is not None else None
        model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=1, validation_data=validation_data)
        y_pred = model.predict(X_test).transpose()[0]
        y_pred = np.clip(y_pred, 0.0001, 0.9999)
    
    return y_pred