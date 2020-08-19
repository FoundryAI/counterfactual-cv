import lightgbm as lgb
import numpy as np

# Original DR Estimator
from sklearn.ensemble import RandomForestClassifier


def twoModel_Tau(X_train,X_test,y_train, y_test, t_train, t_test, is_binary=False):
    if is_binary:
        f1_model = lgb.LGBMClassifier()
        f0_model = lgb.LGBMClassifier()
    else:
        f1_model = lgb.LGBMRegressor()
        f0_model = lgb.LGBMRegressor()
    # Create model f1(X)
    f1_model.fit(X_train.loc[t_train==1,:],y_train.loc[t_train==1])
    f1_pred_test = f1_model.predict(X_test)
    # Create model f0(X)
    f0_model.fit(X_train.loc[t_train==0,:],y_train.loc[t_train==0])
    f0_pred_test = f0_model.predict(X_test)
    # Create propensity model
#     ps_model = RandomForestClassifier(n_estimators=100, max_depth=6,
#                                       class_weight='balanced_subsample',
#                                       min_samples_leaf=int(len(X_train) / 100))
#                                      #min_samples_leaf=int(len(X_test) / 100)) #lgb.LGBMClassifier()
#     ps_model.fit(X_train,t_train)
    ps_model = RandomForestClassifier(n_estimators=100, max_depth=6,
                                      class_weight='balanced_subsample',
                                     min_samples_leaf=int(len(X_test) / 100)) #lgb.LGBMClassifier()
    ps_model.fit(X_test,t_test)
    pscore_test = ps_model.predict_proba(X_test)[:,1]
    # Propensity model mask
    ps_mask = (pscore_test>0.00) & (pscore_test<1)
    # doubly robust-style oracle function τ˜DR( ⋅ )
    # based on https://usaito.github.io/files/cfcv_ws_poster.pdf
    tau_dr = t_test/pscore_test*(y_test-f1_pred_test)-\
        (1-t_test)/(1-pscore_test)*(y_test-f0_pred_test)+\
            f1_pred_test-f0_pred_test
    return tau_dr, ps_mask
# Simple double robust estimator
def threeModel_Tau(X_train,X_test,y_train, y_test, t_train, t_test, is_binary=False):
    # Create Outcome Models
    if is_binary:
        f1_model = lgb.LGBMClassifier()
        f0_model = lgb.LGBMClassifier()
        f_model = lgb.LGBMClassifier()
    else:
        f1_model = lgb.LGBMRegressor()
        f0_model = lgb.LGBMRegressor()
        f_model = lgb.LGBMRegressor()
    f1_model.fit(X_train.loc[t_train==1,:],y_train.loc[t_train==1])
    f1_pred_test = f1_model.predict(X_test)
    f0_model.fit(X_train.loc[t_train==0,:],y_train.loc[t_train==0])
    f0_pred_test = f0_model.predict(X_test)
    # Create full Treatment Model
    newTrain = X_train.assign(t_val = t_train)
    f_model.fit(newTrain,y_train)
    newTest = X_test.assign(t_val = t_test)
    # Create propensity model
#     ps_model = RandomForestClassifier(n_estimators=100, max_depth=6,
#                                       class_weight='balanced_subsample',
#                                      min_samples_leaf=int(len(X_train) / 100))
#                                      #min_samples_leaf=int(len(X_test) / 100)) #lgb.LGBMClassifier()
#     ps_model.fit(X_train,t_train)
    ps_model = RandomForestClassifier(n_estimators=100, max_depth=6,
                                      class_weight='balanced_subsample',
                                     min_samples_leaf=int(len(X_test) / 100)) #lgb.LGBMClassifier()
    ps_model.fit(X_test,t_test)
    pscore_test = ps_model.predict_proba(X_test)[:,1]
    # Propensity model mask
    ps_mask = (pscore_test>0.00) & (pscore_test<1)
    muahat = f_model.predict(newTest)
    tau_dr = (f1_pred_test-f0_pred_test) + (t_test/pscore_test - (1-t_test)/(1-pscore_test)) * (y_test-muahat)
    return tau_dr, ps_mask
# Simple double robust estimator
def oneModel_Tau(X_train,X_test,y_train, y_test, t_train, t_test, is_binary=False):
    # Create full Treatment Model
    if is_binary:
        f_model = lgb.LGBMClassifier()
    else:
        f_model = lgb.LGBMRegressor()
    newTrain = X_train.assign(t_val = t_train)
    f_model.fit(newTrain,y_train)
    newTest = X_test.assign(t_val = t_test)
    # Create propensity model
#     ps_model = RandomForestClassifier(n_estimators=100, max_depth=6,
#                                       class_weight='balanced_subsample',
#                                      min_samples_leaf=int(len(X_train) / 100))
#                                      #min_samples_leaf=int(len(X_test) / 100)) #lgb.LGBMClassifier()
#     ps_model.fit(X_train,t_train)
    ps_model = RandomForestClassifier(n_estimators=100, max_depth=6,
                                      class_weight='balanced_subsample',
                                     min_samples_leaf=int(len(X_train) / 100)) #lgb.LGBMClassifier()
    ps_model.fit(X_train,t_train)
    pscore_test = ps_model.predict_proba(X_test)[:,1]
    # Propensity model mask
    ps_mask = (pscore_test>0.00) & (pscore_test<1)
    muahat = f_model.predict(newTest)
    newTest['t_val'] = np.repeat(1,newTest.shape[0])
    f1_pred_test = f_model.predict(newTest)
    newTest['t_val'] = np.repeat(0,newTest.shape[0])
    f0_pred_test = f_model.predict(newTest)
    tau_dr = (f1_pred_test-f0_pred_test) + (t_test/pscore_test - (1-t_test)/(1-pscore_test)) * (y_test-muahat)
    return tau_dr, ps_mask