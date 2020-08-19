import os

import numpy as np

from econml.data.dgps import ihdp_surface_B as ihdp_B
from sklearn.model_selection import train_test_split

import pandas as pd

import math
from numpy.random import binomial, multivariate_normal, normal, uniform


def generate_data(data: str = 'ihdp_B', iters: int = 50) -> None:
    """Generate ihdp datasets used in semi-synthetic experiments."""
    names = ['Xtr', 'Xval', 'Xte', 'Ttr', 'Tval', 'Ytr', 'Yval', 'ITEte']
    os.makedirs(f'../data/{data}/', exist_ok=True)
    data_lists = [[] for i in np.arange(len(names))]
    for i in np.arange(iters):
        Y, T, X, ITE = ihdp_B(random_state=i)
        # train/test split
        X, Xte, T, _, Y, _, _, ITEte = train_test_split(X, T, Y, ITE, test_size=0.3, random_state=i)
        # train/val split
        Xtr, Xval, Ttr, Tval, Ytr, Yval = train_test_split(X, T, Y, test_size=0.5, random_state=i)
        _data = (Xtr, Xval, Xte, Ttr, Tval, Ytr, Yval, ITEte)
        for j, _ in enumerate(_data):
            data_lists[j].append(_)
    for j in np.arange(len(_data)):
        if j < 3:
            np.save(arr=np.concatenate(data_lists[j]).reshape((iters, _data[j].shape[0], _data[j].shape[1])),
                    file=f'../data/{data}/{names[j]}.npy')
        else:
            np.save(arr=np.c_[data_lists[j]], file=f'../data/{data}/{names[j]}.npy')


def generating_our_data(data: str = 'ihdp_B', iters: int = 50)-> None:
    """Generate our datasets used in semi-synthetic experiments."""
    names = ['Xtr', 'Xval', 'Xte', 'Ttr', 'Tval', 'Ytr', 'Yval', 'ITEte']
    os.makedirs(f'../data/{data}/', exist_ok=True)
    data_lists = [[] for i in np.arange(len(names))]
    for i in np.arange(iters):
        _data = ourDataGenerator(i)
        # _data = generate_curia_synthetic_data(seed=i+10)
        for j, _ in enumerate(_data):
            data_lists[j].append(_)
    for j in np.arange(len(_data)):
        if j < 3:
            np.save(arr=np.concatenate(data_lists[j]).reshape((iters, _data[j].shape[0], _data[j].shape[1])),
                    file=f'../data/{data}/{names[j]}.npy')
        else:
            np.save(arr=np.c_[data_lists[j]], file=f'../data/{data}/{names[j]}.npy')


def ourDataGenerator(state):
    random_state = state
    test_share = 0.5
    binary_baseline_mean = 0.1
    y, t, X, true_ite = ihdp_B(random_state=random_state)
    X = pd.DataFrame(X, columns=['ones', 'bw', 'b.head', 'preterm', 'birth.o', 'nnhealth', 'momage',
                                'sex', 'twin', 'b.marr', 'mom.lths', 'mom.hs', 'mom.scoll', 'cig',
                                'first', 'booze', 'drugs', 'work.dur', 'prenatal', 'site1', 'site2',
                                'site3', 'site4', 'site5', 'site6', 'site7']).reset_index(drop=True)
    t = pd.Series(t).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    true_ite = pd.Series(true_ite).reset_index(drop=True)
    # Convert to binary
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))
    def logit(x):
        return np.log(x/(1-x))
    y = y-y.mean()+logit(binary_baseline_mean)
    y0_prob = pd.Series(sigmoid(y-true_ite*t))
    y1_prob = pd.Series(sigmoid(y+true_ite*(1-t)))
    true_ite = pd.Series(y1_prob-y0_prob)
    y_prob = y1_prob*t + y0_prob*(1-t)
    y = pd.Series([np.random.choice([0, 1], p=[1-y_prob_record, y_prob_record]) for y_prob_record in y_prob])
    # train_test_split
    n_total = X.shape[0]
    n_test = round(test_share * n_total)
    n_train = n_total - n_test


    # train/test split
    X, Xte, T, _, Y, _, _, ITEte = train_test_split(X.to_numpy(), t.to_numpy(), y.to_numpy(), true_ite.to_numpy(), test_size=0.3, random_state=random_state)
    # train/val split
    Xtr, Xval, Ttr, Tval, Ytr, Yval = train_test_split(X, T, Y, test_size=0.5, random_state=random_state)
    data = (Xtr, Xval, Xte, Ttr, Tval, Ytr, Yval, ITEte)

    return data


# Some of these generated datasets have different numbers of confounders, but we need them to be the same
def generate_curia_synthetic_data(
    # Binary or continious
    binary_treatment: bool = True,
    binary_outcome: bool = False,

    # Number of records
    n_train: int = 1000,
    n_test: int = 1000,

    # Number of features to generate by type
    binary_dim: int = 5,
    uniform_dim: int = 5,
    normal_dim: int = 5,

    # Features to have effect on ITE, outcome and treatment propensity
    n_confounders: int = 2,
    n_features_outcome: int = 3,
    n_features_treatment_effect: int = 3,
    n_features_propensity: int = 3,

    # outcome_noise_sd
    outcome_noise_sd: int = 1,

    # Features to drop
    missing_data_scaler: float = 0.5,

    # Treatment share scaler
    treatment_share_scaler: float = 0.05,

    # Random seed
    seed: int = 42) -> object:
    #############################################################
    # Initiate variables and make some checks
    #############################################################

    # Sum train and test together for now
    n_total = n_train + n_test

    # Calculate actual values for the number of the missing features
    n_features_to_drop_outcome_not_counfounders = math.floor(
        (n_features_outcome - n_confounders) * missing_data_scaler)
    n_features_to_drop_treatment_effect_not_counfounders = math.floor(
        (n_features_treatment_effect - n_confounders) * missing_data_scaler)
    n_features_to_drop_confounders = math.floor(
        n_confounders * missing_data_scaler)
    n_features_to_drop_propensity = math.floor(
        n_features_propensity * missing_data_scaler)

    # create empty dataframe
    modeling_df = pd.DataFrame()

    #############################################################
    # Generate features
    #############################################################

    np.random.seed(seed)

    # Generate Age - we will add mean=70 and sd=30 later to avoid high influence of this variable
    modeling_df['age'] = normal(loc=0, scale=1, size=n_total)

    # Generate features with uniform distribution - will multiply to 10 later
    for i in range(0, uniform_dim):
        modeling_df['sdoh_' +
                    str(i)] = np.ceil(uniform(size=n_total) * 10) / 10

    # Generate features with bernoulli distribution
    binary_coefs = uniform(size=binary_dim)
    for i in range(0, binary_dim):
        binary_coef = binary_coefs[i]
        modeling_df['binary_flag_' +
                    str(i)] = binomial(n=1, p=binary_coef, size=n_total)

    # Generate features with normal distribution
    multivariate_df = pd.DataFrame(multivariate_normal(np.zeros(normal_dim),
                                                        np.diag(
                                                            np.ones(normal_dim)),
                                                        n_total),
                                    columns=['vector_' + str(i) for i in range(0, normal_dim)])
    modeling_df = pd.concat([modeling_df, multivariate_df], axis=1)

    # Extract name of the features
    features = pd.Series(modeling_df.columns)


    #############################################################
    # Sample features for the treatment effect and the outcomes
    #############################################################

    # sample features for the confounders
    confounders_features = features.sample(n_confounders, random_state=1)
    outcome_features_not_confounders = features[~features.isin(confounders_features)].sample(
        n_features_outcome - n_confounders, random_state=1)
    outcome_features = pd.concat(
        [outcome_features_not_confounders, confounders_features])

    # sample features for the treatment effect
    treatment_effect_features_not_confounders = features[~features.isin(outcome_features)].sample(
        n_features_treatment_effect - n_confounders, random_state=1)
    treatment_effect_features = pd.concat(
        [treatment_effect_features_not_confounders, confounders_features])

    # sample features for the propensity score
    propensity_score_features = features.sample(n_features_propensity, random_state=1)

    #############################################################
    # Generate outcomes
    #############################################################

    # Generate coefficients
    beta_outcome = normal(0, 1, n_features_outcome)

    # Generate outcomes
    modeling_df['y0'] = np.dot(modeling_df[outcome_features], beta_outcome) + normal(0,
                                                                                        outcome_noise_sd)

    #############################################################
    # Generate treatment effect
    #############################################################

    # Generate coeficients
    beta_te = normal(0, 1, n_features_treatment_effect)

    # Generate outcomes
    modeling_df['true_ite'] = np.dot(
        modeling_df[treatment_effect_features], beta_te)

    #############################################################
    # Generate propensity score
    #############################################################

    # Generate coeficients for propensity score
    # Draw coefficients from beta distributions
    beta_propensity_score = normal(0, 1, n_features_propensity)

    # Generate propensity score and rescale it again from 0 to 1
    modeling_df['true_treatment_propensity'] = np.dot(modeling_df[propensity_score_features],
                                                        beta_propensity_score)

    # Center the distribution first
    modeling_df['true_treatment_propensity'] = modeling_df['true_treatment_propensity'] - \
        modeling_df['true_treatment_propensity'].mean()

    # Rescale to -1 to +1
    modeling_df['true_treatment_propensity'] = modeling_df['true_treatment_propensity'] / \
        modeling_df['true_treatment_propensity'].abs().max()

    # Rescale to get treatment_share_scaler
    modeling_df['true_treatment_propensity'] = modeling_df['true_treatment_propensity'] * \
        min(treatment_share_scaler, 1 - treatment_share_scaler)

    # Move to the right
    modeling_df['true_treatment_propensity'] = modeling_df['true_treatment_propensity'] + \
        treatment_share_scaler

    #############################################################
    # Generate treatment
    #############################################################

    if binary_treatment:
        modeling_df['treatment'] = binomial(n=1, p=modeling_df['true_treatment_propensity'],
                                            size=n_total)
    else:
        modeling_df['treatment'] = modeling_df['true_treatment_propensity']


    #############################################################
    # Generate outcome with treatment effect
    #############################################################

    modeling_df['y1'] = modeling_df['y0'] + modeling_df['true_ite']
    modeling_df['y'] = modeling_df['y0'] + \
        modeling_df['true_ite'] * modeling_df['treatment']

    # Rescale from 0 to 1
    y_min = modeling_df[['y', 'y0', 'y1']].min().min()
    y_max = modeling_df[['y', 'y0', 'y1']].max().max()
    scale_factor = 1 / (y_max - y_min)
    modeling_df['y'] = (modeling_df['y'] - y_min) * scale_factor
    modeling_df['y0'] = (modeling_df['y0'] - y_min) * scale_factor
    modeling_df['y1'] = (modeling_df['y1'] - y_min) * scale_factor

    modeling_df['true_ite_rescaled'] = modeling_df['true_ite'] * scale_factor
    modeling_df['true_ite'] = modeling_df['y1'] - \
        modeling_df['y0']  # modeling_df['true_ite'] * scale_factor

    # If binary - rescale to [0,1] and use as probability to generate bernoulli outcome
    if binary_outcome:
        modeling_df['y'] = binomial(n=1, p=modeling_df['y'], size=n_total)

    #############################################################
    # Features final adjustments
    #############################################################

    # Rescale age feature
    modeling_df['age'] = np.where(modeling_df['age'] * 30 + 70 < 50, 50,
                                    modeling_df['age'] * 30 + 70)

    # Rescale SDOH features
    for i in range(0, uniform_dim):
        modeling_df['sdoh_' + str(i)] = modeling_df['sdoh_' + str(i)] * 10

    #############################################################
    # Drop features
    #############################################################

    # features_to_drop_outcome_not_counfounders
    features_to_drop_outcome_not_counfounders = outcome_features_not_confounders.sample(
        n_features_to_drop_outcome_not_counfounders, random_state=1)
   
    # features_to_drop_treatment_effect_not_confounders
    features_to_drop_treatment_effect_not_confounders = treatment_effect_features_not_confounders.sample(
        n_features_to_drop_treatment_effect_not_counfounders, random_state=1)

    # features_to_drop_confounders
    features_to_drop_confounders = confounders_features.sample(
        n_features_to_drop_confounders, random_state=1)

    # features_to_drop_confounders
    features_to_drop_propensity = propensity_score_features.sample(
        n_features_to_drop_propensity, random_state=1)

    # Now drop all those features
    all_features_to_drop = pd.concat([features_to_drop_outcome_not_counfounders,
                                        features_to_drop_treatment_effect_not_confounders,
                                        features_to_drop_confounders,
                                        features_to_drop_propensity]).drop_duplicates()
    
    for col in all_features_to_drop:
    #         print('Dropping {} from the columns'.format([col]))
        assert (col in modeling_df), 'All features to drop should be in the featureset'
        del modeling_df[col]

    #############################################################
    # Return results
    #############################################################

    # Randomly select train and test
    y = modeling_df['y']
    t = modeling_df['treatment']
    true_ite = modeling_df['true_ite']
    true_treatment_propensity = modeling_df['true_treatment_propensity']
    X = modeling_df.drop(['y', 'y0', 'y1', 'treatment', 'true_ite',
                            'true_treatment_propensity', 'true_ite_rescaled'], axis=1)

    
    X, Xte, T, _, Y, _, _, ITEte = train_test_split(X.to_numpy(), t.to_numpy(), y.to_numpy(), true_ite.to_numpy(), test_size=0.3, random_state=seed)
    Xtr, Xval, Ttr, Tval, Ytr, Yval = train_test_split(X, T, Y, test_size=0.5, random_state=seed)

    data = (Xtr, Xval, Xte, Ttr, Tval, Ytr, Yval, ITEte)
  
    return data