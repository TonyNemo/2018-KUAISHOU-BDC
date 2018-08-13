import pandas as pd
train1 = pd.read_csv('train_feat1.csv')
train2 = pd.read_csv('train_feat2.csv')
test = pd.read_csv('test_feat.csv')
train_feature = train1
valid_feature = train2
test_feature = test

from sklearn.preprocessing import MinMaxScaler,StandardScaler

feature_col = train_feature.columns.drop(['uid','label'])
scalar = MinMaxScaler()
train_feature[feature_col] = scalar.fit_transform(train_feature[feature_col])
valid_feature[feature_col] = scalar.fit_transform(valid_feature[feature_col])
test_feature[feature_col] = scalar.fit_transform(test_feature[feature_col])

all_feature = pd.concat([train_feature, valid_feature], axis=0)

# select_featues
select_features = train_feature.columns.drop(['uid','label'])
select_features.shape

# lgb
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

lgb_params =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'metric': ('multi_logloss', 'multi_error'),
    #'metric_freq': 100,
    'is_training_metric': False,
    'is_unbalanced':True,
    'min_data_in_leaf': 12,
    'num_leaves': 64,
    'learning_rate': 0.1,
    'feature_fraction': 0.5, # 0.5
    'bagging_fraction': 0.5, # 0.5
    'verbosity':-1,
#    'gpu_device_id':2,
#    'device':'gpu'
#    'lambda_l1': 0.001,
#    'skip_drop': 0.95,
#    'max_drop' : 10
#     'lambda_l2': 0.001,
    'num_threads': 8
}

def evalMetric(preds,dtrain):
    
    label = dtrain.get_label()
    
    
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    
    auc = metrics.roc_auc_score(pre.label,pre.preds)

    pre.preds=pre.preds.map(lambda x: 1 if x>=0.4 else 0)

    f1 = metrics.f1_score(pre.label,pre.preds)
    
    
    res = auc
    
    return 'auc',res,True

# CV
dtrain = lgb.Dataset(train_feature[select_features],label=train_feature.label)
cvres = lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=50,verbose_eval=20,num_boost_round=10000,nfold=3,metrics=['evalMetric'])

# Grid Search
dtrain = lgb.Dataset(all_feature[select_features],label=all_feature.label)
gridsearch_params = [
    (feature_fraction, bagging_fraction)
    for feature_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    for bagging_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
]

# Define initial best params and auc
min_auc = -float("Inf")
best_params = None
for feature_fraction, bagging_fraction in gridsearch_params:
    print("CV with feature_fraction={}, bagging_fraction={}".format(
                             feature_fraction,
                             bagging_fraction))

    # Update our parameters
    lgb_params['feature_fraction'] = feature_fraction
    lgb_params['bagging_fraction'] = bagging_fraction

    # Run CV
    cvresult = lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=50,verbose_eval=20
        ,num_boost_round=10000,nfold=5,metrics=['evalMetric'])
        
    # lgb_model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=20,early_stopping_rounds=50,
    #     num_boost_round=1000, valid_sets=[dvalid])
    # pred = lgb_model.predict(valid_feature.drop(['uid'],axis=1),num_iteration=lgb_model.best_iteration)
    # auc = metrics.roc_auc_score(valid_feature.label, pred)
    # Update best auc
    mean_auc = max(cvresult['auc-mean'])
    boost_rounds = np.argmax(cvresult['auc-mean'])
    print("\tauc {} for {} rounds".format(mean_auc, boost_rounds))
    if mean_auc > min_auc:
        min_auc = mean_auc
        best_params = (feature_fraction,bagging_fraction)

print("Best params: {}, {}, auc: {}".format(best_params[0], best_params[1], min_auc))

X_train = all_feature[select_features]
Y_train = all_feature['label']
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
X = X_train
y = Y_train
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.4, random_state = 2018)

dtrain = lgb.Dataset(X_train,label=Y_train)
dvalid = lgb.Dataset(X_test,label=Y_test)

lgb_model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=20,early_stopping_rounds=50,num_boost_round=1000,valid_sets=[dvalid])

lgb.plot_importance(lgb_model)
importance = pd.DataFrame({'feature': lgb_model.feature_name(), 'importance':list(lgb_model.feature_importance())})
importance.sort_values(by='importance', ascending=False, inplace=True)
importance.head()

select_features = list(importance.head(_).feature) # select from the top importance

# multi-lgb the random part did not work
import random
# dtest = lgb.Dataset(test_feature.drop(['uid'],axis=1))
dtest = lgb.Dataset(test_feature[select_features])
# X_Train = train_feature.drop(['uid','label'], axis=1)
X_Train = all_feature[select_features]
Y_Train = all_feature['label']

itter_times = 20
lgb_models = []
for i in range(itter_times):
#     lgb_params =  {
#         'boosting_type': 'gbdt',
#         'objective': 'binary',
#         'is_training_metric': False,
#         'min_data_in_leaf': 20,
#         'num_leaves': 16,
#         'learning_rate': 0.12,
#         'feature_fraction': 0.8,
#         'bagging_fraction': 0.8,
#         'verbosity':-1,
#        'feature_fraction_seed': i
#     }    
    
    lgb_params['feature_fraction'] = random.uniform(2,8)*0.1
    lgb_params['bagging_fraction'] = random.uniform(2,8)*0.1
    lgb_params['learning_rate'] = random.uniform(1,5)*0.1
    print("params: feature_frac: {}, bagging_frac: {}, learning_rate: {}".format(lgb_params['feature_fraction'], 
        lgb_params['bagging_fraction'], lgb_params['learning_rate']))
    X_train, X_test, Y_train, Y_test = train_test_split(X_Train, Y_Train, test_size = 0.4, random_state = 64*i)

    dtrain = lgb.Dataset(X_train,label=Y_train)
    dvalid = lgb.Dataset(X_test,label=Y_test)
    
#     lgb_model =lgb.train(lgb_params,dtrain,feval=lgb_evalMetric,verbose_eval=5,num_boost_round=300,valid_sets=[dtrain], early_stopping_rounds= 50)
    lgb_model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=40,early_stopping_rounds=100,num_boost_round=3000,valid_sets=[dvalid])
    lgb_models.append(lgb_model)



lgb_gather = []
for model in lgb_models:
    pred = model.predict(test_feature.drop(['uid'],axis=1), num_iteration=model.best_iteration)
    lgb_gather.append(pred)

lgb_pred = pd.DataFrame({'uid':test_feature.uid})
for i in range(itter_times):
    lgb_pred['lgb_pred'+str(i)] = lgb_gather[i]
lgb_pred['lgb_mean'] = list(pd.DataFrame(lgb_gather).mean())


# xgb
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

xgb_params = {
        'booster':'gbtree',
        'objective':'binary:logistic',
        'stratified':True,
        'max_depth':20,
         'gamma':0,
        'subsample':0.8,
        'colsample_bytree':0.8,
        #'lambda':0.5,
        'eta':0.05,
#         'seed':7 * i,
        'min_child_weight': 50,
        'silent':1
}


def xgevalMetric(preds,dtrain):
    label = dtrain.get_label()
    pre = pd.DataFrame({'preds':preds,'label':label})
    pre= pre.sort_values(by='preds',ascending=False)
    auc = metrics.roc_auc_score(pre.label,pre.preds)
    pre.preds=pre.preds.map(lambda x: 1 if x>=0.4 else 0)
    f1 = metrics.f1_score(pre.label,pre.preds)
#     res = f1
    return 'auc',auc


# Grid Search
# xgtrain = xgb.DMatrix(train_feature.drop(['uid','label'],axis=1),label=train_feature.label)
# xgtest = xgb.DMatrix(test_feature.drop(['uid'],axis=1))
dtrain = lgb.Dataset(all_feature[select_features],label=all_feature.label)
# dvalid = lgb.Dataset(valid_feature[select_features],label=valid_feature.label)
# dtest = lgb.Dataset(test_feature[select_features])

gridsearch_params = [
    (feature_fraction, bagging_fraction)
    for feature_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    for bagging_fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
]

# Define initial best params and auc
min_auc = -float("Inf")
best_params = None
for feature_fraction, bagging_fraction in gridsearch_params:
    print("CV with feature_fraction={}, bagging_fraction={}".format(
                             feature_fraction,
                             bagging_fraction))

    # Update our parameters
    lgb_params['feature_fraction'] = feature_fraction
    lgb_params['bagging_fraction'] = bagging_fraction

    # Run CV
    cvresult = lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=50,verbose_eval=20
        ,num_boost_round=10000,nfold=5,metrics=['evalMetric'])
        
    # lgb_model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=20,early_stopping_rounds=50,
    #     num_boost_round=1000, valid_sets=[dvalid])
    # pred = lgb_model.predict(valid_feature.drop(['uid'],axis=1),num_iteration=lgb_model.best_iteration)
    # auc = metrics.roc_auc_score(valid_feature.label, pred)
    # Update best auc
    mean_auc = max(cvresult['auc-mean'])
    boost_rounds = np.argmax(cvresult['auc-mean'])
    print("\tauc {} for {} rounds".format(mean_auc, boost_rounds))
    if mean_auc > min_auc:
        min_auc = mean_auc
        best_params = (feature_fraction,bagging_fraction)

print("Best params: {}, {}, auc: {}".format(best_params[0], best_params[1], min_auc))

# multi-xgb
# xgtest = xgb.DMatrix(test_feature.drop(['uid'],axis=1))
# X_Train = train_feature.drop(['uid','label'], axis=1)
xgtest = xgb.DMatrix(test_feature[select_features])
X_Train = train_feature[select_features]
Y_Train = train_feature['label']

itterTimes = 10
xgb_models = []
for i in range(itterTimes):
#     xgb_params = {
#         'booster':'gbtree',
#         'objective':'binary:logistic',
#         'stratified':True,
#         'max_depth':20,
#          'gamma':0,
#         'subsample':0.8,
#         'colsample_bytree':0.8,
#         #'lambda':0.5,
#         'eta':0.05,
#         'seed':7 * i,
#         'min_child_weight': 20,
#         'silent':1
#     }

    X_train, X_test, Y_train, Y_test = train_test_split(X_Train, Y_Train, test_size = 0.3, random_state = 64*i)

    xgtrain = xgb.DMatrix(X_train,label= Y_train)
    xgvalid = xgb.DMatrix(X_test, label= Y_test)
    
    xgb_model=xgb.train(xgb_params,dtrain=xgtrain,num_boost_round=10000,verbose_eval=10,evals=[(xgvalid,'valid')],
                        feval=xgevalMetric,early_stopping_rounds=50, maximize=True)
    xgb_models.append(xgb_model)


xgb_gather = []
xgb_pred = pd.DataFrame({'uid':test_feature.uid})
for model in xgb_models:
    xgb_x = model.predict(xgtest)
    xgb_gather.append(xgb_x)

for i in range(itterTimes):
    xgb_pred['xgb_pred'+str(i)] = xgb_gather[i]
xgb_pred['xgb_mean'] = list(pd.DataFrame(xgb_gather).mean())


# NN FCN
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=5, inter_op_parallelism_threads=5)
from keras.layers import Input, Embedding, LSTM, Dense,Flatten, Dropout, merge, Bidirectional, GRU,MaxoutDense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import  MaxPooling2D,GlobalMaxPooling1D
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D,Merge
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import Convolution1D
from keras.layers import Dense, Dropout, Activation
from keras import backend as K
from keras.optimizers import *
from keras.initializers import *
from keras.callbacks import Callback,EarlyStopping
from keras.regularizers import l1,l2
from sklearn.cross_validation import train_test_split
from keras import backend as K
# K.set_session(tf.Session(config=session_conf))

from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers.merge import _Merge
from keras.engine.topology import Layer,InputSpec
# from keras.utils import conv_utilsv

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import EarlyStopping


def nn_model():
    model = Sequential()
    model.add(Dense(64,  
                    activation='relu',
                    input_shape = (198,), 
                                  kernel_initializer='he_normal',
                                  kernel_regularizer=l2(0.000025)
                                  ))
    model.add(Dropout(0.2))
    
    model.add(Dense(12, 
                    activation='relu', 
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.000025)
                    ))
    model.add(Dropout(0.25))
    
    model.add(Dense(24,
                    activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.000025)
                    ))
    model.add(Dropout(0.1))

    model.add(Dense(units=1, 
                    activation='sigmoid', 
                    kernel_initializer='he_normal',
                    ))
    opt = optimizers.Adadelta(lr=1)
    model.compile(loss='binary_crossentropy', 
                  optimizer=opt,
                  metrics=['binary_accuracy']
                  )
    return(model)

model1 = nn_model()



def nn_model2():
    input1 = Input(shape=(len(other_features),), dtype='float32')
    x1 = BatchNormalization()(input1)
    
    x = Dense(30, activation='sigmoid')(x1)
    x3 = Dense(470)(x1)
    x3 = PReLU()(x3)
    x1 = merge([x,x3], mode='concat')
    x1 = Dropout(0.02)(x1)
    
    x = Dense(256, activation='sigmoid')(x1)
    x2 = Dense(11, activation='tanh')(x1)
    x3 = Dense(11)(x1)
    x3 = PReLU()(x3)
    x1 = merge([x , x2, x3], mode='concat')
    x1 = Dropout(0.02)(x1)
    
    out = Dense(1, activation='sigmoid')(x1)
    model = Model(input=[input1], output=out)
    
    opt = Adam(lr=0.0055,epsilon=8e-5)
    # opt = optimizers.Adadelta(lr=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['binary_accuracy']
                  )
    return model
model2 = nn_model2()


def nn_model3():
    input1 = Input(shape=(len(select_features),), dtype='float32')
    x1 = BatchNormalization()(input1)
    
    x = Dense(64, activation='sigmoid')(x1)
    x3 = Dense(470)(x1)
    x3 = PReLU()(x3)
    x1 = merge([x,x3], mode='concat')
    x1 = Dropout(0.02)(x1)
    
    x = Dense(256, activation='sigmoid')(x1)
    x2 = Dense(11, activation='tanh')(x1)
    x3 = Dense(11)(x1)
    x3 = PReLU()(x3)
    x1 = merge([x , x2, x3], mode='concat')
    x1 = Dropout(0.02)(x1)
    
    out = Dense(1, activation='sigmoid')(x1)
    model = Model(input=[input1], output=out)
    
    opt = Adam(lr=0.0055,epsilon=8e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['binary_accuracy']
                  )
    return model
model3 = nn_model3()

X_train = all_feature[select_features].values
y_train = all_feature['label'].values
X_val = all_feature[select_features].values
y_val = all_feature['label'].values

model3.fit(X_train, y_train, epochs = 70, batch_size=8000, verbose = 2 
          ,validation_data=[X_val, y_val], callbacks=[EarlyStopping(monitor='val_binary_accuracy', patience=10)]
          )

pred_val = model3.predict(X_val)
auc = metrics.roc_auc_score(y_val,pred_val)

nn_pred = pd.DataFrame({'uid':test_feature.uid})
pred_test = model3.predict(test_feature[select_features].values)
nn_pred['auc'+str(int(auc*1000))] = pred_test
