# Imports

# Pandas
import pandas as pd
import numpy as np

# numpy,matplotlib,seaborn,pyecharts
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')  #风格设置近似R这种的ggplot库
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
# from pyecharts import Pie

#  忽略弹出的warnings
import warnings
warnings.filterwarnings('ignore') 

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题



# launch
def concecutive_days(x):
    count = 0
    maxcount = 0
    row = x.values
    for i in row:
        if i>0:
            count +=i
        elif i==0:
            if maxcount < count:
                maxcount = count
            count = 0
    if count>maxcount:
        maxcount = count
    return maxcount

def weighted_day(df):
    col = df.columns.drop(['uid'])
    for i in range(day_gap):
        df[col[i]] *= day_weight[i]
        
    return df

def select_launch_features(launch, uid):
    launch_features = pd.DataFrame()
    launch_features['uid'] = uid['uid']

    minday = launch['day'].min()-1
    launch['day'] -= minday
    launch_day = launch.groupby(['uid'])['day'].agg({'std':'std','max':'max','min':'min','mean':'mean','count':'count'}).add_prefix('launch_day_').reset_index()
    launch_day['launch_day_gap'] = day_gap - launch_day['launch_day_max']
    launch_day['day_ratio'] = launch_day['launch_day_count']/day_gap
    
    launch_times = launch.groupby(['uid','day'])['uid'].count().unstack().add_prefix('launch_times_').reset_index().fillna(0)
    launch_times = weighted_day(launch_times)
    launch_times['times_concecutive'] = launch_times[launch_times.columns.drop(['uid'])].apply(lambda x: concecutive_days(x),axis=1)

    launch_features = pd.merge(launch_features, launch_day, how='left', on='uid')
    launch_features = pd.merge(launch_features, launch_times, how='left', on='uid')
    
    launch_features = launch_features.fillna(0)
    return launch_features


# register 直接对全部的做one-hot
def select_register_features(register, uid):
    register_features = pd.DataFrame()
    register_features['uid'] = uid['uid']

    # types onehot    
    device_type_list = register.groupby(['device_type'])['uid'].count().sort_values(ascending=False).reset_index()['device_type'][0:100].values
    register_device = register[register.device_type.map(lambda x: x in device_type_list)].groupby(['uid','device_type'])['uid'].count().unstack().add_prefix('register_device_').reset_index().fillna(0)
    
    register_type_list = register.groupby(['register_type'])['uid'].count().sort_values(ascending=False).reset_index()['register_type'].values
    register_register = register[register.register_type.map(lambda x: x in register_type_list)].groupby(['uid','register_type'])['uid'].count().unstack().add_prefix('register_register_').reset_index().fillna(0)

    register_features = pd.merge(register_features, register_device, how='left', on='uid')
    register_features = pd.merge(register_features, register_register, how='left', on='uid')
    register_features = register_features.fillna(0)
    
    return register_features

# video
def select_video_features(video, uid):
    video_features = pd.DataFrame()
    video_features['uid'] = uid['uid']
    
    minday = video['day'].min()-1
    video['day'] -= minday
    video_day = video.groupby(['uid'])['day'].agg({'std':'std','max':'max','min':'min','mean':'mean','count':'count'}).add_prefix('vid_day_').reset_index()
    video_day['day_vid_perday'] = video_day['vid_day_count']/day_gap
    video_day['vid_day_gap'] = day_gap - video_day['vid_day_max']
    
    video_times = video.groupby(['uid','day'])['uid'].count().unstack().add_prefix('vid_times_').reset_index().fillna(0)
    video_times = weighted_day(video_times)
    video_times['times_concecutive'] = video_times[video_times.columns.drop(['uid'])].apply(lambda x: concecutive_days(x),axis=1)
    
    video_features = pd.merge(video_features, video_day, how='left', on='uid')
    video_features = pd.merge(video_features, video_times, how='left', on='uid')
    # 不是所有uid都有vid记录
    video_features = video_features.fillna(0)
    
    return video_features


# activity
def select_activity_features(activity, uid):
    activity_features = pd.DataFrame()
    activity_features['uid'] = uid['uid']
    
    minday = activity['day'].min()-1
    activity['day'] -= minday
    
    activity_day = activity.groupby(['uid'])['day'].agg({'std':'std','max':'max','min':'min','mean':'mean','count':'count'}).add_prefix('act_day_').reset_index()
    activity_day['act_day_perday'] = activity_day['act_day_count']/day_gap
    activity_day['act_day_gap'] = day_gap - activity_day['act_day_max']
    
    activity_times = activity.groupby(['uid','day'])['uid'].count().unstack().add_prefix('act_times_').reset_index().fillna(0)
    activity_times = weighted_day(activity_times)
    activity_times['times_concecutive'] = activity_times[activity_times.columns.drop(['uid'])].apply(lambda x: concecutive_days(x),axis=1)
    
    activity_page = activity.groupby(['uid','page'])['uid'].count().unstack().add_prefix('act_page_').reset_index().fillna(0)
#     activity_page['act_page_cntsum'] = activity_page[activity_page.columns.drop(['uid'])].sum(axis=1)   

    activity_actype = activity.groupby(['uid','action_type'])['uid'].count().unstack().add_prefix('act_actype_').reset_index().fillna(0)
    activity_actype[activity_actype.columns.drop(['uid'])] = activity_actype[activity_actype.columns.drop(['uid'])].div(activity_actype[activity_actype.columns.drop(['uid'])].sum(axis=1),axis=0)     
    
    activity_author_id = activity.groupby(['author_id'])['video_id'].agg({'count':'count', 'unique_count': lambda x: len(pd.unique(x))}).add_prefix('act_author_').reset_index()
    activity_author_id.columns=['uid', 'act_author_count', 'act_author_unique_count']
    
    for feat in [activity_day, activity_times, activity_page, activity_actype]:
#                  ,activity_author_id]:
        activity_features = pd.merge(activity_features, feat, how='left', on='uid')

    activity_features = activity_features.fillna(0)

    return activity_features

# construct train/test set
day_gap = 16
window_size = day_gap
stride = 7

day_weight = [i for i in range(1,day_gap+1)]
# 为从远到近的天数赋予权重

datapath = '/mnt/datasets/fusai/'
reginame = 'user_register_log.txt'
actname = 'user_activity_log.txt'
vidname = 'video_create_log.txt'
launchname = 'app_launch_log.txt'
launch_log = pd.read_csv(datapath+launchname ,sep='\t',header=None,names=('uid','day'))
activity_log = pd.read_csv(datapath+actname ,sep='\t',header=None, names=('uid','day','page','video_id','author_id','action_type'))
register_log = pd.read_csv(datapath+reginame ,sep='\t',header=None,names=('uid','day','register_type','device_type'))
video_create_log = pd.read_csv(datapath+vidname ,sep='\t',header=None,names=('uid','day'))

launch = launch_log
activity =activity_log
register = register_log
video = video_create_log

# get features/ determine active users
def find_active_uid(start, end):
    user = pd.DataFrame()
    df1 = launch[((launch['day'] >= start) & (launch['day']<=end))]['uid']
    df2 = activity[((activity['day'] >= start) & (activity['day']<=end))]['uid']
    df3 = register[((register['day'] >= start) & (register['day']<=end))]['uid']
    df4 = video[((video['day'] >= start) & (video['day']<=end))]['uid']
    
    user['uid'] = pd.concat([df1,df2,df3,df4]).unique()
    return user

def get_label(user, start, end):
    user_active = find_active_uid(start, end)
    s = set(user_active['uid'])
    user_label = user.copy()
    user_label['label'] = user_label['uid'].apply(lambda x: 1 if x in s else 0)
    
    return user_label

def get_features(user, start, end):
    # filter by uid
#     reg = pd.merge(register, user, on='uid', how='right')
    lnch = pd.merge(launch, user, on='uid', how='right')
    act = pd.merge(activity, user, on='uid', how='right')
    vid = pd.merge(video, user, on='uid', how='right')
    
    # filter by day
#     reg = reg[((reg['day'] >= start) & (reg['day']<=end))]
    lnch = lnch[((lnch['day'] >= start) & (lnch['day']<=end))]
    act = act[((act['day'] >= start) & (act['day']<=end))]
    vid = vid[((vid['day'] >= start) & (vid['day']<=end))]

    return lnch, act, vid

def data_augmentation_get_features(window_size, stride):
    start = 1
    end = 23
    label_size = 7

    startlist = [start+i*stride for i in range(int((end-window_size)/stride)+1)]
    
    train_feature_list = []
    
    for i in startlist:
        user = find_active_uid(start, i+window_size)
        user_label = get_label(user,i+window_size+1, i+window_size+label_size)
        lnch, act, vid = get_features(user, i, i+window_size)
        
        register_feat = select_register_features(register, user_label)
        launch_feat = select_launch_features(lnch, user_label)
        video_feat = select_video_features(vid, user_label)
        activity_feat = select_activity_features(act, user_label)
        
        train_features = user_label
        for feat in [register_feat, launch_feat, video_feat, activity_feat]:
            train_features = pd.merge(train_features, feat, how='left', on='uid')
            
        train_feature_list.append(train_features)
        
    return train_feature_list # pd.concat(train_feature_list,axis=0)


def get_test_features(window_size):
#     uid = pd.read_csv(datapath+'all_label.csv')
    user = find_active_uid(1, 30)
    lnch, act, vid = get_features(user, 30-window_size, 30)
    
    register_feat = select_register_features(register, user)
    launch_feat = select_launch_features(lnch, user)
    video_feat = select_video_features(vid, user)
    activity_feat = select_activity_features(act, user)
    
    test_features = user
    for feat in [register_feat, launch_feat, video_feat, activity_feat]:
        test_features = pd.merge(test_features, feat, how='left', on='uid')
    
    return test_features





train = data_augmentation_get_features(window_size,stride)
test = get_test_features(window_size)
