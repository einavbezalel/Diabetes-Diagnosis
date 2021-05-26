import pandas as pd
import numpy as np
import copy
from sklearn.linear_model import LinearRegression #for missing data imputation
from sklearn.preprocessing import OrdinalEncoder #for feature encoding
from scipy import stats
import matplotlib.pyplot as plt  # visualization
import seaborn as sns  # visualization
import warnings  # Supress warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import lightgbm as lgb #the model
from sklearn.model_selection import StratifiedKFold #for Cross-Validation
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, f1_score, confusion_matrix

train= pd.read_csv(r'C:\Users\Einav bezalel\PycharmProjects\Datathon2021\input\TrainingWiDS2021.csv')
test= pd.read_csv(r'C:\Users\Einav bezalel\PycharmProjects\Datathon2021\input\UnlabeledWiDS2021.csv')
dictionary= pd.read_csv(r'C:\Users\Einav bezalel\PycharmProjects\Datathon2021\input\DataDictionaryWiDS2021.csv')

# Checking shapes
print('train    ', train.shape)
print('test     ', test.shape)

# test_id = test.encounter_id.values #for submission
# encounter_df=test.encounter_id

print(train.head())
print(test.head())
print(dictionary.head())

target_col='diabetes_mellitus'

#Data Cleaning & missing data imputation
train.nunique().sort_values(ascending=False)#Count distinct observations over requested axis.defult is 0 (over the col)
# we can see that Unnamed: 0 + encounter_id are completely unique-> need to be dropped
#notice also that readmission_status has only one value (0) so I'll drop it
test.nunique().sort_values(ascending=False) #same thing for test

#I suspect that hospital_id comes from different distributions in test and train, let's cheack it:
if np.any((train['hospital_id']).isin(test['hospital_id'])):#should give True if at least one value is in test. o.w False
    print("train's hospital_id is in test")
elif np.any((test['hospital_id']).isin(train['hospital_id'])):
    print("test's hospital_id is in train")
else:
    print("hospital_id is completely different between the sets") #if so, we need to drop it

#make sure with Adversial Validation (first run)
train_copy = train.drop(target_col, axis=1, inplace=False).copy() #dropping because later there is concatenation
test_copy = test.copy()
drop_catego_temp=['ethnicity', 'gender', 'hospital_admit_source', 'icu_admit_source', 'icu_stay_type', 'icu_type','Unnamed: 0'] #because AD requaires encoding categorical features
train_copy.drop(drop_catego_temp,axis=1, inplace=True)
test_copy.drop(drop_catego_temp, axis=1, inplace=True)

def run_adversial_validation(train_set, test_set):
    lgb_params = {'n_estimators': 100,
                  'boosting_type': 'gbdt',
                  'objective': 'binary',
                  'metric': 'auc',
                  }
    # combine train & test features, create label to identify test vs train
    ad_y = np.array([1] * train_set.shape[0] + [0] * test_set.shape[0])
    ad_X = pd.concat([train_set, test_set])

    # evaluate model performance using cross-validation
    lgb_data = lgb.Dataset(ad_X, ad_y)
    cv_lgb = lgb.cv(lgb_params, lgb_data)

    print("Adversarial Validation AUC Score: {}".format(cv_lgb['auc-mean'][-1]))

    # train model & get feature importance
    ad_val_mod = lgb.train(lgb_params, lgb_data)

    print(pd.DataFrame(
        {'feat': ad_X.columns,
         'imp': ad_val_mod.feature_importance()}).sort_values('imp', ascending=False))

    return ad_val_mod

ad_val_mod = run_adversial_validation(train_copy, test_copy)# as expected  hospital_id, encounter_id are in the top, so we need to remove them

#there is one item which is not in train, we want to find it:
variables_of_dict= dictionary['Variable Name']
variables_of_train= train.columns
print(variables_of_dict[~variables_of_dict.isin(variables_of_train)]) #result: icu_admit_type
dictionary.drop([0,1,10,15],axis=0, inplace=True) #'encounter_id', hospital_id, icu_admit_type, readmission_status
#(It is not necessary but it will be more convenient to retrieve data according to their categories that appear in the dictionary)

test[target_col]=0 #For concatenating
# Combining train and test
df = pd.concat([train, test[train.columns]], axis=0)
print('combined df:',df.shape)

df.drop(['Unnamed: 0','readmission_status','encounter_id','hospital_id'], axis=1, inplace=True)

len(dictionary['Variable Name'])
len(train.columns)

df = df.replace([np.inf, -np.inf], np.nan)


min_cols = [f for f in train.columns if f[-4:]=='_min'] #fix min/max
def fix_min_max(data_frame, min_cols):
    for c in min_cols:
        vals = data_frame[[c, c.replace('_min', '_max')]].values.copy() #df[val_min,val_max]
        data_frame[c] = np.nanmin(vals, axis=1) #df[val_min]=min of the two, ignore nan
        data_frame[c.replace('_min', '_max')] = np.nanmax(vals, axis=1) #df[val_max]=max of the two, ignore nan
fix_min_max(df, min_cols)

train_groupby=df.groupby("gender") #Replacing NaNs in weight/height according to the gender's mode
# Explanation in the visualization part
variables=["weight","height"]
for var in variables:
    locals() ["male_" + var ]= train_groupby[var].get_group('M')
    locals() ["female_" + var ]=train_groupby[var].get_group('F')
df["weight"] = np.where((df.height.isna() & (df.gender == 'F')),female_weight.mode() , df["weight"])
df["weight"] = np.where((df.height.isna() & (df.gender == 'M')),male_weight.mode() , df["weight"])
df["height"] = np.where((df.height.isna() & (df.gender == 'F')),female_height.mode() , df["height"])
df["height"] = np.where((df.height.isna() & (df.gender == 'M')),male_height.mode() , df["height"])

df.loc[df.age == 0, 'age'] = np.nan #according to the visualization part


#Freature Engineering:

#counts/frequencies of features
d_f = df['icu_id']
agg = d_f.value_counts().to_dict()
df['icu_id_counts'] = np.log1p(df['icu_id'].map(agg))

d_f = df['age']
agg = d_f.value_counts().to_dict()
df['age_counts'] = np.log1p(df['age'].map(agg))

df['bmi'] = df['weight']/((df['height']/100)**2) #explanation in other notebook (fixing BMI from equation)
df['age_to_bmi'] = df['age']/df['bmi'] #relations
df['weight_to_age'] = df['weight']/df['age']

#Rounding
df['bmi_type'] = df.bmi.fillna(0).apply(lambda x: 5 * (round(int(x)/5)))
df['height_type'] = df.height.fillna(0).apply(lambda x: 5 * (round(int(x)/5)))
df['weight_type'] = df.weight.fillna(0).apply(lambda x: 5 * (round(int(x)/5)))
df['age_type'] = df.age.fillna(0).apply(lambda x: 10 * (round(int(x)/10)))

#medical codes (hierarchal order)
df['apache_3j_diagnosis_prefix'] = df['apache_3j_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]
df['apache_2_diagnosis_prefix'] = df['apache_2_diagnosis'].astype('str').str.split('.',n=1,expand=True)[0]
df["apache_3j_diagnosis_prefix_int"] =  df["apache_3j_diagnosis"].fillna(0).astype(str).str.split(".",expand=True)[0].astype(int)
df["grouped_apache_3j_int"] = df['apache_3j_diagnosis_prefix_int'].fillna(0).astype(int).apply(lambda x: round(x,-2))//100 #A negative value will round to a power of 10. -1 rounds to the nearest 10, -2 to the nearest 100, etc

#cond=df['apache_3j_diagnosis'].isin([0.13, 0.09, 702.01, 1408.02, 1408.11, 704.07, 701.01,0.02]) #explained in other notebook
# df['new_apache_3j_diagnosis']= np.where(cond, 1, 0)
df['new_apache_3j_diagnosis'] = df['apache_3j_diagnosis'].apply(lambda x: 1 if x == 702.01 else 0)

#Bloop Pressure 
df['d1_diasbp_min']= (df[['d1_diasbp_invasive_min', 'd1_diasbp_noninvasive_min']]).min(axis=1)
df['d1_mbp_min']= (df[['d1_mbp_invasive_min', 'd1_mbp_noninvasive_min']]).min(axis=1)
df['d1_sysbp_min']= (df[['d1_sysbp_invasive_min', 'd1_sysbp_noninvasive_min']]).min(axis=1)

df['h1_diasbp_min']= (df[['h1_diasbp_invasive_min', 'h1_diasbp_noninvasive_min']]).min(axis=1)
df['h1_mbp_min']= (df[['h1_mbp_invasive_min', 'h1_mbp_noninvasive_min']]).min(axis=1)
df['h1_sysbp_min']= (df[['h1_sysbp_invasive_min', 'h1_sysbp_noninvasive_min']]).min(axis=1)

df['d1_diasbp_max']= (df[['d1_diasbp_invasive_max', 'd1_diasbp_noninvasive_max']]).max(axis=1)
df['d1_mbp_max']= (df[['d1_mbp_invasive_max', 'd1_mbp_noninvasive_max']]).max(axis=1)
df['d1_sysbp_max']= (df[['d1_sysbp_invasive_max', 'd1_sysbp_noninvasive_max']]).max(axis=1)

df['h1_diasbp_max']= (df[['h1_diasbp_invasive_max', 'h1_diasbp_noninvasive_min']]).max(axis=1)
df['h1_mbp_max']= (df[['h1_mbp_invasive_max', 'h1_mbp_noninvasive_max']]).max(axis=1)
df['h1_sysbp_max']= (df[['h1_sysbp_invasive_max', 'h1_sysbp_noninvasive_max']]).max(axis=1)

categories= dictionary['Category'].unique()
for col in categories[:-1]:
    col_names= col.split(" ")
    col_new='_'.join(col_names)
    locals()['_'.join(col_names)] = dictionary['Variable Name'].loc[dictionary['Category'] == col].values
print(vitals[:-2])

invasive_cols=[c for c in vitals[:-2] if "invasive" in c.split('_')]
noninvasive_cols=[c for c in vitals[:-2] if "noninvasive" in c.split('_')]

bp_drop=noninvasive_cols+invasive_cols
df.drop(bp_drop, axis=1, inplace=True) #because they have more missing data (visualization part)

#ranges 
# tot_lab_cols=np.concatenate((vitals, labs,labs_blood_gas), axis=None).tolist() (other option)
tot_lab_cols = [c for c in df.columns if ((c.startswith("h1")) | (c.startswith("d1")))]

parameters = list(set(list(map(lambda i: i[3: -4], tot_lab_cols))))  # set is for distinct values, then need to return it back to list

for param in parameters:
    df[f"day_range_{param}"] = (df[f"d1_{param}_max"].subtract(df[f"d1_{param}_min"]))  # .div(df[f"d1_{param}_max"])#range day normalizedthe subtraction is col-col
    df[f"hour_range_{param}"] = (df[f"h1_{param}_max"].subtract(df[f"h1_{param}_min"]))  # .div(df[f"h1_{param}_max"])#range hour normalized

    df[f"day_mean_{param}"] = df[[f"d1_{param}_max", f"d1_{param}_min"]].mean(axis=1)  # mean day
    df[f"hour_mean_{param}"] = df[[f"h1_{param}_max", f"h1_{param}_min"]].mean(axis=1)  # mean hour


# all Mathematical operations related to glucose
df["glucose_std"] = df[['h1_glucose_max','h1_glucose_min','d1_glucose_max','d1_glucose_min']].std(axis=1)
df["glucose_mean"]=df[['h1_glucose_max','h1_glucose_min','d1_glucose_max','d1_glucose_min']].mean(axis=1)
df['glucose_type'] = df.glucose_mean.fillna(0).apply(lambda x: 10 * (round(int(x)/10)))


def largest_dis_from(tresh, d_max, d_min, h_min, h_max):
    arr = np.array([d_max, d_min, h_min, h_max])
    if np.isnan(arr).all():
        return np.nan
    else:
        idx = np.nanargmax(np.abs(arr - tresh))
        return arr[idx]


df['glucose_apache'] = df.apply(lambda x: largest_dis_from(tresh=130, d_max=x['d1_glucose_max'], d_min=x['d1_glucose_min'],
                               h_max=x['h1_glucose_max'], h_min=x['h1_glucose_min']), axis=1)


def distance_from(tresh, d_max, d_min, h_min, h_max):
    arr = np.array([d_max, d_min, h_min, h_max])
    if np.isnan(arr).all():
        return np.nan
    else:
        max_dis = np.max(np.abs(arr - tresh))
        return max_dis


df['glucose_distance'] = df.apply(lambda x: distance_from(tresh=200, d_max=x['d1_glucose_max'], d_min=x['d1_glucose_min'],
                            h_max=x['h1_glucose_max'], h_min=x['h1_glucose_min']), axis=1)


def glucose_level(x):
    if pd.isna(x):
        return np.nan
    if x < 100:
        return 1
    elif x >= 100 and x < 210:
        return 2
    elif x >= 210 and x < 250:
        return 3
    elif x >= 250 and x < 350:
        return 4
    elif x >= 350:
        return 5
    else:
        return -1

df['glucose_level'] = df['glucose_mean'].map(glucose_level)
df["glucose_d1_h1_max_eq"] = (df[f"d1_glucose_max"]== df[f"h1_glucose_max"]).astype(np.int8)
df["glucose_h1_bigger"]=(df["hour_mean_glucose"]>df["day_mean_glucose"]).astype(np.int8)
df["glucose_d1_zero_range"] = (df["day_range_glucose"] == 0).astype(np.int8)
df["glucose_h1_zero_range"] = (df["hour_range_glucose"] == 0).astype(np.int8)
df["glucose_tot_zero_range"] = (df["glucose_h1_zero_range"]+df["glucose_d1_zero_range"] ).astype(np.int8)
df['tot_nan_glu']=df[['h1_glucose_max','h1_glucose_min','d1_glucose_max','d1_glucose_min']].isna().sum(axis=1)


def weightedclasst(x):
    if pd.isna(x):
        return np.nan
    if x < 15:
        return 1  # 'very severely underweight'
    elif x >= 15 and x < 16:
        return 2  # 'severely underweight'
    elif x >= 16 and x < 18.5:
        return 3  # 'underweight' 
    elif x >= 18.5 and x < 25:
        return 4  # 'healthy weight' 
    elif x >= 25 and x < 30:
        return 5  # 'overweight'
    elif x >= 30 and x < 35:
        return 6  # 'class 1'  obese
    elif x >= 35:
        return 6  # 'class 2' obese
    else:
        return -1


df['weightclass'] = df['bmi'].map(weightedclasst)  # create new col where every val in bmi col is the input in the function

def bp_group(dia, sys):
    if pd.isna(dia) or pd.isna(sys):
        return np.nan
    if dia != np.nan and sys != np.nan:
        dia = int(dia)
        sys = int(sys)
        if dia < 60 and sys < 90:
            return 0  # 'Low'
        elif dia < 80 and sys < 120:
            return 1  # 'Normal'
        elif dia < 90 and sys < 140:
            return 2  # 'Prehypertension'
        elif dia < 100 and sys < 160:
            return 3  # 'Hypertension Stage 1'
        else:
            return 4  # 'Hypertension Stage 2'

df['bp_level'] = df.apply(lambda x: bp_group(dia=x['day_mean_diasbp'], sys=x['day_mean_sysbp']), axis=1)

#change APACHE scores (see visualization part)
df['albumin_apache']= df[['albumin_apache', 'd1_albumin_max', 'd1_albumin_min', 'h1_albumin_min',  'h1_albumin_max']].min(axis=1)

# High levels of BUN and creatinine in blood for diabetics
df['creatinine_apache']= df[['creatinine_apache', 'd1_creatinine_max', 'd1_creatinine_min', 'h1_creatinine_min',  'h1_creatinine_max']].max(axis=1)
df['bun_apache']= df[['bun_apache', 'd1_bun_max', 'd1_bun_min', 'h1_bun_min',  'h1_bun_max']].max(axis=1)


#don't know the connection of bilirubin to the target, but the highest level is the worst
df['bilirubin_apache']= df[['bilirubin_apache', 'd1_bilirubin_max', 'd1_bilirubin_min', 'h1_bilirubin_min',  'h1_bilirubin_max']].max(axis=1)


#NOTICE: when gcs_unable_apache==1 then we have Nan in all 3 cols(make sense) and 0 in the total sum,
# so I can can drop  gcs_unable_apache since it doesn't give  new info
#train[['gcs_eyes_apache','gcs_motor_apache','gcs_verbal_apache','gcs_total_score' ]].loc[train['gcs_unable_apache'] == 1].head()
df.drop(['gcs_unable_apache'],axis=1, inplace=True)


df['heart_rate_apache'] = df.apply(lambda x: largest_dis_from(tresh=75, d_max = x['d1_heartrate_max'], d_min = x['d1_heartrate_min'],h_max = x['h1_heartrate_max'], h_min = x['h1_heartrate_min']), axis=1)
df['hematocrit_apache'] = df.apply(lambda x: largest_dis_from(tresh=45, d_max = x['d1_hematocrit_max'], d_min = x['d1_hematocrit_min'],h_max = x['h1_hematocrit_max'], h_min = x['h1_hematocrit_min']), axis=1)
df['map_apache'] = df.apply(lambda x: largest_dis_from(tresh=90, d_max = x['d1_mbp_max'], d_min = x['d1_mbp_min'],h_max = x['h1_mbp_max'], h_min = x['h1_mbp_min']), axis=1)

#they're exactly the same so I'll drop one of them:
df.drop(['paco2_for_ph_apache'], axis=1, inplace=True)

#the normal range of p-co2 is 35-45mmHg (blue table) so the further from it, the worse it gets.
df['paco2_apache'] = df.apply(lambda x: largest_dis_from(tresh=40, d_max = x['d1_arterial_pco2_max'], d_min = x['d1_arterial_pco2_min'],h_max = x['h1_arterial_pco2_max'], h_min = x['h1_arterial_pco2_min']), axis=1)

# the normal range of p-co2 is 80mmHg, so the further from it, the worse it gets.
df['pao2_apache'] = df.apply(lambda x: largest_dis_from(tresh=80, d_max = x['d1_arterial_po2_max'], d_min = x['d1_arterial_po2_min'],h_max = x['h1_arterial_po2_max'], h_min = x['h1_arterial_po2_min']), axis=1)
df['ph_apache'] = df.apply(lambda x: largest_dis_from(tresh=7.4, d_max = x['d1_arterial_ph_max'], d_min = x['d1_arterial_ph_min'],h_max = x['h1_arterial_ph_max'], h_min = x['h1_arterial_ph_min']), axis=1)
df['resprate_apache'] = df.apply(lambda x: largest_dis_from(tresh=19, d_max = x['d1_resprate_max'], d_min = x['d1_resprate_min'],h_max = x['h1_resprate_max'], h_min = x['h1_resprate_min']), axis=1)

#A normal blood sodium level is between 135 and 145 milliequivalents per liter (mEq/L)
df['sodium_apache'] = df.apply(lambda x: largest_dis_from(tresh=140, d_max = x['d1_sodium_max'], d_min = x['d1_sodium_min'],h_max = x['h1_sodium_max'], h_min = x['h1_sodium_min']), axis=1)
df['temp_apache'] = df.apply(lambda x: largest_dis_from(tresh=38, d_max = x['d1_temp_max'], d_min = x['d1_temp_min'],h_max = x['h1_temp_max'], h_min = x['h1_temp_min']), axis=1)
df['wbc_apache'] = df.apply(lambda x: largest_dis_from(tresh=11.5, d_max = x['d1_wbc_max'], d_min = x['d1_wbc_min'],h_max = x['h1_wbc_max'], h_min = x['h1_wbc_min']), axis=1)
df['pao2fio2ratio_apache']=df['pao2_apache']/df["fio2_apache"]#new feature:  PaO2/FiO2
df['pao2fio2ratio_apache'] = df[['pao2fio2ratio_apache', 'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min', 'h1_pao2fio2ratio_min',  'h1_pao2fio2ratio_max']].min(axis=1)

#from equations 
df['risk_score'] = 100 / (1 + np.exp(-1*(0.028*df['age'].values + 0.661*np.where(df['gender'].values=="M", 1, 0) +
                                               0.412 * np.where(df['ethnicity'].values=="Native American", 0, 1) +
                                               0.079 * df['glucose_apache'].values + 0.018 * df['d1_diasbp_max'].values +
                                               0.07 * df['bmi'].values + 0.481 * df['cirrhosis'].values - 13.415)))

df["serum_osmolality"]= 2*df['sodium_apache'] + 2*df['d1_potassium_max'] +df['glucose_apache']/18 + df['bun_apache']/2.8

#aggregation
df['gcs_total_score']=(df[['gcs_eyes_apache','gcs_motor_apache','gcs_verbal_apache']]).sum(axis=1)
df['gcs_sum_type'] = df.gcs_total_score.fillna(0).apply(lambda x: 2.5 * (round(int(x)/2.5))).divide(2.5)

df["has_cancer_or_imm"]= 0
cancer_cond= df[['immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']].any(axis=1)
df['has_cancer_or_imm'].mask(cancer_cond, 1, inplace=True) #Replace values where the condition is True.


df['liver_diseases']= 0
liver_cond= (df[['cirrhosis','hepatic_failure']]==1).any(axis=1)
df['liver_diseases'].mask(liver_cond, 1, inplace=True) #Replace values where the condition is True.


df.drop(['immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis','cirrhosis','hepatic_failure','aids'], axis=1, inplace=True)

#Group by a variable and compute new statistics of the numerical features:
IDENTIFYING_COLS = ['age_type', 'height_type',  'ethnicity', 'gender', 'bmi_type']
df['profile'] = df[IDENTIFYING_COLS].apply(lambda x: hash(tuple(x)), axis = 1)

groups = ['apache_3j_diagnosis', 'profile']
for g in groups:
    temp = df[["d1_glucose_max", g]].groupby(g)["d1_glucose_max"].mean().to_dict()
    df[f'd1_glucose_max_{g}_mean'] = df[g].map(temp)

    # distance from the group's mean:
    df[f'd1_glucose_max_{g}_mean_diff'] = df["d1_glucose_max"] - df[g].map(
        temp)

    temp = df[["d1_glucose_max", g]].groupby(g)["d1_glucose_max"].std().to_dict()
    df[f'd1_glucose_max_{g}_std'] = df[g].map(temp)


to_drop=['icu_stay_type','icu_type','icu_admit_source','hospital_admit_source'] #from visualization part looks insignificance
df.drop(to_drop, axis=1, inplace=True)

#Dropping columns with more than 75% missing values.
nans=(df[df.columns]).isna().sum(axis=0).sort_values(ascending=False)#number of nan in each col, from highest to lowest
nans_frac=nans/df.shape[0]
to_drop_nan=list(nans_frac.loc[nans_frac>0.75].index)
df.drop(to_drop_nan, axis=1, inplace=True)

#filling missing values based on linear regression and the most correlated variables
def fillna_using_linear_model(df, feature_cols):

    correl = df[feature_cols].corr()

    for col in tqdm(feature_cols):
        nan_ratio = df[col].isnull().sum() / df.shape[0]
        if nan_ratio > 0:
            best_nan_ratio = nan_ratio
            best_col = None
            for id in correl.loc[np.abs(correl[col]) > 0.7 , col].index:
                nan_temp_ratio = df[id].isnull().sum() / df.shape[0]
                if best_nan_ratio > nan_temp_ratio:
                    best_nan_ratio = nan_temp_ratio
                    best_col = id
            if best_col != None:
                mat = df[[col, best_col]].copy()
                mat = mat.dropna()
                X=np.expand_dims(mat[best_col], axis=1)
                y=mat[col]
                reg = LinearRegression().fit(X, y)
                print(reg.score(X, y))
                if reg.score(X, y)>0.7:
                    if df.loc[(~df[best_col].isnull()) & (df[col].isnull()), col].shape[0] > 0:
                        df.loc[(~df[best_col].isnull()) & (df[col].isnull()), col] = \
                            reg.predict(np.expand_dims(
                                df.loc[(~df[best_col].isnull()) & (df[col].isnull()), best_col], axis=1))

    return df


float_cols=[col for col in df.columns if df[col].dtype=='float64']
df = fillna_using_linear_model(df, float_cols)

#seperate to train and test set
train_len=train.shape[0]
train = copy.copy(df[:train_len])
test = copy.copy(df[train_len:])
print('combined dataset ', df.shape)
print('train             ', train.shape)
print('test              ', test.shape)

#for selecting columns for encoding (categorical)- first find columns with low unique values->most likely to be categorical
print('For Train')
d1 = train.nunique()
print(sorted(d1))
print("==============================")
print('For Test')
d2 = test.nunique()
print(sorted(d2))

col_train = train.columns
col_test = test.columns

l1 = []
for i in col_train:
    if train[i].nunique() <= 14:
        l1.append(i)

l1.remove('diabetes_mellitus')
l1 = l1 + ['icu_id', 'grouped_apache_3j_int', 'profile']
print(l1)

train[l1] = train[l1].apply(lambda x: x.astype('category'), axis=0)
test[l1] = test[l1].apply(lambda x: x.astype('category'), axis=0)
print('train dtypes:')
print(train[l1].dtypes)
print('======================================')
print('test dtypes:')
print(test[l1].dtypes)

#add the non numeric columns
cols = train.columns
num_cols = train._get_numeric_data().columns
not_nuneric_cols = list(set(cols) - set(num_cols))#string/object only
cat_cols= list(set(not_nuneric_cols).union(set(l1))) #union() method returns a new set with distinct elements from all the sets
print(f"Categorical columns are: {cat_cols}")

# fill missing remain missing values
not_cat=[col for col in train.columns if col not in cat_cols]
for col in not_cat:
    train[col] = train[col].fillna(df[col].median())
    test[col] = test[col].fillna(df[col].median())

for col in cat_cols:
    train_only = list(set(train[col].unique()) - set(test[col].unique()))
    test_only = list(set(test[col].unique()) - set(train[col].unique()))
    both = list(set(test[col].unique()).union(set(train[col].unique())))
    #make sure that the same unique variables are in test and train
    train.loc[train[col].isin(train_only), col] = np.nan
    test.loc[test[col].isin(test_only), col] = np.nan
    try:
        lbl = OrdinalEncoder(dtype='int')
        train[col] = lbl.fit_transform(train[col].astype('str').values.reshape(-1,1))
        test[col] = lbl.transform(test[col].astype('str').values.reshape(-1,1))
    except:
        lbl = OrdinalEncoder(dtype='int')
        train[col] = lbl.fit_transform(train[col].astype('str').fillna('').values.reshape(-1,1))
        test[col] = lbl.transform(test[col].astype('str').fillna('').values.reshape(-1,1))
    #for each unique value in those columns compute the ranked frequencies:
    temp = pd.concat([train[[col]], test[[col]]], axis=0)#concatnate train & test
    temp_mapping = temp.groupby(col).size()/len(temp) #Compute group sizes divded by the total lenght(proportion of each group)
    temp['enc'] = temp[col].map(temp_mapping) #replace each value in the col with the corresponding value in the mapping(their frequencies)
    temp['enc'] = stats.rankdata(temp['enc']) #Assign ranks to data, dealing with ties appropriately.defult: by average
    temp = temp.reset_index(drop=True)
    train[f'rank_frqenc_{col}'] = temp[['enc']].values[:train.shape[0]] #divide to train & test
    test[f'rank_frqenc_{col}'] = temp[['enc']].values[train.shape[0]:]
    test[col] = test[col].astype('category') #convert to category dtype so that the model will identify those columns as categories
    train[col] = train[col].astype('category')


#Adversial Validation (run 2)
train_copy = train.drop(target_col, axis=1, inplace=False).copy()
test_copy = test.drop(target_col, axis=1, inplace=False).copy()
ad_val_mod_2 = run_adversial_validation(train_copy, test_copy) #profile is at the top so need to be removed
train.drop(['profile'],axis=1, inplace=True)
test.drop(['profile'],axis=1, inplace=True)


# Prepare training and test data
target_col=['diabetes_mellitus']
X, y = train.drop(target_col, axis=1, inplace=False), train[target_col]
X_test = test.drop(target_col, axis=1, inplace=False)

list_of_cate_in_X=X.select_dtypes(include=['category']).columns.tolist()
print(list_of_cate_in_X)


#the model
SEED = 42

model_params = {
    "objective": "binary",
    "metric": "auc",
    "seed": SEED,
    'num_iterations': 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 32,
    "is_unbalance": True,
    "min_data_in_leaf": 200,
    "lambda_l1": 1,
    "lambda_l2": 0.1,
    "bagging_fraction": 0.7,
    "feature_fraction": 0.8,
    "min_split_gain": 0.5,
    "subsample_for_bin": 200,
    "n_jobs": -1,
}

N_SPLITS = 5

# Initialize variables
y_oof_pred = np.zeros(len(X))
y_test_pred = np.zeros(len(X_test))

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}:")

    # Prepare training and validation data
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    # Define model
    model = lgb.train(params=model_params,
                      train_set=train_data,
                      valid_sets=[train_data, val_data],
                      verbose_eval=50,
                      )

    # Predict out of fold validation set and calculate evaluation metric
    y_val_pred = model.predict(X_val)
    print(f"ROC AUC: {roc_auc_score(y_val, y_val_pred)}")
    y_oof_pred[val_idx] = y_val_pred

    # Make predictions
    y_test_pred += model.predict(X_test)

# Calculate evaluation metric for out of fold validation set
y_test_pred = y_test_pred / N_SPLITS


print(f"Area under the Receiver Operating Characteristic curve (run 1): {roc_auc_score(y, y_oof_pred)}\n")

def get_predictions(y_pred):
    return (y_pred > 0.5).astype(int)

y_oof_pred = get_predictions(y_oof_pred)
print(f"Accuracy Score: {accuracy_score(y, y_oof_pred)}\n")
print(f"F1 Macro Score: {f1_score(y, y_oof_pred, average='macro')}\n")
precision, recall, _, _ = precision_recall_fscore_support(y, y_oof_pred, average=None)
print(f"Precision: {precision} \nRecall: {recall}\n")
cm = confusion_matrix(y, y_oof_pred)
sns.heatmap(cm, cmap='Blues', annot=True, fmt=".6g");


#pseudo labelling
train_df_pseudo = test[(pd.Series(y_test_pred) > 0.8) | (test['new_apache_3j_diagnosis'] == 1)].copy()
train_df_pseudo['diabetes_mellitus'] = 1
print(f"Potential {len(train_df_pseudo)} data points for pseudo labelling.")
print(f"train shape: {train.shape}, test shape: {test.shape},train_df_pseudo shape: {train_df_pseudo.shape}")
train =train.append(train_df_pseudo.sample(frac=0.5)).reset_index(drop=True) #A random 50% sample of train_df_pseudo, then append to train and rest index

# #other option- Upsampling
# from sklearn.utils import resample
# from sklearn.model_selection import train_test_split
#
# #Separate majority and minority classes
# df_majority = train[train.diabetes_mellitus==0]
# df_minority = train[train.diabetes_mellitus==1]
# #Resampling the minority levels to match the majority level
# df_minority_upsampled = resample(df_minority,
#                                  replace=True,  # sample with replacement
#                                  n_samples=102006,  # to match majority class
#                                  random_state=303)  # reproducible results
#
# # Combine majority class with upsampled minority class
# df_upsampled = pd.concat([df_majority, df_minority_upsampled])
#
# # Display new class counts
# df_upsampled.diabetes_mellitus.value_counts()

#run num.2:
# Prepare training and test data
X, y = train.drop(target_col, axis=1, inplace=False), train[target_col]
X_test = test.drop(target_col, axis=1, inplace=False)

list_of_cate_in_X_2=X.select_dtypes(include=['category']).columns.tolist()
print(list_of_cate_in_X_2)

print("---")
diff=[col for col in list_of_cate_in_X if col not in list_of_cate_in_X_2]
print(diff)
X['height_type']=X['height_type'].astype('category')
X['grouped_apache_3j_int']=X['grouped_apache_3j_int'].astype('category')

# Initialize variables
y_oof_pred = np.zeros(len(X))
y_test_pred = np.zeros(len(X_test))

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Fold {fold + 1}:")

    # Prepare training and validation data
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_val = X.iloc[val_idx].reset_index(drop=True)

    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    # Define model
    model = lgb.train(params=model_params,
                      train_set=train_data,
                      valid_sets=[train_data, val_data],
                      verbose_eval=50,
                      )

    # Predict out of fold validation set and calculate evaluation metric
    y_val_pred = model.predict(X_val)
    print(f"ROC AUC: {roc_auc_score(y_val, y_val_pred)}")
    y_oof_pred[val_idx] = y_val_pred

    # Make predictions
    y_test_pred += model.predict(X_test)

# Calculate evaluation metric for out of fold validation set
y_test_pred = y_test_pred / N_SPLITS

#feature importance
feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(importance_type="gain"),X.columns)), columns=['Value','Feature'])
fig, ax = plt.subplots(1,2, figsize=(16, 12))
feature_imp.sort_values('Value', ascending=True).tail(50).set_index('Feature').plot.barh(ax=ax[0])
ax[0].set_title('Top 50 Most Important Features')
feature_imp.sort_values('Value', ascending=True).head(50).set_index('Feature').plot.barh(ax=ax[1])
ax[1].set_title('Bottom 50 Least Important Features')
plt.tight_layout()
plt.show()

print(f"Area under the Receiver Operating Characteristic curve (run 2): {roc_auc_score(y, y_oof_pred)}\n")

y_oof_pred = get_predictions(y_oof_pred)
print(f"Accuracy Score: {accuracy_score(y, y_oof_pred)}\n")
print(f"F1 Macro Score: {f1_score(y, y_oof_pred, average='macro')}\n")
precision, recall, _, _ = precision_recall_fscore_support(y, y_oof_pred, average=None)
print(f"Precision: {precision} \nRecall: {recall}\n")


cm = confusion_matrix(y, y_oof_pred)
sns.heatmap(cm, cmap='Blues', annot=True, fmt=".6g");

#option- drop the least important features
#to_drop=feature_imp.sort_values('Value', ascending=True).head(50)['Feature'].to_list()
#print(f"to drop: {to_drop}")
#if 'new_apache_3j_diagnosis' in to_drop:
#    to_drop.remove('new_apache_3j_diagnosis')
#train.drop(to_drop, inplace=True, axis=1)
#test.drop(to_drop, inplace=True, axis=1)

#option -Ensemble example
#y_test_pred = 0.7 * y_test_pred + 0.3 * y_test_pred_model_2


