from io import StringIO
import math
import platform
import sys
import os
import numpy as np
import pandas as pd
import pandas_profiling as pdp
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pickle
import sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
#from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score
#from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
# import warnings
# warnings.simplefilter('ignore')

import common

# -----------------------------------------------------
# Initialize: Logger
# -----------------------------------------------------

import logging

#
# create logger
#
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#
# create handler and set level to debug
#

# コンソール用

handler_console = logging.StreamHandler()
#handler_console.setLevel(logging.DEBUG)
handler_console.setLevel(logging.INFO)

# ファイル用

handler_file = logging.handlers.RotatingFileHandler("trace.log", maxBytes=1024*1024, backupCount=2)
#handler_file.setLevel(logging.DEBUG)
handler_file.setLevel(logging.INFO)


#
# create formatter
#
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#
# add formatter to handler
#
handler_console.setFormatter(formatter)
handler_file.setFormatter(formatter)

#
# add handler to logger
#
logger.addHandler(handler_console)
logger.addHandler(handler_file)


# -----------------------------------------------------
# Initialize: Dependencies logging
# -----------------------------------------------------

splitterCharNum = 70

logger.info("-" * splitterCharNum)

logger.info("ScriptName: " + os.path.basename(__file__))

logger.info("Environment")
logger.info("  python: " + sys.version)
logger.info("  platform: " + platform.platform())
logger.info("  pandas: " + pd.__version__)
logger.info("  pandas-profiling: " + pdp.__version__)
logger.info("  numpy: " + np.__version__)
#logger.info("  xgboost: " + xgb.__version__)
logger.info("  lightgbm: " + lgb.__version__)
logger.info("  scikit-learn: " + sklearn.__version__)
logger.info("  logging: " + logging.__version__)

logger.info("-" * splitterCharNum)

# -----------------------------------------------------
# main
# -----------------------------------------------------

logger.info("[BEGIN] initialize dataclasses")


logger.info("[BEGIN] read data")

fname_train = "./input/train.csv"
fname_test = "./input/test.csv"
fname_sample_submission = "./input/gender_submission.csv"


data_train_raw = pd.read_csv(fname_train, engine='python', encoding='utf-8')
data_test_raw = pd.read_csv(fname_test, engine='python', encoding='utf-8')
data_concat_raw = pd.concat([data_train_raw, data_test_raw], axis=0, sort=False)
data_concat_sanitized = data_concat_raw.copy()

data_sample_submission = pd.read_csv(fname_sample_submission, engine='python', encoding='utf-8')

logger.info(">> data_train_raw")
common.dfInfoLogger(logger, data_train_raw)
logger.info(">> data_test_raw")
common.dfInfoLogger(logger, data_test_raw)
logger.info(">> data_concat_raw")
common.dfInfoLogger(logger, data_concat_raw)

sturges = lambda n: math.ceil(math.log2(n*2))
nbins = sturges(len(data_train_raw))


# -----------------------------------------------------
# Data Cleaning
# -----------------------------------------------------

logger.info("[BEGIN] data cleaning")

#
# ID除いた状態の重複件数を確認
# - 0件
#
dupNum = data_concat_sanitized.duplicated(subset=data_concat_sanitized.columns.drop("PassengerId")).sum()
logger.info(f">> duplicated: {dupNum}")

#
# Cabin
# - 欠損値多すぎなので削除する
# 
data_concat_sanitized = data_concat_sanitized.drop("Cabin", axis=1)

#
# Embarked
# - 欠損値を埋める（一番多い値で決め打ち）
#
data_concat_sanitized.loc[data_concat_sanitized["Embarked"].isnull(), "Embarked"] = "S"

#
# 年齢の欠損値を埋める
#
AgeAvg = int(data_concat_sanitized["Age"].mean())
data_concat_sanitized.loc[data_concat_sanitized["Age"].isnull(), "Age"] = AgeAvg

common.dfInfoLogger(logger, data_concat_sanitized)

# -----------------------------------------------------
# 特徴量エンジニアリング
# -----------------------------------------------------

logger.info("[BEGIN] feature enginneering")

#
# 年齢をカテゴライズ
# - 幼児ゾーンは死亡率低めなので二極でカテゴライズする
#
data_concat_sanitized["AgeCatTmp"] = pd.cut(data_concat_sanitized["Age"], bins=nbins, precision=1, labels=[i for i in range(nbins)])
data_concat_sanitized["AgeRisk"] = 0
data_concat_sanitized.loc[data_concat_sanitized["AgeCatTmp"] != 0, "AgeRisk"] = 1

data_concat_sanitized = data_concat_sanitized.drop("Age", axis=1)
data_concat_sanitized = data_concat_sanitized.drop("AgeCatTmp", axis=1)

#
# Pclass
# - たぶんこのままでいい気がする
# - one-hot-encodingの対象で
#

#
# Fare
# - 安いゾーンは死亡率高めなので二極でカテゴライズする
#
data_concat_sanitized["FareCatTmp"] = pd.cut(data_concat_sanitized["Fare"], bins=nbins, precision=1, labels=[i for i in range(nbins)])
data_concat_sanitized["FareRisk"] = 0
data_concat_sanitized.loc[data_concat_sanitized["FareCatTmp"] != 0, "FareRisk"] = 1
data_concat_sanitized = data_concat_sanitized.drop("FareCatTmp", axis=1)

#
# SibSp, Parchの合成
#
data_concat_sanitized["FamilyNum"] = data_concat_sanitized["SibSp"] + data_concat_sanitized["Parch"]
data_concat_sanitized["FamilyRisk"] = "Low"
data_concat_sanitized.loc[data_concat_sanitized["FamilyNum"].isin([0, 4, 5, 6]), "FamilyRisk"] = "Middle"
data_concat_sanitized.loc[data_concat_sanitized["FamilyNum"] >= 7, "FamilyRisk"] = "High"

# ↓どこまで残すか・・・
removedCols = []
removedCols += ["FamilyNum"]
removedCols += ["SibSp"]
removedCols += ["Parch"]
data_concat_sanitized = data_concat_sanitized.drop(removedCols, axis=1)

#
# 階級（名前から取得）
# - 階級によって生死が大きく変わるのでカテゴライズ
# - Mr, Miss, Mrs, Masterのみ個別化、他は数が少ないので統合
#
data_concat_sanitized["NameTitle"] = data_concat_sanitized["Name"].str.extract("([A-Za-z]+)\.", expand=False)
data_concat_sanitized["NameTitleRisk"] = 4
data_concat_sanitized.loc[data_concat_sanitized["NameTitle"] == "Mr", "NameTitleRisk"] = 0
data_concat_sanitized.loc[data_concat_sanitized["NameTitle"] == "Miss", "NameTitleRisk"] = 1
data_concat_sanitized.loc[data_concat_sanitized["NameTitle"] == "Mrs", "NameTitleRisk"] = 2
data_concat_sanitized.loc[data_concat_sanitized["NameTitle"] == "Master", "NameTitleRisk"] = 3

data_concat_sanitized = data_concat_sanitized.drop(["Name", "NameTitle"], axis=1)

#
# one-hot-encoding
#

dummyColNames = []
dummyColNames += ["Pclass"]
dummyColNames += ["Embarked"]
dummyColNames += ["Sex"]
dummyColNames += ["FamilyRisk"]
dummyColNames += ["NameTitleRisk"]
#data_concat_sanitized = pd.get_dummies(data_concat_sanitized, drop_first=True, columns=dummyColList)
data_concat_sanitized = pd.get_dummies(data_concat_sanitized, drop_first=False, columns=dummyColNames)
#print(fnData.columns)

#
# 不要列削除
#
data_concat_sanitized = data_concat_sanitized.drop('Ticket',axis=1)

#
# ID削除
#   trainとtestをconcatしてから特徴量エンジニアリングするので、
#   念のためID削除はぎりぎりまで避ける
#
data_concat_sanitized = data_concat_sanitized.drop('PassengerId',axis=1)

# -----------------------------------------------------
# 説明変数、目的変数、学習データ、検証データの分割
# -----------------------------------------------------

data_train_sanitized = data_concat_sanitized[data_concat_sanitized["Survived"].notnull()]
data_train_sanitized_x = data_train_sanitized.drop("Survived", axis=1)
data_train_sanitized_y = data_train_sanitized["Survived"]

data_test_sanitized = data_concat_sanitized[data_concat_sanitized["Survived"].isnull()]
data_test_sanitized_x = data_test_sanitized.drop("Survived", axis=1)

# test = kaggleの予測値提出用のデータ
# valid = 学習モデルの予測確認用
data_train_x, data_valid_x, data_train_y, data_valid_y \
    = train_test_split(data_train_sanitized_x, data_train_sanitized_y, stratify=data_train_sanitized_y, test_size=0.33, random_state=0)

#
# 不均衡データへの対処
# - アップサンプリング
# - ダウンサンプリング
# - 損失関数の調整
#

sm = SMOTE(random_state=42)
data_train_imb_x, data_train_imb_y \
    = sm.fit_resample(data_train_sanitized_x, data_train_sanitized_y)


# -----------------------------------------------------
# 学習、検証
# -----------------------------------------------------

#
# 学習: ホールドアウト法
#

logger.info("[BEGIN] LightGBM learn: hold-out")

lgb_train = lgb.Dataset(data_train_imb_x, data_train_imb_y)
lgb_eval = lgb.Dataset(data_valid_x, data_valid_y)

lgb_params = {"objective":"binary"}

evals_result = {}

lgb_model = lgb.train(params=lgb_params,
                      train_set=lgb_train,
                      valid_sets=[lgb_train, lgb_eval],
                      early_stopping_rounds=20,
                      evals_result=evals_result,
                      verbose_eval=10)

file = 'model_lgb_holdout.pkl'
pickle.dump(lgb_model, open(file, 'wb'))

pred_y = lgb_model.predict(data_valid_x, num_iteration=lgb_model.best_iteration)
pred_y = pred_y.round(0)
logger.info(">> classification_report [predict]:\r\n{}".format(classification_report(data_valid_y, pred_y)))

plt.plot(evals_result["training"]["binary_logloss"], label="train_loss")
plt.plot(evals_result["valid_1"]["binary_logloss"], label="valid_loss")
plt.legend()
plt.show()


#
# 学習: k分割交差検証
#

logger.info("[BEGIN] LightGBM learn: cross-validation")

#kf = KFold(n_splits=10, shuffle=True, random_state=0)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

score_list = []
models = []

for fold_, (train_index, valid_index) in enumerate(kf.split(data_train_imb_x, data_train_imb_y)):
    logger.info(f">> fold:{fold_ + 1} start...")
    train_x = data_train_imb_x.iloc[train_index]
    valid_x = data_train_imb_x.iloc[valid_index]
    train_y = data_train_imb_y[train_index]
    valid_y = data_train_imb_y[valid_index]
    
    lgb_train = lgb.Dataset(data_train_imb_x, data_train_imb_y)
    lgb_valid = lgb.Dataset(data_valid_x, data_valid_y)
    
    lgb_params = {"objective":"binary"}
    
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=20,
        verbose_eval=10
    )

    pred_y = lgb_model.predict(data_valid_x, num_iteration=lgb_model.best_iteration)
    pred_y = pred_y.round(0)
    f1 = classification_report(data_valid_y, pred_y, output_dict=True)["macro avg"]["f1-score"]
    score_list.append(f1)
    models.append(lgb_model)


file = 'model_lgb_crossvalidation.pkl'
pickle.dump(models, open(file, 'wb'))

logger.info(f">> average: {np.mean(score_list)}")


# -----------------------------------------------------
# Submit用の予測
# -----------------------------------------------------

#
# hold-out
#

file = 'model_lgb_holdout.pkl'
model = pickle.load(open(file, 'rb'))

pred = model.predict(data_test_sanitized_x).round(0)
pred = np.array(pred, dtype=np.int32)
data_sample_submission["Survived"] = pred
data_sample_submission.to_csv("pred_lgb_holdout.csv", index=False)

#
# cross validation
#

file = 'model_lgb_crossvalidation.pkl'
models = pickle.load(open(file, 'rb'))

pred_list = []
for m in models:
    pred_list.append(m.predict(data_test_sanitized_x))
    
pred = np.mean(pred_list).round(0)
pred = np.array(pred, dtype=np.int32)
data_sample_submission["Survived"] = pred
data_sample_submission.to_csv("pred_lgb_cv.csv", index=False)


logger.info("finished...")
input()
