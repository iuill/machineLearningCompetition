# Kaggle - Titanic

https://www.kaggle.com/competitions/titanic/overview

## 環境

* python: 3.10.4 | packaged by conda-forge | (main, Mar 30 2022, 08:38:02) [MSC v.1916 64 bit (AMD64)]
* pandas: 1.4.4
* pandas-profiling: 3.3.0
* numpy: 1.23.3
* lightgbm: 3.3.2
* scikit-learn: 1.1.2
* logging: 0.5.1.2

## コミットID、スコアメモ書き

| date | commit id | model file name | score | note |
| ---- | ------ | ----------------| ----- | ---- |
| 2022/10/02 | f24c4d6a980cf0cd342e4cd1ae228360669621f4 | model_lgb_holdout.pkl | 0.75358 | LightGBM + Hold-out |
| 2022/10/02 | f24c4d6a980cf0cd342e4cd1ae228360669621f4 | model_lgb_crossvalidation.pkl | 0.62200 | LightGBM + CV10分割 |
| 2022/10/02 | a521cf0c14b514bfd3cb3990f11cd6a8313e4acc | model_lgb_grid_search_cv.pkl | 0.76315 | LightGBM + GridSearchCV + KFold3分割 |

## 備忘

* このコンペのEvaluationはAccuracy
    * 一方でこのコードではf1を使っているので、ちょっと不適切...
* ネット見ると、特徴量エンジニアリングでさらにスコアを上げる方法がある
