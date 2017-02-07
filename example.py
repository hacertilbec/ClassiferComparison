from functions import *

# convert csv files to dataframes
order_cols = ["server_time", "device", "session_id", "uid", "item_id", "order_id", "quantity"]
order = pd.read_csv("site_order_log.csv000",header=None, names=order_cols)

train_cols = ["impression_id","impression_datetime","uid","platform","inventory_type","app_code","os_version","model","network","is_click","is_conversion"]
train = pd.read_csv("retargeting_ad_data_train.csv000",header=None, names=train_cols)

product_cols = ["item_id", "price", "category1", "category2", "category3", "category4", "brarnd"]
product = pd.read_csv("site_product.csv000",header=None, names=product_cols)

# merge train, order, product dataframes based on 'uid' and
# 'item_id' columns respectively.
merged = mergeDataFrames([train,order,product], ['uid', 'item_id'])


# create classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

RFC = RandomForestClassifier(n_estimators = 10, random_state = 0, max_depth=20)
DTC = DecisionTreeClassifier()
LR = LogisticRegression()
GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
            max_depth=1, random_state=0)

# fit and measure performance of each classifier
classifierPerformances(merged, 'is_conversion', [RFC, DTC, LR, GBC], preprocessing = True, testSize = 0.4)
