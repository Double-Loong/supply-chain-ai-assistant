# 03 - 模型训练
# 供应链销量预测 — 数据集划分 & 模型训练

## 功能说明
1. 按时间顺序划分训练集、验证集（前80%训练，后20%验证）
2. 筛选建模特征列，剔除无关字段
3. 构建训练与验证特征矩阵、目标变量
4. 预留接入传统时序基线模型与LightGBM模型训练接口

```python
# 加载特征数据
feature_df = pd.read_csv('features.csv')
feature_df['order_date'] = pd.to_datetime(feature_df['order_date'])

# 时间拆分（前80%训练，后20%验证）
split_date = feature_df['order_date'].quantile(0.8)
train_df = feature_df[feature_df['order_date'] < split_date].copy()
valid_df = feature_df[feature_df['order_date'] >= split_date].copy()

# 去除目标为空的行
train_df = train_df.dropna(subset=['target_7d'])
valid_df = valid_df.dropna(subset=['target_7d'])

# 特征与目标定义
exclude_cols = [
    'order_date', 'product_id', 'product_name', 'category', 'brand',
    'product_region_level', 'daily_qty', 'daily_amount',
    'daily_orders', 'daily_users', 'target_7d'
]

feature_cols = [c for c in train_df.columns if c not in exclude_cols]

X_train = train_df[feature_cols]
y_train = train_df['target_7d']
X_valid = valid_df[feature_cols]
y_valid = valid_df['target_7d']

# 评估函数
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

results = []

# 基线模型1：朴素预测
naive_pred = X_valid['lag_1'].values * 7
results.append(evaluate(y_valid, naive_pred, "朴素预测(昨日*7)"))

# 基线模型2：7日移动平均
ma_pred = X_valid['rolling_mean_7'].values * 7
results.append(evaluate(y_valid, ma_pred, "7日移动平均*7"))

# 基线模型3：双周同期平均
week_avg_pred = (X_valid['lag_7'].values + X_valid['lag_14'].values) / 2 * 7
results.append(evaluate(y_valid, week_avg_pred, "双周历史同期平均"))

# LightGBM 模型训练
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid)

model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, valid_data],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

# 模型预测
lgb_pred = model.predict(X_valid, num_iteration=model.best_iteration)
results.append(evaluate(y_valid, lgb_pred, "LightGBM"))

# 保存模型
import joblib
joblib.dump(model, 'models/lgb_model.pkl')
