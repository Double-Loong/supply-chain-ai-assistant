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

# LightGBM模型训练
model = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='mae',
    verbose=50
)
