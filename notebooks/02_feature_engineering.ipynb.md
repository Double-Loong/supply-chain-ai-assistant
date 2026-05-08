# 02 - 特征工程
# 供应链销量预测 — 时序特征构建 & 数据集合并

## 功能说明
1. 筛选Top10热销SKU
2. 构建时序特征（滞后、窗口、环比、日期特征）
3. 合并用户分层特征
4. 输出建模用特征表

---

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 重新加载清洗后的数据
daily_sales = pd.read_csv('daily_sales.csv')
daily_sales['order_date'] = pd.to_datetime(daily_sales['order_date'])

# 选择Top 10 SKU做预测
top_skus = daily_sales.groupby('product_id')['daily_qty'].sum().sort_values(ascending=False).head(10).index
sku_data = daily_sales[daily_sales['product_id'].isin(top_skus)].copy()

# 时序特征构建函数
def create_features(df, sku_id):
    sku_df = df[df['product_id'] == sku_id].copy()
    sku_df = sku_df.sort_values('order_date')

    # 补齐日期
    date_range = pd.date_range(start=sku_df['order_date'].min(), end=sku_df['order_date'].max(), freq='D')
    sku_df = sku_df.set_index('order_date').reindex(date_range).fillna(0).reset_index()
    sku_df = sku_df.rename(columns={'index': 'order_date'})

    # 时间特征
    sku_df['year'] = sku_df['order_date'].dt.year
    sku_df['month'] = sku_df['order_date'].dt.month
    sku_df['day'] = sku_df['order_date'].dt.day
    sku_df['dayofweek'] = sku_df['order_date'].dt.dayofweek
    sku_df['weekofyear'] = sku_df['order_date'].dt.isocalendar().week.astype(int)
    sku_df['is_weekend'] = (sku_df['dayofweek'] >= 5).astype(int)
    sku_df['is_month_start'] = sku_df['order_date'].dt.is_month_start.astype(int)
    sku_df['is_month_end'] = sku_df['order_date'].dt.is_month_end.astype(int)

    # 滞后特征
    for lag in [1, 3, 7, 14, 30]:
        sku_df[f'lag_{lag}'] = sku_df['daily_qty'].shift(lag)

    # 滑动窗口
    for window in [7, 14, 30]:
        sku_df[f'rolling_mean_{window}'] = sku_df['daily_qty'].shift(1).rolling(window=window).mean()
        sku_df[f'rolling_std_{window}'] = sku_df['daily_qty'].shift(1).rolling(window=window).std()
        sku_df[f'rolling_max_{window}'] = sku_df['daily_qty'].shift(1).rolling(window=window).max()

    # 周/月环比
    sku_df['lag_7_ratio'] = sku_df['daily_qty'] / (sku_df['lag_7'] + 1)
    sku_df['lag_30_ratio'] = sku_df['daily_qty'] / (sku_df['lag_30'] + 1)

    # 价格前向填充
    sku_df['price'] = sku_df['price'].fillna(method='ffill')

    # 目标：未来7天销量
    sku_df['target_7d'] = sku_df['daily_qty'].shift(-7).rolling(window=7).sum()

    return sku_df

# 批量构建特征
all_sku_features = []
for sku in top_skus:
    features = create_features(sku_data, sku)
    features['product_id'] = sku
    all_sku_features.append(features)

feature_df = pd.concat(all_sku_features, ignore_index=True)

# 合并用户分层数据
segment_daily = pd.read_csv('segment_daily.csv')
segment_daily['order_date'] = pd.to_datetime(segment_daily['order_date'])
feature_df = feature_df.merge(segment_daily, on='order_date', how='left')

# 填充缺失值
feature_df = feature_df.fillna(0)

# 保存特征表
feature_df.to_csv('features.csv', index=False, encoding='utf-8-sig')
