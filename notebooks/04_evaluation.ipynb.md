# 04 - 模型评估
# 供应链销量预测 — 模型指标计算 & 结果可视化

## 功能说明
1. 加载训练好的模型进行验证集预测
2. 计算MAE、RMSE、MAPE等回归评估指标
3. 绘制真实值与预测值对比曲线
4. 输出特征重要性，分析关键影响因子

```python
# 中文绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载模型 & 预测
y_pred = model.predict(X_valid)

# 评估指标
mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
mape = np.mean(np.abs((y_valid - y_pred) / (y_valid + 1))) * 100

print(f"✅ 验证集 MAE: {mae:.2f}")
print(f"✅ 验证集 RMSE: {rmse:.2f}")
print(f"✅ 验证集 MAPE: {mape:.2f}%")

# 特征重要度
plt.figure(figsize=(10,6))
lgb.plot_importance(model, max_num_features=15)
plt.title('特征重要度')
plt.tight_layout()

# 真实值 vs 预测值对比
plt.figure(figsize=(14,5))
plt.plot(y_valid.values[:100], label='真实值', alpha=0.7)
plt.plot(y_pred[:100], label='预测值', alpha=0.7)
plt.title('未来7天销量预测对比')
plt.legend()
plt.tight_layout()
