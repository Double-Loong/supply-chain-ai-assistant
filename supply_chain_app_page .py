import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# ===================== 全局配置 =====================
st.set_page_config(
    page_title="智链AI - 供应链智能预测助手",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session_state
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 加载数据
@st.cache_data
def load_data():
    df = pd.read_csv('daily_sales.csv')
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

daily_sales = load_data()

# ===================== 页面1：登录页 =====================
def page_login():
    st.markdown("""
    <style>
    /* 全局居中 + 顶部1/3定位 */
    .header-box {
        width: 600px;
        margin: 12vh auto 30px auto; /* 页面上方1/3处 */
        padding: 25px 30px;
        border: 2px solid #eee;
        border-radius: 12px;
        background-color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
    }
    /* 左列：大标题 */
    .title-left {
        font-size: 36px;
        font-weight: bold;
        color: #1f2937;
    }
    /* 右列：副标题 */
    .title-right {
        font-size: 18px;
        color: #666;
        font-weight: 500;
    }
    /* 登录表单卡片 */
    .login-card {
        width: 400px;
        margin: 0 auto;
        padding: 40px;
        border: 1px solid #eee;
        border-radius: 12px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # ===================== 顶部标题方框（两列布局 + 居中 + 方框） =====================
    st.markdown("""
    <div class="header-box">
        <div class="title-left">智链AI</div>
        <div class="title-right">供应链智能预测助手</div>
    </div>
    """, unsafe_allow_html=True)

    # ===================== 登录卡片 =====================
    with st.container():
        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        email = st.text_input("企业邮箱", value="your@company.com")
        password = st.text_input("密码", type="password", value="********")

        if st.button("登录", type="primary", use_container_width=True):
            st.session_state['logged_in'] = True
            st.session_state['page'] = 'overview'
            st.rerun()

        st.markdown("<br><p style='color:#aaa; font-size:12px;'>基于LightGBM · 40万订单数据训练</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ===================== 页面2：数据概览仪表盘 =====================
def page_overview():
    # 侧边栏导航
    with st.sidebar:
        st.title("智链AI v1.0")
        if st.button("数据概览", use_container_width=True, type="primary"):
            st.session_state['page'] = 'overview'
        if st.button("预测分析", use_container_width=True):
            st.session_state['page'] = 'forecast'
        if st.button("补货建议", use_container_width=True):
            st.session_state['page'] = 'replenish'
        if st.button("模型报告", use_container_width=True):
            st.session_state['page'] = 'report'
        if st.button("退出", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['page'] = 'login'
            st.rerun()

    # 顶部
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #eee;">
        <h3>数据概览</h3>
        <span style="color:#666;">管理员 | 退出</span>
    </div>
    """, unsafe_allow_html=True)

    # 核心指标
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("今日总销量", "1,234", "↑12%")
    col2.metric("本周预测准确率", "84.2%", "↑2.3%")
    col3.metric("库存预警SKU", "12", "-3")
    col4.metric("AI建议采纳率", "76%", "↑8%")

    # 销量趋势图（修复：生成稳定的模拟数据）
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("销量趋势（近30天）")
        # 生成近30天的日期序列和销量数据
        last_30_days = pd.date_range(end=pd.Timestamp.now(), periods=30)
        # 模拟带波动的销量数据，保证折线图正常显示
        sales_data = np.random.randint(100, 500, size=30)
        trend_df = pd.DataFrame({
            "order_date": last_30_days,
            "daily_qty": sales_data
        })
        fig = px.line(
            trend_df,
            x="order_date",
            y="daily_qty",
            title="",
            line_shape="linear"
        )
        fig.update_layout(
            xaxis_title="日期",
            yaxis_title="销量",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("库存预警")
        st.markdown("""
        - SKU-2001 🔴 库存紧张
        - SKU-2005 🟡 库存偏低
        - SKU-2012 🟡 库存偏低
        - SKU-2034 🔴 库存紧张
        """)

    # ===================== 品类销量分布（无-0.5·无报错·完美对齐） =====================
    st.subheader("品类销量分布")
    
    category_sales = pd.DataFrame({
        "category": ["美妆护肤", "3C电子", "服装鞋帽", "家居日用", "食品饮料"],
        "daily_qty": [3500, 2800, 2000, 1200, 500]
    })

    fig2 = px.bar(category_sales, x="category", y="daily_qty", color="category")

    # ✅ 100%兼容旧版Plotly · 彻底修复-0.5偏移
    fig2.update_layout(
        xaxis_type="category",      # 强制分类轴，消灭小数偏移
        xaxis_tickangle=0,          # 文字水平不旋转
        xaxis_automargin=True,      # 自动边距
        bargap=0.3,                 # 柱子间距
        height=380,
        showlegend=False
    )
    
    # ✅ 强制清除X轴范围，彻底去掉-0.5
    fig2.update_xaxes(range=None)

    st.plotly_chart(fig2, use_container_width=True)

# ===================== 页面3：预测分析（最终版，完全按截图实现） =====================
def page_forecast():
    # 侧边栏导航
    with st.sidebar:
        st.title("智链AI v3.0")
        if st.button("数据概览", use_container_width=True):
            st.session_state['page'] = 'overview'
        if st.button("预测分析", use_container_width=True, type="primary"):
            st.session_state['page'] = 'forecast'
        if st.button("补货建议", use_container_width=True):
            st.session_state['page'] = 'replenish'
        if st.button("模型报告", use_container_width=True):
            st.session_state['page'] = 'report'
        if st.button("退出", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['page'] = 'login'
            st.rerun()

    # 顶部标题
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #eee;">
        <h3>预测分析</h3>
    </div>
    """, unsafe_allow_html=True)

    # 1. 顶部筛选栏（品类 + SKU + 预测周期）
    col_top1, col_top2, col_top3, col_top4 = st.columns([2, 2, 1, 1])
    with col_top1:
        category_list = sorted(daily_sales['category'].unique())
        selected_category = st.selectbox("选择品类", category_list)
    with col_top2:
        sku_filtered = daily_sales[daily_sales['category'] == selected_category]
        sku_list = sorted(sku_filtered['product_id'].unique())
        selected_sku = st.selectbox("选择SKU", sku_list)
    with col_top3:
        forecast_days = st.selectbox("预测周期", [7, 14, 30], index=0)
    with col_top4:
        st.button("开始预测", type="primary", use_container_width=True)

    # 2. 当前商品信息栏
    sku_info = daily_sales[daily_sales['product_id'] == selected_sku].iloc[0]
    st.info(f"当前选中：{sku_info['product_name']}（SKU-{selected_sku}） | 品类：{selected_category} | 价格：{sku_info['price']} | 品牌：{sku_info['brand']}")

    # 3. 销量趋势与预测图
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("销量趋势与预测")
        sku_data = daily_sales[daily_sales['product_id'] == selected_sku].tail(60)
        fig = go.Figure()
        # 历史实际销量
        fig.add_trace(go.Scatter(
            x=sku_data['order_date'],
            y=sku_data['daily_qty'],
            name='历史实际',
            line=dict(color='#1f77b4', width=2)
        ))
        # 模拟AI预测线（随SKU和预测天数变化）
        last_date = sku_data['order_date'].max()
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=forecast_days)
        # 用近7天均值做基础预测，带10%趋势
        base_daily = sku_data.tail(7)['daily_qty'].mean()
        future_qty = [round(base_daily * (1.05 ** i), 1) for i in range(forecast_days)]
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_qty,
            name='AI预测',
            line=dict(color='#2ca02c', dash='dot', width=2)
        ))
        # 7日移动平均线
        fig.add_trace(go.Scatter(
            x=sku_data['order_date'],
            y=sku_data['daily_qty'].rolling(7).mean(),
            name='7日移动平均',
            line=dict(color='orange', dash='dash', width=2)
        ))
        fig.update_layout(
            xaxis_title="日期",
            yaxis_title="销量",
            height=400,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("预测详情")
        # 动态生成预测详情表
        pred_data = pd.DataFrame({
            "日期": future_dates.strftime("%Y-%m-%d %H:%M:%S"),
            "AI预测": future_qty,
            "传统法": [round(qty * 1.15, 1) for qty in future_qty],
            "实际值": ["-"] * forecast_days
        })
        st.dataframe(pred_data.head(5), use_container_width=True, hide_index=True)

    # 4. AI洞察与决策依据（动态变化）
    st.subheader("AI洞察与决策依据")
    # 动态计算关键指标
    total_predicted = round(sum(future_qty))
    start_date = future_dates[0].strftime("%Y-%m-%d")
    end_date = future_dates[-1].strftime("%Y-%m-%d")
    avg_daily = sku_data['daily_qty'].mean()
    avg_future = total_predicted / forecast_days
    growth_rate = (avg_future / avg_daily - 1) * 100

    # 根据增长情况生成不同洞察文案
    if growth_rate > 10:
        insight_text = f"""
        基于近7天趋势，未来{forecast_days}天预计销量{total_predicted}件，日均销量同比上升{round(growth_rate,1)}%，
        建议重点关注{start_date}至{end_date}的库存水位，提前加大备货。
        """
    elif growth_rate < -10:
        insight_text = f"""
        基于近7天趋势，未来{forecast_days}天预计销量{total_predicted}件，日均销量同比下降{round(-growth_rate,1)}%，
        建议控制补货节奏，优先消化现有库存。
        """
    else:
        insight_text = f"""
        基于近7天趋势，未来{forecast_days}天预计销量{total_predicted}件，整体趋势平稳，
        建议维持常规补货节奏，重点关注周末和促销节点的销量波动。
        """
    st.info(insight_text)

    # 5. 下方分两列：TOP5销量影响因素 + 对应营销建议（均动态变化）
    col_feat, col_marketing = st.columns([1, 1])

    with col_feat:
        st.markdown("**TOP5销量影响因素**")
        # 模拟随SKU和预测周期变化的特征重要性
        feature_imp = pd.DataFrame({
            "特征": ["近7天销量", "近30天均价", "周末标记", "用户活跃度", "价格变动"],
            "重要性": np.random.dirichlet(np.ones(5), size=1)[0]
        }).sort_values('重要性', ascending=True)
        fig_feat = px.bar(
            feature_imp,
            x='重要性',
            y='特征',
            orientation='h',
            color='重要性',
            color_continuous_scale='Blues',
            height=250
        )
        fig_feat.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_feat, use_container_width=True)

    with col_marketing:
        st.markdown("**对应营销建议**")
        # 根据TOP3特征动态生成建议
        advice_list = []
        top_features = feature_imp.sort_values('重要性', ascending=False)['特征'].head(3).tolist()

        if "近7天销量" in top_features:
            advice_list.append("✅ 近期销量增长快，建议加大备货")
        if "近30天均价" in top_features:
            advice_list.append("✅ 价格敏感，可适度促销")
        if "周末标记" in top_features:
            advice_list.append("✅ 周末效应明显，周末加大备货")
        if "用户活跃度" in top_features:
            advice_list.append("✅ 用户活跃，可投放站内广告")
        if "价格变动" in top_features:
            advice_list.append("✅ 价格变动影响大，稳定价格策略")

        for advice in advice_list:
            st.success(advice)

# ===================== 页面4：补货建议（已按你的建议改造） =====================
def page_replenish():
    # 侧边栏导航
    with st.sidebar:
        st.title("智链AI v3.0")
        if st.button("数据概览", use_container_width=True):
            st.session_state['page'] = 'overview'
        if st.button("预测分析", use_container_width=True):
            st.session_state['page'] = 'forecast'
        if st.button("补货建议", use_container_width=True, type="primary"):
            st.session_state['page'] = 'replenish'
        if st.button("模型报告", use_container_width=True):
            st.session_state['page'] = 'report'
        if st.button("退出", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['page'] = 'login'
            st.rerun()

    # 顶部
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #eee;">
        <h3>智能补货建议</h3>
    </div>
    """, unsafe_allow_html=True)

    # 【你的建议】：增加品类、SKU选择栏
    col_top1, col_top2 = st.columns(2)
    with col_top1:
        category_list = sorted(daily_sales['category'].unique())
        selected_category = st.selectbox("选择品类", category_list)
    with col_top2:
        sku_filtered = daily_sales[daily_sales['category'] == selected_category]
        sku_list = sorted(sku_filtered['product_id'].unique())
        selected_sku = st.selectbox("选择SKU", sku_list)

    # 补货参数设置
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("参数设置")
        current_stock = st.number_input("当前库存（件）", value=156)
        safety_days = st.slider("安全库存天数", 3, 14, 7)
        forecast_days = st.selectbox("预测周期", [7, 14, 30], index=0)
    with col_right:
        st.subheader("补货建议")
        avg_daily = daily_sales[daily_sales['product_id'] == selected_sku]['daily_qty'].mean()
        predicted_demand = avg_daily * forecast_days
        safety_stock = avg_daily * safety_days
        suggested_order = max(0, predicted_demand + safety_stock - current_stock)
        st.metric("建议补货量（件）", int(suggested_order))
        st.button("生成采购单", type="primary", use_container_width=True)

    # 库存预警与补货优先级
    st.subheader("库存预警与补货优先级")
    st.markdown("""
    <div style="display:flex; gap:20px;">
        <div style="flex:1; background:#ffebeb; padding:15px; border-radius:8px;">
            <h4 style="color:red;">库存预警（低库存+高需求）</h4>
            <p>当前库存仅够2.7天销售，低于安全线7天。建议立即启动采购流程。</p>
            <button style="background:red; color:white; border:none; padding:8px 16px; border-radius:4px;">紧急补货</button>
            <ul>
                <li>SKU-2001 完美日记唇釉</li>
                <li>SKU-2034 兰蔻粉底</li>
            </ul>
        </div>
        <div style="flex:1; background:#fff3e0; padding:15px; border-radius:8px;">
            <h4 style="color:orange;">优先周转（高销量+高库存）</h4>
            <button style="background:orange; color:white; border:none; padding:8px 16px; border-radius:4px;">先进周转</button>
            <ul>
                <li>SKU-2005 Apple iPad</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style="background:#e8f5e9; padding:15px; border-radius:8px;">
            <h4 style="color:green;">常规补货（低销量+低库存）</h4>
            <button style="background:green; color:white; border:none; padding:8px 16px; border-radius:4px;">常规补货</button>
            <ul>
                <li>SKU-2007 常规补货</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style="background:#f5f5f5; padding:15px; border-radius:8px;">
            <h4 style="color:grey;">清销风险（低销量+高库存）</h4>
            <button style="background:grey; color:white; border:none; padding:8px 16px; border-radius:4px;">促销清仓</button>
            <ul>
                <li>SKU-2012 促销清仓</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ===================== 页面5：模型效果报告（按你的要求改造） =====================
def page_report():
    # 侧边栏导航
    with st.sidebar:
        st.title("智链AI v3.0")
        if st.button("数据概览", use_container_width=True):
            st.session_state['page'] = 'overview'
        if st.button("预测分析", use_container_width=True):
            st.session_state['page'] = 'forecast'
        if st.button("补货建议", use_container_width=True):
            st.session_state['page'] = 'replenish'
        if st.button("模型报告", use_container_width=True, type="primary"):
            st.session_state['page'] = 'report'
        if st.button("退出", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['page'] = 'login'
            st.rerun()

    # 顶部标题
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #eee;">
        <h3>AI模型效果报告</h3>
    </div>
    """, unsafe_allow_html=True)

    # 1. 新增：品类 + 时间预测区间筛选栏
    col_top1, col_top2 = st.columns(2)
    with col_top1:
        category_list = ["全品类"] + sorted(daily_sales['category'].unique())
        selected_category = st.selectbox("选择品类", category_list)
    with col_top2:
        forecast_period = st.selectbox("选择时间预测区间", ["7天", "14天", "30天"], index=0)

    # 2. 根据筛选条件动态生成模型指标（模拟真实场景）
    # 为不同品类和预测周期生成略有差异的指标，模拟真实效果变化
    if selected_category == "全品类":
        base_mape = {"7天": 15.8, "14天": 18.2, "30天": 21.5}
        base_rmse = {"7天": 32.5, "14天": 36.8, "30天": 41.2}
        base_mae = {"7天": 18.5, "14天": 21.3, "30天": 24.7}
    elif selected_category == "美妆护肤":
        base_mape = {"7天": 14.2, "14天": 16.5, "30天": 19.8}
        base_rmse = {"7天": 29.8, "14天": 33.5, "30天": 38.1}
        base_mae = {"7天": 16.2, "14天": 19.1, "30天": 22.5}
    elif selected_category == "3C电子":
        base_mape = {"7天": 17.5, "14天": 20.1, "30天": 23.8}
        base_rmse = {"7天": 35.2, "14天": 39.6, "30天": 44.3}
        base_mae = {"7天": 20.8, "14天": 23.9, "30天": 27.4}
    else:
        base_mape = {"7天": 16.1, "14天": 18.9, "30天": 22.3}
        base_rmse = {"7天": 33.7, "14天": 37.9, "30天": 42.5}
        base_mae = {"7天": 19.3, "14天": 22.1, "30天": 25.6}

    # 获取当前选中周期的指标
    current_mape = base_mape[forecast_period]
    current_rmse = base_rmse[forecast_period]
    current_mae = base_mae[forecast_period]
    traditional_mape = {"7天": 25.3, "14天": 28.7, "30天": 32.1}[forecast_period]
    improvement_rate = round((1 - current_mape / traditional_mape) * 100, 1)

    # 3. 预测准确率对比（动态变化）
    st.subheader("预测准确率对比（MAPE - 越低越好）")
    comp_data = pd.DataFrame({
        "模型": ["LightGBM-AI", "历史同期法", "7日移动平均"],
        "MAPE": [current_mape, round(current_mape * 1.4, 1), round(current_mape * 1.6, 1)]
    })
    fig = px.bar(comp_data, x='模型', y='MAPE', color='模型', title="")
    st.plotly_chart(fig, use_container_width=True)

    # 4. 关键指标对比（动态变化）
    st.subheader("关键指标对比")
    metrics_data = pd.DataFrame({
        "模型": ["LightGBM-AI", "历史同期法", "7日移动平均"],
        "MAE": [current_mae, round(current_mae * 1.4, 1), round(current_mae * 1.6, 1)],
        "RMSE": [current_rmse, round(current_rmse * 1.4, 1), round(current_rmse * 1.6, 1)],
        "MAPE": [current_mape, round(current_mape * 1.4, 1), round(current_mape * 1.6, 1)]
    })
    st.dataframe(metrics_data, use_container_width=True)

    # 5. 模型决策依据 & 模型提升效果（动态变化）
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("模型决策依据（TOP5特征）")
        # 根据品类动态调整特征重要性（模拟不同品类的影响因素差异）
        if selected_category == "美妆护肤":
            features = ["近7天销量", "价格变动", "周末标记", "用户活跃度", "近30天均价"]
        elif selected_category == "3C电子":
            features = ["近7天销量", "近30天均价", "用户活跃度", "价格变动", "周末标记"]
        else:
            features = ["近7天销量", "近30天均价", "周末标记", "用户活跃度", "价格变动"]

        feature_imp = pd.DataFrame({
            "特征": features,
            "重要性": np.random.dirichlet(np.ones(5), size=1)[0]
        }).sort_values('重要性', ascending=True)
        fig_feat = px.bar(feature_imp, x='重要性', y='特征', orientation='h', color='重要性', color_continuous_scale='Blues')
        st.plotly_chart(fig_feat, use_container_width=True)

    with col2:
        st.subheader("模型提升效果")
        st.markdown(f"""
        <div style="background:#e8f5e9; padding:20px; border-radius:8px;">
            <h2 style="color:green;">{improvement_rate}%</h2>
            <p>LightGBM vs 传统方法 预测误差降低</p>
            <ul>
                <li>MAPE: {traditional_mape}% → {current_mape}%</li>
                <li>RMSE: {round(current_rmse*1.6,1)} → {current_rmse}</li>
                <li>MAE: {round(current_mae*1.6,1)} → {current_mae}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("模型技术栈")
    st.info(f"""
    - 算法：LightGBM 梯度提升树
    - 数据：40万订单 + 6万用户画像
    - 特征：30+维（时序特征 + 用户行为特征）
    - 训练：时间序列拆分，前80%训练后20%验证
    - 当前查看：{selected_category} | {forecast_period} 预测效果
    """)

# ===================== 主逻辑：页面路由 =====================
if not st.session_state['logged_in']:
    page_login()
else:
    if st.session_state['page'] == 'overview':
        page_overview()
    elif st.session_state['page'] == 'forecast':
        page_forecast()
    elif st.session_state['page'] == 'replenish':
        page_replenish()
    elif st.session_state['page'] == 'report':
        page_report()
