from jqdata import *
import datetime
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import inv
import uuid

# 优化后的ETF池：保留核心高收益赛道
etf_pool = [
    # 境外资产（低相关性）
    "513100.XSHG",  # 纳指ETF
    "513520.XSHG",  # 日经ETF
    # 商品资产（对冲股市风险）
    "518880.XSHG",  # 黄金ETF
    # 港股资产
    "513130.XSHG",  # 恒生科技
    # 核心赛道（表现最好的三个）
    "515070.XSHG",  # 人工智能ETF
    "512480.XSHG",  # 半导体ETF
    "562500.XSHG",  # 机器人ETF
    # 低波动宽基
    "510300.XSHG",  # 沪深300ETF
]

# 全局变量
g_strategys = {}
g_portfolio_value_proportion = [1]
g_positions = {i: {} for i in range(len(g_portfolio_value_proportion))}
g_max_prices = {i: {} for i in range(len(g_portfolio_value_proportion))}
g_weights = {}
g_channel = 'etfld'

# 核心策略参数（已优化）
g_etf_rotation = {
    "index": 0,
    "name": "双ETF动量轮动策略（优化版）",
    "stock_sum": 2,
    "hold_list": [],
    "min_money": 500,
    "etf_pool": etf_pool,
    "m_days": 25,
    "enable_volume_check": True,
    "volume_lookback": 5,
    "volume_threshold": 2.0,
    "ma_filter_days": 20,
    "enable_ma_filter": True,
    "ma_buffer": 0.01,  # ✅ 新增：均线过滤缓冲带1%
    "trailing_stop_loss": 0.07,  # ✅ 优化：移动止损从5%提高到7%
}


def order_(context, security, vol):
    o = order(security, vol)
    return o


def initialize(context):
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    log.info("双ETF动量轮动策略（优化版）初始化完成")
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'info')

    # 设置交易成本（ETF专属）
    set_slippage(FixedSlippage(0.0001), type="fund")
    set_slippage(FixedSlippage(0.003), type="stock")
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0.0001,
            close_commission=0.0001,
            close_today_commission=0,
            min_commission=0,
        ),
        type="fund",
    )
    # 货币ETF交易成本
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0,
            close_commission=0,
            close_today_commission=0,
            min_commission=0,
        ),
        type="mmf",
    )

    # 交易时间：每天14:50先卖后买
    run_daily(etf_rotation_trade, "14:50")
    run_daily(end_trade, "14:59")


def process_initialize(context):
    print("策略重启")
    global g_strategys
    g_strategys = {
        "双ETF动量轮动策略（优化版）": {
            "index": 0,
            "name": "双ETF动量轮动策略（优化版）"
        }
    }


def end_trade(context):
    marked = {s for d in g_positions.values() for s in d}
    current_data = get_current_data()
    for stock in context.portfolio.positions:
        if stock not in marked:
            price = current_data[stock].last_price
            pos = context.portfolio.positions[stock].total_amount
            if my_order(context, stock, -pos, price, 0):
                log.info(f"清理异常持仓：卖出{stock} {pos}份")


def my_order(context, security, vol, price, target_position):
    o = order_(context, security, vol)
    return o


def get_etf_rotation_total_value(context):
    index = g_etf_rotation["index"]
    if not g_positions[index]:
        return 0
    return sum(context.portfolio.positions[key].price * value
               for key, value in g_positions[index].items())


def etf_rotation_order_target_value(context, security, value):
    strategy = g_etf_rotation
    current_data = get_current_data()

    if current_data[security].paused:
        log.info(f"{security} 今日停牌，跳过交易")
        return False
    if current_data[security].last_price == current_data[security].high_limit:
        log.info(f"{security} 涨停，跳过买入")
        return False
    if current_data[security].last_price == current_data[security].low_limit:
        log.info(f"{security} 跌停，跳过卖出")
        return False

    price = current_data[security].last_price
    current_position = g_positions[strategy["index"]].get(security, 0)
    current_position_all = context.portfolio.positions[
        security].total_amount if security in context.portfolio.positions else 0

    # 计算目标持仓（100股整数倍）
    target_position = (int(value / price) // 100) * 100 if price != 0 else 0
    adjustment = target_position - current_position
    target_position_all = current_position_all + adjustment

    # T+1交易限制
    closeable_amount = context.portfolio.positions[
        security].closeable_amount if security in context.portfolio.positions else 0
    if adjustment < 0 and closeable_amount == 0:
        log.info(f"{security} 当日买入不可卖出，跳过")
        return False

    if adjustment != 0:
        o = my_order(context, security, adjustment, price, target_position_all)
        if o:
            filled = o.filled if o.is_buy else -o.filled
            g_positions[strategy["index"]][security] = filled + current_position

            # 记录最高价
            if o.is_buy:
                g_max_prices[strategy["index"]][security] = price

            # 如果当前持仓为零，移除记录
            if g_positions[strategy["index"]][security] == 0:
                g_positions[strategy["index"]].pop(security, None)
                g_max_prices[strategy["index"]].pop(security, None)

            strategy["hold_list"] = list(g_positions[strategy["index"]].keys())
            return True
    return False


def etf_rotation_filter(context):
    strategy = g_etf_rotation
    current_data = get_current_data()

    # 第一步：均线过滤（加入1%缓冲带）
    filtered_pool = strategy["etf_pool"]
    if strategy["enable_ma_filter"]:
        filtered_pool = filter_below_ma(
            stocks=filtered_pool,
            days=strategy["ma_filter_days"],
            buffer=strategy["ma_buffer"]
        )

    # 第二步：计算动量评分
    data = pd.DataFrame(index=filtered_pool,
                        columns=["annualized_returns", "r2", "score"])

    for etf in filtered_pool:
        df = attribute_history(etf, strategy["m_days"], "1d", ["close"])
        prices = np.append(df["close"].values, current_data[etf].last_price)

        y = np.log(prices)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))

        slope, intercept = np.polyfit(x, y, 1, w=weights)
        data.loc[etf, "annualized_returns"] = math.exp(slope * 250) - 1

        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        data.loc[etf, "r2"] = 1 - ss_res / ss_tot if ss_tot else 0

        data.loc[etf, "score"] = data.loc[etf, "annualized_returns"] * data.loc[etf, "r2"]

        # 过滤近3日暴跌超过5%的ETF
        if len(prices) >= 4 and min(prices[-1] / prices[-2], prices[-2] / prices[-3], prices[-3] / prices[-4]) < 0.95:
            data.loc[etf, "score"] = 0

    # 按得分降序排列
    data = data.query("0 < score < 5").sort_values(by="score", ascending=False)
    return data.index.tolist()[:strategy["stock_sum"]], data


def filter_untradeable_stock(stocks):
    current_data = get_current_data()
    return [
        stock for stock in stocks
        if current_data[stock].paused or current_data[stock].last_price in
           (current_data[stock].high_limit, current_data[stock].low_limit)
    ]


def filter_limitup_stock(stocks, days=3):
    """过滤连续涨停的股票"""
    current_data = get_current_data()
    filtered = []
    for stock in stocks:
        try:
            hist = attribute_history(stock, days, '1d', ['close', 'high_limit'])
            limitup_days = 0
            for i in range(days):
                if abs(hist['close'].iloc[i] - hist['high_limit'].iloc[i]) < 0.01:
                    limitup_days += 1
            if limitup_days >= days:
                filtered.append(stock)
        except:
            pass
    return filtered


def etf_rotation_trade(context):
    strategy = g_etf_rotation
    current_data = get_current_data()
    hold_list = list(g_positions[strategy["index"]].keys())

    # 初始化止损冷却字典
    if not hasattr(g, 'stop_loss_cooldown'):
        g.stop_loss_cooldown = {}

    # 清理过期冷却记录（3天）
    today = context.current_dt.date()
    expired = [s for s, d in g.stop_loss_cooldown.items() if (today - d).days > 3]
    for s in expired:
        del g.stop_loss_cooldown[s]

    money_etf = '511880.XSHG'

    # 1. 计算当日目标列表
    targets, data = etf_rotation_filter(context)
    log.info(f"今日动量排名前2：{targets}")

    # 2. 无符合条件标的时切换至货币基金
    if not targets:
        log.info("无符合条件的ETF，切换至货币基金")
        for stock in hold_list:
            if stock != money_etf:
                etf_rotation_order_target_value(context, stock, 0)
        if money_etf not in hold_list:
            etf_rotation_order_target_value(context, money_etf, context.portfolio.total_value)
        return

    # 3. 执行卖出操作
    sell_list = []
    for stock in hold_list:
        if stock == money_etf:
            continue

        current_position = g_positions[strategy["index"]].get(stock, 0)
        if current_position <= 0:
            continue

        current_price = current_data[stock].last_price

        # 移动止损
        max_price = g_max_prices[strategy["index"]].get(stock, current_price)
        if current_price > max_price:
            max_price = current_price
            g_max_prices[strategy["index"]][stock] = max_price

        if current_price < max_price * (1 - strategy["trailing_stop_loss"]):
            sell_reason = f"移动止损（最高价{max_price:.2f}，当前价{current_price:.2f}，回撤{(1 - current_price / max_price) * 100:.1f}%）"
            sell_list.append((stock, sell_reason))
            continue

        # 不在目标列表中则卖出
        if stock not in targets:
            sell_list.append((stock, "不在目标列表中"))

    # 执行卖出
    for stock, reason in sell_list:
        if stock in hold_list:
            log.info(f"卖出 {current_data[stock].name}({stock}) - {reason}，价格：{current_data[stock].last_price:.2f}")
            etf_rotation_order_target_value(context, stock, 0)
            if "移动止损" in reason:
                g.stop_loss_cooldown[stock] = today
                log.info(f"{stock} 进入3天止损冷却期")
            hold_list.remove(stock)

    # 4. 执行买入操作（✅ 优化：动态资金分配）
    total_value = context.portfolio.total_value
    available_cash = context.portfolio.available_cash

    # ✅ 关键优化：根据符合条件的ETF数量动态分配资金
    target_count = len(targets)
    if target_count == 0:
        per_etf_value = 0
    elif target_count == 1:
        per_etf_value = total_value  # 只有1只符合条件，全仓买入
    else:
        per_etf_value = total_value / target_count  # 多只符合条件，等权重分配

    for stock in targets:
        # 跳过冷却期的股票
        if stock in g.stop_loss_cooldown:
            log.info(f"{stock} 处于止损冷却期，跳过买入")
            continue

        current_position = g_positions[strategy["index"]].get(stock, 0)
        current_value = current_position * current_data[stock].last_price

        # 计算需要买入的金额
        need_buy = per_etf_value - current_value
        if need_buy <= max(strategy["min_money"], current_data[stock].last_price * 100):
            continue

        actual_buy = min(need_buy, available_cash)
        if actual_buy > 0:
            log.info(
                f"买入 {current_data[stock].name}({stock})，金额：{actual_buy:.2f}元，价格：{current_data[stock].last_price:.2f}")
            etf_rotation_order_target_value(context, stock, current_value + actual_buy)
            available_cash -= actual_buy


def get_volume_ratio(context, security, lookback_days, threshold):
    try:
        hist_data = attribute_history(security, lookback_days, '1d', ['volume'])
        if hist_data.empty or len(hist_data) < lookback_days:
            return None
        avg_volume = hist_data['volume'].mean()

        today = context.current_dt.date()
        df_vol = get_price(
            security,
            start_date=today,
            end_date=context.current_dt,
            frequency='1m',
            fields=['volume'],
            skip_paused=False,
            fq='pre',
            panel=True,
            fill_paused=False
        )
        if df_vol is None or df_vol.empty:
            return None

        current_volume = df_vol['volume'].sum()
        volume_ratio = current_volume / avg_volume
        return volume_ratio if volume_ratio > threshold else None
    except Exception as e:
        log.warning(f"成交量检测失败 {security}：{e}")
        return None


# ✅ 优化：加入缓冲带参数
def filter_below_ma(stocks, days=20, buffer=0.0):
    """
    过滤掉价格低于N日均线×(1-buffer)的标的
    buffer=0.01表示允许价格在均线下方1%以内
    """
    if not stocks:
        return []

    current_data = get_current_data()
    filtered = []

    for stock in stocks:
        try:
            hist = attribute_history(stock, days, "1d", ["close"])
            if len(hist) < days:
                continue
            ma_n = hist["close"].mean()
            current_price = current_data[stock].last_price

            # 加入缓冲带
            if current_price >= ma_n * (1 - buffer):
                filtered.append(stock)
        except Exception as e:
            log.warning(f"均线过滤失败 {stock}：{e}")
            continue

    return filtered