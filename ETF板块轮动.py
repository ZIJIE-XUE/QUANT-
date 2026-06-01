from jqdata import *
import datetime
import math
import numpy as np
import pandas as pd  # ✅ 修复：缺少pandas导入
from scipy.optimize import minimize
from scipy.linalg import inv
import uuid

# etf_pool_config.py
etf_pool = [
    # 境外
    "513100.XSHG",  # 纳指ETF
    "159509.XSHE",  # 纳指科技ETF
    "513520.XSHG",  # 日经ETF
    "513030.XSHG",  # 德国ETF
    # 商品
    "518880.XSHG",  # 黄金ETF
    "159980.XSHE",  # 有色ETF
    "159985.XSHE",  # 豆粕ETF
    "159981.XSHE",  # 能源化工ETF
    "501018.XSHG",  # 南方原油
    # 债券
    "511090.XSHG",  # 30年国债ETF
    # 国内
    "513130.XSHG",  # 恒生科技
    "513690.XSHG",  # 港股红利
    "513120.XSHG",  # 港股创新药ETF
    "510180.XSHG",  # 上证180
    "159949.XSHE",  # 创业板50ETF
    "510410.XSHG",  # 资源
    "159928.XSHE",  # 消费ETF
    "512290.XSHG",  # 生物医药
    "588000.XSHG",  # 科创50
    "515070.XSHG",  # 人工智能ETF
    "515030.XSHG",  # 新能源车
    "516160.XSHG",  # 新能源ETF
    "512710.XSHG",  # 军工ETF
    "512000.XSHG",  # 券商ETF
    "512480.XSHG",  # 半导体
    "515250.XSHG",  # 智能汽车
    "562500.XSHG",  # 机器人ETF
    "561910.XSHG",  # 电池ETF
    "515050.XSHG",  # 5G通信
    "159995.XSHE",  # 芯片
    "515790.XSHG",  # 光伏
    "515000.XSHG"  # 科技
]

# 全局变量
g_strategys = {}
g_portfolio_value_proportion = [1]  # 测试版
g_positions = {i: {} for i in range(len(g_portfolio_value_proportion))}  # 记录每个子策略的持仓股票
g_weights = {}  # 全天候权重
g_channel = 'etfld'  # 请保持和ThsAutoTrader里面的channel一

# 核心资产轮动策略相关参数
g_etf_rotation = {
    "index": 0,
    "name": "核心资产轮动策略",
    "stock_sum": 1,
    "hold_list": [],
    "min_money": 500,  # 最小交易额(限制手续费)
    "etf_pool": etf_pool,
    "m_days": 25,  # 动量参考天数
    "enable_volume_check": True,  # 是否启用成交量检测
    "volume_lookback": 5,  # 历史成交量参考天数（默认20天）
    "volume_threshold": 2.0,  # 放量阈值（当日成交量/历史平均 > 该值视为放量）
    "ma_filter_days": 20,  # 均线过滤天数（可自定义）
    "enable_ma_filter": True,  # 是否启用均线过滤
}


############打开星球
def order_(context, security, vol):  # 只保留3个必要参数
    o = order(security, vol)
    return o


def initialize(context):
    set_option("avoid_future_data", True)  # 打开防未来函数
    set_option("use_real_price", True)  # 开启动态复权模式(真实价格)
    log.info("初始函数开始运行且全局只运行一次")
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'debug')
    set_slippage(FixedSlippage(0.0001), type="fund")
    set_slippage(FixedSlippage(0.003), type="stock")
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.001,
            open_commission=0.0003,
            close_commission=0.0003,
            close_today_commission=0,
            min_commission=5,
        ),
        type="stock",
    )
    # 设置货币ETF交易佣金0
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

    if g_portfolio_value_proportion[0] > 0:
        # ✅ 修改：交易时间改为每天14:50，先卖后买
        run_daily(etf_rotation_trade, "14:50")
    # 每日剩余资金购买货币ETF（保持不变）
    run_daily(end_trade, "14:59")


def process_initialize(context):
    print("重启程序")
    global g_strategys
    g_strategys = {
        "核心资产轮动策略": {
            "index": 0,
            "name": "核心资产轮动策略"
        }
    }


# 尾盘处理
def end_trade(context):
    marked = {s for d in g_positions.values() for s in d}
    current_data = get_current_data()
    for stock in context.portfolio.positions:
        if stock not in marked:
            price = current_data[stock].last_price
            pos = context.portfolio.positions[stock].total_amount
            if my_order(context, stock, -pos, price, 0):
                log.info(f"卖出{stock}因送股未记录在持仓中", price, pos)


def my_order(context, security, vol, price, target_position):
    o = order_(context, security, vol)
    return o


# 核心资产轮动策略实现
def get_etf_rotation_total_value(context):
    index = g_etf_rotation["index"]
    if not g_positions[index]:
        return 0
    return sum(context.portfolio.positions[key].price * value
               for key, value in g_positions[index].items())


def etf_rotation_order_target_value(context, security, value):
    strategy = g_etf_rotation
    current_data = get_current_data()

    # 检查标的是否停牌、涨停、跌停
    if current_data[security].paused:
        log.info(f"{security}: 今日停牌")
        return False

    if current_data[security].last_price == current_data[security].high_limit:
        log.info(f"{security}: 当前涨停")
        return False

    if current_data[security].last_price == current_data[security].low_limit:
        log.info(f"{security}: 当前跌停")
        return False

    # 获取当前标的的价格
    price = current_data[security].last_price

    # 获取当前策略的持仓数量
    current_position = g_positions[strategy["index"]].get(security, 0)

    # 所有策略中持仓数量
    current_position_all = context.portfolio.positions[
        security].total_amount if security in context.portfolio.positions else 0

    # 计算目标持仓数量
    target_position = (int(value / price) // 100) * 100 if price != 0 else 0

    # 计算需要调整的数量
    adjustment = target_position - current_position

    target_position_all = current_position_all + adjustment

    # 检查是否当天买入卖出
    closeable_amount = context.portfolio.positions[
        security].closeable_amount if security in context.portfolio.positions else 0
    if adjustment < 0 and closeable_amount == 0:
        log.info(f"{security}: 当天买入不可卖出")
        return False

    # 下单并更新持仓
    if adjustment != 0:
        o = my_order(context, security, adjustment, price, target_position_all)
        if o:
            # 更新持仓数量
            filled = o.filled if o.is_buy else -o.filled
            g_positions[strategy["index"]][security] = filled + current_position
            # 如果当前持仓为零，移除该证券
            if g_positions[strategy["index"]][security] == 0:
                g_positions[strategy["index"]].pop(security, None)
            # 更新持有列表
            strategy["hold_list"] = list(g_positions[strategy["index"]].keys())
            return True
    return False


def get_etf_premium_rate_real(context, etf_code):
    """
    在实盘中计算ETF基金的溢价率
    etf_code: ETF代码，如 '510050.XSHG'
    """
    # 获取当前数据对象
    etf_price = get_price(etf_code, start_date=context.previous_date, end_date=context.previous_date).iloc[-1]['close']
    iopv = \
    get_extras('unit_net_value', etf_code, start_date=context.previous_date, end_date=context.previous_date).iloc[
        -1].values[0]

    #  计算溢价率
    if iopv is not None and iopv != 0:
        premium_rate = (etf_price - iopv) / iopv * 100
    else:
        premium_rate = 0

    return premium_rate, etf_price, iopv


def etf_rotation_filter(context):
    strategy = g_etf_rotation
    current_data = get_current_data()

    # 1. 先对原始ETF池进行均线过滤（在动量计算前）
    filtered_pool = strategy["etf_pool"]  # 原始ETF池

    if strategy["enable_ma_filter"]:
        # 调用均线过滤函数，筛选出当前价 >= N日均价的ETF
        filtered_pool = filter_below_ma(
            stocks=filtered_pool,
            days=strategy["ma_filter_days"]
        )

    # 2. 仅对过滤后的ETF池计算动量评分
    data = pd.DataFrame(index=filtered_pool,
                        columns=["annualized_returns", "r2", "score"])

    for etf in filtered_pool:
        # 获取历史数据并计算当前价格
        df = attribute_history(etf, strategy["m_days"], "1d", ["close", "high"])
        prices = np.append(df["close"].values, current_data[etf].last_price)

        # 设置参数
        y = np.log(prices)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))

        # 计算年化收益率
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        data.loc[etf, "annualized_returns"] = math.exp(slope * 250) - 1

        # 计算R²
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        data.loc[etf, "r2"] = 1 - ss_res / ss_tot if ss_tot else 0

        # 计算得分
        data.loc[etf, "score"] = data.loc[etf, "annualized_returns"] * data.loc[etf, "r2"]

        # 过滤近3日跌幅超过5%的ETF
        if len(prices) >= 4 and min(prices[-1] / prices[-2], prices[-2] / prices[-3], prices[-3] / prices[-4]) < 0.95:
            data.loc[etf, "score"] = 0

    # 过滤ETF，并按得分降序排列
    data = data.query("0 < score < 5").sort_values(by="score", ascending=False)

    return data.index.tolist(), data


# 通用工具函数
def filter_untradeable_stock(stocks):
    current_data = get_current_data()
    return [
        stock
        for stock in stocks
        if current_data[stock].paused or current_data[stock].last_price in
           (current_data[stock].high_limit, current_data[stock].low_limit)
    ]


def check_etf_rotation_holdings(context):
    strategy = g_etf_rotation
    hold = list(g_positions[strategy["index"]].keys())
    if not hold:
        return []
    current_data = get_current_data()
    filtered = filter_limitup_stock(hold, 3)
    return [s for s in hold if s not in filtered and current_data[s].last_price < current_data[s].high_limit]


# ✅ 修复：添加缺失的filter_limitup_stock函数
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


# ✅ 合并：将原来的卖出和买入函数合并为一个，在14:50先卖后买
def etf_rotation_trade(context):
    strategy = g_etf_rotation
    current_data = get_current_data()
    hold_list = list(g_positions[strategy["index"]].keys())

    # 初始化止损冷却字典（如果不存在）
    if not hasattr(g, 'stop_loss_cooldown'):
        g.stop_loss_cooldown = {}

    # 清理过期的冷却记录（超过3天）
    today = context.current_dt.date()
    expired_stocks = []
    for stock, cooldown_date in g.stop_loss_cooldown.items():
        if (today - cooldown_date).days > 3:
            expired_stocks.append(stock)
    for stock in expired_stocks:
        del g.stop_loss_cooldown[stock]

    # 1. 市场择时检查
    prohibit_open = check_market_timing(context)  # 直接获取禁止开仓状态
    can_open = not prohibit_open  # 可以开仓状态

    money_etf = '511880.XSHG'  # 华宝添益货币ETF

    # 2. 如果持有货币ETF且市场允许开仓，卖出货币ETF
    if money_etf in hold_list and can_open:
        log.info(f"[市场择时] 市场转为多头信号，卖出货币ETF {money_etf}")
        etf_rotation_order_target_value(context, money_etf, 0)
        # 更新持仓列表
        hold_list.remove(money_etf) if money_etf in hold_list else None

    # 3. 市场禁止开仓时，确保持有货币ETF
    if prohibit_open:
        log.info("[市场择时] 市场处于空头信号，禁止开仓")
        # 如果已经持有货币ETF，直接返回，不做任何操作
        if money_etf in hold_list:
            log.info(f"[市场择时] 已持有货币ETF {money_etf}，保持持仓")
            return

        # 如果没有持仓或持有非货币ETF，切换至货币ETF
        if not hold_list or money_etf not in hold_list:
            # 卖出所有非货币ETF持仓
            for stock in hold_list:
                if stock != money_etf:
                    etf_rotation_order_target_value(context, stock, 0)

            # 买入货币ETF
            try:
                total_value = context.portfolio.total_value
                target_value = total_value * g_portfolio_value_proportion[strategy["index"]]
                last_price = current_data[money_etf].last_price

                if target_value > 0 and last_price > 0:
                    log.info(f"[市场择时] 市场处于空头信号，切换至货币ETF {money_etf}")
                    success = etf_rotation_order_target_value(context, money_etf, target_value)
                    if success:
                        stock_name = current_data[money_etf].name
                        log.info(
                            f"[ETF轮动] 买入：{money_etf}({stock_name}) - 原因：市场择时切换货币ETF，时间：{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')}，买入价格：{last_price:.3f}元")
            except Exception as e:
                log.error(f"买入货币ETF {money_etf}失败: {e}")
        return

    # 4. 计算当日目标列表
    targets, data = etf_rotation_filter(context)
    targets = targets[: strategy["stock_sum"]]

    # 5. 无符合条件的ETF时，切换至货币ETF
    if not targets:
        log.info("无符合条件的买入标的，切换至货币ETF")

        # 如果已经持有货币ETF，直接返回
        if money_etf in hold_list:
            log.info(f"[ETF轮动] 已持有货币ETF {money_etf}，保持持仓")
            return

        # 卖出所有持仓
        for stock in hold_list:
            etf_rotation_order_target_value(context, stock, 0)

        # 买入货币ETF
        try:
            total_value = context.portfolio.total_value
            target_value = total_value * g_portfolio_value_proportion[strategy["index"]]

            if target_value > 0:
                log.info(f"[ETF轮动] 买入货币ETF {money_etf}，金额：{target_value:.2f}元")
                success = etf_rotation_order_target_value(context, money_etf, target_value)
                if success:
                    stock_name = current_data[money_etf].name
                    last_price = current_data[money_etf].last_price
                    log.info(
                        f"[ETF轮动] 买入：{money_etf}({stock_name}) - 原因：无符合条件标的切换货币ETF，时间：{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')}，买入价格：{last_price:.3f}元")
        except Exception as e:
            log.error(f"买入货币ETF失败: {e}")
        return

    # 6. 执行卖出操作
    # 6.1 优先处理回撤控制相关的卖出条件
    sell_list = []
    for stock in hold_list:
        try:
            # 跳过货币ETF的回撤控制检查
            if stock == money_etf:
                continue

            stock_name = current_data[stock].name
            current_position = g_positions[strategy["index"]].get(stock, 0)
            if current_position <= 0:
                continue

            # 获取当前价格和成本价
            current_price = current_data[stock].last_price
            # ✅ 修复：从context获取真实成本价，而不是自己计算
            cost_price = context.portfolio.positions[stock].avg_cost if stock in context.portfolio.positions else 0

            # 计算收益率
            if cost_price > 0:
                return_rate = (current_price - cost_price) / cost_price * 100
            else:
                return_rate = 0

            # 6.1.1 三重硬止损条件
            # 条件1：单日跌幅≥4%
            hist_1d = attribute_history(stock, 1, '1d', ['close'])
            if len(hist_1d) >= 1:
                prev_close = hist_1d['close'].iloc[-1]
                day_return = (current_price - prev_close) / prev_close * 100
                if day_return <= -4:
                    sell_list.append((stock, "单日跌幅≥4%"))
                    continue

            # 条件2：3日累计跌幅≥6%
            hist_3d = attribute_history(stock, 3, '1d', ['close'])
            if len(hist_3d) >= 3:
                three_day_return = (current_price - hist_3d['close'].iloc[0]) / hist_3d['close'].iloc[0] * 100
                if three_day_return <= -6:
                    sell_list.append((stock, "3日累计跌幅≥6%"))
                    continue

            # 条件3：跌破20日均线且放量（≥2.5倍均量）
            hist_20d = attribute_history(stock, 20, '1d', ['close', 'volume'])
            if len(hist_20d) >= 20:
                ma20 = hist_20d['close'].mean()
                avg_volume = hist_20d['volume'].mean()

                # 获取当日成交量
                today_vol = attribute_history(stock, 1, '1d', ['volume']).iloc[-1]['volume']
                volume_ratio = today_vol / avg_volume

                if current_price < ma20 and volume_ratio >= 2.5:
                    sell_list.append((stock, "跌破20日均线且放量"))
                    continue

            # 6.1.2 动态止损上移条件
            stop_loss_trigger = False
            if return_rate >= 15:
                stop_loss_price = cost_price * 1.05  # 盈利15%，止损上移至成本+5%
                stop_loss_trigger = True
            elif return_rate >= 10:
                stop_loss_price = cost_price * 1.02  # 盈利10%，止损上移至成本+2%
                stop_loss_trigger = True
            elif return_rate < -2:  # 仅在亏损超过2%时触发止损
                stop_loss_price = cost_price * 0.98  # 亏损2%，止损线设为成本-2%
                stop_loss_trigger = True

            # 检查是否触发动态止损
            if stop_loss_trigger and current_price <= stop_loss_price:
                sell_reason = f"动态止损触发（止损线：{stop_loss_price:.2f}，当前价：{current_price:.2f}）"
                sell_list.append((stock, sell_reason))
                continue

        except Exception as e:
            log.warning(f"处理{stock}回撤控制条件失败: {e}")
            continue

    # 执行硬止损和动态止损的卖出操作
    for stock, reason in sell_list:
        if stock in hold_list:
            try:
                stock_name = current_data[stock].name
                log.info(
                    f"[ETF轮动] 卖出：{stock}({stock_name}) - 原因：{reason}，时间：{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')}，卖出价格：{current_data[stock].last_price:.3f}元")
                etf_rotation_order_target_value(context, stock, 0)

                # 为动态止损的股票添加3天冷却期
                if "动态止损" in reason:
                    if not hasattr(g, 'stop_loss_cooldown'):
                        g.stop_loss_cooldown = {}
                    g.stop_loss_cooldown[stock] = context.current_dt.date()
                    log.info(f"[ETF轮动] 为{stock}添加3天止损冷却期")

                # 从持仓列表中移除
                if stock in hold_list:
                    hold_list.remove(stock)
            except Exception as e:
                log.error(f"卖出{stock}失败: {e}")

    # 6.2 优先卖出放量的持仓ETF（若启用成交量检测，排除货币ETF）
    if strategy["enable_volume_check"]:
        for stock in hold_list:
            # 跳过货币ETF的放量检测
            if stock == money_etf:
                continue

            # 获取当前价格和昨日收盘价
            current_price = current_data[stock].last_price
            hist_data = attribute_history(stock, 1, '1d', ['close'])
            if hist_data.empty:
                continue
            prev_close = hist_data['close'].iloc[-1]
            price_change = (current_price - prev_close) / prev_close

            # 仅在价格下跌超过3%且放量时才进行放量检测
            if price_change < -0.03:
                vol_ratio = get_volume_ratio(
                    context,
                    stock,
                    strategy["volume_lookback"],
                    strategy["volume_threshold"]
                )
                if vol_ratio is not None:
                    stock_name = current_data[stock].name
                    # 放量，强制卖出
                    sell_reason = f"放量异常（{vol_ratio:.2f}倍）"
                    log.info(
                        f"[ETF轮动] 卖出：{stock}({stock_name}) - 原因：{sell_reason}，时间：{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')}，卖出价格：{current_data[stock].last_price:.3f}元")
                    etf_rotation_order_target_value(context, stock, 0)
                    # 从持仓列表中移除（避免重复处理）
                    if stock in hold_list:
                        hold_list.remove(stock)

    # 6.3 清仓不在目标列表中的标的（排除货币ETF）
    for stock in hold_list:
        if stock not in targets and stock != money_etf:
            stock_name = current_data[stock].name
            sell_reason = "不在目标列表中"
            log.info(
                f"[ETF轮动] 卖出：{stock}({stock_name}) - 原因：{sell_reason}，时间：{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')}，卖出价格：{current_data[stock].last_price:.3f}元")
            etf_rotation_order_target_value(context, stock, 0)

    # 6.4 若持仓超标，卖出目标列表中排名靠后的（排除货币ETF）
    current_hold_in_targets = [s for s in hold_list if s in targets]
    if len(current_hold_in_targets) > strategy["stock_sum"]:
        log.info(
            f"[ETF轮动] 持仓超标（当前{len(current_hold_in_targets)}只，上限{strategy['stock_sum']}只），卖出排名靠后标的")
        # 对持仓的ETF重新排序（根据动量评分）
        hold_data = data.loc[[etf for etf in data.index if etf in current_hold_in_targets]]
        if not hold_data.empty:
            hold_data = hold_data.sort_values(by="score", ascending=False)
            keep = hold_data.index.tolist()[: strategy["stock_sum"]]
            for stock in hold_data.index.tolist():
                if stock not in keep:
                    stock_name = current_data[stock].name
                    sell_reason = "持仓超标，排名靠后"
                    log.info(
                        f"[ETF轮动] 卖出：{stock}({stock_name}) - 原因：{sell_reason}，时间：{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')}，卖出价格：{current_data[stock].last_price:.3f}元")
                    etf_rotation_order_target_value(context, stock, 0)

    # 7. 执行买入操作
    # 更新持仓列表
    hold_list = list(g_positions[strategy["index"]].keys())
    current_hold_in_targets = [s for s in hold_list if s in targets]
    current_hold_count = len(current_hold_in_targets)

    portfolio = context.portfolio
    total_value = portfolio.total_value
    available_cash = portfolio.available_cash
    target_value = total_value * g_portfolio_value_proportion[strategy["index"]]

    for stock in targets:
        # 检查是否在止损冷却期内
        if hasattr(g, 'stop_loss_cooldown') and stock in g.stop_loss_cooldown:
            cooldown_days = (context.current_dt.date() - g.stop_loss_cooldown[stock]).days
            if cooldown_days < 3:
                log.info(f"[ETF轮动] {stock}处于止损冷却期（剩余{3 - cooldown_days}天），跳过买入")
                continue

        stock_name = current_data[stock].name
        weight = 1 / len(targets)
        target = target_value * weight
        last_price = current_data[stock].last_price
        current_position = g_positions[strategy["index"]].get(stock, 0)
        current_value = current_position * last_price

        if current_hold_count == 0:
            # 未持仓，计算买入需求
            need_buy_value = target - current_value
            actual_buy_value = min(need_buy_value, available_cash)
            if actual_buy_value <= max(strategy["min_money"], last_price * 100):
                log.info(
                    f"[ETF轮动] 跳过买入：{stock}({stock_name}) - 买入金额不足（需{need_buy_value:.2f}元，可用{available_cash:.2f}元）")
                continue

            # 执行买入
            actual_order_amount = etf_rotation_order_target_value(context, stock, target)
            if actual_order_amount:
                # 获取动量评分
                momentum_score = data.loc[stock, 'score'] if stock in data.index else 0
                log.info(
                    f"[ETF轮动] 买入：{stock}({stock_name}) - 时间：{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')}，买入价格：{last_price:.3f}元，动量评分：{momentum_score:.6f}")

        else:
            # 已持仓，判断是否补仓
            if current_value < target * 0.9:
                rebalance_amount = target - current_value
                actual_rebalance = min(rebalance_amount, available_cash)
                if actual_rebalance > max(strategy["min_money"], last_price * 100):
                    # 执行补仓
                    actual_rebalance_amount = etf_rotation_order_target_value(context, stock, target)
                    if actual_rebalance_amount:
                        # 获取动量评分
                        momentum_score = data.loc[stock, 'score'] if stock in data.index else 0
                        log.info(
                            f"[ETF轮动] 补仓：{stock}({stock_name}) - 时间：{context.current_dt.strftime('%Y-%m-%d %H:%M:%S')}，买入价格：{last_price:.3f}元，动量评分：{momentum_score:.6f}")


def get_volume_ratio(context, security, lookback_days, threshold):
    """
    计算标的成交量比值（当日成交量/历史平均成交量）
    返回：若放量（>threshold）则返回比值，否则返回None，异常时返回None
    """
    try:
        # 1. 获取历史成交量（N天平均）
        hist_data = attribute_history(security, lookback_days, '1d', ['volume'])
        if hist_data.empty or len(hist_data) < lookback_days:
            return None
        avg_volume = hist_data['volume'].mean()

        # 2. 获取当日实时成交量（分钟数据累加）
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

        # 3. 超过阈值视为放量
        return volume_ratio if volume_ratio > threshold else None
    except Exception as e:
        log.warning(f"成交量检测失败 {security}：{e}")
        return None


def calculate_rsrs(security, n=18):
    """
    计算RSRS（阻力支撑相对强度）指标
    参数:
        security: 标的代码
        n: 计算周期（默认18天）
    返回:
        rsrs_score: RSRS标准化分值
    """
    try:
        # 获取历史数据
        df = attribute_history(security, n + 5, '1d', ['high', 'low'])
        if len(df) < n:
            return 0

        # 计算每日的最高价和最低价
        high = df['high'].values
        low = df['low'].values

        # 计算线性回归斜率
        slopes = []
        for i in range(n):
            x = low[i:i + 10]
            y = high[i:i + 10]
            if len(x) < 10 or np.std(x) == 0:
                continue

            # 计算线性回归斜率
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)

        if not slopes:
            return 0

        # 计算标准化分值
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        if std_slope == 0:
            return 0

        rsrs_score = (slopes[-1] - mean_slope) / std_slope
        return rsrs_score

    except Exception as e:
        log.warning(f"计算{security} RSRS指标失败: {e}")
        return 0


def check_market_timing(context):
    """
    市场择时判断 - 取消择时功能，永远允许开仓
    返回:
        prohibit_open: 是否禁止开仓（True: 禁止开仓，False: 允许开仓）
    """
    # 取消择时功能，永远返回False，即允许开仓
    prohibit_open = False
    return prohibit_open


def filter_below_ma(stocks, days=20):
    """
    过滤掉当前价格小于N日均价的股票/ETF（N可自定义）
    参数:
        stocks: 待过滤的标的列表
        days: 均线天数（默认20日，可自定义为5/10/60等）
    返回:
        过滤后的标的列表（仅保留当前价 >= N日均价的标的）
    """
    if not stocks:
        return []

    current_data = get_current_data()
    filtered = []

    for stock in stocks:
        try:
            # 获取N日历史收盘价数据
            hist = attribute_history(stock, days, "1d", ["close"])
            if len(hist) < days:  # 确保有足够的历史数据（避免新股/刚上市ETF）
                log.debug(f"{stock} 历史数据不足{days}天，跳过过滤")
                continue

            # 计算N日均价
            ma_n = hist["close"].mean()
            # 获取当前价格
            current_price = current_data[stock].last_price

            # 保留当前价 >= N日均价的标的
            if current_price >= ma_n:
                filtered.append(stock)
            else:
                log.debug(f"{stock} 过滤（当前价 {current_price:.2f} < {days}日均价 {ma_n:.2f}）")

        except Exception as e:
            log.warning(f"计算{stock} {days}日均价失败: {e}")
            continue

    return filtered