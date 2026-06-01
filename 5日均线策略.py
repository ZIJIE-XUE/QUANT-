def initialize(context):
    g.security = '000001.XSHE'
    run_daily(market_open,time='every_bar') # 定义策略函数的名称

def market_open(context):
    security = g.security
    close_data = attribute_history(security,5,'1d',['close'])
    MA5 = close_data['close'].mean()
    current_price = close_data['close'][-1]

    cash = context.portfolio.available_cash

    if current_price > 1.01*MA5:
        order_value(security,cash)
        log.info('买入 %s' % (security))

    elif current_price < MA5 and context.portfolio.positions[security].closeable_amount > 0:
        order_target(security,0)
        log.info('卖出 %s' % (security))

    record(stock_price = current_price)