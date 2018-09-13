#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Filename: conns
# @Date:   : 2017-11-29 18:42
# @Author  : YuJun
# @Email   : yujun_mail@163.com


import src.settings as SETTINGS

# 因子数据库根目录
# DB_PATH = '/Users/davidyujun/Dropbox/FactorDB'
DB_PATH = SETTINGS.FACTOR_DB_PATH
# 个股基本信息数据相对目录
BASIC_INFO_PATH = 'ElementaryFactor/basic_info'
# 个股停牌信息数据相对目录
SUSPENSION_INOF_PATH = 'ElementaryFactor/suspension_info'
# 日行情复权数据相对目录
MKT_DAILY_FQ = 'ElementaryFactor/mkt_daily_FQ'
# 日行情非复权数据相对目录
MKT_DAILY_NOFQ = 'ElementaryFactor/mkt_daily_NoFQ'
# 分钟行情复权数据相对目录
MKT_MIN_FQ = 'ElementaryFactor/mkt_1min_FQ'
# 分钟行情非复权数据相对目录
MKT_MIN_NOFQ = 'ElementaryFactor/mkt_1min_NoFQ'
# 股本结构数据相对目录
CAP_STRUCT = 'ElementaryFactor/cap_struct'
# 主要财务指标相对目录
FIN_BASIC_DATA_PATH = 'ElementaryFactor/fin_data/fin_data_basics'
# 财务报表摘要相对目录
FIN_SUMMARY_DATA_PATH = 'ElementaryFactor/fin_data/fin_data_cwbbzy'
# 个股行业分类相对目录
INDUSTRY_CLASSIFY_DATA_PATH = 'ElementaryFactor/industry_classify'
# 个股IPO信息数据相对目录
IPO_INFO_PATH = 'ElementaryFactor/ipo_info'
# 一致预期数据相对目录
CONSENSUS_PATH = 'ElementaryFactor/consensus_data'

# 日行情复权数据的表头
MKT_DAILY_FQ_HEADER = ['code', 'date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'turnover1', 'turnover2', 'factor']
# 日行情非复权数据的表头
MKT_DAILY_NOFQ_HEADER = ['date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'turnover1', 'turnover2']
# 分钟行情复权数据的表头
MKT_MIN_FQ_HEADER = ['code', 'datetime', 'open', 'high', 'low', 'close', 'vol', 'amount', 'factor']
# 股票股本结构数据的表头
CAP_STRUCT_HEADER = ['code', 'date', 'reason', 'total', 'liquid_a', 'liquid_b', 'liquid_h']

# 主要财务数据的表头
FIN_BASIC_DATA_HEADER = ['ReportDate', 'BasicEPS', 'UnitNetAsset', 'UnitNetOperateCashFlow', 'MainOperateRevenue',
                         'MainOperateProfit', 'OperateProfit', 'InvestIncome', 'NonOperateNetIncome', 'TotalProfit',
                         'NetProfit', 'DeductedNetProfit', 'NetOperateCashFlow', 'CashEquivalentsChg', 'TotalAsset',
                         'CurrentAsset', 'TotalLiability', 'CurrentLiability', 'ShareHolderEquity', 'ROE']
# 财务报表摘要的表头
FIN_SUMMARY_DATA_HEADER = ['ReportDate', 'OperatingIncome', 'OperatingCost', 'OperatingProfit', 'TotalProfit',
                           'IncomeTax', 'NetProfit', 'EarningsPerShare', 'Cash', 'AccountsReceivable', 'Inventories',
                           'TotalCurrentAssets', 'NetFixedAssets', 'TotalAssets', 'TotalCurrentLiabilities',
                           'TotalNonCurrentLiabilities', 'TotalLiabilities', 'TotalShareholderEquity',
                           'InitialCashAndCashEquivalentsBalance', 'NetCashFlowsFromOperatingActivities',
                           'NetCashFlowsFromInvestingActivities', 'NetCashFlowsFromFinancingActivities',
                           'NetIncreaseInCashAndCashEquivalents', 'FinalCashAndCashEquivalentsBalance']

# 申万行业分类信息表头
SW_INDUSTRY_CLASSIFY_HEADER = ['ind_code', 'ind_name']

# 因子载荷文件持久化形式
# FACTOR_LOADING_PERSISTENCE_TYPE='shelve,csv'
FACTOR_LOADING_PERSISTENCE_TYPE='csv'
# 读取因子载荷采用的持久化形式，csv或shelve
USING_PERSISTENCE_TYPE='csv'

# 去极值方法中mad的乘数
CLEAN_EXTREME_VALUE_MULTI_CONST=5.2

# 财务数据中金额的单位
FIN_DATA_AMOUNT_UNIT=10000

# 绝对极小值
TINY_ABS_VALUE = 0.000001


if __name__ == '__main__':
    pass
