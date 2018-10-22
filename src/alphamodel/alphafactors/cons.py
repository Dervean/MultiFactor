#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Filename: cons
# @Date:   : 2017-12-06 19:26
# @Author  : YuJun
# @Email   : yujun_mail@163.com

from src.util.dottabledict import DottableDict
import src.settings as SETTINGS

# 基准代码, 用于因子载荷计算中日期序列提取等
BENCHMARK_CODE = 'SH000001'

# SmartMoney因子的配置参数
SMARTMONEY_CT = DottableDict({'days_num': 10,                               # 计算因子载荷所需分钟行情的天数
                              'db_file': 'AlphaFactor/SmartMoney',          # 因子载荷的保存文件路径名（相对于因子数据库根目录的相对路径）
                              'month_end': True,                            # 是否只计算月末的因子载荷值
                              'component': [],                              # 成分因子列表
                              'backtest_path': 'FactorBackTest/SmartQ',     # 历史回测结果文件保存路径
                              'constituent_ratio': 0.1                      # 多头组合的选股比例
                              })
# APM因子的配置参数
APM_CT = DottableDict({'index_code': '000001',                              # 指数代码
                       'days_num': 20,                                      # 计算因子载荷所需分钟行情的天数
                       'db_file': 'AlphaFactor/APM',                        # 因子载荷的保存文件路径名（相对于因子数据库根目录的相对路径）
                       'month_end': True,                                   # 是否只计算月末的因子载荷
                       'component': [],                                     # 成分因子列表
                       'pure_apm_db_file': 'Sentiment/APM/PureAPM',
                       'backtest_path': 'FactorBackTest/APM',               # 非纯净因子历史回测结果文件的保存路径（相对于因子数据库根目录的相对路径）
                       'pure_backtest_path': 'FactorBackTest/PureAPM',      # 纯净因子历史回测结果文件的相对保存路径
                       'constituent_ratio': 0.1                             # 多头组合的选股比例
                       })
# IntradayMementum因子的配置参数
INTRADAYMOMENTUM_CT = DottableDict({'days_num': 20,                                             # 计算因子载荷所需分钟行情的天数
                                    'db_file': 'AlphaFactor/IntradayMomentum',  # 日内时点动量因子载荷的保存文件路径名（相对于因子数据库根目录的相对路径）
                                    'month_end': True,                          # 是否只计算月末的因子载荷
                                    'optimized': False,                         # 是否计算最优化权重
                                    'synthesized': True,                       # 是否计算合成日内动量因子
                                    'factor_ic_file': 'AlphaFactor/IntradayMomentum/raw/intradaymomentum_ic.csv',   # 日内各时段动量因子的IC数据文件
                                    'optimal_weight_file': 'AlphaFactor/IntradayMomentum/raw/optimal_weight.csv',   # 日内因子最优权重文件相对路径
                                    'backtest_path': 'FactorBackTest/IntradayMomentum'          # 历史回测结果文件的保存路径（相对于因子数据库根目录的相对路径）
                                    })
# CYQ筹码分布因子的配置参数
# CYQ_CT = DottableDict({'days_num': 60,                                      # 计算因子载荷所需日K线行情的天数
#                        'db_file': 'Sentiment/CYQ/CYQ',                      # 筹码分布因子载荷的保存文件相对路径名
#                        'proxies_db_file': 'Sentiment/CYQ/CYQ_proxies',      # 筹码分布代理变量的保存文件相对路径名
#                        'proxies_weight_file': 'Sentiment/CYQ/CYQ_weight.csv',   # 筹码分布代理变量权重文件相对路径名
#                        'backtest_path': 'FactorTest/CYQ'                    # 历史回测结果文件的相对路径名
#                        })

# # CYQ筹码分布因子的配置参数
# CYQ_CT = DottableDict({'db_file': 'AlphaFactor/CYQ/',                     # 筹码分布因子载荷的保存文件相对路径
#                        'CYQ_rp_file': 'cyq_rp/CYQ_rp',                  # 筹码分布相对价格因子保存文件的相对路径
#                        'backtext_path': 'FactorBackTest/CYQ'            # 历史回测结果文件的相对路径
#                        })

# CYQRP筹码分布因子的配置参数
CYQRP_CT = DottableDict({'db_file': 'AlphaFactor/CYQ/CYQRP/',
                         'listed_days': 90})

# INTRADAYLIQUIDITY日内流动性因子的配置参数
INTRADAYLIQUIDITY_CT = DottableDict({'db_file': 'AlphaFactor/IntradayLiquidity',
                                     'days_num': 20,
                                     'listed_days': 90})
LIQ1_CT = DottableDict({'db_file': 'AlphaFactor/IntradayLiquidity/liq1'})

LIQ2_CT = DottableDict({'db_file': 'AlphaFactor/IntradayLiquidity/liq2'})

LIQ3_CT = DottableDict({'db_file': 'AlphaFactor/IntradayLiquidity/liq3'})

LIQ4_CT = DottableDict({'db_file': 'AlphaFactor/IntradayLiquidity/liq4'})

# EPTTM因子的配置参数
EPTTM_CT = DottableDict({'db_file': 'AlphaFactor/Value/EPTTM',
                         'listed_days': 90})

# SPTTM因子的配置参数
SPTTM_CT = DottableDict({'db_file': 'AlphaFactor/Value/SPTTM',
                         'listed_days': 90})

# OperateRevenueYoY因子的配置参数
OPERATEREVENUEYOY_CT = DottableDict({'db_file': 'AlphaFactor/Growth/OperateRevenueYoY',
                                     'listed_days': 90})

# OperateProfitYoY因子的配置参数
OPERATEPROFITYOY_CT = DottableDict({'db_file': 'AlphaFactor/Growth/OperateProfitYoY',
                                    'listed_days': 90})

# NetProfitYoY因子的配置参数
NETPROFITYOY_CT = DottableDict({'db_file': 'AlphaFactor/Growth/NetProfitYoY',
                                'listed_days': 90})

# OperateCashFlowYoY因子的配置参数
OPERATECASHFLOWYOY_CT = DottableDict({'db_file': 'AlphaFactor/Growth/OperateCashFlowYoY',
                                      'listed_days': 90})

# OperateRevenueQYoY因子的配置参数
OPERATEREVENUEQYOY_CT = DottableDict({'db_file': 'AlphaFactor/Growth/OperateRevenueQYoY',
                                      'listed_days': 90})

# OperateProfitQYoY因子的配置参数
OPERATEPROFITQYOY_CT = DottableDict({'db_file': 'AlphaFactor/Growth/OperateProfitQYoY',
                                     'listed_days': 90})

# NetProfitQYoY因子的配置参数
NETPROFITQYOY_CT = DottableDict({'db_file': 'AlphaFactor/Growth/NetProfitQYoY',
                                 'listed_days': 90})

# OperateCashFlowQYoY因子的配置参数
OPERATECASHFLOWQYOY_CT = DottableDict({'db_file': 'AlphaFactor/Growth/OperateCashFlowQYoY',
                                       'listed_days': 90})

# OperateRevenueQoQ因子的配置参数
OPERATEREVENUEQOQ_CT = DottableDict({'db_file': 'AlphaFactor/Growth/OperateRevenueQoQ',
                                     'listed_days': 90})

# OperateProfitQoQ因子的配置参数
OPERATEPROFITQOQ_CT = DottableDict({'db_file': 'AlphaFactor/Growth/OperateProfitQoQ',
                                    'listed_days': 90})

# NetProfitQoQ因子的配置参数
NETPROFITQOQ_CT = DottableDict({'db_file': 'AlphaFactor/Growth/NetProfitQoQ',
                                'listed_days': 90})

# OperateCashFlowQoQ因子的配置参数
OPERATECASHFLOWQOQ_CT = DottableDict({'db_file': 'AlphaFactor/Growth/OperateCashFlowQoQ',
                                      'listed_days': 90})

# ROE因子的配置参数
ROE_CT = DottableDict({'db_file': 'AlphaFactor/Quality/ROE',
                       'listed_days': 90})

# alpha因子列表
ALPHA_FACTORS = ['APM', 'IntradayMomentum']

# 因子载荷类型
FACTORLOADING_TYPE = {'RAW': 'raw',                         # 原始因子载荷
                      'STANDARDIZED': 'standardized',       # 去极值标准化后的因子载荷
                      'ORTHOGONALIZED': 'orthogonalized'    # 正交化后的因子载荷
                      }


# 因子数据库的路径
# FACTOR_DB = DottableDict({'db_path': '/Users/davidyujun/Dropbox/FactorDB'})
FACTOR_DB = DottableDict({'db_path': SETTINGS.FACTOR_DB_PATH})


if __name__ == '__main__':
    print(FACTOR_DB.db_path)
