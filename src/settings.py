#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 
# @Filename: settings.py
# @Date:   : 2018-07-07 20:36
# @Author  : YuJun
# @Email   : yujun_mail@163.com


# 因子数据库根目录
FACTOR_DB_PATH = '/Volumes/DB/FactorDB'

# 因子数据库文件夹结构
FACTOR_DB_DIR = {
                'AlphaFactor': {
                               'APM': {
                                      'mvpfp': None,
                                      'orthogonalized': None,
                                      'raw': None,
                                      'standardized': None
                                      },
                               'CYQ': None,
                               'IntradayMomentum': {
                                                   'mvpfp': None,
                                                   'orthogonalized': None,
                                                   'raw': None,
                                                   'standardized': None
                                                   },
                               'SmartMoney': {
                                             'mvpfp': None,
                                             'orthogonalized': None,
                                             'performance': None,
                                             'raw': None,
                                             'standardized': None
                                             }
                               },
                'ElementaryFactor': {
                                    'basic_info': None,
                                    'cap_struct': None,
                                    'consensus_data': {
                                                      'growth_data': None,
                                                      'predicted_earnings': None
                                                      },
                                    'fin_data': {
                                                'fin_data_basics': None,
                                                'fin_data_cwbbzy': None
                                                },
                                    'future_ret': None,
                                    'index_cons': None,
                                    'industry_classify': None,
                                    'ipo_info': None,
                                    'mkt_1min_FQ': None,
                                    'mkt_daily_FQ': None,
                                    'mkt_daily_NoFQ': None,
                                    'suspension_info': None
                                    },
                'portfolio': {
                             'holdings': None
                             },
                'RiskFactor': {
                              'Beta': {
                                      'BETA': None
                                      },
                              'EarningsYield': {
                                               'CETOP': None,
                                               'EPFWD': None,
                                               'ETOP': None
                                               },
                              'Growth': {
                                        'EGRLF': None,
                                        'EGRO': None,
                                        'EGRSF': None,
                                        'SGRO': None
                                        },
                              'Leverage': {
                                          'BLEV': None,
                                          'DTOA': None,
                                          'MLEV': None
                                          },
                              'Liquidity': {
                                           'STOA': None,
                                           'STOM': None,
                                           'STOQ': None
                                           },
                              'Momentum': {
                                          'RSTR': None
                                          },
                              'NonlinearSize': {
                                               'NLSIZE': None
                                               },
                              'ResVolatility': {
                                               'CMRA': None,
                                               'DASTD': None,
                                               'HSIGMA': None
                                               },
                              'Size': {
                                      'CAP': None,
                                      'LNCAP': None
                                      },
                              'Value': {
                                       'BTOP': None
                                       }
                              },
                'riskmodel': {
                             'cov': {
                                    'factor_cov': None,
                                    'naive_cov': None,
                                    'naive_specvar': None,
                                    'spec_var': None
                                    },
                             'dailyret': None,
                             'factorloading': {
                                              'indfactorloading': None,
                                              'stylefactorloading': None
                                              },
                             'factorret': None,
                             'residual': None
                             }
                }

# 是否进行并行计算
CONCURRENCY_ON = True

# 并行计算时使用的核心数
CONCURRENCY_KERNEL_NUM = 4

# 数据文件编码
DATA_ENCODING_TYPE = 'utf-8'


if __name__ == '__main__':
    pass
