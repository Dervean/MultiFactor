#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 风险模型的cons
# @Filename: cons
# @Date:   : 2018-03-21 18:21
# @Author  : YuJun
# @Email   : yujun_mail@163.com

from src.util.dottabledict import DottableDict

# Size风险因子的配置参数
SIZE_CT = DottableDict({'db_file': 'RiskFactor/Size/Size',
                        'component': ['LNCAP'],
                        'weight': {'LNCAP': 1.0}})

LNCAP_CT = DottableDict({'db_file': 'RiskFactor/Size/LNCAP/LNCAP',
                         'liquidcap_dbfile': 'RiskFactor/Size/CAP/CAP',
                         'listed_days': 0})

# BETA风险因子的配置参数
DBETA_CT = DottableDict({'db_file': 'RiskFactor/Beta/BETA/BETA',            # Beta因子在因子数据库的相对路径
                         'benchmark': 'SH000300',                           # 2007年以前采用上证综指作为基准
                         'trailing': 252,
                         'half_life': 63,
                         'listed_days': 0})
BETA_CT = DottableDict({'db_file': 'RiskFactor/Beta/Beta',
                        'component': ['DBETA'],
                        'weight': {'DBETA': 1.0}})

# RSTR风险因子的配置参数
RSTR_CT = DottableDict({'db_file': 'RiskFactor/Momentum/RSTR/RSTR',
                        'trailing_start': 504,
                        'trailing_end': 21,
                        'half_life': 126,
                        'listed_days': 0})
MOMENTUM_CT = DottableDict({'db_file': 'RiskFactor/Momentum/Momentum',
                            'component': ['RSTR'],
                            'weight': {'RSTR': 1.0}})

# DASTD风险因子的配置参数
DASTD_CT = DottableDict({'db_file': 'RiskFactor/ResVolatility/DASTD/DASTD',
                         'trailing': 252,
                         'half_life': 42,
                         'listed_days': 0})

# CMRA风险因子的配置参数
CMRA_CT = DottableDict({'db_file': 'RiskFactor/ResVolatility/CMRA/CMRA',
                        'trailing': 12,
                        'days_scale': 21,
                        'listed_days': 0})

# HSIGMA风险因子的配置参数
HSIGMA_CT = DottableDict({'db_file': 'RiskFactor/ResVolatility/HSIGMA/HSIGMA',
                          'benchmark': 'SH000300',
                          'trailing': 252,
                          'half_life': 63})

# ResVolatility风险因子的配置参数
RESVOLATILITY_CT = DottableDict({'db_file': 'RiskFactor/ResVolatility/ResVolatility',
                                 'component': ['DASTD', 'CMRA', 'HSIGMA'],
                                 'weight': {'DASTD': 0.74, 'CMRA': 0.16, 'HSIGMA': 0.1}})

# NLSIZE风险因子的配置参数
NLSIZE_CT = DottableDict({'db_file': 'RiskFactor/NonlinearSize/NLSIZE/NLSIZE'
                          })
NONLINEARSIZE_CT = DottableDict({'db_file': 'RiskFactor/NonlinearSize/NonlinearSize',
                                 'component': ['NLSIZE'],
                                 'weight': {'NLSIZE': 1.0}})

# BTOP风险因子的配置参数
BTOP_CT = DottableDict({'db_file': 'RiskFactor/Value/BTOP/BTOP',
                        'listed_days': 0
                        })

VALUE_CT = DottableDict({'db_file': 'RiskFactor/Value/Value',
                         'component': ['BTOP'],
                         'weight': {'BTOP': 1.0}})

# LIQUIDITY风险因子的配置参数
STOM_CT = DottableDict({'db_file': 'RiskFactor/Liquidity/STOM/STOM',
                        'month_days': 21,
                        'months': 1,
                        'listed_days': 0})
STOQ_CT = DottableDict({'db_file': 'RiskFactor/Liquidity/STOQ/STOQ',
                        'month_days': 21,
                        'months': 3,
                        'listed_days': 0})
STOA_CT = DottableDict({'db_file': 'RiskFactor/Liquidity/STOA/STOA',
                        'month_days': 21,
                        'months': 12,
                        'listed_days': 0})
LIQUIDITY_CT = DottableDict({'db_file': 'RiskFactor/Liquidity/Liquidity',
                             'component': ['STOM', 'STOQ', 'STOA'],
                             'weight': {'STOM': 0.35, 'STOQ': 0.35, 'STOA': 0.3}})
# LIQUIDITY_CT = DottableDict({'db_file': 'RiskFactor/Liquidity/Liquidity',
#                           'stom_days': 21,
#                           'stom_weight': 0.35,
#                           'stoq_months': 3,
#                           'stoq_weight': 0.35,
#                           'stoa_months': 12,
#                           'stoa_weight': 0.3,
#                           'listed_days': 180
#                           })

# EarningsYield风险因子的配置参数
EPFWD_CT = DottableDict({'db_file': 'RiskFactor/EarningsYield/EPFWD/EPFWD',
                         'listed_days': 0})
CETOP_CT = DottableDict({'db_file': 'RiskFactor/EarningsYield/CETOP/CETOP',
                         'listed_days': 0})
ETOP_CT = DottableDict({'db_file': 'RiskFactor/EarningsYield/ETOP/ETOP',
                        'listed_days': 0})
EARNINGSYIELD_CT = DottableDict({'db_file': 'RiskFactor/EarningsYield/EarningsYield',
                                 'component': ['EPFWD', 'CETOP', 'ETOP'],
                                 'weight': {'EPFWD': 0.68, 'CETOP': 0.21, 'ETOP': 0.11}})

# Growth风险因子的配置参数
EGRLF_CT = DottableDict({'db_file': 'RiskFactor/Growth/EGRLF/EGRLF',
                         'listed_days': 0})
EGRSF_CT = DottableDict({'db_file': 'RiskFactor/Growth/EGRSF/EGRSF',
                         'listed_days': 0})
EGRO_CT = DottableDict({'db_file': 'RiskFactor/Growth/EGRO/EGRO',
                        'listed_days': 0})
SGRO_CT = DottableDict({'db_file': 'RiskFactor/Growth/SGRO/SGRO',
                        'listed_days': 0})
GROWTH_CT = DottableDict({'db_file': 'RiskFactor/Growth/Growth',
                          'component': ['EGRLF', 'EGRSF', 'EGRO', 'SGRO'],
                          'weight': {'EGRLF': 0.18, 'EGRSF': 0.11, 'EGRO': 0.24, 'SGRO': 0.47}})

# Leverage风险因子的配置参数
MLEV_CT = DottableDict({'db_file': 'RiskFactor/Leverage/MLEV/MLEV',
                        'listed_days': 0})
DTOA_CT = DottableDict({'db_file': 'RiskFactor/Leverage/DTOA/DTOA',
                        'listed_days': 0})
BLEV_CT = DottableDict({'db_file': 'RiskFactor/Leverage/BLEV/BLEV',
                       'listed_days': 0})
LEVERAGE_CT = DottableDict({'db_file': 'RiskFactor/Leverage/Leverage',
                            'component': ['MLEV', 'DTOA', 'BLEV'],
                            'weight': {'MLEV': 0.38, 'DTOA': 0.35, 'BLEV': 0.27}})

# 风险模型风格因子列表
STYLE_RISK_FACTORS = ['Size', 'Beta', 'Momentum', 'ResVolatility', 'NonlinearSize', 'Value', 'Liquidity',
                      'EarningsYield', 'Growth', 'Leverage']
# 风险模型行业因子列表
INDUSTRY_FACTORS = ['sw_cj', 'sw_cm', 'sw_dqsb', 'sw_dz', 'sw_fdc', 'sw_fzfz', 'sw_zq', 'sw_bx', 'sw_dyjr', 'sw_gt',
                    'sw_gysy', 'sw_gfjg', 'sw_hg', 'sw_jxsb', 'sw_jsj', 'sw_jydq', 'sw_jzcl', 'sw_jzzs', 'sw_jtys',
                    'sw_nlmy', 'sw_qc', 'sw_qgzz', 'sw_symy', 'sw_spyl', 'sw_tx', 'sw_xxfw', 'sw_yysw', 'sw_yh',
                    'sw_ysjs', 'sw_zh']

# 风险模型的市场因子列表
MARKET_FACTORS = ['market']

# 风险模型的风险因子总列表
RISK_FACTORS = MARKET_FACTORS + INDUSTRY_FACTORS + STYLE_RISK_FACTORS   # 含市场因子
RISK_FACTORS_NOMARKET = INDUSTRY_FACTORS + STYLE_RISK_FACTORS           # 不含市场因子


if __name__ == '__main__':
    pass
