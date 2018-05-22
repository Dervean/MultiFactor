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
                         'listed_days': 180})

# BETA风险因子的配置参数
BETA_CT = DottableDict({'db_file': 'RiskFactor/BETA/BETA',           # Beta因子在因子数据库的相对路径
                        'benchmark': 'SH000300',
                        'trailing': 252,
                        'half_life': 63})

# HSIGMA风险因子的配置参数
HSIGMA_CT = DottableDict({'db_file': 'RiskFactor/HSIGMA/HSIGMA',
                          'benchmark': 'SH000300',
                          'trailing': 252,
                          'half_life': 63})

# RSTR风险因子的配置参数
RSTR_CT = DottableDict({'db_file': 'RiskFactor/RSTR/RSTR',
                        'trailing_start': 504,
                        'trailing_end': 21,
                        'half_life': 126})

# DASTD风险因子的配置参数
DASTD_CT = DottableDict({'db_file': 'RiskFactor/DASTD/DASTD',
                         'trailing': 252,
                         'half_life': 42,
                         'listed_days': 180})

# CMRA风险因子的配置参数
CMRA_CT = DottableDict({'db_file': 'RiskFactor/CMRA/CMRA',
                        'trailing': 12,
                        'days_scale': 21,
                        'listed_days': 180})

# NLSIZE风险因子的配置参数
NLSIZE_CT = DottableDict({'db_file': 'RiskFactor/NLSIZE/NLSIZE'
                          })

# BTOP风险因子的配置参数
BTOP_CT = DottableDict({'db_file': 'RiskFactor/BTOP/BTOP',
                        'listed_days': 180
                        })

# LIQUIDITY风险因子的配置参数
LIQUID_CT = DottableDict({'db_file': 'RiskFactor/LIQUIDITY/LIQUIDITY',
                          'stom_days': 21,
                          'stom_weight': 0.35,
                          'stoq_months': 3,
                          'stoq_weight': 0.35,
                          'stoa_months': 12,
                          'stoa_weight': 0.3,
                          'listed_days': 180
                          })

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


if __name__ == '__main__':
    pass
