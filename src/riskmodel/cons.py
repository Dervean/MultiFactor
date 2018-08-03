#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: constant variable for risk model
# @Filename: cons
# @Date:   : 2018-07-10 17:42
# @Author  : YuJun
# @Email   : yujun_mail@163.com


# 风险模型中存储个股日收益率数据的相对路径
DAILY_RET_PATH = 'riskmodel/dailyret/dailyret'
# 风险模型中行业因子载荷数据的相对路径
INDUSTRY_FACTORLOADING_PATH = 'riskmodel/factorloading/indfactorloading/ind_factorloading'
# 风险模型中风格因子载荷数据的相对路径
STYLE_FACTORLOADING_PATH = 'riskmodel/factorloading/stylefactorloading/style_factorloading'
# 风险模型因子报酬数据的相对路径
FACTOR_RETURN_PATH = 'riskmodel/factorret/risk_factor_ret.csv'
# 个股风险模型残差数据的相对路径
RISKMODEL_RESIDUAL_PATH = 'riskmodel/residual/residual_ret.csv'
# 风险因子协方差矩阵相对路径
FACTOR_COVMAT_PATH = 'riskmodel/cov/factor_cov'
# 风险因子朴素协方差矩阵相对路径
FACTOR_NAIVE_COVMAT_PATH = 'riskmodel/cov/naive_cov'
# 特质收益率方差矩阵相对路径
SPECIFICRISK_VARMAT_PATH = 'riskmodel/cov/spec_var'
# 特质收益率朴素方差矩阵相对路径
SPECIFICRISK_NAIVE_VARMAT_PATH = 'riskmodel/cov/naive_specvar'

# 因子协方差矩阵估计的参数定义
FACTOR_COVMAT_PARAMS = {
                        'EWMA':                     # 指数移动加权平均
                            {
                                'trailing': 252,           # 样本长度
                                'vol_half_life': 90,       # 方差半衰期
                                'cov_half_life': 90        # 协方差半衰期
                             },
                        'Newey_West':               # Newey_West调整
                            {
                                'trailing': 252,           # 样本长度
                                'vol_lags': 2,             # 方差滞后期数
                                'vol_half_life': 90,       # 方差半衰期
                                'cov_lags': 2,             # 协方差滞后期数
                                'cov_half_life': 90        # 协方差半衰期
                             },
                        'Eigenfactor_Risk_Adj':     # 特征值调整
                            {
                                'sim_num': 10000,          # 蒙特卡洛模拟次数
                                'adj_coef': 1.2            # 调整系数
                             },
                        'Vol_Regime_Adj':           # 波动率偏误调整
                            {
                                'trailing': 252,           # 波动率乘数样本长度
                                'half_life': 42            # 波动率乘数半衰期
                             }
                        }
# 个股特质风险方差矩阵估计的参数定义
SPECIFICRISK_VARMAT_PARAMS = {
                              'EWMA':                   # 指数移动加权平均
                                  {
                                      'trailing': 252,      # 样本长度
                                      'half_life': 90       # 半衰期
                                  },
                              'Newey_West':             # Newey_West调整
                                  {
                                      'trailing': 252,      # 样本长度
                                      'half_life': 90,      # 半衰期
                                      'lags': 5             # 滞后期数
                                  },
                              'Structural_Model_Adj':   # 结构化模型调整
                                  {
                                      'adj_coef': 1.05      # 调整系数
                                  },
                              'Bayesian_Shrinkage':     # 贝叶斯压缩调整
                                  {
                                      'shrinkage_coef': 1.0 # 压缩系数
                                  },
                              'Vol_Regime_Adj':     # 波动率偏误调整
                                  {
                                      'trailing': 252,      # 样本长度
                                      'half_life': 42       # 半衰期
                                  }
                              }


if __name__ == '__main__':
    pass
