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


if __name__ == '__main__':
    pass
