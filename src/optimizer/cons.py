#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 组合优化器的常量定义文件
# @Filename: cons
# @Date:   : 2018-09-14 14:22
# @Author  : YuJun
# @Email   : yujun_mail@163.com


# 组合优化约束条件
## 1.中证500指数增强组合
CSI500_Enhancement = {'riskfactor_const':                   # 风格因子约束
                          {'Beta':          (-0.1, 0.1),
                           'EarningsYield': (-0.1, 0.1),
                           'Growth':        (-0.1, 0.1),
                           'Leverage':      (-0.1, 0.1),
                           'Liquidity':     (-0.1, 0.1),
                           'Momentum':      (-0.1, 0.1),
                           'NonlinearSize': (-0.1, 0.1),
                           'ResVolatility': (-0.1, 0.1),
                           'Size':          (-0.1, 0.1),
                           'Value':         (-0.1, 0.1)},
                      'industry_neutral': True,             # 是否行业中性
                      'weight_bound': (0.0, 0.005),          # 权重上下限
                      'weight_sum': 1.0,                    # 权重之和
                      'secu_num_cons': False,               # 是否约束股票数量
                      'n_max': 200,                         # 股票数量上限
                      'lambda': 0.75,                       # 风险厌恶系数
                      'rho': 0.3,                           # 交易成本惩罚系数
                      'benchmark': 'SZ399905'               # 基准代码
                      }

# 需要优化的投资组合名称
portfolios = ['CSI500_Enhancement']

if __name__ == '__main__':
    pass
