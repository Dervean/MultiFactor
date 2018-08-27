#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: alpha模型的常量定义文件
# @Filename: cons
# @Date:   : 2018-08-09 22:12
# @Author  : YuJun
# @Email   : yujun_mail@163.com

# 比较基准代码
BENCHMARK = 'SH000905'

# alpha因子绩效评估结果数据的header
FACTOR_PERFORMANCE_HEADER = {'daily_performance': ['date', 'port_daily_ret', 'bnk_daily_ret', 'hedge_daily_ret',
                                                   'port_nav', 'bnk_nav', 'hedge_nav', 'port_accu_ret', 'bnk_accu_ret',
                                                   'hedge_accu_ret'],
                             'monthly_performance': [],
                             'summary_performance': []}


if __name__ == '__main__':
    pass
