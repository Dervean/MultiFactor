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
FACTOR_PERFORMANCE_HEADER = {'daily_performance': ['date', 'daily_ret', 'nav', 'accu_ret'],
                             'monthly_performance': ['date', 'monthly_ret'],
                             'summary_performance': ['date', 'total_ret', 'annual_ret', 'max_drawdown', 'volatility',
                                                     'monthly_winrate', 'IR', 'type']
                             }

# alpha因子汇总绩效的时间区间类型
SUMMARY_PERFORMANCE_MONTH_LENGTH = ['total', 60, 48, 36, 24, 12]


if __name__ == '__main__':
    pass
