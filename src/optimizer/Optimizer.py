#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 组合优化器
# @Filename: Optimizer
# @Date:   : 2018-09-14 14:15
# @Author  : YuJun
# @Email   : yujun_mail@163.com


def calc_optimized_portfolio(start_date, end_date=None, port_name=None, month_end=True, save=False):
    """
    计算最优化组合权重
    Parameters:
    --------
    :param start_date: datetime-like, str
        开始计算日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param end_date: datetime-like, str
        结束计算日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param port_name: list of str, str
        需优化的组合名称
        默认为None, 优化所有组合
    :param month_end: bool
        是否仅优化月末数据, 默认True
    :param save: bool
        是否保存优化组合权重数据, 默认为False
    :return: pd.DataFrame
        返回组合优化权重数据
    --------
        0.date: 日期
        1.code: 个股代码
        2.weight: 个股权重
    """



if __name__ == '__main__':
    pass
