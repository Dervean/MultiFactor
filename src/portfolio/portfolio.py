#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 投资组合类
# @Filename: portfolio
# @Date:   : 2018-08-10 15:52
# @Author  : YuJun
# @Email   : yujun_mail@163.com

import pandas as pd
from collections import OrderedDict


class WeightHolding(object):
    """以权重方式表示的组合持仓类"""

    def __init__(self):
        """
        初始化
        --------
        持仓数据self._data的数据结构为:
        0. code: 证券代码
        1. name: 证券名称
        2. weight: 持仓权重
        """
        self._data = pd.DataFrame(columns=['code', 'name', 'weight'])

    @property
    def holding(self):
        return self._data

    @property
    def count(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data['code'].values

    def __getitem__(self, key):
        """取得证券代码为key的持仓数据"""
        if isinstance(key, str):
            if key in self._data['code'].values:
                return self._data[self._data['code'] == key]
            else:
                return pd.Series()
        else:
            raise KeyError("仅支持单只证券读取持仓信息")

    def append(self, data):
        """
        添加持仓数据
        Parameters:
        --------
        :param data: pd.Series, dict
            单条持仓数据, 字段格式:
            0. code: 证券代码
            1. name: 证券名称
            2. weight: 持仓权重
        :return:
        """
        if data['code'] in self._data['code'].values:
            idx = self._data[self._data['code'] == data['code']].index
            self._data.loc[idx, 'weight'] += data['weight']
        else:
            self._data.append(data, ignore_index=True)


class PortHolding(object):
    """投资组合持仓类"""

    def __init__(self):
        """
        初始化
        ----
        持仓数据self._data的数据结构为:
        0. code: 证券代码
        1. name: 证券名称
        2. volume: 持仓量
        3. value: 持仓市值
        4. weight: 持仓权重
        """
        self._data = pd.DataFrame(columns=['code', 'name', 'volume', 'value', 'weight'])     # 持仓数据, pd.DataFrame

    @property
    def holding(self):
        return self._data

    @property
    def count(self):
        return len(self._data)

    def __getitem__(self, key):
        """取得证券代码为key的持仓数据"""
        if isinstance(key, str):
            return self._data[self._data['code'] == key]
        else:
            raise KeyError("仅支持单只证券读取持仓信息")

    def append(self, data):
        """
        添加持仓数据
        Parameters:
        --------
        :param data: pd.Series, dict
            单条持仓数据, 字段格式:
            0. code: 证券代码
            1. name: 证券名称
            2. volume: 持仓量
        :return:
        """
        if data['code'] in self._data['code'].values:
            idx = self._data[self._data['code'] == data['code']].index
            self._data.loc[idx, 'volume'] += data['volume']
        else:
            self._data.append(data, ignore_index=True)


class Portfolio(object):
    """投资组合类"""

    def __init__(self):
        """初始化"""
        self.__holdings = OrderedDict()     # 组合持仓数据(OrderedDict of pd.DataFrame)

    @property
    def holdings(self):
        return self.__holdings

    def load_holdings_fromfile(self, holding_filepath):
        """
        从文件导入组合持仓数据
        Parameters:
        --------
        :param holding_filepath: str
            持仓文件的路径
        :return: 导入持仓数据至self.__holdings
        --------
        持仓文件格式: .csv文件, 各列数据为：
            0. date: 日期
            1. code: 证券代码
            2. name: 证券名称
            3. volume: 持仓量
            4. value: 持仓市值
            5. weight: 权重
        """


if __name__ == '__main__':
    pass
