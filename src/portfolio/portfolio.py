#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 投资组合类
# @Filename: portfolio
# @Date:   : 2018-08-10 15:52
# @Author  : YuJun
# @Email   : yujun_mail@163.com

import pandas as pd
from collections import OrderedDict
import os
import src.settings as SETTINGS
import src.portfolio.cons as portfolio_ct


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
            self._data = self._data.append(data, ignore_index=True)


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


def load_holding_data(port_name=None, holding_name=None, holding_path=None):
    """
    读取组合持仓数据
    Parameters:
    ------
    :param port_name: str
        组合名称
    :param holding_name: str
        持仓名称
    :param holding_path: str
        持仓文件路径
    :return: WeightHolding
    ------
        1. 如果port_name和holding_name均不为None, 那么从FactorDB/portfolio/port_name/holding_name.csv文件中读取持仓数据
        2. 否则从holding_path指定的持仓文件中读取持仓数据
    """
    if holding_path is None:
        if (port_name is None) or (holding_name is None):
            raise ValueError("请指定(组合名称、持仓名称)或者持仓文件路径.")
        else:
            holding_path = os.path.join(SETTINGS.FACTOR_DB_PATH, portfolio_ct.PORTFOLIO_HOLDING_PATH, port_name, ''.join([holding_name, '.csv']))
    if not os.path.isfile(holding_path):
        raise FileExistsError("持仓文件不存在:%s" % holding_path)

    df_holdings = pd.read_csv(holding_path, header=0)
    if not all([col in df_holdings.columns for col in portfolio_ct.WEIGHTHOLDING_DATA_HEADER]):
        raise ValueError("持仓数据应包含%s" % str(portfolio_ct.WEIGHTHOLDING_DATA_HEADER))

    holding_data = WeightHolding()
    for _, data in df_holdings.iterrows():
        holding_data.append(data)
    return holding_data


if __name__ == '__main__':
    pass
    holding_data = load_holding_data('tmp', 'sh50')
    print(holding_data.holding)
