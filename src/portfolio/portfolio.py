#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 投资组合类
# @Filename: portfolio
# @Date:   : 2018-08-10 15:52
# @Author  : YuJun
# @Email   : yujun_mail@163.com

import pandas as pd
# from collections import OrderedDict
import os
import src.settings as SETTINGS
import src.portfolio.cons as portfolio_ct
import src.util.cons as util_ct
from src.util.utils import Utils


class CWeightHolding(object):
    """以权重方式表示的组合持仓类"""

    def __init__(self):
        """
        初始化
        --------
        持仓数据self._data的数据结构为:
        0. date: 日期, datetimelike, 每条持仓数据的日期必须一致
        1. code: 证券代码, str
        2. weight: 持仓权重, float
        """
        self._data = pd.DataFrame(columns=['date', 'code', 'weight'])

    @property
    def holding(self):
        return self._data

    @property
    def count(self):
        return len(self._data)

    @property
    def date(self):
        if self.count > 0:
            return self.holding.iloc[0]['date']
        else:
            return None

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
            0. date: 日期
            1. code: 证券代码
            2. weight: 持仓权重
        :return:
        """
        if (self.date is not None) and data['date'] != self.date:
            raise ValueError("添加持仓数据的日期(%s)与原持仓日期(%s)不一致." %
                             (Utils.datetimelike_to_str(data['date'], dash=False),
                              Utils.datetimelike_to_str(self.date, dash=False)))
        if data['code'] in self._data['code'].values:
            idx = self._data[self._data['code'] == data['code']].index
            self._data.loc[idx, 'weight'] += data['weight']
        else:
            self._data = self._data.append(data, ignore_index=True)

    def from_dataframe(self, df_holdingdata, cancel_tinyweight=False):
        """
        从给定的pd.DataFrame中导入持仓数据
        Parameters:
        --------
        :param df_holdingdata: pd.DataFrame
            持仓数据
        :param cancel_tinyweight: bool
            是否剔除小权重数据, 默认为False
        :return:
        """
        if not all([col in df_holdingdata.columns for col in portfolio_ct.WEIGHTHOLDING_DATA_HEADER]):
            raise ValueError("持仓数据应包含%s." % str(portfolio_ct.WEIGHTHOLDING_DATA_HEADER))

        if cancel_tinyweight:
            df_holdingdata = df_holdingdata[abs(df_holdingdata['weight']) > util_ct.TINY_ABS_VALUE]

        for _, holding_data in df_holdingdata.iterrows():
            self.append(holding_data)

    def from_file(self, holdingfile_path, cancel_tinyweight=False):
        """
        从持仓文件导入持仓数据
        Parameters:
        --------
        :param holdingfile_path: str
            持仓文件路径
        :param cancel_tinyweight: bool
            是否剔除小权重个股, 默认False
        :return:
        --------
            持仓文件的第一列应该为日期数据
        """
        if not os.path.isfile(holdingfile_path):
            raise FileExistsError("持仓文件: %s, 不存在." % holdingfile_path)
        df_holding = pd.read_csv(holdingfile_path, header=0, parse_dates=[0])
        if not all([col in df_holding.columns for col in portfolio_ct.WEIGHTHOLDING_DATA_HEADER]):
            raise ValueError("持仓数据应包含%s." % str(portfolio_ct.WEIGHTHOLDING_DATA_HEADER))

        if cancel_tinyweight:
            df_holding = df_holding[abs(df_holding['weight']) > util_ct.TINY_ABS_VALUE]

        for _, holding_data in df_holding.iterrows():
            self.append(holding_data)

    def save_data(self, holding_path):
        """
        保存持仓数据
        Parameters:
        --------
        :param holding_path: 持仓保存路径
        :return:
        """
        self._data.to_csv(holding_path, index=False)


class CPortHolding(object):
    """投资组合持仓类"""

    def __init__(self):
        """
        初始化
        ----
        持仓数据self._data的数据结构为:
        0. date: 日期, datetimelike, 每条持仓数据的日期必须一致
        1. code: 证券代码, str
        2. volume: 持仓量, int
        3. value: 持仓市值, float
        4. weight: 持仓权重, float
        """
        self._data = pd.DataFrame(columns=['date', 'code', 'volume', 'value', 'weight'])     # 持仓数据, pd.DataFrame

    @property
    def holding(self):
        return self._data

    @property
    def count(self):
        return len(self._data)

    @property
    def date(self):
        if self.count > 0:
            return self.holding.iloc[0]['date']
        else:
            return None

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
            0. date: 日期
            1. code: 证券代码
            2. volume: 持仓量
        :return:
        """
        if (self.date is not None) and data['date'] != self.date:
            raise ValueError("添加持仓数据的日期(%)与原持仓日期(%s)不一致." %
                             (Utils.datetimelike_to_str(data['date'], dash=False),
                              Utils.datetimelike_to_str(self.date, dash=False)))
        if data['code'] in self._data['code'].values:
            idx = self._data[self._data['code'] == data['code']].index
            self._data.loc[idx, 'volume'] += data['volume']
        else:
            self._data.append(data, ignore_index=True)

    def from_dataframe(self, df_holdingdata, cancel_tinyweight=False):
        """
        从给定的pd.DataFrame中导入持仓数据
        Parameters:
        --------
        :param df_holdingdata: pd.DataFrame
            持仓数据
        :param cancel_tinyweight: bool
            是否剔除小权重个股, 默认False
        :return:
        """
        if not all([col in df_holdingdata.columns for col in portfolio_ct.PORTHOLDING_DATA_HEADER]):
            raise ValueError("持仓数据应包含%s." % str(portfolio_ct.PORTHOLDING_DATA_HEADER))

        if cancel_tinyweight:
            df_holdingdata = df_holdingdata[abs(df_holdingdata['weight']) > util_ct.TINY_ABS_VALUE]

        for _, holding_data in df_holdingdata.iterrows():
            self.append(holding_data)

    def from_file(self, holdingfile_path, cancel_tinyweight=False):
        """
        从持仓文件导入持仓数据
        Parameters:
        --------
        :param holdingfile_path: str
            持仓文件路径
        :param cancel_tinyweight: bool
            是否剔除小权重个股, 默认False
        :return:
        --------
            持仓文件的第一列应该为日期数据
        """
        if not os.path.isfile(holdingfile_path):
            raise FileExistsError("持仓文件: %s, 不存在." % holdingfile_path)
        df_holding = pd.read_csv(holdingfile_path, header=0, parse_dates=[0])
        if not all([col in df_holding.columns for col in portfolio_ct.PORTHOLDING_DATA_HEADER]):
            raise ValueError("持仓数据应包含%s." % str(portfolio_ct.PORTHOLDING_DATA_HEADER))

        if cancel_tinyweight:
            df_holding = df_holding[abs(df_holding['weight']) > util_ct.TINY_ABS_VALUE]

        for _, holding_data in df_holding.iterrows():
            self.append(holding_data)

    def save_data(self, holding_path):
        """
        保存持仓数据
        Parameters:
        --------
        :param holding_path: str
            持仓文件路径
        :return:
        """
        self._data.to_csv(holding_path, index=False)


class CPortfolio(object):
    """投资组合类"""

    def __init__(self, holding_type):
        """
        初始化
        :param holding_type: str
            持仓类型, 'port_holding'=投资组合持仓, 'weight_holding'=权重形式的持仓
        """
        if holding_type not in ['weight_holding', 'port_holding']:
            raise ValueError("持仓类型必须为'weight_holding'或'port_holding'.")
        self._holdings = dict()     # 组合持仓数据(dict of CHoldingData)
        self._holdingtype = holding_type

    @property
    def holdings(self):
        return self._holdings

    @property
    def holdingtype(self):
        return self._holdingtype

    @property
    def count(self):
        """持仓数量"""
        return len(self._holdings)

    def append_holding(self, holding_data):
        """
        添加持仓数据
        Parameters:
        --------
        :param holding_data: CWeightHolding, CPortHolding
            持仓数据
        :return:
        --------
            将持仓数据添加至组合持仓数据中
        """
        if not (isinstance(holding_data, CWeightHolding) or isinstance(holding_data, CPortHolding)):
            raise ValueError("持仓数据应该是CWeightHolding类型或CPortHolding类型.")
        if holding_data.count == 0:
            raise ValueError("持仓数据不能为空.")
        self._holdings[holding_data.date] = holding_data

    def load_holdings_fromfile(self, holdingfile_path, cancel_tinyweight=False):
        """
        从文件导入组合持仓数据
        Parameters:
        --------
        :param holdingfile_path: str
            持仓文件的路径
        :param cancel_tinyweight: bool
            是否剔除小权重个股, 默认为False
        :return: 导入持仓数据至self.__holdings
        --------
        持仓文件格式: .csv文件, 各列数据为：
            0. date: 日期
            1. code: 证券代码
            3. volume: 持仓量(持仓类型为CPortHolding含有)
            4. value: 持仓市值(持仓类型为CWeightHolding含有)
            5. weight: 权重
        """
        if self.holdingtype == 'weight_holding':
            holding_data = CWeightHolding()
        elif self.holdingtype == 'port_holding':
            holding_data = CPortHolding()
        else:
            raise ValueError("持仓类型必须为'weight_holding'或'port_holding'.")
        holding_data.from_file(holdingfile_path, cancel_tinyweight)
        self.append_holding(holding_data)


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

    holding_data = CWeightHolding()
    for _, data in df_holdings.iterrows():
        holding_data.append(data)
    return holding_data


if __name__ == '__main__':
    pass
    holding_data = load_holding_data('tmp', 'sh50')
    print(holding_data.holding)
