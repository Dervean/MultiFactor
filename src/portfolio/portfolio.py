#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 投资组合类
# @Filename: portfolio
# @Date:   : 2018-08-10 15:52
# @Author  : YuJun
# @Email   : yujun_mail@163.com

import pandas as pd
import numpy as np
import os
import src.settings as SETTINGS
import src.portfolio.cons as portfolio_ct
import src.util.cons as util_ct
from src.util.utils import Utils
# from src.riskmodel.RiskModel import Barra
# import src.alphamodel.AlphaModel as AlphaModel
# import src.riskmodel.riskfactors.cons as riskfactor_ct
# import src.alphamodel.alphafactors.cons as alphafactor_ct


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

    def __init__(self, holding_type, benchmark=None):
        """
        初始化
        :param holding_type: str
            持仓类型, 'port_holding'=投资组合持仓, 'weight_holding'=权重形式的持仓
        :param benchmark: str
            基准代码
        """
        if holding_type not in ['weight_holding', 'port_holding']:
            raise ValueError("持仓类型必须为'weight_holding'或'port_holding'.")
        self._holdings = dict()     # 组合持仓数据(dict of CHoldingData)
        self._holdingtype = holding_type
        self._benchmark = benchmark

    @property
    def holdings(self):
        return self._holdings

    @property
    def holdingtype(self):
        return self._holdingtype

    @property
    def benchmark(self):
        return self._benchmark

    @property
    def count(self):
        """持仓数量"""
        return len(self._holdings)

    @property
    def holding_dates(self):
        """返回持仓日期列表"""
        if self.count == 0:
            return []
        else:
            return sorted(list(self._holdings.keys()))

    def holding_data(self, date):
        """
        取得date日期对应的组合持仓数据
        Parameters:
        --------
        :param date: datetime-like, str
            持仓日期, e.g: YYYY-MM-DD, YYYYMMDD
        :return: pd.DataFrame
        --------
            组合持仓数据, columns = ['date', 'code', 'weight']
            如果指定日期持仓数据不存在, raise ValueError
        """
        date = Utils.to_date(date)
        if date not in self.holdings:
            raise ValueError("不存在%s的持仓数据." % Utils.datetimelike_to_str(date))
        if self.holdingtype == 'weight_holding':
            df_holding = self.holdings[date].holding
        elif self.holdingtype == 'port_holding':
            df_holding = self.holdings[date].holding[['date', 'code', 'weight']]
        else:
            raise ValueError("组合持仓类型错误: %s." % self.holdingtype)

        return df_holding

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
'''
    def _holding_analysis(self, date, holding_data):
        """
        对持仓数据从风险因子配置、alpha因子配置及风险预测进行分析
        Parameters:
        --------
        :param date: datetime-like, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param holding_data: pd.DataFrame
            持仓数据, index=个股代码, columns=['weight']
        :return: 持仓数据的风险因子配置、alpha因子配置及风险预测
        --------
        1.risk_allocation:
        2.alpha_allocation:
        3.risk_contribution:
        """
        date = Utils.to_date(date)
        # 取得风险模型数据(风险因子暴露矩阵、风险因子协方差矩阵、特质波动率方差矩阵)
        CRiskModel = Barra()
        df_riskfactor_loading, arr_risk_covmat, ser_spec_var = CRiskModel.get_riskmodel_data(date)
        # 取得alpha模型相关的数据(alpha因子载荷矩阵、alpha因子收益向量)
        df_alphafactor_loading, ser_alphafactor_ret = AlphaModel.get_alphamodel_data(date)

        df_riskfactor_data = pd.merge(left=holding_data, right=df_riskfactor_loading, how='inner', left_index=True, right_index=True)
        df_riskfactor_data = pd.merge(left=df_riskfactor_data, right=ser_spec_var.to_frame(name='spec_var'), how='inner', left_index=True, right_index=True)
        arr_riskfactor_loading = np.array(df_riskfactor_data.loc[:, riskfactor_ct.RISK_FACTORS])
        arr_weight = np.array(df_riskfactor_data.loc[:, ['weight']])
        arr_specvar = np.diag(df_riskfactor_data.loc[:, 'spec_var'])

        df_alphafactor_data = pd.merge(left=holding_data, right=df_alphafactor_loading, how='inner', left_index=True, right_index=True)
        arr_alphafactor_loading = np.array(df_alphafactor_data.loc[:, alphafactor_ct.ALPHA_FACTORS])

        # 持仓数据的风险因子配置
        risk_allocation = pd.Series(np.dot(arr_weight.T, arr_riskfactor_loading), index=riskfactor_ct.RISK_FACTORS)

        # 持仓数据的风险contribution
        fsigma = float(np.sqrt(np.linalg.multi_dot([arr_weight.T, arr_riskfactor_loading, arr_risk_covmat, arr_riskfactor_loading.T, arr_weight]) + np.linalg.multi_dot([arr_weight.T, arr_specvar, arr_weight])))

        Psi = np.dot(arr_weight.T, arr_riskfactor_loading).transpose()
        risk_contribution = pd.Series(1.0 / fsigma * Psi * np.dot(arr_risk_covmat, Psi), index=riskfactor_ct.RISK_FACTORS)
        fselection = fsigma - risk_contribution.sum()
        falloction = risk_contribution.sum() - risk_contribution['market']
        risk_contribution['sigma'] = fsigma
        risk_contribution['selection'] = fselection
        risk_contribution['allocation'] = falloction
        risk_contribution['industry'] = risk_contribution[riskfactor_ct.INDUSTRY_FACTORS].sum()
        risk_contribution['style'] = risk_contribution[riskfactor_ct.STYLE_RISK_FACTORS].sum()

        # 持仓数据的alpha因子配置
        alpha_alloction = pd.Series(np.dot(arr_weight.T, arr_alphafactor_loading), index=alphafactor_ct.ALPHA_FACTORS)

        return risk_allocation, alpha_alloction, risk_contribution


    def port_analysis(self, date, benchmark=None):
        """
        对组合从风险因子配置、alpha因子配置及风险预测进行分析
        Parameters:
        --------
        :param date: datetime-like, str
            计算日期, e.g: YYYY-MM-DD, YYYYMMDD
        :param benchmark: str
            基准代码, 默认为None
            benchmark=None时, 采用self._benchmark作为基准
        :return: 组合及基准的风险因子配置、alpha因子配置及风险贡献
        --------
        1.risk_allocation:
        2.alpha_allocation:
        3.risk_contribution:
        """
        date = Utils.to_date(date)
        # 判断是否存在指定日期的持仓数据
        if date not in self.holdings:
            raise ValueError("不存在%s的持仓数据." % Utils.datetimelike_to_str(date))
        # 取得持仓数据
        if 'weight_holding' == self.holdingtype:
            df_holding = self.holdings[date].holding
        elif 'port_holding' == self.holdingtype:
            df_holding = self.holdings[date].holding[['date', 'code', 'weight']]
        else:
            raise ValueError("组合持仓类型错误：%s" % self.holdingtype)
        df_holding.drop(columns='date', inplace=True)
        df_holding.set_index('code', inplace=True)
        # 取得基准持仓数据
        if benchmark is None:
            benchmark = self.benchmark
        df_ben_holding = Utils.get_index_weight(benchmark, date)
        if df_ben_holding is None:
            raise ValueError("无法读取基准权重数据：%s" % benchmark)
        df_ben_holding.drop(columns='date', inplace=True)
        df_ben_holding.set_index('code', inplace=True)

        port_risk_allocation, port_alpha_allocation, port_risk_contribution = self._holding_analysis(date, df_holding)
        ben_risk_allocation, ben_alpha_allocation, ben_risk_contribution = self._holding_analysis(date, df_ben_holding)

        risk_allocation = pd.merge(left=port_risk_allocation.to_frame(name='port'), right=ben_risk_allocation.to_frame(name='ben'), how='outer', left_index=True, right_index=True)
        risk_allocation['active'] = risk_allocation['port'] - risk_allocation['ben']

        alpha_allocation = pd.merge(left=port_alpha_allocation.to_frame(name='port'), right=ben_alpha_allocation.to_frame(name='ben'), how='outer', left_index=True, right_index=True)
        alpha_allocation['active'] = alpha_allocation['port'] - alpha_allocation['ben']

        return risk_allocation, alpha_allocation
'''

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
    # holding_data = load_holding_data('tmp', 'sh50')
    # print(holding_data.holding)

    port = CPortfolio('weight_holding')
    holdingfile_path = os.path.join(SETTINGS.FACTOR_DB_PATH, 'portfolio/opt_port/CSI500_Enhancement/20180928.csv')
    port.load_holdings_fromfile(holdingfile_path)
    risk_allocation, alpha_allocation = port.port_analysis('2018-09-28', 'SZ399905')
