#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 组合分析模块
# @Filename: analysis
# @Date:   : 2018-10-12 19:05
# @Author  : YuJun
# @Email   : yujun_mail@163.com

import pandas as pd
import numpy as np
from src.util.utils import Utils
from src.riskmodel.RiskModel import Barra
import src.alphamodel.AlphaModel as AlphaModel
import src.riskmodel.riskfactors.cons as riskfactor_ct
import src.alphamodel.alphafactors.cons as alphafactor_ct
from src.portfolio.portfolio import CPortfolio


def port_allocation(portfolio, date, benchmark=None):
    """
    分析组合的风险因子配置和alpha因子配置
    Parameters:
    --------
    :param portfolio: CPortfolio类
        投资组合
    :param date: datetime-like, str
        计算日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param benchmark: str
        基准代码, 默认为None
        benchmark=None时, 采用portfolio.benchmark作为基准
    :return: tuple(pd.DataFrame, pd.DataFrame)
        组合及基准的风险因子配置、alpha因子配置数据
    --------
    1.risk_allocation
    2.alpha_allocation
    """
    date = Utils.to_date(date)
    # 取得持仓数据
    df_holding = portfolio.holding_data(date)
    df_holding.drop(columns='date', inplace=True)
    df_holding.set_index('code', inplace=True)
    # 取得基准持仓数据
    if benchmark is None:
        benchmark = portfolio.benchmark
    if benchmark is None:
        df_ben_holding = None
    else:
        df_ben_holding = Utils.get_index_weight(benchmark, date)
        if df_ben_holding is None:
            raise ValueError("无法读取基准%s在%s的权重数据." % (benchmark, Utils.datetimelike_to_str(date)))
        df_ben_holding.drop(columns='date', inplace=True)
        df_ben_holding.set_index('code', inplace=True)

    port_risk_allocation, port_alpha_allocation = _holding_allocation(date, df_holding)
    ben_risk_allocation, ben_alpha_allocation = _holding_allocation(date, df_ben_holding)

    risk_allocation = pd.merge(left=port_risk_allocation.to_frame(name='port'), right=ben_risk_allocation.to_frame(name='ben'), how='outer', left_index=True, right_index=True)
    risk_allocation['active'] = risk_allocation['port'] - risk_allocation['ben']

    alpha_allocation = pd.merge(left=port_alpha_allocation.to_frame(name='port'), right=ben_alpha_allocation.to_frame(name='ben'), how='outer', left_index=True, right_index=True)
    alpha_allocation['active'] = alpha_allocation['port'] - alpha_allocation['ben']

    return risk_allocation, alpha_allocation


def port_risk_contribution(portfolio, date, benchmark=None):
    """
    分析组合的风险因子归因
    Parameters:
    --------
    :param portfolio: CPortfolio类
        投资组合
    :param date: datetime-like, str
        计算日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param benchmark: str
        基准代码, 默认为None
        benchmark=None时, 采用Portfolio.benchmark作为基准
    :return: pd.DataFrame
        组合及基准的风险因子归因
    --------
    """
    date = Utils.to_date(date)
    # 取得组合持仓数据
    df_port_holding = portfolio.holding_data(date)
    df_port_holding.drop(columns='date', inplace=True)
    df_port_holding.rename(index=str, columns={'weight': 'port_weight'}, inplace=True)
    df_port_holding.set_index('code', inplace=True)
    # 取得基准持仓数据
    if benchmark is None:
        benchmark = portfolio.benchmark
    if benchmark is None:
        df_ben_holding = None
    else:
        df_ben_holding = Utils.get_index_weight(benchmark, date)
        if df_ben_holding is None:
            raise ValueError("无法读取基准%s在%s的权重数据." % (benchmark, Utils.datetimelike_to_str(date)))
        df_ben_holding.drop(columns='date', inplace=True)
        df_ben_holding.rename(index=str, columns={'weight': 'ben_weight'}, inplace=True)
        df_ben_holding.set_index('code', inplace=True)

    # 取得风险模型数据(风险因子暴露矩阵、风险因子协方差矩阵、特质波动率方差矩阵)
    CRiskModel = Barra()
    df_riskfactor_loading, arr_risk_covmat, ser_spec_var = CRiskModel.get_riskmodel_data(date)

    df_weight = pd.merge(left=df_port_holding, right=df_ben_holding, how='outer', left_index=True, right_index=True)
    # df_weight.fillna(0, inplace=True)
    df_riskfactor_data = pd.merge(left=df_weight, right=df_riskfactor_loading, how='left', left_index=True, right_index=True)
    df_riskfactor_data = pd.merge(left=df_riskfactor_data, right=ser_spec_var.to_frame(name='spec_var'), how='left', left_index=True, right_index=True)
    df_riskfactor_data.fillna(0, inplace=True)

    arr_port_weight = np.array(df_riskfactor_data[['port_weight']])
    arr_ben_weight = np.array(df_riskfactor_data[['ben_weight']])
    arr_active_weight = arr_port_weight - arr_ben_weight
    arr_riskfactor_loading = np.array(df_riskfactor_data.loc[:, riskfactor_ct.RISK_FACTORS])
    arr_specvar = np.diag(df_riskfactor_data.loc[:, 'spec_var'])

    # 计算组合的risk contribution
    fsigma = float(np.sqrt(np.linalg.multi_dot([arr_port_weight.T, arr_riskfactor_loading, arr_risk_covmat, arr_riskfactor_loading.T, arr_port_weight]) + np.linalg.multi_dot([arr_port_weight.T, arr_specvar, arr_port_weight])))
    Psi = np.dot(arr_port_weight.T, arr_riskfactor_loading).transpose()
    risk_contribution_port = pd.Series((1.0 / fsigma * Psi * np.dot(arr_risk_covmat, Psi)).flatten(), index=riskfactor_ct.RISK_FACTORS)
    fselection = fsigma - risk_contribution_port.sum()
    fallocation = risk_contribution_port.sum() - risk_contribution_port['market']
    risk_contribution_port['sigma'] = fsigma
    risk_contribution_port['selection'] = fselection
    risk_contribution_port['allocation'] = fallocation
    risk_contribution_port['industry'] = risk_contribution_port[riskfactor_ct.INDUSTRY_FACTORS].sum()
    risk_contribution_port['style'] = risk_contribution_port[riskfactor_ct.STYLE_RISK_FACTORS].sum()

    # 计算基准的risk contribution
    fsigma = float(np.sqrt(np.linalg.multi_dot([arr_ben_weight.T, arr_riskfactor_loading, arr_risk_covmat, arr_riskfactor_loading.T, arr_ben_weight]) + np.linalg.multi_dot([arr_ben_weight.T, arr_specvar, arr_ben_weight])))
    Psi = np.dot(arr_ben_weight.T, arr_riskfactor_loading).transpose()
    risk_contribution_ben = pd.Series((1.0 / fsigma * Psi * np.dot(arr_risk_covmat, Psi)).flatten(), index=riskfactor_ct.RISK_FACTORS)
    fselection = fsigma - risk_contribution_ben.sum()
    fallocation = risk_contribution_ben.sum() - risk_contribution_ben['market']
    risk_contribution_ben['sigma'] = fsigma
    risk_contribution_ben['selection'] = fselection
    risk_contribution_ben['allocation'] = fallocation
    risk_contribution_ben['industry'] = risk_contribution_ben[riskfactor_ct.INDUSTRY_FACTORS].sum()
    risk_contribution_ben['style'] = risk_contribution_ben[riskfactor_ct.STYLE_RISK_FACTORS].sum()

    # 计算组合相对于基准的risk contribution
    fsigma = float(np.sqrt(np.linalg.multi_dot([arr_active_weight.T, arr_riskfactor_loading, arr_risk_covmat, arr_riskfactor_loading.T, arr_active_weight]) + np.linalg.multi_dot([arr_active_weight.T, arr_specvar, arr_active_weight])))
    Psi = np.dot(arr_active_weight.T, arr_riskfactor_loading).transpose()
    risk_contribution_active = pd.Series((1.0 / fsigma * Psi * np.dot(arr_risk_covmat, Psi)).flatten(), index=riskfactor_ct.RISK_FACTORS)
    fselection = fsigma - risk_contribution_active.sum()
    fallocation = risk_contribution_active.sum() - risk_contribution_active['market']
    risk_contribution_active['sigma'] = fsigma
    risk_contribution_active['selection'] = fselection
    risk_contribution_active['allocation'] = fallocation
    risk_contribution_active['industry'] = risk_contribution_active[riskfactor_ct.INDUSTRY_FACTORS].sum()
    risk_contribution_active['style'] = risk_contribution_active[riskfactor_ct.STYLE_RISK_FACTORS].sum()

    risk_contribution = pd.DataFrame({'port': risk_contribution_port, 'ben': risk_contribution_ben, 'active': risk_contribution_active})
    return risk_contribution[['port', 'ben','active']]


def _holding_allocation(date, holding_data):
    """
    分析持仓数据的风险因子配置和alpha因子配置
    Parameters:
    --------
    :param date: datetime-like, str
        计算日期, e.g: YYYY-MM-DD, YYYYMMDD
    :param holding_data: pd.DataFrame
        持仓数据, index=个股代码, columns=['weight']
    :return: tuple(pd.Series, pd.Series)
        持仓数据的风险因子配置、alpha因子配置
    --------
    1.risk_allocation: Series的index为风险因子名称
    2.alpha_allocation: Series的index为alpha因子名称
    """
    if holding_data is None:
        risk_allocation = pd.Series([0]*len(riskfactor_ct.RISK_FACTORS), index=riskfactor_ct.RISK_FACTORS)
        alpha_alloction = pd.Series([0]*len(alphafactor_ct.ALPHA_FACTORS), index=alphafactor_ct.ALPHA_FACTORS)
    else:
        date = Utils.to_date(date)
        # 取得风险模型数据(风险因子暴露矩阵、风险因子协方差矩阵、特质波动率方差矩阵)
        CRiskModel = Barra()
        df_riskfactor_loading, arr_risk_covmat, ser_spec_var = CRiskModel.get_riskmodel_data(date)
        # 取得alpha模型相关的数据(alpha因子载荷矩阵、alpha因子收益向量)
        df_alphafactor_loading, ser_alphafactor_ret = AlphaModel.get_alphamodel_data(date)

        # 持仓数据的风险因子配置
        df_riskfactor_data = pd.merge(left=holding_data, right=df_riskfactor_loading, how='inner', left_index=True,
                                      right_index=True)
        arr_riskfactor_loading = np.array(df_riskfactor_data.loc[:, riskfactor_ct.RISK_FACTORS])
        arr_weight = np.array(df_riskfactor_data.loc[:, ['weight']])

        risk_allocation = pd.Series(np.dot(arr_weight.T, arr_riskfactor_loading).flatten(),
                                    index=riskfactor_ct.RISK_FACTORS)

        # 持仓数据的alpha因子配置
        df_alphafactor_data = pd.merge(left=holding_data, right=df_alphafactor_loading, how='inner', left_index=True,
                                       right_index=True)
        arr_weight = np.array(df_alphafactor_data.loc[:, ['weight']])
        arr_alphafactor_loading = np.array(df_alphafactor_data.loc[:, alphafactor_ct.ALPHA_FACTORS])

        alpha_alloction = pd.Series(np.dot(arr_weight.T, arr_alphafactor_loading).flatten(), index=alphafactor_ct.ALPHA_FACTORS)

    return risk_allocation, alpha_alloction


if __name__ == '__main__':
    pass
    port = CPortfolio('weight_holding', 'SZ399905')
    holdingfile_path = '/Volumes/DB/FactorDB/portfolio/holdings/CSI500_Enhancement/20180928.csv'
    port.load_holdings_fromfile(holdingfile_path)
    risk_allocation, alpha_allocation = port_allocation(port, '2018-09-28')
    print(risk_allocation)
    print(alpha_allocation)

    risk_contribution = port_risk_contribution(port, '2018-09-28')
    print(risk_contribution)
