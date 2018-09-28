#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 组合优化器
# @Filename: Optimizer
# @Date:   : 2018-09-14 14:15
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.util.utils import Utils
import src.alphamodel.AlphaModel as AlphaModel
from src.riskmodel.RiskModel import Barra
import src.optimizer.cons as opt_ct


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
    start_date = Utils.to_date(start_date)
    if end_date is None:
        trading_days_series = Utils.get_trading_days(end=start_date, ndays=1)
    else:
        end_date = Utils.to_date(end_date)
        trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)

    # 遍历交易日, 计算最优化组合
    for calc_date in trading_days_series:
        if month_end and (not Utils.is_month_end(calc_date)):
            continue
        # 读取alpha模型数据
        df_alphafactor_loading, ser_alphafactor_ret = AlphaModel.get_alphamodel_data(calc_date)
        # 计算个股预期收益向量
        df_alphafactor_loading.set_index('code', inplace=True)
        ser_secu_ret = df_alphafactor_loading * ser_alphafactor_ret
        df_secu_ret = ser_secu_ret.to_frame(name='ret')
        df_secu_ret.index.name = 'code'
        df_secu_ret.reset_index(inplace=True)

        # 读取风险模型数据
        BarraModel = Barra()
        df_riskfactor_loading, arr_riskfactor_covmat, ser_spec_var = BarraModel.get_riskmodel_data(calc_date)
        df_spec_var = ser_spec_var.to_frame(name='spec_var')
        df_spec_var.index.name = 'code'
        df_spec_var.reset_index(inplace=True)

        # 合并个股预期收益向量、个股风险因子暴露矩阵、个股特质波动率向量数据


        # 遍历投资组合名称, 计算最优化权重
        if port_name is None:
            port_names = opt_ct.portfolios
        else:
            port_names = [port_name]
        for portfolio_name in port_names:
            # 组合优化的约束条件
            opt_constraints = eval('opt_ct.'+portfolio_name)
            # 读取基准成份股权重数据
            df_ben_weight = Utils.get_index_weight(opt_constraints['benchmark'], calc_date)



if __name__ == '__main__':
    pass
