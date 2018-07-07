#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
# @Abstract: 风险模型类文件
# @Filename: RIskModel
# @Date:   : 2018-06-21 20:03
# @Author  : YuJun
# @Email   : yujun_mail@163.com


from src.util.utils import Utils
from src.riskmodel.riskfactors.Size import Size
from src.riskmodel.riskfactors.Beta import Beta
from src.riskmodel.riskfactors.Momentum import Momentum
from src.riskmodel.riskfactors.ResVolatility import ResVolatility
from src.riskmodel.riskfactors.NonlinearSize import NonlinearSize
from src.riskmodel.riskfactors.Value import Value
from src.riskmodel.riskfactors.Liquidity import Liquidity
from src.riskmodel.riskfactors.EarningsYield import EarningsYield
from src.riskmodel.riskfactors.Growth import Growth
from src.riskmodel.riskfactors.Leverage import Leverage


class Barra(object):
    """Barra风险模型基类"""

    def calc_factorloading(self, start_date, end_date=None, multi_prc=False):
        """
        计算风险因子的因子载荷
        Parameters:
        --------
        :param start_date: datetime-like, str
            计算开始日期, 格式: YYYY-MM-DD
        :param end_date: datetime-like, str
            计算结束日期, 格式: YYYY-MM-DD
        :param multi_prc: bool
            是否并行计算, 默认为False
        :return: None
        """
        # 读取交易日序列
        start_date = Utils.to_date(start_date)
        if not end_date is None:
            end_date = Utils.to_date(end_date)
            trading_days_series = Utils.get_trading_days(start=start_date, end=end_date)
        else:
            trading_days_series = Utils.get_trading_days(start=start_date, ndays=1)
        # 遍历交易日序列, 计算风险因子的因子载荷
        for calc_date in trading_days_series:
            Size.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Beta.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Momentum.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            ResVolatility.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            NonlinearSize.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Value.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Liquidity.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            EarningsYield.calc_factor_loading(start_date=starcalc_datet_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Growth.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)
            Leverage.calc_factor_loading(start_date=calc_date, end_date=None, month_end=False, save=True, multi_proc=multi_prc)

    def _get_factorloading_matrix(self, date):
        """

        :param date:
        :return:
        """


if __name__ == '__main__':
    BarraModel = Barra()
    BarraModel.calc_factorloading('2017-12-29')
