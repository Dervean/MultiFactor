#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Abstract: 集中处理多因子数据库的数据导入
# @Filename: DataHandler
# @Date:   : 2018-01-09 19:32
# @Author  : YuJun
# @Email   : yujun_mail@163.com

import src.datahandler.StockBasicsHandler as st_basics
import src.datahandler.MktDataHandler as mkt
import src.datahandler.CapStructHandler as cap_struct
import src.datahandler.IndustryClassifyHandler as ind_cls
import src.datahandler.IPOInfoHandler as ipo_info

if __name__ == '__main__':
    str_date = '2018-10-19'
    print(str_date)
    # 导入个股基本信息相关数据
    print('导入个股基本信息数据...')
    st_basics.load_stock_basics()
    print('导入个股st信息数据...')
    st_basics.load_st_info()
    print('导入交易日历数据...')
    st_basics.load_calendar()
    # 导入复权分钟数据
    print('导入复权分钟行情数据...')
    mkt.load_mkt_1min(str_date.replace('-', ''), 'D')
    # 导入日线行情数据
    print('导入日线行情数据...')
    mkt.load_mkt_daily(True, str_date, False)
    # 生成停牌个股数据
    print('生成停牌个股数据...')
    mkt.calc_suspension_info(str_date)
    # 导入股票股本结构数据
    print('导入股票股本结构数据...')
    cap_struct.load_cap_struct(str_date)
    # 导入最新行业分类数据
    print('导入申万行业分类数据...')
    ind_cls.download_sw_fyjr_classify()
    ind_cls.load_industry_classify(standard='sw', date=str_date)
    # 下载个股ipo数据（增量下载）
    print('下载(增量)个股IPO数据...')
    ipo_info.load_ipo_info()
