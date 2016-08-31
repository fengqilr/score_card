# -*- coding: utf-8 -*-
"""
此代码用于：
变量的信息和配置，包括类型类型，中文名， 分箱函数，连续变量分箱的临界点，是否显示woe图，是否显示过程

数据来源：
手工输入

注意事项：
建立新模型时，需要新建此配置文件

调用方法：
无
"""

import pandas as pd
from data_platform.tools import null_tools


class Grouping:
    """
    分箱类，可配置各变量的基本信息，和分箱函数，及分箱临界点，是否系那是图形，和过程
    """
    def __init__(self):
        # 分箱配置，变量类型，中文释义，分箱函数，连续变量的分箱临界点，是否显示图像，和过程
        self.var_config_dict = {"app_age": {"type": "continuous",
                                            "var_name_desp": u"年龄",
                                            "grouping_func": self.age_grouping,
                                            "bin_edge_list": [25, 29, 35],
                                            "show_chart": True,
                                            "verbose": True},
                                "cp_max_cumm_tf_days": {"type": "continuous",
                                                        "var_name_desp": u"最长关机天数",
                                                        "grouping_func": self.get_continuous_group,
                                                        "bin_edge_list": [0.5, 1.5, 4.5],
                                                        "show_chart": True,
                                                        "verbose": True},
                                "cpc_mthly_call_01_cnt": {"type": "continuous",
                                                          "var_name_desp": u"每月至少通话一次的号码个数",
                                                          "grouping_func": self.get_continuous_group,
                                                          "bin_edge_list": [1.5, 6.5, 14.5],
                                                          "show_chart": True,
                                                          "verbose": True},
                                "app_prov": {"type": "discrete",
                                             "var_name_desp": u"申请人省份",
                                             "grouping_func": self.app_prov_grouping,
                                             "show_chart": True,
                                             "verbose": True},
                                "cpc_bank_out_cnt": {"type": "continuous",
                                                     "var_name_desp": u"银行呼叫次数",
                                                     "grouping_func": self.get_continuous_group,
                                                     "bin_edge_list": [1, 8, 25],
                                                     "show_chart": True,
                                                     "verbose": True},
                                }
    # def __init__(self, dict_info):
    #     分箱配置，变量类型，中文释义，分箱函数，连续变量的分箱临界点，是否显示图像，和过程
    #     self.var_config_dict = dict_info

    @staticmethod
    def get_continuous_group(x, bin_list):
        """
        根据bin_list, 对连续变量进行分箱， 通用的分箱函数
        """

        # x = x.map(lambda x: np.nan if null_tools.is_null(x) else x)
        proc_bin_list = bin_list[:]
        proc_bin_list.append(x.min()-1)
        proc_bin_list.append(x.max())
        proc_bin_list = list(set(proc_bin_list))
        sort_bin_list = sorted(proc_bin_list)
        x_bin = pd.cut(x, sort_bin_list, labels=["%02d_(%.2f,%.2f]" % (i, sort_bin_list[i], sort_bin_list[i+1]) for i in range(len(sort_bin_list) - 1)])
        # x_bin = pd.cut(x, sort_bin_list)
        x_bin = x_bin.apply(lambda x: x if pd.notnull(x) else 'no_data')
        return x_bin

    # 各变量独特的分箱函数的分箱函数
    @staticmethod
    def age_grouping(x):
        if pd.isnull(x):
            return '[0, 23)'
        elif x < 23.5:
            return '[0, 23)'
        elif 24 <= x < 27:
            return '[23, 27)'
        elif 27 <= x < 35:
            return '[27, 35)'
        elif x >= 35:
            return '[35, +)'

    @staticmethod
    def app_prov_grouping(x):
        if x in [u'云南省', u'海南省', u'贵州省', u'广东省', u'广西壮族自治区', u'湖南省', u'江西省', u'河南省',
                 u'黑龙江省', u'辽宁省', u'内蒙古自治区', u'吉林省', u'陕西省', u'山西省', u'北京市', u'天津市',
                 u'湖北省', u'安徽省', u'河北省', u'新疆维吾尔自治区', u'甘肃省', u'四川省', u'宁夏回族自治区',
                 u'青海省', u'未知', u'西藏自治区', u'重庆市', u'福建省', u'山东省']:
            return u'Not East'
        elif x in [u'浙江省', u'江苏省', u'上海市']:
            return u'East'
        elif x in [u"无数据"]:
            return u"Other"
        else:
            return u'Other'

    @staticmethod
    def bank_grouping(x):
        if pd.isnull(x):
            return u'无数据'
        elif x < 1.1:
            return '[0, 1)'
        elif 1.1 <= x < 5:
            return '[1.13, 2.5)'
        elif (x >= 5) and (x < 10):
            return '[2.5, 5)'
        elif x >= 10:
            return '[5, +)'

    @staticmethod
    def cpc_type_grouping(x):
        if pd.isnull(x):
            return u'无数据'
        elif x < 5:
            return '[0, 05)'
        elif 5 <= x < 45:
            return '[05, 45)'
        elif x >= 45:
            return '[45, +)'

    @staticmethod
    def cpc_h11_grouping(x):
        if pd.isnull(x):
            return u'无数据'
        elif x < 140:
            return '[0, 220)'
        elif 140 <= x < 260:
            return '[220, 380)'
        elif 260 <= x < 400:
            return '[230, 370)'
        elif 400 <= x < 650:
            return '[370, 600)'
        elif x >= 650:
            return '[650, +)'

    @staticmethod
    def cpc_h11_grouping_2(x):
        if pd.isnull(x):
            return u'无数据'
        elif x < 140:
            return '[0, 220)'
        elif 140 <= x < 260:
            return '[220, 380)'
        elif 260 <= x < 400:
            return '[230, 370)'
        elif 400 <= x < 650:
            return '[370, 600)'
        elif x >= 650:
            return '[650, +)'

    @staticmethod
    def cpc_bank_in_len_grouping(x):
        if pd.isnull(x):
            return u'无数据'
        elif x < 1:
            return '[0, 01)'
        elif 1 <= x < 8:
            return '[01, 8)'
        elif x >= 8:
            return '[8, +)'

    @staticmethod
    def cps_cnt_grouping(x):
        if pd.isnull(x):
            return "无数据"
        elif x < 20:
            return '[0, 0200)'
        elif 20 <= x < 100:
            return '[0150, 0450)'
        elif 100 <= x < 150:
            return '[0200, 0750)'
        elif 150 <= x:
            return '[0750, +)'

# 定义连续变量类型
CONTINUITY_LIST = ['float', 'int']
# 定义离散变量类型
DISCRETE_LIST = ['str', 'bool']
# 需要移除的变量
REMOVE_LIST = ['error_cnt', 'creat_time', '_id', 'env', 'update_time', 'org']
# 各变量类型
VARIATE_TYPE = {
    'cpc_mthly_call_02_cnt': u'int',
    'cpc_bank_m1_avg_len': u'float',
    'cpc_h01_cnt_pct': u'float',
    'cpc_h06_len_pct': u'float',
    'cpc_card_m3_out_len_pct':  u'float',
    'cpc_bi_phn_len':  u'float',
    'cpc_h07_cnt': u'int',
    'eb_item_P50_amt':  u'float',
    'cpc_car_m3_len':  u'float',
    'cpc_house_m3_out_cnt':  u'float',
    'cpc_loan_in_len':  u'float',
    'cpc_loan_m2_in_len_pct':  u'float',
    'cpc_h15_len_pct':  u'float',
    'cpc_car_m2_cnt':  u'float',
    'cpn_h11_amt_pct':  u'float',
    'cpc_bank_out_cnt':  u'float',
    'cpc_car_out_len_pct':  u'float',
    'cpc_loan_m3_avg_cnt':  u'float',
    'cpc_loan_m3_cnt':  u'float',
    'cpc_house_m2_avg_len_pct':  u'float',
    'cpc_h07_cnt_pct':  u'float',
    'cpc_card_m2_avg_cnt_pct':  u'float',
    'cpc_car_mth_avg_len':  u'float',
    'cpa_ut':  u'str',
    'cpc_loan_in_len_pct':  u'float',
    'cpc_cp_len':  u'float',
    'cpc_card_len_pct':  u'float',
    'cpc_card_m2_len_pct':  u'float',
    'cpc_loan_m3_out_len':  u'float',
    'cpc_car_m2_in_cnt':  u'float',
    'cpc_loan_m2_out_len_pct':  u'float',
    'cpc_card_out_cnt':  u'float',
    'cpc_h08_cnt_pct':  u'float',
    'cpc_h22_len_pct':  u'float',
    'cpc_house_m1_in_cnt':  u'float',
    'cpc_card_m1_avg_len':  u'float',
    'eb_has_app_live_addr':  u'bool',
    'cpc_car_m1_in_len':  u'float',
    'cpc_bank_len_pct':  u'float',
    'app_qstn_work_time':  u'str',
    'cpc_card_len':  u'float',
    'cpc_h04_len':  u'float',
    'cpc_loan_m2_avg_len':  u'float',
    'cp_cntk_all_day_cnt':  u'int',
    'cpc_house_m3_avg_cnt_pct':  u'float',
    'cpc_h10_len_pct':  u'float',
    'cpn_h05_amt_pct':  u'float',
    'cpn_h11_amt':  u'float',
    'cpc_h02_len_pct':  u'float',
    'cpc_bank_m3_out_len':  u'float',
    'cpc_loan_m2_in_cnt_pct':  u'float',
    'cpc_h10_len':  u'float',
    'eb_addr_5k_cnt':  u'int',
    'cpn_h21_amt':  u'float',
    'cpc_h06_cnt':  u'int',
    'cpn_h09_amt':  u'float',
    'cpc_loan_mth_avg_cnt':  u'float',
    'cpc_h08_len':  u'float',
    'cpn_h12_amt_pct':  u'float',
    'app_idcn':  u'str',
    'cpc_car_m2_avg_len':  u'float',
    'cpc_bank_m3_len_pct':  u'float',
    'cpc_car_m2_cnt_pct':  u'float',
    'cpc_bank_m1_in_len_pct':  u'float',
    'cpc_house_len_pct':  u'float',
    'cpc_bi_phn_cnt':  u'int',
    'cpc_h18_cnt':  u'int',
    'app_cnct_type1':  u'str',
    'cpc_h14_len_pct':  u'float',
    'cpc_hldy_len':  u'float',
    'app_cnct_type2':  u'str',
    'cps_car_cnt':  u'int',
    'cpc_loan_m3_len':  u'float',
    'cpn_h15_amt_pct':  u'float',
    'cpc_bank_m2_avg_cnt_pct':  u'float',
    'eb_ord_amt':  u'float',
    'app_qstn_car':  u'str',
    'eb_shr_addr_cnt':  u'int',
    'cpc_h23_len_pct':  u'float',
    'cpc_car_m3_out_len':  u'float',
    'cpc_bank_m1_in_cnt_pct':  u'float',
    'cpc_bank_in_cnt':  u'int',
    'cpc_bank_m3_cnt_pct':  u'float',
    'cpc_bank_out_cnt_pct':  u'float',
    'cpc_loan_m1_cnt':  u'float',
    'eb_live_addr_ord_avg_amt':  u'float',
    'cpc_car_m1_out_cnt_pct':  u'float',
    'cpc_card_m1_out_cnt_pct':  u'float',
    'cpn_h17_amt':  u'float',
    'cpc_bank_m3_avg_len_pct':  u'float',
    'cpc_house_m1_out_cnt':  u'float',
    'cpc_h11_cnt_pct':  u'float',
    'cpc_h07_len':  u'float',
    'cpc_card_cnt':  u'float',
    'eb_ord_m1_amt':  u'float',
    'cpc_house_in_cnt_pct':  u'float',
    'cpc_h23_cnt_pct':  u'float',
    'cpc_card_m1_cnt_pct':  u'float',
    'cpc_house_m1_avg_len_pct':  u'float',
    'cpc_bank_m1_avg_cnt':  u'float',
    'cpc_mthly_call_01_cnt':  u'int',
    'cpc_loan_out_cnt_pct':  u'float',
    'cpc_car_in_cnt_pct':  u'float',
    'eb_ord_5h_cnt_pct':  u'float',
    'cpc_card_mth_avg_cnt':  u'float',
    'cpc_car_mth_avg_len_pct':  u'float',
    'cp_use_days':  u'int',
    'app_idcn_city':  u'str',
    'cpc_h03_len_pct':  u'float',
    'cpc_h09_len_pct':  u'float',
    'cpn_h00_amt_pct':  u'float',
    'cpc_card_m2_out_cnt':  u'float',
    'cpc_loan_m1_in_cnt':  u'float',
    'cpc_card_m2_avg_cnt':  u'float',
    'eb_ord_2k_cnt_pct':  u'float',
    'cpc_h22_cnt_pct':  u'float',
    'cpc_loan_m1_avg_cnt_pct':  u'float',
    'eb_item_P99_amt':  u'float',
    'cpc_card_m3_len_pct':  u'float',
    'cpc_card_mth_avg_cnt_pct':  u'float',
    'cpc_card_m1_in_cnt_pct':  u'float',
    'cpc_car_m1_cnt_pct':  u'float',
    'cpc_house_m2_cnt':  u'float',
    'cpc_h21_cnt':  u'int',
    'cpc_h16_cnt_pct':  u'float',
    'cp_reg_rgn_rank':  u'int',
    'cpc_bank_m1_cnt':  u'float',
    'cpc_mthly_call_05_cnt_pct':  u'float',
    'cpn_h23_amt_pct':  u'float',
    'cpc_card_m3_avg_cnt':  u'float',
    'cpc_loan_m2_out_len':  u'float',
    # 'in_risk_rules_TD':  u'bool',
    'cpc_card_m1_cnt':  u'float',
    'cpc_loan_m2_avg_cnt':  u'float',
    'cpc_h19_len':  u'float',
    'cpc_bank_cnt_pct':  u'float',
    'cpc_loan_m3_in_len_pct':  u'float',
    'cpc_mthly_call_02_cnt_pct':  u'float',
    'cpc_last_no_data_day_cnt':  u'int',
    'cpc_house_m1_out_cnt_pct':  u'float',
    'cpc_car_m1_out_cnt':  u'float',
    'cpc_h02_cnt_pct':  u'float',
    'cpc_house_m2_out_cnt':  u'float',
    'eb_self_ord_amt':  u'float',
    'cpc_h12_len_pct':  u'float',
    'cpc_card_m1_out_len_pct':  u'float',
    'cpc_h17_cnt':  u'int',
    'cpc_h13_cnt':  u'int',
    'cpc_house_out_cnt_pct':  u'float',
    'cpc_loan_out_len_pct':  u'float',
    'cpc_h18_cnt_pct':  u'float',
    'cpc_card_out_len_pct':  u'float',
    'cpc_car_m3_in_cnt':  u'float',
    'cpc_car_m3_cnt_pct':  u'float',
    'cpc_card_m2_out_len_pct':  u'float',
    'cpc_h14_len':  u'float',
    'cpc_card_m2_avg_len':  u'float',
    'cpc_car_m2_len_pct':  u'float',
    'cpc_h17_len':  u'float',
    'cpc_bank_m2_out_len_pct':  u'float',
    'eb_item_avg_amt':  u'float',
    'app_qstn_crdt_card':  u'str',
    'cpc_bank_in_cnt_pct':  u'float',
    'cpc_h20_len':  u'float',
    'cpc_bank_m3_in_cnt_pct':  u'float',
    'eb_ord_m1_cnt_pct':  u'float',
    'cpc_card_m1_in_len_pct':  u'float',
    'cpc_house_m2_len_pct':  u'float',
    'cpc_card_out_cnt_pct':  u'float',
    'cpa_rt':  u'str',
    'cpc_h08_len_pct':  u'float',
    'cpc_card_m2_out_cnt_pct':  u'float',
    'cps_car_mth_avg_cnt':  u'int',
    'cps_cnt':  u'int',
    'cpc_loan_mth_avg_cnt_pct':  u'float',
    'cpc_bank_m1_len_pct':  u'float',
    'cpc_h00_len_pct':  u'float',
    'eb_ord_m2_amt_pct':  u'float',
    'cpn_h16_amt':  u'float',
    'cpc_mthly_call_01_cnt_pct':  u'float',
    'eb_ord_item_cnt':  u'int',
    'cpc_bank_m3_out_cnt_pct':  u'float',
    'cpc_house_m2_out_len_pct':  u'float',
    'cpc_bank_mth_avg_len':  u'float',
    'cpc_h21_len_pct':  u'float',
    'cpn_h14_amt':  u'float',
    'cpc_h03_cnt':  u'int',
    'cpc_bank_in_len':  u'float',
    'cpc_car_m1_avg_cnt':  u'float',
    'cpn_h13_amt':  u'float',
    'cpc_house_m1_avg_cnt_pct':  u'float',
    'cpc_card_m1_in_len':  u'float',
    'cpa_auth_match_all':  u'bool',
    'eb_item_P75_amt':  u'float',
    'cpc_house_m2_in_cnt':  u'float',
    'si_is_valid':  u'bool',
    'cpc_loan_m1_avg_cnt':  u'float',
    'cpc_house_m1_len':  u'float',
    'cpc_h16_len_pct':  u'float',
    'cpc_h01_len_pct':  u'float',
    'cpc_card_m2_avg_len_pct':  u'float',
    'cpn_h09_amt_pct':  u'float',
    'cpc_house_m1_out_len_pct':  u'float',
    'cpc_bank_m2_in_len':  u'float',
    'cpc_h12_cnt':  u'int',
    'cpc_wked_cnt_pct':  u'float',
    'app_prov':  u'str',
    'cpc_loan_m3_avg_len':  u'float',
    'eb_ord_1k_cnt_pct':  u'float',
    'eb_ord_m2_cnt':  u'int',
    'eb_self_ord_cnt_pct':  u'float',
    'cpc_h13_len':  u'float',
    'eb_item_P01_amt':  u'float',
    'cpn_h04_amt_pct':  u'float',
    'cpc_loan_m1_out_cnt_pct':  u'float',
    'cpc_loan_in_cnt':  u'int',
    'cpc_house_m1_avg_len':  u'float',
    'eb_ord_m1_amt_pct':  u'float',
    'cpc_3rd_call_rgn':  u'str',
    'cpc_h22_cnt':  u'int',
    'cpn_h03_amt':  u'float',
    'cpc_house_m3_in_cnt':  u'float',
    'cpc_bank_m2_cnt':  u'float',
    'cpa_auth_match_1':  u'bool',
    'cpc_card_cnt_pct':  u'float',
    'cpc_card_m2_cnt_pct':  u'float',
    'cpn_h03_amt_pct':  u'float',
    'cpc_mth_avg_out_cnt':  u'float',
    'eb_ord_m1_cnt':  u'int',
    'cpc_loan_m3_cnt_pct':  u'float',
    'eb_addr_10k_cnt':  u'int',
    'eb_live_addr_ord_amt':  u'float',
    'cpc_loan_m3_len_pct':  u'float',
    'app_cp':  u'str',
    'app_ct':  u'str',
    'cpc_card_m1_len_pct':  u'float',
    'cpc_h13_cnt_pct':  u'float',
    'cpc_h05_cnt':  u'int',
    'cpc_h11_len_pct':  u'float',
    'cpc_car_m3_avg_len_pct':  u'float',
    'cpc_house_m3_len':  u'float',
    'cpc_loan_m3_out_cnt':  u'float',
    'cpc_house_cnt':  u'float',
    'eb_ord_avg_amt':  u'float',
    'cpc_h19_cnt':  u'int',
    'eb_self_ord_cnt':  u'int',
    'eb_addr_1k_cnt':  u'int',
    'cpc_in_len':  u'float',
    'cpc_loan_m2_avg_len_pct':  u'float',
    'cpc_h15_cnt_pct':  u'float',
    'cpc_loan_len_pct':  u'float',
    'cpc_bank_m1_avg_cnt_pct':  u'float',
    'eb_self_ord_amt_pct':  u'float',
    'cpc_h16_cnt':  u'int',
    'cpc_loan_m2_out_cnt':  u'float',
    'cpn_h08_amt_pct':  u'float',
    'cpc_house_m2_len':  u'float',
    'cpc_h09_cnt_pct':  u'float',
    'cpc_loan_m2_in_cnt':  u'float',
    'cpc_call_rank2':  u'int',
    'si_degree':  u'str',
    'cpc_bank_cnt':  u'float',
    'cpc_h09_cnt':  u'int',
    'eb_ord_avg_item_cnt':  u'int',
    'cpc_out_cnt':  u'int',
    'eb_ord_m2_cnt_pct':  u'float',
    'cpc_h19_cnt_pct':  u'float',
    'cpc_h08_cnt':  u'int',
    'cpc_in_cnt':  u'int',
    'cpc_house_mth_avg_len_pct':  u'float',
    'cpc_h05_len':  u'float',
    'cpc_bank_m1_out_cnt':  u'float',
    'cpc_bi_phn_pct':  u'float',
    'cpc_loan_m3_avg_cnt_pct':  u'float',
    'app_org_acc':  u'str',
    'cpc_bank_m3_out_len_pct':  u'float',
    'app_qstn_kid':  u'str',
    'cps_card_mth_avg_cnt':  u'int',
    'cpc_wked_len':  u'float',
    'cpc_bi_cp_len_pct':  u'float',
    'cpn_h22_amt_pct':  u'float',
    'eb_ord_mth_avg_amt':  u'float',
    'cpc_car_m3_in_cnt_pct':  u'float',
    'cpn_h04_amt':  u'float',
    'cpc_bank_m2_out_cnt_pct':  u'float',
    'cpc_card_m3_in_len_pct':  u'float',
    'cpc_bank_in_len_pct':  u'float',
    'cpc_phn_cnt':  u'int',
    'app_cnct_nm2':  u'str',
    'cpc_car_m1_in_cnt':  u'float',
    'cpc_loan_m3_in_cnt':  u'float',
    'cpc_h15_cnt':  u'int',
    'cpc_h06_len':  u'float',
    'cpc_h15_len':  u'float',
    'cps_out_cnt':  u'int',
    'cpb_mth_cnt':  u'int',
    'cpc_loan_m2_cnt_pct':  u'float',
    'cpc_loan_m2_out_cnt_pct':  u'float',
    'cpc_wked_len_pct':  u'float',
    'cpc_card_m2_in_len':  u'float',
    'cpc_h20_cnt':  u'int',
    'cpc_card_m1_len':  u'float',
    'cpc_card_m3_len':  u'float',
    'cpc_card_m1_out_cnt':  u'float',
    'cpc_bank_mth_avg_len_pct':  u'float',
    'cpc_bank_m2_cnt_pct':  u'float',
    'cpc_h16_len':  u'float',
    'cpc_car_m2_out_cnt_pct':  u'float',
    'cpc_h11_cnt':  u'int',
    'cpc_house_m3_out_len':  u'float',
    'cpn_h19_amt_pct':  u'float',
    'cpc_card_m3_cnt':  u'float',
    'cpc_bank_m2_len_pct':  u'float',
    'cpn_h20_amt_pct':  u'float',
    'cpc_loan_m1_avg_len_pct':  u'float',
    'cpc_cp_pct':  u'float',
    'cpc_house_m2_in_len':  u'float',
    'cpc_card_m1_avg_cnt_pct':  u'float',
    'cpc_loan_m1_cnt_pct':  u'float',
    'cpc_card_m1_avg_cnt':  u'float',
    'cpc_house_mth_avg_len':  u'float',
    'cpc_house_m3_out_cnt_pct':  u'float',
    'cpc_house_m2_avg_cnt_pct':  u'float',
    'cpc_car_m3_out_cnt_pct':  u'float',
    'cpc_loan_in_cnt_pct':  u'float',
    'cpc_h12_cnt_pct':  u'float',
    'cp_live_rgn_rank':  u'int',
    'cpc_card_m3_out_cnt':  u'float',
    'cpn_h10_amt_pct':  u'float',
    'cpc_h03_len':  u'float',
    'eb_ord_5h_cnt':  u'int',
    'cpc_car_m3_avg_len':  u'float',
    'cpc_loan_m1_out_cnt':  u'float',
    'cpc_car_m1_in_cnt_pct':  u'float',
    'cpc_h00_cnt_pct':  u'float',
    'app_cp_carrier':  u'str',
    'app_idcn_region':  u'str',
    'cpn_h06_amt_pct':  u'float',
    'cpa_rt_months': u'float',
    'eb_ord_1k_cnt':  u'int',
    'cpc_house_out_cnt':  u'float',
    'cpc_house_m1_out_len':  u'float',
    'cpc_h02_len':  u'float',
    'cpc_loan_cnt_pct':  u'float',
    'cpc_loan_m1_len':  u'float',
    'cpc_mth_avg_in_cnt':  u'float',
    'cpc_card_m2_len':  u'float',
    'cpc_car_m3_avg_cnt_pct':  u'float',
    'cpc_house_out_len_pct':  u'float',
    'cpc_card_m3_in_len':  u'float',
    'cpc_bank_m2_len':  u'float',
    'cpc_call_rank1':  u'int',
    'app_qstn_income':  u'str',
    'cpc_card_in_cnt':  u'int',
    'cpc_car_m3_out_len_pct':  u'float',
    'cpc_car_m3_len_pct':  u'float',
    'cpc_loan_m2_len_pct':  u'float',
    'cpc_card_m2_in_cnt':  u'float',
    'cpc_car_m2_avg_len_pct':  u'float',
    'cpc_car_mth_avg_cnt':  u'float',
    'cpc_house_m3_avg_len_pct':  u'float',
    'cpn_h22_amt':  u'float',
    'cpc_car_in_len':  u'float',
    'cpc_house_in_cnt':  u'int',
    'cpc_mth_avg_in_len':  u'float',
    'cpc_mthly_call_10_cnt':  u'int',
    'cpc_hldy_cnt':  u'float',
    'cpc_car_m2_out_cnt':  u'float',
    'cpc_car_out_cnt':  u'float',
    'cpc_h23_len':  u'float',
    'cpc_house_in_len_pct':  u'float',
    'cpc_bank_m3_len':  u'float',
    'cpc_car_m2_in_len':  u'float',
    'cpc_car_m1_avg_cnt_pct':  u'float',
    'cpc_loan_m1_in_len':  u'float',
    'cpc_bank_m3_out_cnt':  u'float',
    'cpc_house_m1_in_cnt_pct':  u'float',
    'cpc_2nd_call_rgn':  u'str',
    'cpn_h01_amt':  u'float',
    'cpc_h18_len':  u'float',
    'in_blacklist':  u'bool',
    'cpc_out_len':  u'float',
    'cpc_house_m2_cnt_pct':  u'float',
    'cpc_house_m3_in_len_pct':  u'float',
    'cpc_bank_m1_in_len':  u'float',
    'cpc_h10_cnt':  u'int',
    'cpc_bank_m3_avg_cnt_pct':  u'float',
    'cpc_h00_cnt':  u'int',
    'cpc_car_len':  u'float',
    'cpc_card_m2_in_len_pct':  u'float',
    'cpc_car_m2_in_cnt_pct':  u'float',
    'cpc_bank_m2_out_cnt':  u'float',
    'cpc_house_m3_cnt':  u'float',
    'cpc_car_cnt':  u'float',
    'cpn_h07_amt_pct':  u'float',
    'cps_loan_mth_avg_cnt':  u'int',
    'cpc_h00_len':  u'float',
    'cps_mth_avg_cnt':  u'int',
    'cpc_card_m3_cnt_pct':  u'float',
    'cpc_car_m2_out_len':  u'float',
    'cpc_house_m3_avg_cnt':  u'float',
    'app_cnct_nm1':  u'str',
    'cpc_loan_m2_in_len':  u'float',
    'cpn_h17_amt_pct':  u'float',
    'cpc_house_in_len':  u'float',
    'cpc_loan_mth_avg_len':  u'float',
    'eb_ord_cnt':  u'int',
    'cpc_1st_call_rgn':  u'str',
    'eb_item_P25_amt':  u'float',
    'cpb_mth_avg_pay':  u'float',
    'cpc_h17_cnt_pct':  u'float',
    'cpc_bank_m2_in_cnt_pct':  u'float',
    'cpn_h15_amt':  u'float',
    'cpc_mth_avg_out_len':  u'float',
    'cpc_house_len':  u'float',
    'eb_addr_5h_cnt':  u'int',
    'cpc_loan_m1_out_len_pct':  u'float',
    'cpc_bank_m2_out_len':  u'float',
    'cpc_card_m3_avg_cnt_pct':  u'float',
    'app_qstn_mari':  u'str',
    'cpc_loan_out_cnt':  u'int',
    'cpc_card_m3_out_cnt_pct':  u'float',
    'cpc_bank_out_len_pct':  u'float',
    'cpc_h17_len_pct':  u'float',
    'cps_house_cnt':  u'int',
    'cpc_house_m3_in_cnt_pct':  u'float',
    'app_age':  u'int',
    'cps_loan_cnt_pct':  u'float',
    'eb_addr_cnt':  u'int',
    'cps_loan_cnt':  u'int',
    'cpc_loan_m3_in_len':  u'float',
    'cpc_car_m1_len_pct':  u'float',
    'cpc_h19_len_pct':  u'float',
    'cpc_bank_m3_avg_cnt':  u'float',
    'app_nm':  u'str',
    'cpc_house_m2_avg_len':  u'float',
    'cpc_bank_m2_avg_len_pct':  u'float',
    'cpc_car_m1_len':  u'float',
    'cpn_h14_amt_pct':  u'float',
    'app_valid_idcn':  u'bool',
    'cpc_h14_cnt':  u'int',
    'cpc_card_m2_in_cnt_pct':  u'float',
    'cpc_loan_m1_avg_len':  u'float',
    'cps_card_cnt':  u'int',
    'cpn_h01_amt_pct':  u'float',
    'cpc_bank_mth_avg_cnt':  u'float',
    'cpc_card_in_len':  u'float',
    'cpn_h18_amt':  u'float',
    'cpc_bank_m1_out_cnt_pct':  u'float',
    'cpc_wkdt_cnt':  u'float',
    'cpc_out_cnt_pct':  u'float',
    'cpc_house_m1_in_len':  u'float',
    'cpc_loan_m2_cnt':  u'float',
    'cpc_car_m1_avg_len_pct':  u'float',
    'cpc_card_mth_avg_len':  u'float',
    'cpc_house_m2_in_cnt_pct':  u'float',
    'cpc_loan_m1_in_cnt_pct':  u'float',
    'cpc_house_out_len':  u'float',
    'cpn_h13_amt_pct':  u'float',
    'cpc_h07_len_pct':  u'float',
    'cpc_bank_mth_avg_cnt_pct':  u'float',
    'cpc_loan_m3_avg_len_pct':  u'float',
    'cpc_car_m1_out_len_pct':  u'float',
    'cpc_card_out_len':  u'float',
    'cpc_house_m1_in_len_pct':  u'float',
    'cpc_loan_out_len':  u'float',
    'cps_mth_avg_out_cnt':  u'int',
    'cpc_h22_len':  u'float',
    'eb_ord_mth_avg_item_cnt':  u'float',
    'cpc_bi_cp_cnt_pct':  u'float',
    'cpc_bank_m1_len':  u'float',
    'cpc_house_m1_cnt_pct':  u'float',
    'eb_ord_m2_amt':  u'float',
    'cpc_bank_m1_out_len':  u'float',
    'cpn_amt':  u'float',
    'cpn_h21_amt_pct':  u'float',
    'cpc_car_m3_in_len_pct':  u'float',
    'eb_live_addr_use_mths':  u'int',
    'cpc_bank_m2_avg_len':  u'float',
    'cpc_h20_cnt_pct':  u'float',
    'cpc_car_m3_in_len':  u'float',
    'cpc_house_m1_cnt':  u'float',
    'cpc_h10_cnt_pct':  u'float',
    'cpc_car_m2_len':  u'float',
    'cpn_h02_amt':  u'float',
    'cpc_h05_len_pct':  u'float',
    'cpc_h21_len':  u'float',
    'cpc_no_call_day_cnt_pct':  u'float',
    'cpc_h04_cnt':  u'int',
    'cps_bank_cnt_pct':  u'float',
    'cpc_hldy_cnt_pct':  u'float',
    'cpc_len':  u'float',
    'cpc_bank_m1_out_len_pct':  u'float',
    'cpc_card_in_len_pct':  u'float',
    'cps_car_cnt_pct':  u'float',
    'eb_item_P10_amt':  u'float',
    'eb_addr_avg_use_mths':  u'float',
    'cpc_car_cnt_pct':  u'float',
    'cpc_loan_len':  u'float',
    'app_gender':  u'str',
    'cpc_house_cnt_pct':  u'float',
    'eb_ord_valid_cnt':  u'int',
    'cpc_car_out_len':  u'float',
    'cpc_h02_cnt':  u'int',
    'cpc_bank_m1_cnt_pct':  u'float',
    'cpn_h18_amt_pct':  u'float',
    'cp_max_cumm_tf_days':  u'int',
    'cpc_loan_m3_in_cnt_pct':  u'float',
    'cpc_bank_out_len':  u'float',
    'eb_live_addr_ord_cnt':  u'int',
    'cpc_bank_m3_cnt':  u'float',
    'cpa_rt_days':  u'int',
    'cpc_loan_m2_len':  u'float',
    'cpn_h16_amt_pct':  u'float',
    'cpc_h23_cnt':  u'int',
    'cpc_house_m2_avg_cnt':  u'float',
    'cpc_bank_m1_avg_len_pct':  u'float',
    'cpc_car_m1_in_len_pct':  u'float',
    'cpc_house_m3_in_len':  u'float',
    'eb_live_addr_use_days':  u'int',
    'cpc_card_m3_avg_len':  u'float',
    'cpc_car_in_cnt':  u'int',
    'cpc_card_m3_in_cnt_pct':  u'float',
    'cps_in_cnt':  u'int',
    'app_cp_prov':  u'str',
    'cpc_card_m1_in_cnt':  u'float',
    'cpc_h18_len_pct':  u'float',
    'cps_card_cnt_pct':  u'float',
    'cpc_loan_m3_out_len_pct':  u'float',
    'cpc_bank_m2_in_cnt':  u'float',
    'cpc_house_m3_cnt_pct':  u'float',
    'cpc_bank_m1_in_cnt':  u'float',
    'cpb_bill_cnt':  u'int',
    'cpc_card_in_cnt_pct':  u'float',
    'cpc_bank_m3_in_cnt':  u'float',
    'cpc_h05_cnt_pct':  u'float',
    'eb_addr_2k_cnt':  u'int',
    'eb_item_P95_amt':  u'float',
    'cpn_h06_amt':  u'float',
    'cpc_cp_cnt':  u'int',
    'cp_tf_days':  u'int',
    'cpc_car_m1_out_len':  u'float',
    'cpc_loan_m3_out_cnt_pct':  u'float',
    'cpc_in_cnt_pct':  u'float',
    'cps_mth_avg_in_cnt':  u'int',
    'cpc_house_m2_out_len':  u'float',
    'cpn_h19_amt':  u'float',
    'eb_addr_avg_ord_cnt':  u'float',
    'cpc_house_mth_avg_cnt_pct':  u'float',
    'cpc_card_m3_out_len':  u'float',
    'cpn_h10_amt':  u'float',
    'cpn_h20_amt':  u'float',
    'cpc_h13_len_pct':  u'float',
    'cpc_car_m1_cnt':  u'float',
    'cpc_bi_phn_len_pct':  u'float',
    'cpc_wkdt_cnt_pct':  u'float',
    'cpc_h20_len_pct':  u'float',
    'cpc_card_m2_out_len':  u'float',
    'cpc_car_m2_avg_cnt_pct':  u'float',
    'cpc_h12_len':  u'float',
    'cpc_no_call_day_cnt':  u'int',
    'eb_item_P90_amt':  u'float',
    'cpc_wkdt_len_pct':  u'float',
    'cpc_house_m2_in_len_pct':  u'float',
    'cpc_bi_cp_len':  u'float',
    'eb_addr_avg_use_days':  u'float',
    'cpc_loan_m2_avg_cnt_pct':  u'float',
    'eb_live_addr_gap_days':  u'int',
    'cpc_bank_m3_avg_len':  u'float',
    'eb_ord_2k_cnt':  u'int',
    'cpc_bank_m3_in_len_pct':  u'float',
    'cpc_card_m3_avg_len_pct':  u'float',
    'cpc_bank_m3_in_len':  u'float',
    'cpc_house_m2_out_cnt_pct':  u'float',
    'cpn_h05_amt':  u'float',
    'cpc_cnt':  u'int',
    'eb_use_days':  u'int',
    'cpc_car_len_pct':  u'float',
    'cpc_bi_cp_cnt':  u'int',
    'cpc_h11_len':  u'float',
    'cpn_mth_avg_amt':  u'float',
    'cpc_bank_m2_avg_cnt':  u'float',
    'cpn_h08_amt':  u'float',
    'cps_house_mth_avg_cnt':  u'int',
    'cpc_card_m2_cnt':  u'float',
    'cpc_car_out_cnt_pct':  u'float',
    'cpc_card_m1_out_len':  u'float',
    'cpn_h12_amt':  u'float',
    'cpc_mth_avg_len':  u'float',
    'cpc_house_m1_avg_cnt':  u'float',
    'si_is_in_sch':  u'bool',
    'eb_live_addr_gap_mths':  u'int',
    'cpc_card_mth_avg_len_pct':  u'float',
    'cpc_bank_len':  u'float',
    'cpc_car_m3_avg_cnt':  u'float',
    'cpc_loan_m1_out_len':  u'float',
    'cpc_h14_cnt_pct':  u'float',
    'cpc_h04_len_pct':  u'float',
    'cpc_h04_cnt_pct':  u'float',
    'cps_bank_cnt':  u'int',
    'cpc_car_m2_in_len_pct':  u'float',
    'cpc_mthly_call_05_cnt':  u'int',
    'cpc_wkdt_len':  u'float',
    'cpc_loan_m1_in_len_pct':  u'float',
    'cpc_car_m1_avg_len':  u'float',
    'cpc_loan_mth_avg_len_pct':  u'float',
    'cpc_house_m3_avg_len':  u'float',
    'cpn_h23_amt':  u'float',
    'cpc_car_mth_avg_cnt_pct':  u'float',
    'cpn_h02_amt_pct':  u'float',
    'cpc_car_m2_out_len_pct':  u'float',
    'cpc_h06_cnt_pct':  u'float',
    'cpc_h21_cnt_pct':  u'float',
    'cpn_h07_amt':  u'float',
    'cps_house_cnt_pct':  u'float',
    'cpn_h00_amt':  u'float',
    'cpc_house_mth_avg_cnt':  u'float',
    'cpc_h01_cnt':  u'int',
    'cpc_car_m2_avg_cnt':  u'float',
    'cps_bank_mth_avg_cnt':  u'int',
    'eb_ord_mth_avg_cnt':  u'float',
    'cpc_house_m1_len_pct':  u'float',
    'app_cp_city':  u'str',
    'cpc_loan_cnt':  u'int',
    'cpc_card_m3_in_cnt':  u'float',
    'cpc_bank_m2_in_len_pct':  u'float',
    'cpc_house_m3_out_len_pct':  u'float',
    'cpc_h01_len':  u'float',
    'cpc_mthly_call_10_cnt_pct':  u'float',
    'cpc_loan_m1_len_pct':  u'float',
    'cpc_h03_cnt_pct':  u'float',
    'cpc_house_m3_len_pct':  u'float',
    'cpc_car_m3_out_cnt':  u'float',
    'cpc_hldy_len_pct':  u'float',
    'cpc_h09_len':  u'float',
    'cpc_card_m1_avg_len_pct':  u'float',
    'cpc_car_in_len_pct':  u'float',
    'eb_item_P05_amt':  u'float',
    'cpc_wked_cnt':  u'float',
    'eb_use_mths':  u'int',
    'cpc_car_m3_cnt':  u'float',
    'cpc_bi_phn_cnt_pct':  u'float',
    'cpc_mth_avg_cnt':  u'float',
    'app_qstn_house': u'str',
    'cpc_nonlocal_cnt': u'int',
    'cpc_nonlocal_in_cnt': u'int',
    'cpc_nonlocal_out_cnt': u'int',
    'cpc_nonlocal_cnt_pct': u'float',
    'cpc_nonlocal_in_cnt_pct': u'float',
    'cpc_nonlocal_out_cnt_pct': u'float',
    'cpc_nonlocal_len': u'float',
    'cpc_nonlocal_in_len': u'float',
    'cpc_nonlocal_out_len': u'float',
    'cpc_nonlocal_len_pct': u'float',
    'cpc_nonlocal_in_len_pct': u'float',
    'cpc_nonlocal_out_len_pct': u'float',
    'cpc_roam_cnt': u'int',
    'cpc_roam_in_cnt': u'int',
    'cpc_roam_out_cnt': u'int',
    'cpc_roam_cnt_pct': u'float',
    'cpc_roam_in_cnt_pct': u'float',
    'cpc_roam_out_cnt_pct': u'float',
    'cpc_roam_len': u'float',
    'cpc_roam_in_len': u'float',
    'cpc_roam_out_len': u'float',
    'cpc_roam_len_pct': u'float',
    'cpc_roam_in_len_pct': u'float',
    'cpc_roam_out_len_pct': u'float',
    'cpc_chain_len_mth1_rate': u'float',
    'cpc_chain_len_mth2_rate': u'float',
    'cpc_chain_len_mth3_rate': u'float',
    'cpc_chain_len_mth4_rate': u'float',
    'cpc_chain_cnt_mth1_rate': u'float',
    'cpc_chain_cnt_mth2_rate': u'float',
    'cpc_chain_cnt_mth3_rate': u'float',
    'cpc_chain_cnt_mth4_rate': u'float',
    'cpc_ins_cnt': u'float',
    'cpc_ins_len': u'float',
    'cpc_ins_cnt_pct': u'float',
    'cpc_ins_len_pct': u'float',
    'cpc_ins_out_cnt': u'float',
    'cpc_ins_out_len': u'float',
    'cpc_ins_out_cnt_pct': u'float',
    'cpc_ins_out_len_pct': u'float',
    'cpc_ins_in_cnt': u'int',
    'cpc_ins_in_len': u'float',
    'cpc_ins_in_cnt_pct': u'float',
    'cpc_ins_in_len_pct': u'float',
    'cpc_ins_mth_avg_cnt': u'float',
    'cpc_ins_mth_avg_len': u'float',
    'cpc_ins_mth_avg_cnt_pct': u'float',
    'cpc_ins_mth_avg_len_pct': u'float',
    'cpc_ins_m1_cnt': u'float',
    'cpc_ins_m1_len': u'float',
    'cpc_ins_m1_cnt_pct': u'float',
    'cpc_ins_m1_len_pct': u'float',
    'cpc_ins_m1_out_cnt': u'float',
    'cpc_ins_m1_out_len': u'float',
    'cpc_ins_m1_out_cnt_pct': u'float',
    'cpc_ins_m1_out_len_pct': u'float',
    'cpc_ins_m1_in_cnt': u'float',
    'cpc_ins_m1_in_len': u'float',
    'cpc_ins_m1_in_cnt_pct': u'float',
    'cpc_ins_m1_in_len_pct': u'float',
    'cpc_ins_m1_avg_cnt': u'float',
    'cpc_ins_m1_avg_len': u'float',
    'cpc_ins_m1_avg_cnt_pct': u'float',
    'cpc_ins_m1_avg_len_pct': u'float',
    'cpc_ins_m2_cnt': u'float',
    'cpc_ins_m2_len': u'float',
    'cpc_ins_m2_cnt_pct': u'float',
    'cpc_ins_m2_len_pct': u'float',
    'cpc_ins_m2_out_cnt': u'float',
    'cpc_ins_m2_out_len': u'float',
    'cpc_ins_m2_out_cnt_pct': u'float',
    'cpc_ins_m2_out_len_pct': u'float',
    'cpc_ins_m2_in_cnt': u'float',
    'cpc_ins_m2_in_len': u'float',
    'cpc_ins_m2_in_cnt_pct': u'float',
    'cpc_ins_m2_in_len_pct': u'float',
    'cpc_ins_m2_avg_cnt': u'float',
    'cpc_ins_m2_avg_len': u'float',
    'cpc_ins_m2_avg_cnt_pct': u'float',
    'cpc_ins_m2_avg_len_pct': u'float',
    'cpc_ins_m3_cnt': u'float',
    'cpc_ins_m3_len': u'float',
    'cpc_ins_m3_cnt_pct': u'float',
    'cpc_ins_m3_len_pct': u'float',
    'cpc_ins_m3_out_cnt': u'float',
    'cpc_ins_m3_out_len': u'float',
    'cpc_ins_m3_out_cnt_pct': u'float',
    'cpc_ins_m3_out_len_pct': u'float',
    'cpc_ins_m3_in_cnt': u'float',
    'cpc_ins_m3_in_len': u'float',
    'cpc_ins_m3_in_cnt_pct': u'float',
    'cpc_ins_m3_in_len_pct': u'float',
    'cpc_ins_m3_avg_cnt': u'float',
    'cpc_ins_m3_avg_len': u'float',
    'cpc_ins_m3_avg_cnt_pct': u'float',
    'cpc_ins_m3_avg_len_pct': u'float',
    'cps_chain_mth1_rate': u'float',
    'cps_chain_mth2_rate': u'float',
    'cps_chain_mth3_rate': u'float',
    'cps_chain_mth4_rate': u'float',
    'cps_card_mth_avg_cnt_pct': u'float',
    'cps_card_m1_cnt': u'float',
    'cps_card_m1_cnt_pct': u'float',
    'cps_card_m1_avg_cnt': u'float',
    'cps_card_m1_avg_cnt_pct': u'float',
    'cps_card_m2_cnt': u'float',
    'cps_card_m2_cnt_pct': u'float',
    'cps_card_m2_avg_cnt': u'float',
    'cps_card_m2_avg_cnt_pct': u'float',
    'cps_card_m3_cnt': u'float',
    'cps_card_m3_cnt_pct': u'float',
    'cps_card_m3_avg_cnt': u'float',
    'cps_card_m3_avg_cnt_pct': u'float',
    'cps_house_mth_avg_cnt_pct': u'float',
    'cps_house_m1_cnt': u'float',
    'cps_house_m1_cnt_pct': u'float',
    'cps_house_m1_avg_cnt': u'float',
    'cps_house_m1_avg_cnt_pct': u'float',
    'cps_house_m2_cnt': u'float',
    'cps_house_m2_cnt_pct': u'float',
    'cps_house_m2_avg_cnt': u'float',
    'cps_house_m2_avg_cnt_pct': u'float',
    'cps_house_m3_cnt': u'float',
    'cps_house_m3_cnt_pct': u'float',
    'cps_house_m3_avg_cnt': u'float',
    'cps_house_m3_avg_cnt_pct': u'float',
    'cps_car_mth_avg_cnt_pct': u'float',
    'cps_car_m1_cnt': u'float',
    'cps_car_m1_cnt_pct': u'float',
    'cps_car_m1_avg_cnt': u'float',
    'cps_car_m1_avg_cnt_pct': u'float',
    'cps_car_m2_cnt': u'float',
    'cps_car_m2_cnt_pct': u'float',
    'cps_car_m2_avg_cnt': u'float',
    'cps_car_m2_avg_cnt_pct': u'float',
    'cps_car_m3_cnt': u'float',
    'cps_car_m3_cnt_pct': u'float',
    'cps_car_m3_avg_cnt': u'float',
    'cps_car_m3_avg_cnt_pct': u'float',
    'cps_loan_mth_avg_cnt_pct': u'float',
    'cps_loan_m1_cnt': u'float',
    'cps_loan_m1_cnt_pct': u'float',
    'cps_loan_m1_avg_cnt': u'float',
    'cps_loan_m1_avg_cnt_pct': u'float',
    'cps_loan_m2_cnt': u'float',
    'cps_loan_m2_cnt_pct': u'float',
    'cps_loan_m2_avg_cnt': u'float',
    'cps_loan_m2_avg_cnt_pct': u'float',
    'cps_loan_m3_cnt': u'float',
    'cps_loan_m3_cnt_pct': u'float',
    'cps_loan_m3_avg_cnt': u'float',
    'cps_loan_m3_avg_cnt_pct': u'float',
    'cps_bank_mth_avg_cnt_pct': u'float',
    'cps_bank_m1_cnt': u'float',
    'cps_bank_m1_cnt_pct': u'float',
    'cps_bank_m1_avg_cnt': u'float',
    'cps_bank_m1_avg_cnt_pct': u'float',
    'cps_bank_m2_cnt': u'float',
    'cps_bank_m2_cnt_pct': u'float',
    'cps_bank_m2_avg_cnt': u'float',
    'cps_bank_m2_avg_cnt_pct': u'float',
    'cps_bank_m3_cnt': u'float',
    'cps_bank_m3_cnt_pct': u'float',
    'cps_bank_m3_avg_cnt': u'float',
    'cps_bank_m3_avg_cnt_pct': u'float',
    'cps_ins_cnt': u'int',
    'cps_ins_cnt_pct': u'float',
    'cps_ins_mth_avg_cnt': u'int',
    'cps_ins_mth_avg_cnt_pct': u'float',
    'cps_ins_m1_cnt': u'float',
    'cps_ins_m1_cnt_pct': u'float',
    'cps_ins_m1_avg_cnt': u'float',
    'cps_ins_m1_avg_cnt_pct': u'float',
    'cps_ins_m2_cnt': u'float',
    'cps_ins_m2_cnt_pct': u'float',
    'cps_ins_m2_avg_cnt': u'float',
    'cps_ins_m2_avg_cnt_pct': u'float',
    'cps_ins_m3_cnt': u'float',
    'cps_ins_m3_cnt_pct': u'float',
    'cps_ins_m3_avg_cnt': u'float',
    'cps_ins_m3_avg_cnt_pct': u'float',
    'cpn_chain_mth1_rate': u'float',
    'cpn_chain_mth2_rate': u'float',
    'cpn_chain_mth3_rate': u'float',
    'cpn_chain_mth4_rate': u'float',
    'cpb_chain_mth1_rate': u'float',
    'cpb_chain_mth2_rate': u'float',
    'cpb_chain_mth3_rate': u'float',
    'cpb_chain_mth4_rate': u'float',
    'cpb_var_mth1': u'float',
    'cpb_var_mth2': u'float',
    'cpb_var_mth3': u'float',
    'cpb_var_mth4': u'float',
    'cpb_var_mth5': u'float',
    'cpc_work_hour_cnt': u'int',
    'cpc_h00_h02_cnt': u'int',
    'cpc_h22_h00_cnt_pct': u'float',
    'live_prov_match_cp_prov': u'str',
    'app_idcn_prov_rank': u'int',
    'cpc_call_1_cnt': u'int',
    'cpc_call_2_cnt': u'int',
    'cpc_call_1_len': u'float',
    'cpc_call_2_len': u'float',
    'cpc_main_prov_rank': u'int',
    'cpc_bank_card_ins_loan_cnt': u'int',
    'cpc_local_cnt': u'int',
    'cpc_local_len': u'float',
    'cpc_cnt_day_var': u'float',
    'cpc_cnt_day_stddev': u'float',
    'cpc_cnt_day_cv': u'float',
    'cpc_cnt_week_var': u'float',
    'cpc_cnt_week_stddev': u'float',
    'cpc_cnt_week_cv': u'float',
    'cpc_cnt_mth_var': u'float',
    'cpc_cnt_mth_stddev': u'float',
    'cpc_cnt_mth_cv': u'float',
    'cpc_len_day_var': u'float',
    'cpc_len_day_stddev': u'float',
    'cpc_len_day_cv': u'float',
    'cpc_len_week_var': u'float',
    'cpc_len_week_stddev': u'float',
    'cpc_len_week_cv': u'float',
    'cpc_len_mth_var': u'float',
    'cpc_len_mth_stddev': u'float',
    'cpc_len_mth_cv': u'float',
    'mother_baby_cnt': u'float',
    'mother_baby_price': u'float',
    'book_cnt': u"float",
    'book_price': u"float",
    'home_appliances_cnt': u'float',
    'home_appliances_price': u'float',
    'home_improvement_cnt': u'float',
    'home_improvement_price': u'float',
    'cellphone_cnt': u'float',
    'cellphone_price': u'float',
    'clothes_cnt': u'float',
    'clothes_price': u'float',
    'food_drink_cnt': u'float',
    'food_drink_price': u'float',
    'computer_cnt': u'float',
    'computer_price': u'float',
    'makeup_cnt': u'float',
    'makeup_price': u'float',
    'gift_luggage_cnt': u'float',
    'gift_luggage_price': u'float',
    'sport_cnt': u'float',
    'sport_price': u'float',
    'car_accessory_cnt': u'float',
    'car_accessory_price': u'float',
    'digital_cnt': u'float',
    'digital_price': u'float',
    'eb_auth_match': u'bool',
    'eb_binding_jingdong': u'bool',
    'eb_binding_taobao': u'bool'
}
