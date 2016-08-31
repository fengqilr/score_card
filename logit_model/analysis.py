# -*- coding: utf-8 -*-
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import null_tools
import modeling.model_demo.logit_model.logit_model_demo as md

def get_feature_cfg(feature):
    cfg_dict = {
        'cps_bank_cnt_pct': {
            'type': "continuous"
        },
        'app_age': {
            'type': "continuous"
        },
        'cpc_loan_cnt': {
            'type': "continuous"
        },
        'app_cp_city': {
            'type': "discrete"
        },
        'cpc_chain_cnt_mth1_rate': {
            'type': "continuous"
        },
    }
    return cfg_dict[feature]

def time_2_mth(dt):
    """
    将时间转化为月份
    :param dt_str: 2015/6/23 19:25
    :return:
    """
    return dt.strftime('%Y-%m')


def time_2_week(dt):
    """
    将datetime转换为今年第几周
    :param dt:
    :return:
    """
    week = "%s-%s" % (dt.strftime('%Y'), dt.strftime('%U'))
    return week


def analysis_features(data, features, cfg_fun, tgt):
    for each_feature in features:
        cfg = cfg_fun(each_feature)
        analysis_single_feature(data, each_feature, cfg, tgt)


def analysis_single_feature(data, feature, cfg, tgt):
    if cfg['type'] == 'continuous':
        edge = data[feature].quantile(np.arange(0.01, 1, 0.01))
        edge.drop_duplicates(inplace=True)
        feature_group = '%s_group' % feature
        data[feature_group] = data[feature].map(lambda x: get_continuous_group(x, list(edge)))

        draw_distribution(data, feature_group, tgt, feature_group, type='continuous')
    else:
        draw_distribution(data, feature, tgt, feature, type='discrete')


def get_continuous_group(x_value, bin_list):
    """
    根据bin_list,对连续变量进行分箱， 通用的连续变量分箱函数
    :param x_value:
    :param bin_list:
    :return:
    """
    if null_tools.is_null(x_value) or x_value == "NoData":
        return u"无数据"
    if len(bin_list) == 0:
        group_name = "0"
        return group_name
    else:
        sort_bin_list = sorted(bin_list)
        bin_list_len = len(sort_bin_list)
        for i in range(bin_list_len):
            if x_value < sort_bin_list[i]:
                if i == 0:
                    group_name = "%02d_<%.2f" % (i, sort_bin_list[i])
                else:
                    group_name = "%02d_%.2f-%.2f" % (i, sort_bin_list[i-1], sort_bin_list[i])
                return group_name
        return "%02d_>%.2f" % (bin_list_len, sort_bin_list[bin_list_len-1])


def draw_distribution(data, var, tgt, title, type='continuous'):
    group = data.groupby(var)
    group_size = group[tgt].count()/(len(data)*1.0)
    group_bad_rate = group[tgt].apply(lambda x: 1.0*x.sum()/len(x))
    group_data = pd.concat([group_size, group_bad_rate], axis=1)
    group_data.columns = ['size', 'bad_rate']
    if type == "discrete":
        group_data.sort('size', ascending=False, inplace=True)
        group_data = group_data.head(50)
        group_data.sort('bad_rate', inplace=True)
    print group_data
    group_data['avg_bad_rate'] = data[tgt].sum()*1.0/len(data)
    draw_by_var_chart(group_data, title)


def draw_org_distribution(data, var, tgt, n=20):
    group = data.groupby(var)
    group_size = group[tgt].count()/(len(data)*1.0)
    group_bad_rate = group[tgt].apply(lambda x: 1.0*x.sum()/len(x))
    group_data = pd.concat([group_size, group_bad_rate], axis=1)
    group_data.columns = ['size', 'bad_rate']
    group_data.sort(columns='size', ascending=False, inplace=True)
    group_data = group_data.head(n)
    draw_by_var_chart(group_data, 'org_distribution')
    return list(group_data.index)


def draw_org_time_distribution(data, orgs, tgt):
    for each_org in orgs:
        org_data = data[data.app_last_org_each_idcn == each_org]
        draw_distribution(org_data, 'app_ct_week', tgt, each_org)

# 绘制组分布和bad_rate曲线
def draw_by_var_chart(grp_df, title, y_range_low=0, y_range_up=0.5, y2_range_low=0, y2_range_up=1):
    """
    绘制组分布和woe曲线
    :param grp_df: 分组结果数据集
    :param y_range_low: y轴下限
    :param y_range_up: y轴上限
    :return:
    """
    # 创建新图
    # font = FontProperties(fname=r"c:\Windows\Fonts\simsun.ttc", size=14)
    fig, ax = plt.subplots(1, 1)

    # 设定左边y轴
    ax.yaxis.grid()
    ax.set_ylabel('group size')
    ax.set_ylim((y_range_low, y_range_up))

    # 设定右边y轴
    ax2 = ax.twinx()
    ax2.set_ylabel('bad_rate')
    ax2.set_ylim((y2_range_low, y2_range_up))

    # 画出基准线
    plt.axhline(0, color='k')
    # 画出组分布
    # ax.bar(grp_df.total_dist, style='b', alpha=0.9, label="group dist")
    grp_df['size'].plot(kind='bar', style='b', alpha=0.9, ax=ax, label="group size")
    handles, labels = ax.get_legend_handles_labels()
    # 画出woe曲线
    ax2.plot(grp_df['bad_rate'], linestyle='-', marker='o', color='g', label="bad_rate")
    ax2.plot(grp_df['avg_bad_rate'], linestyle='-',  color='k', label="avg_bad_rate")
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2)
    ax2.legend(handles2, labels2, loc=9)
    # 渲染
    # plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    file_path = ur'd:/数据资料/建模/审批模型/raw_data'
    files_name = u'sample_2015_07_24_yy.pkl'
    # org_name_file = u'latest_org_name_for_each_idcn.pkl'
    # org_num_file = u'num_org_for_each_idcn.pkl'
    # model_data = pd.read_pickle(file_path+files_name)
    get_data = md.GetData(file_path, files_name)
    model_data = get_data.read_data()
    not_test = model_data.app_last_org_each_idcn.map(lambda x: x not in ['juxinli', 'demo1', 'jdTest', 'juxinlidemo', 'tangchaolizi'])
    model_data = model_data[not_test]
    print len(model_data)
    # get_data = md.GetData(file_path, org_name_file)
    # org_name_data = get_data.read_data()
    # org_data = pd.DataFrame(org_name_data, columns=['org_name'])
    # get_data = md.GetData(file_path, org_num_file)
    # org_num_data = get_data.read_data()
    # org_data['org_num'] = org_num_data
    # org_num_data = pd.DataFrame(org_num_data, columns=['org_num'])
    # model_data = pd.merge(model_data, org_data, left_on='app_idcn', right_index=True, how='left')
    # model_data = pd.merge(model_data, org_num_data, left_on='app_idcn', right_index=True, how='left')
    visible_missing = md.StatisticalMissingData('in_bl_std')
    continuous_var = visible_missing.save_variate_name(model_data.columns)
    model_data = visible_missing.def_target(model_data)
    var_list = ['cps_bank_cnt_pct', 'app_age', 'cpc_loan_cnt', 'app_cp_city', 'cpc_chain_cnt_mth1_rate']
    tgt = 'target'
    analysis_features(model_data, var_list, cfg_fun=get_feature_cfg, tgt=tgt)