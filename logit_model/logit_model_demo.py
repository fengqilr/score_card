# -*- coding: utf-8 -*-

"""
此模块用于放置做评分卡模型（逻辑回归）的公用函数
建立新模型时，调用此函数即可
"""

import math

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold, metrics

import modeling.tools as mt
from modeling.model_demo.logit_model.logit_model_config import VARIATE_TYPE as var_type
from modeling.model_demo.logit_model.logit_model_config import CONTINUITY_LIST as continuity_list
from modeling.model_demo.logit_model.logit_model_config import DISCRETE_LIST as discrete_list
from modeling.model_demo.logit_model.logit_model_config import REMOVE_LIST as remove_list
from data_platform.tools import null_tools


class GetData():
    """
    读入数据
    """
    def __init__(self, path, file_name):
        self.files_path = u"%s/%s" % (path, file_name)
        self.files_name = file_name
        self.path = path

    def read_data(self):
        if u'.pkl' in self.files_name:
            data_df = pd.read_pickle(self.files_path)
            return data_df
        elif (u'.xlsx' in self.files_name) or (u'xls' in self.files_name):
            data_df = pd.read_excel(self.files_path)
            return data_df
        elif u'pkl' in self.files_path:
            data_df = pd.read_pickle(self.files_path)
            return data_df

    def save_file(self, data_df):
        if u'.pkl' in self.files_name:
            data_df = data_df.to_pickle(self.files_path)
            return data_df
        elif (u'.xlsx' in self.files_name) or (u'xls' in self.files_name):
            data_df = data_df.to_excel(self.files_path)
            return data_df
        elif u'pkl' in self.files_path:
            data_df = data_df.read_pickle(self.files_path)
            return data_df


class StatisticalMissingData():
    """
    对数据各个变量的缺失值进行统计，和处理
    """
    def __init__(self, y_variate_name):
        self.y_variate = y_variate_name

    @staticmethod
    def variate_type():
        """
        将变量名按照变量类型进行分类
        :return:
        continuity_variate_list, 连续数据的变量名
        discrete_variate_list， 离散数据的变量名
        """
        continuity_variate_list = []
        discrete_variate_list = []
        for i in var_type:
            variate_type = var_type.get(i)
            if variate_type in continuity_list:
                continuity_variate_list.append(i)
            elif variate_type in discrete_list:
                discrete_variate_list.append(i)

        return continuity_variate_list, discrete_variate_list

    @staticmethod
    def save_variate_name(save_name):
        """
        将需要保留的变量名中是连续型变量的保留下来
        :param save_name:
        :return:
        """
        result_list = []
        for each_one in save_name:
            each_type = var_type.get(each_one)
            result_list.append(each_one) if (each_one == 'target') or (each_type in continuity_list) else result_list
        return result_list

    def def_target(self, data_df):
        """
        将数据 y变量名：y_variate_name 替换成 target， 并把 juid 设为索引
        :param data_df:
        :return:
        """
        is_not_null = data_df[self.y_variate].map(lambda x: not pd.isnull(x))
        data_df = data_df[is_not_null]
        data_df = data_df.reset_index(drop=True)
        data_df['target'] = data_df.loc[:, self.y_variate]
        data_df = data_df.drop(self.y_variate, axis=1)
        return data_df

    @staticmethod
    def count_data(data_df, rank_count, x_variate):
        """
        根据某一个变量进行分组，计算出每组包含多少个样本
        :param data_df: 需要分组的数据
        :param rank_count: 需要分多少组
        :param x_variate: 分组所依据的变量名
        :return:
        """
        count_df = data_df.count()
        sum_count = count_df.to_dict().get(x_variate)
        rank_num = int(sum_count/rank_count)
        return rank_num

    def collection_missing_data(self, data_df, save_pct=None, fill_pct=None):
        """
        统计每个变量的缺失值的比率情况，并按从大到小排序, 并返回缺失率大于pct的变量名
        :param data_df:  需要统计缺失变量的数据
        :param save_pct:  有数据的比率达到 save_pct 的保留
        :param fill_pct:  有数据的比率大于 fill_pct 时可以用来填充数据的变量
        :return:
        remove_variate_list, 需要移除的变量名
        save_variate_list, 需要保留的变量名
        missing_list,  cpc字段中需要 补零 的变量名
        fill_name，需要 补数据的 变量名
        """
        total_count = len(data_df.index.values)
        data_count = data_df.count()
        data_dict = (data_count/total_count).to_dict()
        num = data_dict.get('cpc_cnt')
        sorted_data = sorted(data_dict.iteritems(), key=lambda x: x[1], reverse=True)
        pct = save_pct if save_pct is not None else 0
        fill_pct = fill_pct if fill_pct is not None else 0
        remove_variate_list = []
        save_variate_list = []
        missing_list = []
        fill_name_list = []
        continuity_variate_list, discrete_variate_list = self.variate_type()
        remove_list.append(discrete_variate_list)
        print "----"*20, u'以下是各变量缺失率', "----"*20
        for each_one in sorted_data:
            print each_one[0], each_one[1]
            if "cpc_" in each_one[0] and each_one[1] != num:
                missing_list.append(each_one[0])
            if each_one[1] < pct:
                remove_variate_list.append(each_one[0])
            elif (each_one[1] >= pct) and (each_one[0] not in remove_list):
                save_variate_list.append(each_one[0])
                if fill_pct < each_one[1] < 1.0:
                    fill_name_list.append(each_one[0])
        print "----"*20, u'以上是各变量缺失率', "----"*20
        return remove_variate_list, save_variate_list, missing_list, fill_name_list

    @staticmethod
    def dev_sample(data_df, path, dev_rate_num=0.7, max_rate_diff_num=0.01):
        """
        将样本数据分成训练与测试样本
        :param data_df:  需要分的数据
        :param path:  分割后的数据保存路径
        :param dev_rate_num:  分割比率
        :param max_rate_diff_num:  分割样本后的样本 good 与 bad 的差距比率
        :return:
        """
        dev_sample, val_sample = mt.split_sample(dataset=data_df, dev_rate=dev_rate_num,
                                                 max_rate_diff=max_rate_diff_num)
        pd.to_pickle(dev_sample, path+u'/dev_sample.pkl')
        pd.to_pickle(val_sample, path+u'/val_sample.pkl')
        return dev_sample, val_sample

    @staticmethod
    def deal_missing_data(data_df, save_variate_list):
        for each_one in save_variate_list:
            data_df[each_one] = data_df[each_one].map(lambda x: x if not pd.isnull(x) else None)
        return data_df

    def describe_statistic(self, data_df, path):
        """
        计算描述性统计量 在指定路径下保留 describe_statistic.csv 文件
        :param data_df:
        :param path:
        :return:
        """
        file_name = u"%s/describe_statistic.csv" % path
        continuity_variate_list, discrete_variate_list = self.variate_type()
        use_data_df = pd.DataFrame(data_df, columns=continuity_variate_list)
        use_data_df = use_data_df.applymap(lambda x: None if pd.isnull(x) else x)
        result_list = []
        for each_one in continuity_variate_list:
            each_one_df = use_data_df[each_one].dropna()
            each_df = each_one_df.describe()
            result_list.append(each_df)
        result_df = pd.DataFrame(result_list)
        result_df.to_csv(file_name)
        print "==="*10, u"连续变量描述统计量输出", "==="*10
        print result_df
        return result_df


class Correlation():
    def __init__(self, data_df, x_variate):
        self.data_df = data_df
        self.x_variate = x_variate

    def correlation_matrix(self, path):
        """
        计算相关系数矩阵
        :param path: 保存路径
        :return:
        """
        data_df = pd.DataFrame(self.data_df, columns=self.x_variate)
        name_list = self.x_variate
        save_path = u"%s/correlation_matrix.csv" % path
        variate_name_list = []
        data_df = data_df.dropna()
        for each_variate in self.x_variate:
            try:
                data_df[each_variate] = data_df[each_variate].map(lambda x: float(x))
            except Exception as e:
                print e, each_variate
                name_list.remove(each_variate)
                continue
            each_values = data_df[each_variate].values
            variate_name_list.append(each_values)
        result_df = pd.DataFrame(np.corrcoef(variate_name_list), columns=name_list, index=name_list)
        result_df.to_csv(save_path)

        print "==="*10, u"相关系数矩阵结果输出", "==="*10
        print result_df
        return result_df, name_list

    @staticmethod
    def clustering(data_df, variate_list, path):
        """
        根据输出的 相关系数矩阵 进行聚类
        :param data_df: 需要聚类的数据表
        :param variate_list: 需要聚类的变量名
        :param path: 保存聚类后的结果 csv 文件
        :return:
        """
        data_df = pd.DataFrame(data_df, columns=variate_list)
        data_df = data_df.dropna()
        data_df = data_df.applymap(lambda x: float(x))
        edge_model = covariance.GraphLassoCV()
        data_values = data_df.values
        edge_model.fit(data_values)
# =======================================计算相关系数，并根据相关系数聚类==============================================
        grouping, labels = cluster.affinity_propagation(edge_model.covariance_)
        result_df = pd.DataFrame()
        result_df['variate'] = variate_list
        result_df['cluster'] = list(labels)
        result_df['variate'] = result_df['variate'].map(lambda i: "%s, " % i)
        result_df = result_df.groupby(['cluster']).sum()
        result_df['variate'] = result_df['variate'].map(lambda j: j[:-2])

        print "==="*10, u"聚类分组结果输出", "==="*10
        for cluster_index, values_data in result_df.iterrows():
            print('Cluster %i: %s' % ((cluster_index + 1), values_data.to_dict().get('variate')))
# =========================================保存聚类结果==============================================
        result_df.to_csv(path)
# =======================================把聚类结果可视化==============================================
        node_position_model = manifold.LocallyLinearEmbedding(
            n_components=2, eigen_solver='dense', n_neighbors=6)

        embedding = node_position_model.fit_transform(data_values.T).T

        plt.figure(1, facecolor='w', figsize=(10, 8))
        plt.clf()
        ax = plt.axes([0., 0., 1., 1.])
        plt.axis('off')

# =======================================显示局部相关性==============================================
        partial_correlations = edge_model.precision_.copy()
        d = 1 / np.sqrt(np.diag(partial_correlations))
        partial_correlations *= d
        partial_correlations *= d[:, np.newaxis]
        non_zero = (np.abs(np.triu(np.abs(partial_correlations)+1, k=1))-1 >= 0)
# =======================================将各点嵌入坐标图============================================
        plt.scatter(embedding[0], embedding[1], s=d*10.0, c=labels,
                    cmap=plt.cm.spectral)
# =========================================设置图形边缘==============================================
        start_idx, end_idx = np.where(non_zero)
        segments = [[embedding[:, start], embedding[:, stop]]
                    for start, stop in zip(start_idx, end_idx)]
        values = np.abs(partial_correlations[non_zero])
        lc = LineCollection(segments,
                            zorder=0, cmap=plt.cm.hot_r,
                            norm=plt.Normalize(0, .7 * values.max()))
        lc.set_array(values)
        lc.set_linewidths(15*values)
        ax.add_collection(lc)
# =======================================对每个坐标点加标签，并设置图框的边际========================
        for index, (name, label, (x, y)) in enumerate(
                zip(variate_list, labels, embedding.T)):

            dx = x - embedding[0]
            dx[index] = 1
            dy = y - embedding[1]
            dy[index] = 1
            this_dx = dx[np.argmin(np.abs(dy))]
            this_dy = dy[np.argmin(np.abs(dx))]
            if this_dx > 0:
                horizontalalignment = 'left'
                x += .002
            else:
                horizontalalignment = 'right'
                x -= .002
            if this_dy > 0:
                verticalalignment = 'bottom'
                y += .002
            else:
                verticalalignment = 'top'
                y -= .002
            n_labels = labels.max()
            plt.text(x, y, name, size=10,
                     horizontalalignment=horizontalalignment,
                     verticalalignment=verticalalignment,
                     bbox=dict(facecolor='w',
                               edgecolor=plt.cm.spectral(label / float(n_labels)),
                               alpha=.6))

        plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
                 embedding[0].max() + .10 * embedding[0].ptp(),)

        plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
                 embedding[1].max() + .03 * embedding[1].ptp())

        plt.show()


class AutoDev():
    """
    此类用于对变量进行初步的划分，比较iv值筛选变量
    """
    def __init__(self, variate_name_list):
        self.variate_list = variate_name_list

    def original_iv(self, data_df):
        """
        原始 iv 值每一个值分成一组
        :param data_df:
        :return:
        """
        use_data_df = pd.DataFrame(data_df, columns=self.variate_list)
        columns_name = use_data_df.columns.values
        name_list = []
        iv_list = []
        iv_df = pd.DataFrame()
        print "==="*10, u"原始 IV 值结果输出", "==="*10
        for each_variate in columns_name:
            print each_variate
            name_woe = "%s_woe" % each_variate
            each_grp_df = mt.var_grouping(dataset=data_df, grp_var_name=each_variate, tgt_var_name='target')
            name_list.append(each_variate)
            iv_list.append(each_grp_df.iv.sum())
            mt.get_woe_var(dataset=data_df, grp_dataset=each_grp_df, grp_var_name=each_variate, woe_var_name=name_woe)

        iv_df['variate_name'] = name_list
        iv_df['iv_values'] = iv_list
        iv_sorted_df = iv_df.sort('iv_values', ascending=False).reset_index(drop=True)
        print "==="*10, u"原始 IV 值结果输出", "==="*10
        print iv_sorted_df
        return iv_sorted_df

    @staticmethod
    def variate_group(group_limit_dict, x_values):
        """
        将数据进行分箱操作（等深分箱）
        :param group_limit_dict:
        :param x_values:
        :return:
        """
        for each_group in group_limit_dict:
            if pd.isnull(x_values):
                return u'无数据'
            else:
                limit_list = group_limit_dict.get(each_group)
                if "+" in each_group:
                    if x_values > limit_list[0]:
                        return each_group
                else:
                    if limit_list[0] <= x_values < limit_list[1]:
                        return each_group

    def grouping_data(self, data_df, group_limit_dict):
        """
        将数据分箱 然后计算 iv 值
        :param data_df:
        :param group_limit_dict:
        :return:
        """
        iv_bin_df = pd.DataFrame()

        print "==="*10, u"分位数分箱 IV 值结果输出", "==="*10
        for each_group in group_limit_dict:
            try:
                each_variate = group_limit_dict.get(each_group)
                tgt_name = "%s_bin" % each_group
                data_df[tgt_name] = data_df[each_group].map(lambda x: self.variate_group(each_variate, x))
                group_bin_df = mt.var_grouping(dataset=data_df, grp_var_name=tgt_name, tgt_var_name='target')
            except Exception as e:
                print e, each_group
                continue

            each_iv = pd.DataFrame({'total_iv': {tgt_name: group_bin_df.iv.sum()}})
            iv_bin_df = iv_bin_df.append(each_iv)

        return iv_bin_df

    @staticmethod
    def variate_type():
        """
        将变量名按照变量类型进行分类
        :return:
        continuity_variate_list, 连续数据的变量名
        discrete_variate_list， 离散数据的变量名
        """
        continuity_variate_list = []
        discrete_variate_list = []
        for i in var_type:
            variate_type = var_type.get(i)
            if variate_type in continuity_list:
                continuity_variate_list.append(i)
            elif variate_type in discrete_list:
                discrete_variate_list.append(i)

        return continuity_variate_list, discrete_variate_list

    def variate_grouping(self, data_df, func=True, group_num=5, name_list=None):
        """
        定义分组的上下限
        :param data_df: 需要建模的数据
        :param group_num: 需要分组的个数
        :param func: 变量分组的方法 为 True 时， name_list 的变量名不需要分组建模；反之，则对name_list中的变量进行分组
        :param name_list: 变量名列表
        :return:
        """

        continuity_variate_list,  discrete_variate_list = self.variate_type()

        if func:
            name_list = name_list if name_list is not None else []
            result_list = []
            for each_variate in continuity_variate_list:
                result_list.append(each_variate) if each_variate not in name_list else result_list
        else:
            result_list = name_list

        variate_dict = {}
        if len(result_list) > 0:

            print "==="*10, u"分位数分组情况输出", "==="*10

            for each_one in result_list:
                data_one = data_df[each_one].dropna()
                data_values = data_one.values
                group_dict = {}
                group_down = 0
                for i in range(1, group_num+1):
                    try:
                        group_up = np.percentile(data_values, i*100.0/group_num)
                    except Exception as e:
                        print each_one, e
                        continue
                    group_range = '[%s, %s)' % (group_down, group_up) if i < group_num else '[%s, +)' % group_down
                    group_dict.update({group_range: [group_down, group_up]})
                    group_down = group_up
                variate_dict.update({each_one: group_dict})

        print variate_dict
        return variate_dict

    # def pdf_plt(self, data_df, columns_list):
    #     """
    #
    #     :param data_df:
    #     :param columns_list:
    #     :return:
    #     """
    #     for each_one in columns_list:
    #         df = data_df[each_one].dropna()
    #         df_values = df.tolist()
    #         plt.subplot(111)


class AutoGroupIv():
    """
    等深分箱，并结算iv值
    """
    def __init__(self, model_df, var_type_dict, tgt):
        self.var_type_dict = var_type_dict
        self.model_df = model_df
        self.tgt = tgt

    def group_iv(self, group_num):
        var_iv_list = []
        var_list = self.var_type_dict.keys()
        for each_var in var_list:
            edge_list = []
            var_df = pd.DataFrame(self.model_df, columns=[each_var, self.tgt])
            print each_var, self.tgt
            if self.var_type_dict[each_var] == "continuous":
                data = var_df[each_var]
                data.dropna(inplace=True)
                for i in range(group_num):
                    try:
                        edge = np.percentile(data.values, i*100.0/group_num)
                        edge_list.append(edge)
                    except Exception as e:
                        print each_var, e
                        continue
                edge_list = list(set(edge_list))
                var_group_name = each_var + '_bin'
                var_df[var_group_name] = self.get_continuous_group(var_df[each_var], edge_list)
            else:
                var_group_name = each_var
            group_bin_df = mt.var_grouping(dataset=var_df, grp_var_name=var_group_name, tgt_var_name=self.tgt)
            iv_value = group_bin_df['iv'].sum()
            var_iv_dict = {'var': each_var, 'iv': iv_value}
            var_iv_list.append(var_iv_dict)
            print each_var, iv_value
        var_iv_df = pd.DataFrame(var_iv_list)
        return var_iv_df

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


    # @staticmethod
    # def get_continuous_group(x_value, bin_list):
    #     """
    #     根据bin_list,对连续变量进行分箱， 通用的连续变量分箱函数
    #     :param x_value:
    #     :param bin_list:
    #     :return:
    #     """
    #     if null_tools.is_null(x_value) or x_value == "NoData":
    #         return u"no data"
    #     if len(bin_list) == 0:
    #         group_name = "0"
    #         return group_name
    #     else:
    #         sort_bin_list = sorted(bin_list)
    #         bin_list_len = len(sort_bin_list)
    #         for i in range(bin_list_len):
    #             if x_value < sort_bin_list[i]:
    #                 if i == 0:
    #                     group_name = "%02d_<%s" % (i, sort_bin_list[i])
    #                 else:
    #                     group_name = "%02d_%s-%s" % (i, sort_bin_list[i-1], sort_bin_list[i])
    #                 return group_name
    #         return "%02d_>%s" % (bin_list_len, sort_bin_list[bin_list_len-1])


class ModelingProcess:
    def __init__(self, logit_config, dev, val, variable_list, target):
        self.var_group_config = logit_config              # 逻辑回归配置，包括变量类型，变量分箱函数
        self.dev_sample = dev                             # 开发样本
        self.val_sample = val                             # 测试样本
        self.var_list = variable_list                     # 变量列表
        self.var_woe_list = []                            # 变量woe名列表
        self.tgt = target                                 # 目标变量名
        self.var_grp_df_dict = {}                         # 各变量分箱结果字典
        self.attr_score_dict = {}                         # 变量分箱打分字典
        self.var_dict = {}                                # 各变量信息表，包括原变量名，分箱函数，分箱结果
        self.var_iv = []
        self.std_offset = 60
        self.std_odds = 3.36
        self.pdo = 12
        self.offset = 0
        self.factor = 0
        self.log_model = object
        self.log_model_res = object
        self.dev_err_rate = 0
        self.val_err_rate = 0
        self.dev_gains = None
        self.val_gains = None

    def model_std_process(self):
        # st = time.clock()
        self.dev_variables_treatment()
        # print time.clock() - st
        self.developing_model()
        # print time.clock() - st
        self.val_variables_treatment()
        # print time.clock() - st
        self.validating_model()
        # print time.clock() - st
        self.gen_roc_curve_func()
        self.gen_pr_curve_func()

    def gen_std_offset_std_odds(self, std_offset, pdo, tgt):
        self.std_offset = std_offset
        self.pdo = pdo
        self.std_odds = 1.0*(self.dev_sample[tgt].count() + self.val_sample[tgt].count())/(self.dev_sample[tgt].sum() + self.val_sample[tgt].sum()) - 1

    def dev_variables_treatment(self):
        """
        变量备选变量太多，通过此函数可基于训练数据快速选择变量，分箱，进行调试, 返回各变量的grp_df和iv
        :return:
        """

        for each_var in self.var_list:
            var_woe_name = each_var + "_woe"
            self.var_woe_list.append(var_woe_name)
            print each_var
            var_grp_df = self.dev_one_variable_treatment(self.dev_sample,
                                                         self.var_group_config.var_config_dict[each_var],
                                                         each_var,
                                                         self.tgt)

            self.var_grp_df_dict.update({each_var: var_grp_df})
            self.var_iv.append({'var_name': each_var, 'total_iv': var_grp_df.iv.sum()})

        return self.var_grp_df_dict, self.var_iv

    def val_variables_treatment(self):
        """
        处理训练样本变量，分箱，计算woe值
        :return:
        """
        for each_var in self.var_list:
            self.val_one_variable_treatment(self.val_sample,
                                            self.var_group_config.var_config_dict[each_var],
                                            each_var,
                                            self.var_grp_df_dict)
        return None

    def developing_model(self):

        self.log_model, self.log_model_res = mt.gen_logistic_model(dataset=self.dev_sample, var_list=self.var_woe_list, target_var=self.tgt)

        print "==="*10, 'VIF 值输出，值小于2，最好小于1.5', "==="*10
        mt.gen_vif(dataset=self.dev_sample, var_list=self.var_woe_list)

        (self.offset, self.factor) = mt.gen_scoring_param(self.std_offset, self.std_odds, self.pdo)
        print self.offset, self.factor
        # 打分
        print "====="*10, '开发样本打分输出', "====="*10
        self.dev_sample['log_odds'] = mt.gen_log_odds_score(dataset=self.dev_sample,
                                                            var_list=self.var_woe_list, scr_model=self.log_model_res)

        self.dev_sample['score'] = self.dev_sample['log_odds'].map(lambda x: mt.gen_score(self.offset, self.factor, x))

        self.dev_gains = mt.gen_gains(dataset=self.dev_sample, scr_var_name='score', tgt_var_name=self.tgt, group_num=10)

        print self.dev_gains

        mt.draw_gains_chart(self.dev_gains)
        mt.draw_score_chart(self.dev_sample, 'score', target_var=self.tgt, bins=20)

    def validating_model(self):
        # 打分
        print "====="*10, '测试样本打分输出', "====="*10
        self.val_sample['log_odds'] = mt.gen_log_odds_score(
            dataset=self.val_sample, var_list=self.var_woe_list, scr_model=self.log_model_res)
        self.val_sample['score'] = self.val_sample['log_odds'].map(lambda x: mt.gen_score(self.offset, self.factor, x))

        self.val_gains = mt.gen_gains(dataset=self.val_sample, scr_var_name='score', tgt_var_name=self.tgt, group_num=10)
        print self.val_gains

        mt.draw_gains_chart(self.val_gains)
        mt.draw_score_chart(self.val_sample, 'score', bins=20, target_var=self.tgt)
        mt.draw_gains_compare_chart(self.dev_gains, self.val_gains)

    def gen_roc_curve_func(self):
        """
        生成develop， validation样本的roc曲线
        :return:
        """
        val_resp_prob = self.gen_prob_reponse(self.val_sample)
        dev_resp_prob = self.gen_prob_reponse(self.dev_sample)
        dev_fpr, dev_tpr, thresholds = metrics.roc_curve(self.dev_sample[self.tgt], dev_resp_prob)
        dev_area = metrics.auc(dev_fpr, dev_tpr)
        plt.plot(dev_fpr, dev_tpr, lw=1, label="dev_area = %0.2f" % dev_area)
        val_fpr, val_tpr, thresholds = metrics.roc_curve(self.val_sample[self.tgt], val_resp_prob)
        val_area = metrics.auc(val_fpr, val_tpr)
        plt.plot(val_fpr, val_tpr, lw=1, label="val_area = %0.2f" % val_area)
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def gen_pr_curve_func(self):
        """
        生成develop, validation样本的pr曲线
        :return:
        """
        val_resp_prob = self.gen_prob_reponse(self.val_sample)
        dev_resp_prob = self.gen_prob_reponse(self.dev_sample)
        dev_precision, dev_recall, thresh = metrics.precision_recall_curve(self.dev_sample[self.tgt], dev_resp_prob)
        dev_area = metrics.auc(dev_recall, dev_precision)
        plt.plot(dev_recall, dev_precision, lw=1, label="dev_pr_area = %0.2f" % dev_area)
        val_precision, val_recall, thresholds = metrics.precision_recall_curve(self.val_sample[self.tgt], val_resp_prob)
        # f1_score = metrics.f1_score(self.val_sample[self.tgt], self.log_model_res.predict(self.val_sample))
        val_area = metrics.auc(val_recall, val_precision)
        plt.plot(val_recall, val_precision, lw=1, label="val_pr_area = %0.2f" % val_area)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('precision_recall_curve')
        plt.legend(loc="lower right")
        plt.show()

    def gen_scorecard_table(self):
        """
        产生评分卡，并保存
        :return:
        """
        self.attr_score_dict = mt.gen_attr_score(var_grp_dict=self.var_grp_df_dict, scr_model=self.log_model_res,
                                                 factor=self.factor, offset=self.offset, num_vars=len(self.var_list))
        for each_var in self.var_list:
            tmp_dict = {}
            tmp_dict.update({"bin_func": self.var_group_config().var_config_dict[each_var]["grouping_func"]})
            var_woe_name = each_var + "_woe"
            tmp_dict.update({"grp_woe_map": self.var_grp_df_dict[each_var]})
            self.var_dict.update(tmp_dict)
        import pickle
        f = open(u'final_model.pkl', 'wb')
        pickle.dump(self.attr_score_dict, f)
        pickle.dump(self.log_model_res, f)
        pickle.dump(self.offset, f)
        pickle.dump(self.factor, f)
        f.close()

    def gen_error_rate(self):
        self.dev_sample["predict_target"] = self.dev_sample.apply(
            lambda x: 1 if x["score"] < len(self.var_list)*self.attr_score_dict["neutral_score"] else 0, axis=1)
        self.val_sample["predict_target"] = self.val_sample.apply(
            lambda x: 1 if x["score"] < len(self.var_list)*self.attr_score_dict["neutral_score"] else 0, axis=1)
        self.dev_sample["predict_result"] = (self.dev_sample["predict_target"] - self.dev_sample["target"]).map(math.fabs)
        self.dev_err_rate = 1.0*self.dev_sample["predict_result"].sum()/self.dev_sample["predict_result"].count()
        self.val_sample["predict_result"] = (self.val_sample["predict_target"] - self.val_sample["target"]).map(math.fabs)
        self.val_err_rate = 1.0*self.val_sample["predict_result"].sum()/self.val_sample["predict_result"].count()
        print "dev_err_rate: %s" % self.dev_err_rate
        print "val_err_rate: %s" % self.val_err_rate
        return self.dev_err_rate, self.val_err_rate

    def gen_prob_reponse(self, sample):
        """
        产生预测的概率
        :param sample:
        :return:
        """
        sample["intercept"] = 1
        data = pd.DataFrame(sample, columns=self.log_model_res.params.index)
        return self.log_model.predict(self.log_model_res.params, exog=data)

    def gen_var_distribution(self, x1, x2):
        """
        产生各变量woe值得分布
        :return:
        """
        x1_bin = x1 + "_bin"
        x2_bin = x2 + "_bin"
        x1_woe = x1 + "_woe"
        x2_woe = x2 + "_woe"
        total_count = self.dev_sample["target"].count()
        group = self.dev_sample.groupby([x1_bin, x2_bin])
        group_data = []
        for each_idx, group in group:
            d = {}
            x1_value, x2_value = each_idx
            d[x1_bin] = x1_value
            d[x2_bin] = x2_value
            d[x1_woe] = group[x1_woe].iloc[0]
            d[x2_woe] = group[x2_woe].iloc[0]
            d["total_count"] = group["target"].count()
            d['bad_count'] = group["target"].sum()
            d['good_count'] = d["total_count"] - d['bad_count']
            d["radius"] = (1.0*d["total_count"]/total_count)**0.5
            group_data.append(d)
        group_df = pd.DataFrame(group_data, columns=[x1_bin, x2_bin, x1_woe, x2_woe, 'total_count', 'bad_count',
                                                     'good_count', "radius"])
        x1_woe_max = group_df[x1_woe].max()*1.1
        x1_woe_min = group_df[x1_woe].min()*1.1
        x2_woe_max = group_df[x2_woe].max()*1.1
        x2_woe_min = group_df[x2_woe].min()*1.1

        fig = plt.figure()
        out_ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        out_ax.set_xlim(x1_woe_min, x1_woe_max)
        out_ax.set_ylim(x2_woe_min, x2_woe_max)
        out_ax.set_xlabel(x1_woe)
        out_ax.set_ylabel(x2_woe)
        for each_idx in group_df.index:
            each_group = group_df.ix[each_idx]
            x1_position = self.pie_position_std(each_group[x1_woe], x1_woe_min, x1_woe_max)
            x2_position = self.pie_position_std(each_group[x2_woe], x2_woe_min, x2_woe_max)
            ax = fig.add_axes([x1_position, x2_position, 0.001, 0.001])
            ax.pie([each_group["good_count"], each_group["bad_count"]], radius=each_group["radius"]*200, colors=['k', 'w'])
        plt.show()
        return group_df

    def gen_all_var_interaciton(self):
        """
        生成所有变量之间的关系图
        :return:
        """
        l = len(self.var_list)
        for i in range(l):
            for j in range(l):
                print "+"*20 + self.var_list[i] + ',' + self.var_list[j] + "+"*20
                ax = plt.subplot(l, l, l*i+j+1)
                self.gen_var_distribution_v2(i, j, ax)
        label_name = ""
        for each_var in self.var_list:
            label_name = label_name + '_' + each_var
        plt.show()

    def gen_var_distribution_v2(self, i, j, ax):
        """
        产生各变量之间相关关系分布图，画出spine图
        :return:
        """
        l = len(self.var_list)
        x1 = self.var_list[j]
        x2 = self.var_list[i]
        bad_rate_gray_matrix = np.ones((1000, 1000))*0.5                 # 初始化灰度矩阵
        x1_idx_dict = self.gen_var_idx_dict(self.dev_sample, x1)
        x2_idx_dict = self.gen_var_idx_dict(self.dev_sample, x2)

        x1_bin = x1 + "_bin"
        x2_bin = x2 + "_bin"
        group = self.dev_sample.groupby([x1_bin, x2_bin])
        # 计算得到灰度矩阵
        for each_idx, group in group:
            if len(each_idx) == 2:
                x1_bin_value, x2_bin_value = each_idx
            else:
                x1_bin_value = each_idx
                x2_bin_value = each_idx
            bad_rate = 1.0*group["target"].sum()/group["target"].count()
            print each_idx, bad_rate
            bad_rate_gray_matrix = self.deal_gray_matrix(bad_rate_gray_matrix, x1_idx_dict, x2_idx_dict, x1_bin_value,
                                                         x2_bin_value, bad_rate)

        ax.imshow(bad_rate_gray_matrix, cmap=plt.cm.gray, interpolation='bilinear')
        if j == 0:
            ax.set_ylabel(self.var_list[i])
        if i == l-1:
            ax.set_xlabel(self.var_list[j])
        for each_item in x1_idx_dict:
            y = -50
            x = x1_idx_dict[each_item]['start_idx']
            woe = x1_idx_dict[each_item]['woe']
            group_name = "%s: %.2f" % (each_item, woe)
            ax.text(x, y, s=group_name)
        for each_item in x2_idx_dict:
            x = 1010
            y = 1000-(x2_idx_dict[each_item]['start_idx'] + x2_idx_dict[each_item]['end_idx'])/2
            woe = x2_idx_dict[each_item]['woe']
            group_name = "%s: %.2f" % (each_item, woe)
            ax.text(x, y, s=group_name)

    def validate_org(self, org, org_num=10):
        org_count = self.val_sample[org].value_counts().head(org_num)
        for each_org in org_count.index:
            print "==="*10, each_org, "==="*10
            org_val_sample = self.val_sample[self.val_sample[org] == each_org]
            val_gains = mt.gen_gains(dataset=org_val_sample, scr_var_name='score', tgt_var_name=self.tgt, group_num=20)
            print val_gains
            mt.draw_gains_chart(val_gains)
            mt.draw_score_chart(org_val_sample, 'score', bins=20, target_var=self.tgt)

    @staticmethod
    def deal_gray_matrix(matrix, x1_dict, x2_dict, x1_group_name, x2_group_name, gray_value):
        x1_start_idx = x1_dict[x1_group_name]["start_idx"]
        x1_end_idx = x1_dict[x1_group_name]["end_idx"]
        x2_start_idx = 1000-x2_dict[x2_group_name]["end_idx"]
        x2_end_idx = 1000-x2_dict[x2_group_name]["start_idx"]
        for i in range(x2_start_idx, x2_end_idx):
            for j in range(x1_start_idx, x1_end_idx):
                matrix[i, j] = gray_value
        return matrix

    @staticmethod
    def gen_var_idx_dict(df, var):
        var_bin = var + "_bin"
        var_woe = var + "_woe"
        group = df.groupby(var_bin)
        data = []
        for each_idx, each_group in group:
            d = dict()
            d["group_name"] = each_idx
            d["woe"] = each_group[var_woe].iloc[0]
            d["cnt"] = each_group[var_woe].count()
            data.append(d)
        data_df = pd.DataFrame(data, columns=["group_name", "woe", "cnt"])
        sorted_data_df = data_df.sort(columns="woe", ascending=True)
        total_cnt = len(df)
        sorted_data_df["cnt_pct"] = 1.0*sorted_data_df["cnt"]/total_cnt
        sorted_data_df.index = range(len(sorted_data_df))
        last_idx = 0
        total_group_dict = {}
        for each_idx in sorted_data_df.index:
            sub_group_dict = {}
            if each_idx == 0:
                start_idx = 0
            else:
                start_idx = last_idx
            if each_idx != (len(sorted_data_df) - 1):
                end_idx = last_idx + int(sorted_data_df["cnt_pct"].ix[each_idx]*1000)
            else:
                end_idx = 1000
            last_idx = end_idx
            sub_group_dict.update({"start_idx": start_idx})
            sub_group_dict.update({"end_idx": end_idx})
            sub_group_dict.update({"woe": sorted_data_df["woe"].ix[each_idx]})
            total_group_dict.update({sorted_data_df["group_name"].ix[each_idx]: sub_group_dict})
        return total_group_dict

    @staticmethod
    def pie_position_std(x, x_min, x_max):
        """
         根据woe值计算相对位置
        :return:
        """
        return 0.8*(x-x_min)/(x_max - x_min)+0.1


    @staticmethod
    def dev_one_variable_treatment(ds_name, var_config, var_name, target_name):
        """
        这是一个类似于宏函数作用的简便函数，用于对单个变量的相同代码做了简单替换，会对输入数据集做写操作
        :param ds_name: 输入数据集
        :param var_config: 变量配置， 包括中文名字， 变量类型， 变量分箱，是否画出woe图，是否显示过程
        :param var_name: 分组变量名称
        :param target_name: 目标变量名称
        :return: 分组结果
        """
        # 打印标题行
        print "-" * 40, var_config["var_name_desp"], "-" * 40

        # 变量分组
        var_bin_name = var_name + "_bin"
        if var_config["type"] == "bool":
            ds_name[var_bin_name] = ds_name[var_name]
        elif var_config["type"] == "continuous" and var_config["grouping_func"].__name__ == "get_continuous_group":
            # ds_name[var_bin_name] = ds_name[var_name].map(lambda x:
            #                                               var_config["grouping_func"](x, var_config["bin_edge_list"]))
            ds_name[var_bin_name] = var_config["grouping_func"](ds_name[var_name], var_config["bin_edge_list"])
        else:
            ds_name[var_bin_name] = ds_name[var_name].map(lambda x: var_config["grouping_func"](x))

        tmp_grp_df = mt.var_grouping(dataset=ds_name, grp_var_name=var_bin_name, tgt_var_name=target_name)

        # 输出分组后的结果
        if var_config["verbose"]:
            print "%s total iv: %8.6f" % (var_bin_name, tmp_grp_df.iv.sum())
            print tmp_grp_df

        # woe值显示
        if var_config["show_chart"]:
            mt.draw_by_var_chart(tmp_grp_df, var_bin_name)

        # woe值替换
        var_woe_name = var_name + "_woe"
        mt.get_woe_var(dataset=ds_name, grp_dataset=tmp_grp_df, grp_var_name=var_bin_name, woe_var_name=var_woe_name)

        return tmp_grp_df

    @staticmethod
    def val_one_variable_treatment(ds_name, var_config, var_name, grp_dict):
        """
        对单个测试变量进行分箱，并计算woe值
        :param ds_name:
        :param var_config:
        :param var_name:
        :param grp_dict:
        :return:
        """
        # 变量分组
        var_bin_name = var_name + "_bin"
        if var_config["type"] == "bool":
            ds_name[var_bin_name] = ds_name[var_name]
        elif var_config["type"] == "continuous" and var_config["grouping_func"].__name__ == "get_continuous_group":
            # ds_name[var_bin_name] = ds_name[var_name].map(lambda x:
            #                                               var_config["grouping_func"](x, var_config["bin_edge_list"]))
            ds_name[var_bin_name] = var_config["grouping_func"](ds_name[var_name], var_config["bin_edge_list"])
        else:
            ds_name[var_bin_name] = ds_name[var_name].map(lambda x: var_config["grouping_func"](x))

        # woe值替换
        var_woe_name = var_name + "_woe"
        mt.get_woe_var(dataset=ds_name, grp_dataset=grp_dict[var_name], grp_var_name=var_bin_name,
                       woe_var_name=var_woe_name)


class ConfigParameter():
    """
    此类用于将配置参数化
    """
    def __init__(self, dict_data):
        self.dict_data = dict_data

    @staticmethod
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
                        group_name = "%02d_<%s" % (i, sort_bin_list[i])
                    else:
                        group_name = "%02d_%s-%s" % (i, sort_bin_list[i-1], sort_bin_list[i])
                    return group_name
            return "%02d_>%s" % (bin_list_len, sort_bin_list[bin_list_len-1])

    def grouping_limit(self):
        new_dict = dict()
        for each_one in self.dict_data:
            parameter = self.dict_data.get(each_one)
            variate_name = parameter.get('type', "continuous")
            var_name_desp = u"%s变量" % variate_name
            grouping_func = parameter.get('grouping_func', self.get_continuous_group)
            bin_edge_list = parameter.get('bin_edge_list')
            show_chart = parameter.get('show_chart', True)
            verbose = parameter.get('verbose', True)
            new_dict.update({variate_name: {"type": variate_name,
                                            "var_name_desp": var_name_desp,
                                            "grouping_func": grouping_func,
                                            "bin_edge_list": bin_edge_list,
                                            "show_chart": show_chart,
                                            "verbose": verbose}
                             })

        return new_dict

if __name__ == '__main__':
    # =======================================读取原始数据文件==============================================
    file_path = u'c:/users/wuwen/model_data'
    files_name = u'lost_connection_model2015-05-31.pkl'
    # files_name = u'/dev_sample.pkl'
    get_data = GetData(file_path, files_name)
    df0 = get_data.read_data()

#     print "====="*10, u'读取原始数据文件', "====="*10
#     # =======================================查看数据整体情况==============================================
    # 定义 target 变量
    visible_missing = StatisticalMissingData('lost_cnct_status')
    df0 = visible_missing.def_target(df0)
    # 演示：保留无缺失数据的变量
    remove_variate, save_variate, missing, fill_name = visible_missing.collection_missing_data(df0, save_pct=0)

    # 保留 save_variate 中连续变量
    name_list = visible_missing.save_variate_name(save_variate)

    # 对筛选出的变量进行样本分割
    df1 = pd.DataFrame(df0, columns=name_list)
     # 在文件路径下生成 dev_simple.pkl, val_simple.pkl文件
    dev_simple, val_simple = visible_missing.dev_sample(df1, file_path)
#     # 输出连续变量描述性统计量， 并在文件路径下生成 describe_statistic.csv 文件
#     visible_missing.describe_statistic(df1, path=file_path)
#
# # =========================================初步计算变量的iv值==========================================
# #  计算原始 IV 值
#     dev_func = AutoDev(df1.columns)
#     a = dev_func.original_iv(df1)
#     choose_variate = a.head(100)['variate_name'].tolist()
#     dict_grouping = dev_func.variate_grouping(dev_simple, func=False, group_num=5, name_list=choose_variate)
#     dev_func.grouping_data(dev_simple, dict_grouping)
#     # ===================================计算各变量的相关系数矩阵==========================================
#     csv = u"%s/clustering.csv" % file_path
#     correlation_func = Correlation(dev_simple, choose_variate)
#     # 在文件路径下生成 correlation_matrix.xlsx 文件
#     res_df, save_variate = correlation_func.correlation_matrix(file_path)
#     # 在文件路径下生成 clustering.csv 文件
#     correlation_func.clustering(dev_simple, save_variate, csv)
    # ===============================================筛选变量==============================================
#     # var_list = ["app_age", "cp_max_cumm_tf_days", "cpc_mthly_call_01_cnt"]
#     # print "app_age" in dev_simple.columns
#     # model_process = ModelingProcess(dev_simple, val_simple, var_list, "target")
#     # model_process.model_std_process()
#
# #     # =========================================将选出的变量进行细调========================================
# #     var_iv = []
# #     print "====="*10, u'年龄变量', "====="*10
# #     dev_sample['rpt_age_bin'] = dev_sample.app_age.map(grp_conf.age_grouping)
# #     age_grp_df = mt.var_grouping(dataset=dev_sample, grp_var_name='app_age', tgt_var_name='target')
# #     var_iv.append({'var_name': 'app_age', 'total_iv': age_grp_df.iv.sum()})
# #     mt.draw_by_var_chart(age_grp_df)
# #     mt.get_woe_var(dataset=dev_sample, grp_dataset=age_grp_df, grp_var_name='app_age', woe_var_name='rpt_age_woe')
# #
# #     print "====="*10, u'银行变量', "====="*10
# #     dev_sample['rpt_cpc_bank_bin'] = dev_sample.cpc_bank_in_len.map(grp_conf.cpc_bank_in_len_grouping)
# #     cpc_bank_grp_df = mt.var_grouping(dataset=dev_sample, grp_var_name='rpt_cpc_bank_bin', tgt_var_name='target')
# #     var_iv.append({'var_name': 'rpt_cpc_bank_bin', 'total_iv': cpc_bank_grp_df.iv.sum()})
# #     mt.draw_by_var_chart(cpc_bank_grp_df)
# #     mt.get_woe_var(dataset=dev_sample, grp_dataset=cpc_bank_grp_df, grp_var_name='rpt_cpc_bank_bin', woe_var_name='rpt_cpc_bank_woe')
# #
# #     print "====="*10, u'上午通话变量', "====="*10
# #     dev_sample['h11_h15'] = dev_sample.cpc_h08_cnt + dev_sample.cpc_h21_cnt + dev_sample.cpc_h20_cnt + dev_sample.cpc_h12_cnt
# #     dev_sample['rpt_cpc_am_bin'] = dev_sample.h11_h15.map(grp_conf.cpc_h11_grouping)
# #     cpc_h11_grp_df = mt.var_grouping(dataset=dev_sample, grp_var_name='rpt_cpc_am_bin', tgt_var_name='target')
# #     var_iv.append({'var_name': 'rpt_cpc_am_bin', 'total_iv': cpc_h11_grp_df.iv.sum()})
# #     mt.draw_by_var_chart(cpc_h11_grp_df)
# #     mt.get_woe_var(dataset=dev_sample, grp_dataset=cpc_h11_grp_df, grp_var_name='rpt_cpc_am_bin',
# #                    woe_var_name='rpt_cpc_am_woe')
# #
# #     print "====="*10, u'汽车与贷款变量', "====="*10
# #     dev_sample['type_len'] = dev_sample.cpc_car_len + dev_sample.cpc_loan_len
# #     dev_sample['rpt_type_len_bin'] = dev_sample.type_len.map(grp_conf.cpc_type_grouping)
# #     cpc_type_len_df = mt.var_grouping(dataset=dev_sample, grp_var_name='rpt_type_len_bin', tgt_var_name='target')
# #     var_iv.append({'var_name': 'rpt_type_len_bin', 'total_iv': cpc_type_len_df.iv.sum()})
# #     mt.draw_by_var_chart(cpc_type_len_df)
# #     mt.get_woe_var(dataset=dev_sample, grp_dataset=cpc_type_len_df, grp_var_name='rpt_type_len_bin',
# #                    woe_var_name='rpt_type_len_woe')
# #
# #     print "====="*10, u'短信变量', "====="*10
# #     dev_sample['rpt_cps_cnt_bin'] = dev_sample.cpc_bi_cp_cnt.map(grp_conf.cps_cnt_grouping)
# #     cps_cnt_df = mt.var_grouping(dataset=dev_sample, grp_var_name='rpt_cps_cnt_bin', tgt_var_name='target')
# #     var_iv.append({'var_name': 'rpt_cps_cnt_bin', 'total_iv': cps_cnt_df.iv.sum()})
# #     mt.draw_by_var_chart(cps_cnt_df)
# #     mt.get_woe_var(dataset=dev_sample, grp_dataset=cps_cnt_df, grp_var_name='rpt_cps_cnt_bin',
# #                    woe_var_name='rpt_cps_cnt_woe')
# #
# #     var_list = ['rpt_age_woe', 'rpt_type_len_woe', "rpt_cpc_am_woe", "rpt_cpc_bank_woe"]
# #
# #     log_model, log_model_res = mt.gen_logistic_model(dataset=dev_sample, var_list=var_list)
# #
# #     print "==="*10, 'VIF 值输出，值小于2，最好小于1.5', "==="*10
# #     mt.gen_vif(dataset=dev_sample, var_list=var_list)
# #
# #     std_offset = 600
# #     std_odds = 50
# #     pdo = -20
# #     (offset, factor) = mt.gen_scoring_param(std_offset, std_odds, pdo)
# #
# #     # 打分
# #     print "====="*10, '开发样本打分输出', "====="*10
# #     dev_sample['log_odds'] = mt.gen_log_odds_score(dataset=dev_sample, var_list=var_list, scr_model=log_model_res)
# #     dev_sample['score'] = dev_sample['log_odds'].map(lambda x: mt.gen_score(offset, factor, x))
# #
# #     dev_gains = mt.gen_gains(dataset=dev_sample, scr_var_name='score', tgt_var_name='target', group_num=10)
# #     print dev_gains
# #
# #     mt.draw_gains_chart(dev_gains)
# #     mt.draw_score_chart(dev_sample, 'score', bins=20)
# #
# # # ===================================================测试样本=====================================================
# #     print "====="*10, u'测试样本短信变量', "====="*10
# #     val_date = val_sample.reset_index(drop=True)
# #     val_date['rpt_age_bin'] = val_date.app_age.map(grp_conf.age_grouping)
# #     mt.get_woe_var(dataset=val_date, grp_dataset=age_grp_df, grp_var_name='rpt_age_bin', woe_var_name='rpt_age_woe')
# #
# #     print "====="*10, u'测试样本银行变量', "====="*10
# #     val_date['rpt_cpc_bank_bin'] = val_date.cpc_bank_in_len.map(grp_conf.cpc_bank_in_len_grouping)
# #     mt.get_woe_var(dataset=val_date, grp_dataset=cpc_bank_grp_df, grp_var_name='rpt_cpc_bank_bin',
# #                    woe_var_name='rpt_cpc_bank_woe')
# #
# #     print "====="*10, u'测试样本分时段通话次数变量', "====="*10
# #     val_date['rpt_h11_h10_bin'] = \
# #         val_date.cpc_h08_cnt + val_date.cpc_h12_cnt + val_date.cpc_h21_cnt + val_date.cpc_h20_cnt
# #     val_date['rpt_cpc_h11_bin'] = val_date.rpt_h11_h10_bin.map(grp_conf.cpc_h11_grouping)
# #     mt.get_woe_var(dataset=val_date, grp_dataset=cpc_h11_grp_df, grp_var_name='rpt_cpc_h11_bin',
# #                    woe_var_name='rpt_cpc_am_woe')
# #
# #     print "====="*10, u'测试样本汽车与贷款变量', "====="*10
# #     val_date['type_len'] = val_date.cpc_car_out_len + val_date.cpc_loan_len
# #     val_date['rpt_type_len_bin'] = val_date.type_len.map(grp_conf.cpc_type_grouping)
# #     mt.get_woe_var(dataset=val_date, grp_dataset=cpc_type_len_df, grp_var_name='rpt_type_len_bin',
# #                    woe_var_name='rpt_type_len_woe')
# #
# #     print "====="*10, u'测试样本短信变量', "====="*10
# #     val_date['rpt_cps_cnt_bin'] = val_date.cpc_bi_cp_cnt.map(grp_conf.cps_cnt_grouping)
# #     mt.get_woe_var(dataset=val_date, grp_dataset=cps_cnt_df, grp_var_name='rpt_cps_cnt_bin',
# #                    woe_var_name='rpt_cps_cnt_woe')
# #
# #     #打分
# #     print "====="*10, '测试样本打分输出', "====="*10
# #     val_date['log_odds'] = mt.gen_log_odds_score(dataset=val_date, var_list=var_list, scr_model=log_model_res)
# #     val_date['score'] = val_date['log_odds'].map(lambda x: mt.gen_score(offset, factor, x))
# #
# #     val_gains = mt.gen_gains(dataset=val_date, scr_var_name='score', tgt_var_name='target', group_num=10)
# #     print val_gains
# #
# #     mt.draw_gains_chart(val_gains)
# #     mt.draw_score_chart(val_date, 'score', bins=20)