import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys


class Data_pre():
    def __init__(self):
        self.attributes = []  # 存储属性名
        self.column = []  # 按列存储原始数据
        self.length = 0  # 数据条数
        self.loss = []  # 按列存储将缺失部分剔除后的数据
        self.max_frequent = []  # 按列存储用最高频率值来填补缺失值的数据
        self.scattered = []  # 按列存储标称属性转化后的数据
        self.related = []  # 按列存储通过属性的相关关系来填补缺失值的数据
        self.similarity = []  # 按列通过数据对象之间的相似性来填补缺失值的数据
        self.result = []  # 按行存储原始数据

    # 读取csv文件并初始化
    def read_csv(self, path):
        with open(path, 'r', encoding='mac_roman') as f:
            reader = csv.reader(f)
            self.result = list(reader)
            self.length = len(self.result) - 2
            self.attributes = self.result[0]
            for i in range(len(self.attributes)):
                ss = [row[i] for row in self.result]
                self.column.append(ss[1:-2])

    # 输出测试
    def print(self):
        print(self.length)
        print(self.attributes)
        for src in self.column:
            print(src[0:50])

    def isNum2(self, value):
        try:
            x = float(value)  # 此处更改想判断的类型
        except TypeError:
            return False
        except ValueError:
            return False
        except Exception as e:
            return False
        else:
            return True

    # 求原始数据数值属性的五数概括并画盒图，缺失值用0.0填充
    def five_points(self, column_num):
        nums = []
        count = 0
        for ss in self.column[column_num]:
            if ss == "":
                count += 1
                nums.append(0.0)
            else:
                nums.append(float(ss))
        Minimum = min(nums)
        Maximum = max(nums)
        Q1 = np.percentile(nums, 25)
        Median = np.median(nums)
        Q3 = np.percentile(nums, 75)

        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR  # 下限值
        upper_limit = Q3 + 1.5 * IQR  # 上限值
        print(self.attributes[column_num] + "的五数概括为：")
        print("Min:", str(Minimum), "  Q1:", str(Q1), "  Median:", str(Median), "  Q3:", str(Q3), "  Max:",
              str(Maximum))
        print("IQR= " + str(IQR))
        print(self.attributes[column_num] + "数值属性缺失数据的值得个数为：" + str(count))

        # 画盒图
        df = pd.DataFrame(nums)
        title = self.attributes[column_num] + "_box"
        df.plot.box(title=title)
        plt.grid(linestyle="--", alpha=0.3)
        plt.show()

    # 求原始数据标称属性的值和频数的对应关系，并用柱状图显示同时统计缺失值个数，column_count对应属性的列数
    def Nominal_Attribute(self, column_count):
        values = []
        map = {}  # 存储值和频数的对应关系
        count = 0  # 统计缺失值个数
        for ss in self.column[column_count]:
            if ss == "":
                count += 1
            else:
                if ss not in values:
                    values.append(ss)
                    map[ss] = 1
                else:
                    map[ss] += 1
        # print(map.keys())
        # print(map.values())
        for key in list(map.keys()):
            print(key + ":" + str(map[key]))

        # 画柱状图
        fig = plt.figure(figsize=(30, 11))
        plt.bar(range(len(map.values())), map.values(), tick_label=list(map.keys()))
        for a, b in zip(range(len(map.values())), map.values()):
            plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
        plt.xlabel(self.attributes[column_count])
        plt.ylabel('frequent')
        plt.title('Nominal_Attribute')
        plt.xticks(rotation=90)
        plt.show()
        print(self.attributes[column_count] + "标称缺失数据的值得个数为：" + str(count))

    # 画直方图
    # def histogram(self, column_count):
    #     data = []
    #     count = 0
    #     for ss in self.column[column_count]:
    #         if ss == "":
    #             count += 1
    #         else:
    #             data.append(ss)
    #     plt.hist(data, bins=40, normed=0, facecolor="green", edgecolor="black", alpha=0.7)
    #     # 显示横轴标签
    #     plt.xlabel(self.attributes[column_count])
    #     # 显示纵轴标签
    #     plt.ylabel("frequent")
    #     # 显示图标题
    #     title = "histogram: " + self.attributes[column_count]
    #     plt.title(title)
    #     plt.show()

    # 处理缺失数据，将缺失部分剔除，并存入self.loss
    def handle_loss(self, mark):
        count = 0
        for i in mark:
            # 0代表不用处理
            if i == 0:
                self.loss.append(self.column[count])
            else:
                tmp = []
                for ss in self.column[count]:
                    if ss != "":
                        tmp.append(ss)
                self.loss.append(tmp)
            count += 1

    # 处理缺失数据，用最高频率值来填补缺失值，并存入self.max_frequent
    def handle_max_frequent(self, mark):
        count = 0
        for i in mark:
            # 0代表不用处理
            if i == 0:
                self.max_frequent.append(self.column[count])
            else:
                values = []
                map = {}
                for ss in self.column[count]:
                    if ss == "":
                        continue
                    else:
                        if ss not in values:
                            values.append(ss)
                            map[ss] = 1
                        else:
                            map[ss] += 1
                # 得到最高频率值
                s1 = list(map.keys())
                s2 = list(map.values())
                index = s2.index(max(s2))
                max_num = s1[index]
                print(max_num)

                tmp = []
                for ss in self.column[count]:
                    if ss != "":
                        tmp.append(ss)
                    else:
                        tmp.append(max_num)
                self.max_frequent.append(tmp)
            count += 1

    # 处理缺失数据，标称属性转化，便于计算属性相关性
    def handle_scattered(self, mark1, mark2):
        count = 0
        for i in mark2:
            # 0代表不用处理
            if i == 0:
                if mark1[i] == 0:
                    self.scattered.append(self.column[count])
                else:
                    tmp = []
                    for x in self.column[i]:
                        if x == "":
                            tmp.append(x)
                        else:
                            tmp.append(int(x))
                    self.scattered.append(tmp)
                print("<<<<<<")
            else:
                tmp = []
                values = []
                map = {}
                num = 2
                for ss in self.column[count]:
                    if ss not in values:
                        values.append(ss)
                        map[ss] = num
                        num += 2
                for ss in self.column[count]:
                    tmp.append(map[ss])
                self.scattered.append(tmp)
                print(">>>>>")
            count += 1

    # 处理缺失数据，通过属性的相关关系来填补缺失值，并存入self.related
    def handle_related(self, mark1, mark2):
        count = 0
        for i in mark1:
            # 0代表不用处理
            if i == 0:
                self.related.append(self.scattered[count])
            else:
                tmp = []
                src = self.scattered
                indexs = [i for i, x in enumerate(src[count][0:5000]) if x == ""]
                for x in range(len(src)):
                    src[x] = [src[x][i] for i in range(0, 5000, 1) if i not in indexs]
                max1 = -10
                k = 0
                for ss in range(len(mark1)):
                    if mark1[ss] == 0:
                        continue
                    else:
                        # src[ss] = [int(src[ss][i]) for i in range(0, len(src[ss]), 1)]
                        # 处理数值属性的情况
                        src[ss] = [float(src[ss][i]) for i in range(0, len(src[ss]), 1)]

                        # 处理标称属性的情况
                        src[ss] = [src[ss][i] for i in range(0, len(src[ss]), 1)]

                # 得到最相关的属性
                for ss in range(len(mark2)):
                    if mark2[ss] == 0 or ss == count:
                        continue
                    else:
                        s1 = pd.Series(src[ss][0:5000])
                        s2 = pd.Series(src[count][0:5000])
                        # print(s1)
                        # print(s2)
                        value = s1.corr(s2)
                        # print(value)
                        if value > max1:
                            max1 = value
                            k = ss

                # 缺失值用和它最相关属性相等的对应属性的平均值填充（数值属性）
                # 缺失值用和它最相关属性相等的对应属性出现次数最多的值填充（标称属性）
                for ss in range(len(self.column[count])):
                    # 数值属性情况
                    # if self.isNum2(self.column[count][ss]) and ss != "":
                    # 标称属性情况
                    if self.column[count][ss] != "":
                        # print(count)
                        # print(self.column[count][ss])
                        # 数值属性情况
                        # tmp.append(float(self.column[count][ss]))
                        # 标称属性情况
                        tmp.append(self.column[count][ss])
                    else:
                        # 标称属性
                        value = self.column[k][ss]
                        maps = {}
                        values = []
                        # for v in range(len(self.column[k])):
                        for v in range(10000):
                            if self.column[k][v] == value:
                                if self.column[count][v] not in values:
                                    values.append(self.column[count][v])
                                    maps[self.column[count][v]] = 1
                                else:
                                    maps[self.column[count][v]] += 1
                        s1 = list(maps.keys())
                        s2 = list(maps.values())
                        if len(s2) != 0:
                            sss = max(s2)
                            kk = s2.index(sss)
                            print(s1[kk])
                            tmp.append(s1[kk])
                        # 数值属性情况
                        # value = self.column[k][ss]
                        # nums = 0
                        # sum = 0
                        # # for v in range(len(self.column[k])):
                        # for v in range(10000):
                        #     if self.column[k][v] == value:
                        #         if self.isNum2(self.column[count][v]):
                        #             nums += 1
                        #             sum += float(self.column[count][v])
                        # if nums == 0:
                        #     ans = 22.0  # 由于计算时间过长进行简化，若出现nums=0的情况下，用中位数替代
                        # else:
                        #     ans = sum * 1.0 / nums
                        # # print(ans)
                        # tmp.append(ans)
                self.related.append(tmp)
            count += 1

    # 处理缺失数据，通过数据对象之间的相似性来填补缺失值，并存入self.similarity
    def handle_similarity(self, mark):
        count = 0
        for i in mark:
            # 0代表不用处理
            if i == 0:
                self.similarity.append(self.column[count])
            else:
                tmp = []
                max = 0
                index = 0
                test = 0
                # 找到最相似的对象的索引
                for ss in range(len(self.column[count])):
                    if self.column[count][ss] != "":
                        tmp.append(self.column[count][ss])
                    else:
                        # for m in range(len(self.result)):
                        for m in range(5000):
                            if ss == m:
                                continue
                            else:
                                num = 0
                                for k in range(len(self.attributes)):
                                    if self.result[ss][k] == self.result[m][k]:
                                        num += 1
                                    if num > max:
                                        max = num
                                        index = m

                        # 处理数值属性
                        # if self.isNum2(self.column[count][index]):
                        #     tmp.append(float(self.column[count][index]))

                        # 处理标称属性
                        tmp.append(self.column[count][index])
                        test += 1
                        # print(test)
                self.similarity.append(tmp)
            count += 1

    def autolabel(self, rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % float(height))

    # 标称属性处理缺失值前后柱状图对比
    def nonimal_compared(self, column_count):
        # 得到处理前后指定列的值
        values1 = []
        map1 = {}
        count = 0
        for ss in self.column[column_count]:
            if ss == "":
                count += 1
            else:
                if ss not in values1:
                    values1.append(ss)
                    map1[ss] = 1
                else:
                    map1[ss] += 1

        values2 = []
        map2 = {}
        count = 0
        # for ss in self.loss[column_count]:
        # for ss in self.max_frequent[column_count]:
        # for ss in self.related[column_count]:
        for ss in self.similarity[column_count]:
            if ss == "":
                count += 1
            else:
                if ss not in values2:
                    values2.append(ss)
                    map2[ss] = 1
                else:
                    map2[ss] += 1
        p1 = map1.keys()
        p2 = map1.values()
        q1 = map2.keys()
        q2 = map2.values()

        # 柱状图对比
        plt.figure(figsize=(100, 11))
        a = plt.bar(np.arange(len(p1)) * 2, p2, tick_label=list(p1), label='before')
        b = plt.bar(np.arange(len(q1)) * 2 + 0.8, q2, label='after', color='red')
        plt.legend()
        # for a, b in zip(range(len(p1)), p2):
        #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
        self.autolabel(a)
        self.autolabel(b)
        plt.xlabel(self.attributes[column_count])
        plt.ylabel('frequent')
        plt.title('Nominal_Attribute')
        plt.xticks(rotation=90)
        plt.show()

    # 数值属性处理缺失值前后盒图对比，并前后输出五数概括
    def box_compared(self, column_count):
        # 得到处理前后指定列的值并输出五数概括
        nums1 = []
        count = 0
        for ss in self.column[column_count]:
            if ss == "":
                count += 1
                nums1.append(0.0)
            else:
                nums1.append(float(ss))
        # tmp = self.loss[column_count]
        # tmp = self.max_frequent[column_count]
        tmp = self.related[column_count]
        # print(tmp)
        # tmp = self.similarity[column_count]
        nums2 = []
        for ss in tmp:
            nums2.append(float(ss))
        # print(nums2)
        nums = [nums1, nums2]
        Minimum = min(nums1)
        Maximum = max(nums1)
        Q1 = np.percentile(nums1, 25)
        Median = np.median(nums1)
        Q3 = np.percentile(nums1, 75)

        IQR = Q3 - Q1
        print(self.attributes[column_count] + "属性处理缺失值前的五数概括为：")
        print("Min:", str(Minimum), "  Q1:", str(Q1), "  Median:", str(Median), "  Q3:", str(Q3), "  Max:",
              str(Maximum))
        print("IQR= " + str(IQR))

        Minimum = min(nums2)
        Maximum = max(nums2)
        Q1 = np.percentile(nums2, 25)
        Median = np.median(nums2)
        Q3 = np.percentile(nums2, 75)

        IQR = Q3 - Q1
        print(self.attributes[column_count] + "属性处理缺失值后的五数概括为：")
        print("Min:", str(Minimum), "  Q1:", str(Q1), "  Median:", str(Median), "  Q3:", str(Q3), "  Max:",
              str(Maximum))
        print("IQR= " + str(IQR))

        # 前后盒图对比
        title = self.attributes[column_count] + "_box_compared"
        plt.title(title)
        plt.boxplot(nums)
        plt.grid(linestyle="--", alpha=0.3)
        plt.show()


data = Data_pre()
# path = "D:/data/oakland_crime/records-for-2016.csv"
# data.read_csv(path)
# # data.Nominal_Attribute(4)
# mark1 = [0,0,0,0,1,0,0,0,0,0]
# # data.handle_loss(mark1)
# data.handle_max_frequent(mark1)
# data.nonimal_compared(4)

# path = "D:/data/wine_reviews/winemag-data_first150k.csv"
# path = "D:/data/wine_reviews/winemag-data-130k-v2.csv"
# path = "D:/data/oakland_crime/records-for-2011.csv"
# path = "D:/data/oakland_crime/records-for-2012.csv"
# path = "D:/data/oakland_crime/records-for-2013.csv"
# path = "D:/data/oakland_crime/records-for-2014.csv"
# path = "D:/data/oakland_crime/records-for-2015.csv"
path = "D:/data/oakland_crime/records-for-2016.csv"
data.read_csv(path)
# data.five_points(5)
# data.Nominal_Attribute(7)
# mark1 = [0,0,0,0,0,1,0,0,0,0,0]  # 要处理的列位置索引
mark1 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # 要处理的列位置索引
# mark2 = [0,1,0,1,0,0,1,1,1,1,1]  # 标称属性位置索引
mark2 = [0,0,0,1,1,1,1,1,0,0]  # 标称属性位置索引
data.handle_loss(mark1)
data.handle_max_frequent(mark1)
data.handle_scattered(mark1, mark2)
data.handle_related(mark1, mark2)
data.handle_similarity(mark1)
# data.box_compared(5)  # 数值属性指定列属性处理前后盒图对比

data.nonimal_compared(4)  # 标称属性指定列属性处理前后柱状图对比
