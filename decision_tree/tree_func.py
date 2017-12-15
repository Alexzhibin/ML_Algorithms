# -*-coding:utf-8-*-
from math import log
import operator

class DecisionNode:
    def __init__(self, col = -1, label=None, value = None, results = None, tb = None,fb = None,children = None):
        self.col = col   # col是待检验的判断条件所对应的列索引值
        self.label = label
        self.value = value # value对应于为了使结果为True，当前列必须匹配的值
        self.results = results #保存的是针对当前分支的结果，它是一个字典
        self.tb = tb ## desision node,对应于结果为true时，树上相对于当前节点的子树上的节点
        self.fb = fb ## desision node,对应于结果为true时，树上相对于当前节点的子树上的节点
        self.children = children

def unique_counts(rows):
    '''
    :param rows:
    :return: results 字典
    对y的各种可能的取值出现的个数进行计数.。其他函数利用该函数来计算数据集和的混杂程度
    '''
    results = {}
    for row in rows:
        #计数结果在最后一列
        r = row[-1]
        if r not in results: results[r] = 0
        results[r]+=1
    return results # 返回一个字典



def get_entropy(rows):
    '''
    :param rows:
    :return: entropy 熵
    '''
    results = unique_counts(rows)
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log(p,2)
    return ent

def gini_impurity(rows):
    '''
    随机放置的数据项出现于错误分类中的概率
    :param rows:
    :return:
    '''
    total = len(rows)
    counts = unique_counts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        imp += p1 * p1
    imp = 1- imp
    return imp


def divide_discrete_set(rows, column, value):
    """
    输入：数据集，选择维度，选择值
    输出：划分数据集
    描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
    """
    tb = []
    fb = []
    for row in rows:
        features = row[:column]
        features.extend(row[column+1:])
        if row[column] == value:
            tb.append(features)
        else:
            fb.append(features)

    return (tb, fb)

def divide_set(rows,column,value):
    """
    :param rows: 数据集
    :param column: 字段的序号
    :param value: 字段中的选择值
    :return:
    """
    split_function = None
    if isinstance(value,int) or isinstance(value,float):
        split_function = lambda record: record[column] >= value
    else:
        split_function = lambda record: record[column] == value

    tb = []
    fb = []
    for row in rows:
        features = row[:column]
        features.extend(row[column+1:])
        if split_function(row):
            tb.append(features)
        else:
            fb.append(features)
    return (tb,fb)




def print_tree(tree, indent = '\t', left=''):
    '''
    决策树的显示
    '''
    # 是否是叶节点
    if tree.results != None:
        print str(tree.results)
    else:
        if tree.children != None:
            # 打印判断条件
            print str(tree.col) + ":" + str(tree.value) + "? "
            #打印分支
            for node in tree.children:
                print left + str(node.value) + "->",
                print_tree(node, indent, left = left + indent)
        else:
            # 打印判断条件
            print str(tree.col) + ":" + str(tree.value) + "? "
            #打印分支
            print left + "T->",
            print_tree(tree.tb, indent, left = left + indent)
            print left + "F->",
            print_tree(tree.fb, indent, left = left + indent)


def classify(observation,tree):
    """
    :param observation: 新的测试样本(1个)
    :param tree: 数模型
    :return: 最终返回单一路径下的结果(字典)
    """
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = observation[tree.col]

        if tree.children == None:
            if isinstance(v,int) or isinstance(v,float):
                if v >= tree.value:
                    branch = tree.tb
                else: branch = tree.fb
            else:
                if v == tree.value : branch = tree.tb
                else: branch = tree.fb
        else:
            for node in tree.children:
                if v == node.value:
                    branch = node
        print branch
        return classify(observation,branch)

def classify_all(tree,testDataSet):
    """
    :param tree: 数模型
    :param testDataSet: 全部测试数据
    :return: 所有决策结果
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(testVec,tree))
    return classLabelAll



def createDataSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    dataSet = [[0, 0, 0, 0, 'N'],
               [0, 0, 0, 1, 'N'],
               [1, 0, 0, 0, 'Y'],
               [2, 1, 0, 0, 'Y'],
               [2, 2, 1, 0, 'Y'],
               [2, 2, 1, 1, 'N'],
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels

def createDataSet2():
    my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]

    labels = ['source', 'country', 'clicked', 'age']
    return (my_data, labels)


def majority_class(classList):
    """
    输入：分类类别列表
    输出：子节点的分类
    描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
          采用多数判决的方法决定该子节点的分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return {sortedClassCount[0][0]: sortedClassCount[0][1]}