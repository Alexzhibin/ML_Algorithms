# -*-coding:utf-8-*-
import tree_func as dt
from tree_func import DecisionNode
import dt_plotter

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

def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    testSet = [[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]]
    return testSet

def chooseBestFeatureToSplit(dataSet):
    """
    :param dataSet:
    :return:bestFeature,bestInfoGainRatio
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = dt.get_entropy(dataSet)
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # splitInfo = 0.0
        for value in uniqueVals:
            subDataSet,_ = dt.divide_discrete_set(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * dt.get_entropy(subDataSet)
            # splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy
        #if (splitInfo == 0 ): #fix the overflow bug
        #   splitInfo += 0.000000001
        #else:
        #   infoGain /= splitInfo
        infoGainRatio = infoGain / baseEntropy
        if (infoGainRatio > bestInfoGainRatio):
            bestInfoGainRatio = infoGainRatio
            bestFeature = i
    return (bestFeature,bestInfoGainRatio)


def createTree(dataSet,labels,epsilon,decisionValues=None):
    """
    :param dataSet: 输入的数据集,最后一个字段是标签
    :param labels: 标签
    :param epsilon: 阈值，当信息增益比小于该阈值，返回类别最多的节点
    :param decisionValues: 当前列必须匹配的值
    :return:
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return DecisionNode(results = dt.unique_counts(dataSet),value=decisionValues)
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return DecisionNode(results = dt.majority_class(classList),value=decisionValues)

    bestFeat, bestInfoGainRatio = chooseBestFeatureToSplit(dataSet)

    # print bestInfoGainRatio
    # if bestInfoGainRatio < epsilon:
    #   # 如果特征的信息增益比小于阈值，返回类别最多的节点
    #   return DecisionNode(results = dt.majority_class(classList),value=decisionValues)

    bestFeatLabel = labels[bestFeat]
    del(labels[bestFeat])

    children = []
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels
        children.append(createTree(dt.divide_discrete_set(dataSet,bestFeat,value)[0],subLabels,epsilon,decisionValues=value))

    return DecisionNode(col = bestFeat,label = bestFeatLabel,value = decisionValues,children=children)



####1.Test####
def main():
    dataSet, labels = createDataSet()

    desicionTree = createTree(dataSet, labels, 0.1)

    dt.print_tree(desicionTree)

    #dt_plotter.createPlot(desicionTree)

    testSet = createTestSet()

    print('classifyResult:\n', dt.classify_all(desicionTree, testSet))


if __name__ == '__main__':
    main()
