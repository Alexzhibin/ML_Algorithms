# -*-coding:utf-8-*-

import tree_func as dt
from tree_func import DecisionNode
import dt_plotter

def chooseBestFeatureToSplit(dataSet):
    """
    :param dataSet: 数据集
    :return:选择最好的划分特征和划分数据集
    """
    numFeatures = len(dataSet[0]) - 1
    numEntries = len(dataSet)
    bestGini = 1.0
    bestFeature = -1
    bestSets = None
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        for value in uniqueVals:
            tb, fb = dt.divide_set(dataSet,i,value)
            prob = len(tb) / float(numEntries)
            gini = prob * dt.gini_impurity(tb) + (1-prob)*dt.gini_impurity(fb)
            if (gini < bestGini and len(fb) > 0 and len(fb) > 0 ):
                bestGini = gini
                bestFeature = (i, value)
                bestSets = (tb,fb)

    print "best gini: ",bestGini
    return (bestFeature,bestSets)

def createTree(dataSet,labels):
    """
    :param dataSet:数据集
    :param labels:标签
    :return:递归的决策树
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return DecisionNode(results=dt.unique_counts(dataSet))
    if len(dataSet[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return DecisionNode(results=dt.majority_class(classList))

    bestFeatValue, bestSets = chooseBestFeatureToSplit(dataSet)
    bestFeat = bestFeatValue[0]
    bestValue = bestFeatValue[1]

    bestFeatLabel = labels[bestFeat]

    subLabels = labels[:bestFeat]
    subLabels.extend(labels[bestFeat + 1:])
    trueBranch = createTree(bestSets[0], subLabels)  # 递归调用
    falseBranch = createTree(bestSets[1], subLabels)
    return DecisionNode(col=bestFeat, label=bestFeatLabel, value=bestValue,
                        tb=trueBranch, fb=falseBranch)


def pruneTree(inputTree):
    k = 0
    tree = inputTree
    a = float("inf")

    # while tree != None:
    #     if not isinstance(tree, list):
    #         leafCounts = countLeafs(tree)
    #         missingRate = calcMissingRate(tree)
    #         leafMissingRate = calcLeafMissingRate(tree)
    #         g = (missingRate - leafMissingRate) / (leafCounts - 1)

    #         if a == g:
    #             # prune
    #             pass
    #         else:
    #             k = k + 1


###test###
def main():
    dataSet, labels = dt.createDataSet2()

    desicionTree = createTree(dataSet, labels)

    dt_plotter.createPlot(desicionTree)

if __name__ == '__main__':
    main()