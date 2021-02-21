import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

class performance():
    def __init__(self, label_prediction: list):
        # 将压缩的list解压，返回矩阵形式
        _, self.label, self.prediction = zip(*label_prediction)

        self.accurancy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1score = 0.0
        self.AUC = 0.0

    def get_base_metrics(self):
        # 由于是二分类问题，所以只区分正常和异常数据
        self.label = [1 if i != 0 else i for i in self.label ]
        # 以 label= 1为正例
        self.accurancy = metrics.accuracy_score(self.label, self.prediction)
        self.precision = metrics.precision_score(self.label, self.prediction)
        self.recall = metrics.recall_score(self.label, self.prediction)
        self.f1score = metrics.f1_score(self.label, self.prediction)

    def AUC_ROC(self):
        label_array = np.array(self.label)
        prediction_array = np.array(self.prediction)
        self.AUC = metrics.roc_auc_score(label_array, prediction_array)
        # ROC曲线
        FPR, TPR, threshold = metrics.roc_curve(label_array, prediction_array)

        # 画ROC曲线
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.plot(FPR, TPR, color='green',linewidth=3.0, linestyle='-')
        plt.show()





