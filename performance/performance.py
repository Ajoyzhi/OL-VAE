import numpy as np
from sklearn import metrics

class performance():
    def __init__(self, label_prediction: list):
        # 将压缩的list解压，返回矩阵形式
        _, self.label, self.prediction = zip(*label_prediction)

        self.accurancy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1score = 0.0

        self.AUC = 0.0
        self.FPR = 0.0
        self.TPR = 0.0

    def get_base_metrics(self):
        # 由于是二分类问题，所以只区分正常和异常数据
        temp = [1 for i in self.label if i != 0]
        self.label = temp

        self.accurancy = metrics.accuracy_score(self.label, self.prediction)
        self.precision = metrics.precision_score(self.label, self.prediction)
        self.recall = metrics.recall_score(self.label, self.prediction)
        self.f1score = metrics.f1_score(self.label, self.prediction)

    def AUC_ROC(self):
        label_array = np.array(self.label)
        prediction_array = np.array(self.prediction)
        self.AUC = metrics.roc_auc_score(label_array, prediction_array)
        # ROC曲线
        self.FPR, self.TPR, threshold = metrics.roc_curve(label_array, prediction_array, pos_label=2)





