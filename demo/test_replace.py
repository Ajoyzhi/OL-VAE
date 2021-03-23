from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

label = [0,1,2,0,3,2,0]
label = [1 if i != 0 else i for i in label ]
print(label)

# test 二分类的acc，pre,recall，f1score
gd = [0, 1, 1, 0, 0]
pre = [0, 1, 1, 1, 0]

acc = metrics.accuracy_score(gd, pre)
preci = metrics.precision_score(gd, pre)
recall = metrics.recall_score(gd, pre)
f1score = metrics.f1_score(gd, pre)
AUC = metrics.roc_auc_score(np.array(gd), np.array(pre))
FPR, TPR, threshold = metrics.roc_curve(np.array(gd), np.array(pre))

# 可以根据合适的FPR和TPR画图
plt.plot(FPR, TPR, '-g')
plt.show()

print(
    "accurancy:", acc,
    "precision:", preci,
    "recall:", recall,
    "f1score:",f1score,
    "AUC:", AUC,
    "FPR:", FPR,
    "TPR:",TPR,
    "thre:", threshold,
)
org = [1,2 ,3]
x = np.arange(len(org))
print(x)


