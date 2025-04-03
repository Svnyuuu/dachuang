from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def model_training(request):
    # 加载数据
    dataset = pd.read_csv('./processed/randel.csv')

    # 数据处理
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    
    # 数据标准化
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 训练 LightGBM 模型
    lgbm_classifier = LGBMClassifier(n_estimators=100, random_state=42)
    lgbm_classifier.fit(X_train, y_train)
    y_pred = lgbm_classifier.predict(X_test)

    # 计算准确率和ROC
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # 计算ROC曲线
    y_score = lgbm_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 生成ROC曲线图
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # 将图像转为base64编码并传递给模板
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    context = {
        'accuracy': accuracy,
        'report': report,
        'roc_image': image_base64,
    }

    return render(request, 'mlapp/model_results.html', context)
