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
from .models import MedicalData, ModelResults

def model_training(request):
    # 1️⃣ 从数据库中读取数据
    data = list(MedicalData.objects.all().values())
    if not data:
        return JsonResponse({'error': '数据库中没有数据！请先导入数据。'})

    dataset = pd.DataFrame(data)

    # 2️⃣ 进行数据预处理
    X = dataset[['age', 'average_resp_rate_score', 'average_po2', 'sofa_score']].values
    y = dataset['ards_label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # 3️⃣ 数据标准化
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # 4️⃣ 训练 LightGBM 模型
    lgbm_classifier = LGBMClassifier(n_estimators=100, random_state=42)
    lgbm_classifier.fit(X_train, y_train)
    y_pred = lgbm_classifier.predict(X_test)

    # 5️⃣ 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # 计算 ROC 曲线
    y_score = lgbm_classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 6️⃣ 将结果存入数据库
    ModelResults.objects.create(accuracy=accuracy, roc_auc=roc_auc, report=report)

    # 7️⃣ 生成 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # 8️⃣ 转为 Base64 以供前端显示
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # 9️⃣ 渲染模板
    context = {
        'accuracy': accuracy,
        'report': report,
        'roc_image': image_base64,
    }

    return render(request, 'mlapp/model_results.html', context)
