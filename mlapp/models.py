from django.db import models

# 存储原始数据
class MedicalData(models.Model):
    age = models.IntegerField()
    average_resp_rate_score = models.FloatField()
    average_po2 = models.FloatField()
    sofa_score = models.FloatField()
    ards_label = models.IntegerField()

# 存储计算结果
class ModelResults(models.Model):
    accuracy = models.FloatField()
    roc_auc = models.FloatField()
    report = models.TextField()  # 存储 classification_report
    created_at = models.DateTimeField(auto_now_add=True)
