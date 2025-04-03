# import pandas as pd
# from mlapp.models import MedicalData

# def import_data_from_csv():
#     # 读取 CSV 文件
#     df = pd.read_csv('./data/randel.csv')

#     # 遍历 DataFrame，把数据存入数据库
#     for _, row in df.iterrows():
#         MedicalData.objects.create(
#             age=row['age'],
#             average_resp_rate_score=row['average_resp_rate_score'],
#             average_po2=row['average_po2'],
#             sofa_score=row['sofa_score'],
#             ards_label=row['ards_label']
#         )

#     print("✅ 数据导入成功！")


# 运行数据导入函数
# 已经在 Django shell 中运行过，注释掉以避免重复导入

# 步骤 1.python manage.py shell
# 步骤 from mlapp.datain import import_data_from_csv
# import_data_from_csv()
