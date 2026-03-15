import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

csv_path = 'D:/Project/Skin_Disease_Classfication/data/ISIC_2019_Training_GroundTruth.csv'

df = pd.read_csv(csv_path)

# 라벨 컬럼
label_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# 클래스별 이미지 수 합산
class_counts = df[label_columns].sum().sort_values(ascending=False)

# 시각화 스타일
plt.figure(figsize=(10, 6))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='Set2')

# 그래프 제목 및 라벨
plt.title('ISIC 2019 클래스별 이미지 수', fontsize=16)
plt.xlabel('클래스 이름')
plt.ylabel('이미지 수')
plt.tight_layout()
plt.show()
