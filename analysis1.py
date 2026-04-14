import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Загрузка данных
df = pd.read_csv('data/cvss_all_11042026.csv.gz')  # замените на ваш путь

# 1. Анализ целевой переменной (например, availability или составной риск)
# Если нет отдельной целевой метки, можно создать score
df['severity_score'] = df[['confidentiality', 'integrity', 'availability']].sum(axis=1)

# 2. Распределение классов в целевой переменной
target_col = 'severity_score'  # или 'availability', или другой
print("=== Распределение целевого класса ===")
print(df[target_col].value_counts(normalize=True))
print(df[target_col].value_counts())

# 3. Перекосы в категориальных признаках
cat_cols = ['attack_vector', 'attack_complexity', 'privileges_required', 
            'user_interaction', 'scope']

print("\n=== Перекосы в признаках ===")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True))

# 4. Корреляции признаков с целевой переменной
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_encoded = df.copy()
for col in cat_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

print("\n=== Корреляция с целевой переменной ===")
print(df_encoded[cat_cols + [target_col]].corr()[target_col].sort_values(ascending=False))

# 5. Зависимости между признаками (Chi-square тест)
def chi2_square_test(df, col1, col2):
    """
    Выполняет тест хи-квадрат для двух категориальных переменных
    Возвращает: (chi2_statistic, p_value, dof, expected_freq)
    """
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, dof, expected

print("\n=== Зависимости между признаками (p-value) ===")
for i in range(len(cat_cols)):
    for j in range(i+1, len(cat_cols)):
        chi2, p, dof, expected = chi2_square_test(df, cat_cols[i], cat_cols[j])
        if p < 0.05:
            print(f"{cat_cols[i]} - {cat_cols[j]}: chi2={chi2:.2f}, p={p:.4f} (зависимы)")
        else:
            print(f"{cat_cols[i]} - {cat_cols[j]}: p={p:.4f} (независимы)")

# 6. Визуализация перекосов
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, col in enumerate(cat_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[idx])
    axes[idx].set_title(f'{col} Distribution')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')

plt.tight_layout()
plt.show()

# 7. Cross-tabulation для обнаружения комбинаторных перекосов
print("\n=== Пример редких комбинаций ===")
rare_pairs = df.groupby(cat_cols).size().reset_index(name='count')
rare_pairs = rare_pairs[rare_pairs['count'] < 5]  # редкие комбинации
print(f"Найдено {len(rare_pairs)} редких комбинаций (<5 примеров)")
if len(rare_pairs) > 0:
    print(rare_pairs.head(10))

# 8. Дополнительно: Cramér's V для силы зависимости
def cramers_v(df, col1, col2):
    """
    Вычисляет Cramér's V - меру силы связи между категориальными переменными
    Значения: 0 = нет связи, 1 = полная связь
    """
    confusion_matrix = pd.crosstab(df[col1], df[col2])
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

print("\n=== Сила зависимостей (Cramér's V) ===")
for i in range(len(cat_cols)):
    for j in range(i+1, len(cat_cols)):
        v = cramers_v(df, cat_cols[i], cat_cols[j])
        print(f"{cat_cols[i]} - {cat_cols[j]}: {v:.3f}")
