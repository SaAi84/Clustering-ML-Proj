import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import os
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

#----------------------------------------------------------------------------------------

# Output File
if not os.path.exists('output'):
    os.makedirs('output')

#----------------------------------------------------------------------------------------

# Graphics Settings
sns.set_theme(style="darkgrid", palette='viridis')           
plt.rcParams['figure.figsize'] = (10, 5)  
plt.rcParams['font.size'] = 10

#----------------------------------------------------------------------------------------

# Data Preprocessing
customer_data = pd.read_csv('data/shopping-data.csv')

# Data Exploration
print('\n')
print(customer_data.head())


# delete customerID
data = customer_data.iloc[:, 1:3].values

# Initialization of the model
kmeans = KMeans(n_clusters=5)

# fitting the model
kmeans.fit(data)

print('='*100)
# Centers of the clusters displayed
print('Centers of the clusters displayed\n',kmeans.cluster_centers_)

print('='*100)
# The labels of each point 
print('The labels of each point:\n',kmeans.labels_)

# Drawing the clusters
plt.scatter(data[:,0],data[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black', marker='X', s=200)

plt.title("Clusters")
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig('output/01_Clusters.png', dpi=300, bbox_inches='tight')
print("\nFig 1: saved: output/01_Clusters.png")

plt.show()

#----------------------------------------------------------------------------------------

# fitting the model
cluster_labels = kmeans.fit_predict(data)


print('='*100)

# Sample size
range_n_clusters = [2, 3, 4, 5, 6, 7]
print('silhouette Average:\n')

for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(data)

    # silhouette coefficient
    silhouette_avg = silhouette_score(data, cluster_labels)
    print(
        "For n =",
        n_clusters,
        "silhouette_score is :",
        silhouette_avg,
    )

print('='*100)

#----------------------------------------------------------------------------------------

# Ward Linkage
linked = linkage(data, 'ward')

# dendrogram
threshold = 0.7*np.max(linked[:,2])
dendrogram(linked,color_threshold=threshold,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)


# Drawing the dendrogram
plt.title("Customer Dendograms")
plt.savefig('output/02_Dendrogram.png', dpi=300, bbox_inches='tight')
print("Fig 2: saved: output/02_Dendrogram.png")

plt.show()

#----------------------------------------------------------------------------------------

# fitting the model
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
cluster.fit_predict(data)


# Drawing the clusters
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')

plt.title("Culstering labels (Agglomerative)")
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig('output/03_scatter.png', dpi=300, bbox_inches='tight')
print("Fig 3: saved: output/03_scatter.png")

plt.show()

#--------------------------------------------------------------------------------------
# Evaluation another method
#--------------------------------------------------------------------------------------

# توحيد البيانات للحصول على مقاييس دقيقة
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --- 1. مقاييس K‑means (k=5) ---------------------------------------------------------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(data_scaled)

sil = silhouette_score(data_scaled, labels_kmeans)
db  = davies_bouldin_score(data_scaled, labels_kmeans)
ch  = calinski_harabasz_score(data_scaled, labels_kmeans)

print("\n========== Scales K-means (k=5) =================================================")
print(f"Silhouette          : {sil:.4f}")
print(f"Davies-Bouldin      : {db:.4f}")
print(f"Calinski-Harabasz   : {ch:.4f}")

# --- 2. منحنى الكوع (Elbow) -----------------------------------------------------------
inertias = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(data_scaled)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve for Optimal Number of Clusters')
plt.savefig('output/04_Elbow.png', dpi=300, bbox_inches='tight')
plt.show()
print("\nFig 4: saved: output/04_Elbow.png")

# --- 3. معامل Silhouette مقابل عدد التجمعات -------------------------------------------
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(data_scaled)
    sil_scores.append(silhouette_score(data_scaled, labels))

plt.figure()
plt.plot(k_range, sil_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Average')
plt.title('Silhouette Coefficient Evaluation')
plt.savefig('output/05_Silhouette_vs_k.png', dpi=300, bbox_inches='tight')
plt.show()
print("Fig 5: saved: output/05_Silhouette_vs_k.png")

# --- 4. تقييم التجميع الهرمي (Agglomerative) -------------------------------------------
agg = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
labels_agg = agg.fit_predict(data_scaled)

sil_agg = silhouette_score(data_scaled, labels_agg)
db_agg  = davies_bouldin_score(data_scaled, labels_agg)
ch_agg  = calinski_harabasz_score(data_scaled, labels_agg)

print("\n========== Agglomerative (k=5) =================================================")
print(f"Silhouette          : {sil_agg:.4f}")
print(f"Davies-Bouldin      : {db_agg:.4f}")
print(f"Calinski-Harabasz   : {ch_agg:.4f}")


# =======================================================================================
# تقسيم البيانات لعرض التنبؤ (Train/Test)
# =======================================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# استخدام 80% تدريب و 20% اختبار
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# تدريب النموذج على مجموعة التدريب فقط
kmeans_train = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_train.fit(X_train)

# التنبؤ بعناقيد مجموعة الاختبار (نماذج لعملاء جدد)
preds_test = kmeans_train.predict(X_test)

# رسم مجموعة الاختبار مع ألوان العناقيد المتوقعة
plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=preds_test, cmap='rainbow')
plt.title("توقع عناقيد العملاء الجدد (مجموعة الاختبار)")
plt.savefig('output/07_TestSet_Predictions.png', dpi=300, bbox_inches='tight')
plt.show()
print("Fig 7: saved: output/07_TestSet_Predictions.png")

# =======================================================================================
# التحقق المتقاطع (Cross-Validation) لتقييم النموذج
# =======================================================================================
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_silhouette_scores = []

for train_idx, val_idx in kf.split(data_scaled):
    # تدريب النموذج على جزء التدريب من الطية
    km_cv = KMeans(n_clusters=5, random_state=42, n_init=10)
    km_cv.fit(data_scaled[train_idx])
    
    # التنبؤ بعناقيد جزء التحقق من الطية
    labels_val = km_cv.predict(data_scaled[val_idx])
    
    # حساب معامل Silhouette على جزء التحقق فقط
    score = silhouette_score(data_scaled[val_idx], labels_val)
    cv_silhouette_scores.append(score)

# طباعة متوسط معامل Silhouette عبر جميع الطيات
print("\n=== تقييم النموذج باستخدام Cross-Validation (5 folds) ===")
print(f"متوسط معامل Silhouette: {np.mean(cv_silhouette_scores):.4f}")
print(f"الانحراف المعياري: {np.std(cv_silhouette_scores):.4f}")