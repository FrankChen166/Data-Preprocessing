from pycaret.datasets import get_data

dataset = get_data(dataset="facebook")

dataset.isnull().sum()
dataset.shape 


data_train = dataset.sample(frac=0.9, random_state=786)
data_unseen = dataset.drop(data_train.index)

data_train.reset_index(inplace=True, drop=True) #指標重製重零開始排
data_unseen.reset_index(inplace=True, drop=True) #指標重製

('Data for Modeling: ' + str(data_train.shape)) 
('Unseen Data For Predictions: ' + str(data_unseen.shape)) 





from pycaret.clustering import*
#資料前處理
exp = setup(data=data_train, 
            normalize=True, 
            transformation=True, 
            session_id=123,
            use_gpu = True,
            pca = True,
            pca_method = "linear", 
            pca_components = 2,
            silent=True)

data_train_pre = get_config(variable="X")

models()

# model_trained_kmaens = create_model(model='kmeans', num_clusters=6) 
# # plot_model(model=model_trained, plot='elbow', scale=2)

# model_trained_meanshift = create_model(model='meanshift', num_clusters=6) 

# model_trained_optics = create_model(model='optics', num_clusters=6) 


# data_train_clustered_kmeans = assign_model(model=model_trained_kmeans)

# data_train_clustered_meanshift = assign_model(model=model_trained_meanshift)

# data_train_clustered_optics = assign_model(model=model_trained_optics)

#%% kmeans

# 分二到11群，分別計算三個驗證指標
cluster = range(2, 11)



SC_col_kmeans = []
CHI_col_kmeans = []
DBI_col_kmeans = []

for  i in cluster:
    pd.set_option('display.max_columns', None)
    model_train_kmeans = create_model(model="kmeans", num_clusters=i)
    tmp = pull()
    # 堆疊分群驗證指標
    SC_col_kmeans.append(tmp.loc[0]["Silhouette"])
    CHI_col_kmeans.append(tmp.loc[0]["Calinski-Harabasz"])
    DBI_col_kmeans.append(tmp.loc[0]["Davies-Bouldin"])
    
    
import pandas as pd
import numpy as np    
opt_unm_clusters_col_kmeans = []
SC = pd.DataFrame(data = SC_col_kmeans, index=cluster, columns=["Silhouette"])

SC.Silhouette.idxmax() #取SC最大值
opt_unm_clusters_col_kmeans.append(SC["Silhouette"].idxmax())

CHI = pd.DataFrame(data = CHI_col_kmeans, index=cluster, columns=["Calinski-Harabasz"])
opt_unm_clusters_col_kmeans.append(CHI["Calinski-Harabasz"].idxmax())

DBI = pd.DataFrame(data = DBI_col_kmeans, index=cluster, columns=["Davies-Bouldin"])
opt_unm_clusters_col_kmeans.append(DBI["Davies-Bouldin"].idxmin())

# 採取投票 取眾數的方式決定最佳分群 若有同票的壯框發生以Silhouette判斷
opt_num_cluster_kmeans = max(set(opt_unm_clusters_col_kmeans), key=opt_unm_clusters_col_kmeans.count)

#以最佳劃分群樹的資訊再分群
model_train_kmeans = create_model(model="kmeans", num_clusters=opt_num_cluster_kmeans)
#指派分群標籤
data_train_clustered_kmeans = assign_model(model=model_train_kmeans)

#未見資料的推論
data_unseen_clustered_kmeans = predict_model(model = model_train_kmeans, data = data_unseen) 

import matplotlib.pyplot as plt
data_pca = get_config('X')
data_pca = pd.DataFrame(dict(x=data_pca['Component_1'], y=data_pca['Component_2'], label=data_train_clustered_kmeans.Cluster))
groups = data_pca.groupby('label')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.show()

data_train_clustered_mean_kmeans= data_train_clustered_kmeans.groupby("Cluster").mean()

#%% kmodes
cluster = range(2, 11)



SC_col_kmodes  = []
CHI_col_kmodes = []
DBI_col_kmodes  = []

for  i in cluster:
    pd.set_option('display.max_columns', None)
    model_train_kmodes  = create_model(model="kmodes", num_clusters=i)
    tmp = pull()
    # 堆疊分群驗證指標
    SC_col_kmodes.append(tmp.loc[0]["Silhouette"])
    CHI_col_kmodes.append(tmp.loc[0]["Calinski-Harabasz"])
    DBI_col_kmodes.append(tmp.loc[0]["Davies-Bouldin"])
    
    
import pandas as pd
import numpy as np    
opt_unm_clusters_col_kmodes = []
SC_kmodes = pd.DataFrame(data = SC_col_kmodes , index=cluster, columns=["Silhouette"])

SC_kmodes.Silhouette.idxmax() #取SC最大值
opt_unm_clusters_col_kmodes.append(SC_kmodes["Silhouette"].idxmax())

CHI_kmodes  = pd.DataFrame(data = CHI_col_kmodes, index=cluster, columns=["Calinski-Harabasz"])
opt_unm_clusters_col_kmodes.append(CHI_kmodes["Calinski-Harabasz"].idxmax())

DBI_kmodes  = pd.DataFrame(data = DBI_col_kmodes , index=cluster, columns=["Davies-Bouldin"])
opt_unm_clusters_col_kmodes.append(DBI_kmodes["Davies-Bouldin"].idxmin())

# 採取投票 取眾數的方式決定最佳分群 若有同票的壯框發生以Silhouette判斷
opt_num_cluster_kmodes = max(set(opt_unm_clusters_col_kmodes ), key=opt_unm_clusters_col_kmodes .count)

#以最佳劃分群樹的資訊再分群
model_train_kmodes = create_model(model="kmodes", num_clusters=opt_num_cluster_kmodes )
#指派分群標籤
data_train_clustered_kmodes = assign_model(model=model_train_kmodes )

#未見資料的推論
data_unseen_clustered_kmodes = predict_model(model = model_train_kmodes , data = data_unseen) 

import matplotlib.pyplot as plt
data_pca = get_config('X')
data_pca = pd.DataFrame(dict(x=data_pca['Component_1'], y=data_pca['Component_2'], label=data_train_clustered_kmodes.Cluster))
groups = data_pca.groupby('label')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.show()

data_train_clustered_mean_kmodes= data_train_clustered_kmodes.groupby("Cluster").mean()

#%% hclust

models()

SC_col_hclust  = []
CHI_col_hclust = []
DBI_col_hclust  = []

for  i in cluster:
    pd.set_option('display.max_columns', None)
    model_train_hclust = create_model(model="hclust", num_clusters=i)
    tmp = pull()
    # 堆疊分群驗證指標
    SC_col_hclust.append(tmp.loc[0]["Silhouette"])
    CHI_col_hclust.append(tmp.loc[0]["Calinski-Harabasz"])
    DBI_col_hclust.append(tmp.loc[0]["Davies-Bouldin"])
    
    
import pandas as pd
import numpy as np    
opt_unm_clusters_col_hclust = []
SC_hclust = pd.DataFrame(data = SC_col_hclust, index=cluster, columns=["Silhouette"])

SC_hclust.Silhouette.idxmax() #取SC最大值
opt_unm_clusters_col_hclust.append(SC_hclust["Silhouette"].idxmax())

CHI_hclust = pd.DataFrame(data = CHI_col_hclust, index=cluster, columns=["Calinski-Harabasz"])
opt_unm_clusters_col_hclust.append(CHI_hclust["Calinski-Harabasz"].idxmax())

DBI_hclust = pd.DataFrame(data = DBI_col_hclust , index=cluster, columns=["Davies-Bouldin"])
opt_unm_clusters_col_hclust.append(DBI_hclust["Davies-Bouldin"].idxmin())

# 採取投票 取眾數的方式決定最佳分群 若有同票的壯框發生以Silhouette判斷
opt_num_cluster_hclust = max(set(opt_unm_clusters_col_hclust ), key=opt_unm_clusters_col_hclust.count)

#以最佳劃分群樹的資訊再分群
model_train_hclust = create_model(model="hclust", num_clusters=opt_num_cluster_hclust)
#指派分群標籤
data_train_clustered_hclust = assign_model(model=model_train_hclust)

#未見資料的推論
# data_unseen_clustered_hclust = predict_model(model = model_train_hclust, data = data_unseen) 

import matplotlib.pyplot as plt
data_pca = get_config('X')
data_pca = pd.DataFrame(dict(x=data_pca['Component_1'], y=data_pca['Component_2'], label=data_train_clustered_hclust.Cluster))
groups = data_pca.groupby('label')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.show()

data_train_clustered_mean_hclust= data_train_clustered_hclust.groupby("Cluster").mean()

#%% ap

SC_col_ap  = []
CHI_col_ap = []
DBI_col_ap  = []

for i in cluster:
    pd.set_option('display.max_columns', None)
    model_train_ap = create_model(model="ap", num_clusters=i)
    tmp = pull()
    # 堆疊分群驗證指標
    SC_col_ap.append(tmp.loc[0]["Silhouette"])
    CHI_col_ap.append(tmp.loc[0]["Calinski-Harabasz"])
    DBI_col_ap.append(tmp.loc[0]["Davies-Bouldin"])
    
    
import pandas as pd
import numpy as np    
opt_unm_clusters_col_ap = []
SC_ap = pd.DataFrame(data = SC_col_ap, index=cluster, columns=["Silhouette"])

SC_ap.Silhouette.idxmax() #取SC最大值
opt_unm_clusters_col_ap.append(SC_ap["Silhouette"].idxmax())

CHI_ap = pd.DataFrame(data = CHI_col_ap, index=cluster, columns=["Calinski-Harabasz"])
opt_unm_clusters_col_ap.append(CHI_ap["Calinski-Harabasz"].idxmax())

DBI_ap = pd.DataFrame(data = DBI_col_ap , index=cluster, columns=["Davies-Bouldin"])
opt_unm_clusters_col_ap.append(DBI_ap["Davies-Bouldin"].idxmin())

# 採取投票 取眾數的方式決定最佳分群 若有同票的壯框發生以Silhouette判斷
opt_num_cluster_ap = max(set(opt_unm_clusters_col_ap ), key=opt_unm_clusters_col_ap.count)

#以最佳劃分群樹的資訊再分群
model_train_ap = create_model(model="ap", num_clusters=opt_num_cluster_ap)
#指派分群標籤
data_train_clustered_ap = assign_model(model=model_train_ap)

#未見資料的推論
data_unseen_clustered_ap = predict_model(model = model_train_ap, data = data_unseen) 

import matplotlib.pyplot as plt
data_pca = get_config('X')
data_pca = pd.DataFrame(dict(x=data_pca['Component_1'], y=data_pca['Component_2'], label=data_train_clustered_ap.Cluster))
groups = data_pca.groupby('label')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.show()

data_train_clustered_mean_ap= data_train_clustered_ap.groupby("Cluster").mean()


#%% sc

SC_col_sc  = []
CHI_col_sc = []
DBI_col_sc  = []

for i in cluster:
    pd.set_option('display.max_columns', None)
    model_train_sc = create_model(model="sc", num_clusters=i)
    tmp = pull()
    # 堆疊分群驗證指標
    SC_col_sc.append(tmp.loc[0]["Silhouette"])
    CHI_col_sc.append(tmp.loc[0]["Calinski-Harabasz"])
    DBI_col_sc.append(tmp.loc[0]["Davies-Bouldin"])
    
    
import pandas as pd
import numpy as np    
opt_unm_clusters_col_sc = []
SC_sc = pd.DataFrame(data = SC_col_sc, index=cluster, columns=["Silhouette"])

SC_sc.Silhouette.idxmax() #取SC最大值
opt_unm_clusters_col_sc.append(SC_sc["Silhouette"].idxmax())

CHI_sc = pd.DataFrame(data = CHI_col_sc, index=cluster, columns=["Calinski-Harabasz"])
opt_unm_clusters_col_sc.append(CHI_sc["Calinski-Harabasz"].idxmax())

DBI_sc = pd.DataFrame(data = DBI_col_sc , index=cluster, columns=["Davies-Bouldin"])
opt_unm_clusters_col_sc.append(DBI_sc["Davies-Bouldin"].idxmin())

# 採取投票 取眾數的方式決定最佳分群 若有同票的壯框發生以Silhouette判斷
opt_num_cluster_sc = max(set(opt_unm_clusters_col_sc ), key=opt_unm_clusters_col_sc.count)

#以最佳劃分群樹的資訊再分群
model_train_sc = create_model(model="sc", num_clusters=opt_num_cluster_sc)
#指派分群標籤
data_train_clustered_sc = assign_model(model=model_train_sc)

#未見資料的推論
data_unseen_clustered_sc = predict_model(model = model_train_sc, data = data_unseen) 
import matplotlib.pyplot as plt
data_pca = get_config('X')
data_pca = pd.DataFrame(dict(x=data_pca['Component_1'], y=data_pca['Component_2'], label=data_train_clustered_sc.Cluster))
groups = data_pca.groupby('label')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.show()

data_train_clustered_mean_sc= data_train_clustered_sc.groupby("Cluster").mean()


#%% dbscan

SC_col_dbscan  = []
CHI_col_dbscan = []
DBI_col_dbscan  = []

for i in cluster:
    pd.set_option('display.max_columns', None)
    model_train_dbscan = create_model(model="dbscan", num_clusters=i)
    tmp = pull()
    # 堆疊分群驗證指標
    SC_col_dbscan.append(tmp.loc[0]["Silhouette"])
    CHI_col_dbscan.append(tmp.loc[0]["Calinski-Harabasz"])
    DBI_col_dbscan.append(tmp.loc[0]["Davies-Bouldin"])
    
    
import pandas as pd
import numpy as np    
opt_unm_clusters_col_dbscan = []
SC_dbscan = pd.DataFrame(data = SC_col_dbscan, index=cluster, columns=["Silhouette"])

SC_dbscan.Silhouette.idxmax() #取SC最大值
opt_unm_clusters_col_dbscan.append(SC_dbscan["Silhouette"].idxmax())

CHI_dbscan = pd.DataFrame(data = CHI_col_dbscan, index=cluster, columns=["Calinski-Harabasz"])
opt_unm_clusters_col_dbscan.append(CHI_dbscan["Calinski-Harabasz"].idxmax())

DBI_dbscan = pd.DataFrame(data = DBI_col_dbscan , index=cluster, columns=["Davies-Bouldin"])
opt_unm_clusters_col_dbscan.append(DBI_dbscan["Davies-Bouldin"].idxmin())

# 採取投票 取眾數的方式決定最佳分群 若有同票的壯框發生以Silhouette判斷
opt_num_cluster_dbscan = max(set(opt_unm_clusters_col_dbscan ), key=opt_unm_clusters_col_dbscan.count)

#以最佳劃分群樹的資訊再分群
model_train_dbscan = create_model(model="dbscan", num_clusters=opt_num_cluster_dbscan)
#指派分群標籤
data_train_clustered_dbscan = assign_model(model=model_train_dbscan)

#未見資料的推論
data_unseen_clustered_dbscan = predict_model(model = model_train_dbscan, data = data_unseen) 
import matplotlib.pyplot as plt
data_pca = get_config('X')
data_pca = pd.DataFrame(dict(x=data_pca['Component_1'], y=data_pca['Component_2'], label=data_train_clustered_dbscan.Cluster))
groups = data_pca.groupby('label')
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
ax.legend()
plt.show()


data_train_clustered_mean_dbscan= data_train_clustered_dbscan.groupby("Cluster").mean()








