
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv(r"reco_assignment_training.csv")

features_df=df.pivot_table(index='Customer_num',columns='Product_num',values='Tran_qty').fillna(0)

features_df_matrix = csr_matrix(features_df.values)

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(features_df_matrix)

cust_name=[]
similar_cust_1=[]
similar_cust_2=[]
similar_cust_3=[]
similar_cust_4=[]
similar_cust_5=[]
for query_index in range(features_df.shape[0]):
    distances, indices = model_knn.kneighbors(features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            cust_name.append(features_df.index[query_index])
        elif i == 1:
            similar_cust_1.append((features_df.index[indices.flatten()[i]],distances.flatten()[i]))
        elif i == 2:
            similar_cust_2.append((features_df.index[indices.flatten()[i]],distances.flatten()[i]))
        elif i == 3:
            similar_cust_3.append((features_df.index[indices.flatten()[i]],distances.flatten()[i]))
        elif i == 4:
            similar_cust_4.append((features_df.index[indices.flatten()[i]],distances.flatten()[i]))
        elif i == 5:
            similar_cust_4.append((features_df.index[indices.flatten()[i]],distances.flatten()[i]))

# initialize data of lists.
data = {'Customer_Name': cust_name,
        'Similar_Customer_1': similar_cust_1,
        'Similar_Customer_2': similar_cust_2,
        'Similar_Customer_3': similar_cust_3,
        'Similar_Customer_4': similar_cust_4,
        'Similar_Customer_5': similar_cust_5}

out_df = pd.DataFrame(data)
out_df.to_csv("similar_customer_data.csv")