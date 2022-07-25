import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# read train dataset using pandas
data = pd.read_csv(r"reco_assignment_training.csv")

# take 'Product_num', 'Tran_qty', 'Customer_num' col
DataPrep = data[['Product_num', 'Tran_qty', 'Customer_num']]

#to see what product items have been purchased by what Customer
DataGrouped = DataPrep.groupby(['Customer_num', 'Product_num']).sum().reset_index()

# our collaborative filtering will be based on binary data. For every dataset we will add a 1 as purchased
def create_DataBinary(DataGrouped):
    DataBinary = DataGrouped.copy()
    DataBinary['PurchasedYes'] = 1
    return DataBinary
DataBinary = create_DataBinary(DataGrouped)

# get rid of Tran_qty col
purchase_data=DataBinary.drop(['Tran_qty'], axis=1)

# Function that calculate the Item-Item cosine similarity
def GetItemItemSim(user_ids, product_ids):
    SalesItemCustomerMatrix = csr_matrix(([1]*len(user_ids), (product_ids, user_ids)))
    similarity = cosine_similarity(SalesItemCustomerMatrix)
    return similarity, SalesItemCustomerMatrix

"""Receiving the top 3 SalesItem recommendations per Customer in a dataframe,
 we will use the Item-Item Similarity Matrix from above function via creating 
 a SalesItemCustomerMatrixs (product_num per rows and Customer as columns filled binary incidence)."""
def get_recommendations_from_similarity(similarity_matrix, SalesItemCustomerMatrix, top_n=3):
    CustomerSalesItemMatrix = csr_matrix(SalesItemCustomerMatrix.T)
    CustomerSalesItemScores = CustomerSalesItemMatrix.dot(similarity_matrix) # sum of similarities to all purchased products
    RecForCust = []
    for user_id in range(CustomerSalesItemScores.shape[0]):
        scores = CustomerSalesItemScores[user_id, :]
        purchased_items = CustomerSalesItemMatrix.indices[CustomerSalesItemMatrix.indptr[user_id]:
        CustomerSalesItemMatrix.indptr[user_id+1]]
        scores[purchased_items] = -1 # do not recommend already purchased SalesItems
        top_products_ids = np.argsort(scores)[-top_n:][::-1]
        recommendations = pd.DataFrame(
        top_products_ids.reshape(1, -1),
        index=[user_id],
        columns=['Top%s' % (i+1) for i in range(top_n)])
        RecForCust.append(recommendations)
    return pd.concat(RecForCust)

# get recommendations
def get_recommendations(purchase_data):
    user_label_encoder = LabelEncoder()
    user_ids = user_label_encoder.fit_transform(purchase_data.Customer_num)
    product_label_encoder = LabelEncoder()
    product_ids = product_label_encoder.fit_transform(purchase_data.Product_num)
    # compute recommendations
    similarity_matrix, SalesItemCustomerMatrix = GetItemItemSim(user_ids, product_ids)
    recommendations = get_recommendations_from_similarity(similarity_matrix, SalesItemCustomerMatrix)
    recommendations.index = user_label_encoder.inverse_transform(recommendations.index)
    for i in range(recommendations.shape[1]):
        recommendations.iloc[:, i] = product_label_encoder.inverse_transform(recommendations.iloc[:, i])
    return recommendations

recommendations = get_recommendations(purchase_data)
dfrec = recommendations #recommendations dataframe
dfrec.to_excel("3_purchase_recommendation_scikit.xlsx")

