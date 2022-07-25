import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

import sys

data = pd.read_csv("/content/drive/MyDrive/reco_assignment_training.csv")


# Dummy for marking whether a customer bought that item or not
def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy


data_dummy = create_data_dummy(data)
print(data.head())
data_dummy.head()

# normalize purchase frequency of each item across users
def normalize_data(data):
    # first creating a user-item matrix
    df_matrix = pd.pivot_table(data, values='Tran_qty', index='Customer_num', columns='Product_num')
    df_matrix_norm = (df_matrix - df_matrix.min()) / (df_matrix.max() - df_matrix.min())
    # create a table for input to the modeling
    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']
    return pd.melt(d, id_vars=['Customer_num'], value_name='scaled_purchase_freq').dropna()


data_norm = normalize_data(data)


def split_data(data):
    '''
    Splits dataset into training and test set.

    Args:
        data (pandas.DataFrame)

    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = train_test_split(data, test_size=.2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data


# dataset with transaction quantity as a coloumn
train_data, test_data = split_data(data)
# dataset with purchase dummy (i.e. purchase yes or no) as a coloumn
train_data_dummy, test_data_dummy = split_data(data_dummy)
# dataset with scaled purchase count (i.e. normalized purchase freq) as a coloumn
train_data_norm, test_data_norm = split_data(data_norm)

# constant variables to define field names include:
user_id = 'Customer_num'
item_id = 'Product_num'
users_to_recommend = list(data[user_id].unique())
n_rec = 3  # number of items to recommend
n_display = 30  # to display the first few rows in an output dataset


# function for all models as follows:
def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data,
                                                 user_id=user_id,
                                                 item_id=item_id,
                                                 target=target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data,
                                                      user_id=user_id,
                                                      item_id=item_id,
                                                      target=target,
                                                      similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data,
                                                      user_id=user_id,
                                                      item_id=item_id,
                                                      target=target,
                                                      similarity_type='pearson')

    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model


# Popularity Model as Baseline:
# Using Tran_qty
name = 'popularity'
target = 'Tran_qty'
popularity = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# using purchase_dummy
name = 'popularity'
target = 'purchase_dummy'
pop_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Using scaled_purchase_freq
name = 'popularity'
target = 'scaled_purchase_freq'
pop_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Collaborative Filtering Model:
# Using Cosine similarity with Tran_qty
name = 'cosine'
target = 'Tran_qty'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Using Cosine similarity with purchase_dummy
name = 'cosine'
target = 'purchase_dummy'
cos_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Using Cosine similarity with scaled_purchase_freq

name = 'cosine'
target = 'scaled_purchase_freq'
cos_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Using pearson similarity with Tran_qty
name = 'pearson'
target = 'Tran_qty'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Using pearson similarity with purchase_dummy
name = 'pearson'
target = 'purchase_dummy'
pear_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# Using pearson similarity with scaled_purchase_freq
name = 'pearson'
target = 'scaled_purchase_freq'
pear_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)

# created initial callable variables for model evaluation
models_w_counts = [popularity, cos, pear]
models_w_dummy = [pop_dummy, cos_dummy, pear_dummy]
models_w_norm = [pop_norm, cos_norm, pear_norm]
names_w_counts = ['Popularity Model on Tran_qty', 'Cosine Similarity on Tran_qty', 'Pearson Similarity on Tran_qty']
names_w_dummy = ['Popularity Model on Purchase Dummy', 'Cosine Similarity on Purchase Dummy',
                 'Pearson Similarity on Purchase Dummy']
names_w_norm = ['Popularity Model on Scaled Purchase Counts', 'Cosine Similarity on Scaled Purchase Counts',
                'Pearson Similarity on Scaled Purchase Counts']

# compare all the models we have built based on RMSE and precision-recall characteristics
eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)
eval_dummy = tc.recommender.util.compare_models(test_data_dummy, models_w_dummy, model_names=names_w_dummy)
eval_norm = tc.recommender.util.compare_models(test_data_norm, models_w_norm, model_names=names_w_norm)

"""
Based on Recall:

Popularity Model on Tran_qty:  0.0004737470885896125 
Cosine Similarity on Tran_qty: 0.014189874927791538
Pearson Similarity on Tran_qty: 0.0004934406016595423

Popularity Model on Purchase Dummy: 0.00012868196178010942
Cosine Similarity on Purchase Dummy: 0.025122097347211856 
Pearson Similarity on Purchase Dummy: 0.00023594210406964906

Popularity Model on Scaled Purchase Counts: 0.00034914047814751497 
Cosine Similarity on Scaled Purchase Counts: 0.02435555878348629
Pearson Similarity on Scaled Purchase Counts: 0.00029620917952794885


Therefore By looking at the recall we choose Cosine Similarity on Purchase Dummy as the final model

"""

# Train Final Model using target as purchase coloumn and cosine similarity with whole training data
final_model = tc.item_similarity_recommender.create(tc.SFrame(data_dummy),
                                                    user_id=user_id,
                                                    item_id=item_id,
                                                    target='purchase_dummy', similarity_type='cosine')


# Function to recommend next 3 purchase for each customer
def create_output(model, users_to_recommend, n_rec, print_csv=True):
    recomendation = model.recommend(users=users_to_recommend, k=n_rec)
    df_rec = recomendation.to_dataframe()
    df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id] \
        .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['Customer_num', 'recommendedProducts']].drop_duplicates() \
        .sort_values('Customer_num').set_index('Customer_num')
    if print_csv:
        df_output.to_csv('option_recommendation.csv')
        print("An output file can be found in 'output' folder with name 'option1_recommendation.csv'")
    return df_output


# recommend next 3 purchase of all the users
df_output = create_output(final_model, users_to_recommend, 3, print_csv=True)
print(df_output.shape)
df_output.head()

# data manipilation to calculate recall on the test holdout
rec_out_df = pd.read_csv("option_recommendation.csv")
rec_out_df.head()

new_df = rec_out_df["recommendedProducts"].str.split("|", expand=True)
new_df["Customer_num"] = rec_out_df["Customer_num"]
new_df.head()
new_df1 = pd.DataFrame()
new_df1["Customer_num"] = new_df["Customer_num"]
new_df1["Product_num"] = new_df[0]
new_df1["Customer_num"].append(new_df["Customer_num"], ignore_index=True)
new_df1["Product_num"].append(new_df[1], ignore_index=True)
new_df1["Customer_num"].append(new_df["Customer_num"], ignore_index=True)
new_df1["Product_num"].append(new_df[2], ignore_index=True)
new_df1.to_csv("next_3_recommendation.csv")

# Calculate Recall
pred_df = new_df1.copy()
test_df = pd.read_csv("reco_assignment_holdout.csv")
TP = pd.merge(pred_df, test_df, how="inner", left_on=["Customer_num", "Product_num"],
              right_on=["Customer_num", "Product_num"])
recall = TP.shape[0] / pred_df.shape[0]
print(recall)
# Here Recall is 0.10940518556176919 i.e. approx 11%



















