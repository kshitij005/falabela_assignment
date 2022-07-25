# falabela_assignment

run: pip install -r requirements.txt in your shell.

run: similar_customer_search to find similar customer of each customer.

run recommendation_systeam_analysis_and_evaluation_and_recall.py

In this file I have calculated recall using pupularity, cosine similarity and pearson similarity.
with that I have also added purchase coloumn and scaled_purchase coloumn to analyse variation in the results.

Based on Recall(Calculated on the test set which is splitted from train set):

Popularity Model on Tran_qty:  0.0004737470885896125 
Cosine Similarity on Tran_qty: 0.014189874927791538
Pearson Similarity on Tran_qty: 0.0004934406016595423

Popularity Model on Purchase Dummy: 0.00012868196178010942
Cosine Similarity on Purchase Dummy: 0.025122097347211856 
Pearson Similarity on Purchase Dummy: 0.00023594210406964906

Popularity Model on Scaled Purchase Counts: 0.00034914047814751497 
Cosine Similarity on Scaled Purchase Counts: 0.02435555878348629
Pearson Similarity on Scaled Purchase Counts: 0.00029620917952794885


Therefore By looking at the recall I choose Cosine Similarity on Purchase Dummy as the final model.

I have calculated the recall on the test holdout which is approx 0.11 

these model were created using turicreate python library.

Using same Insight's I have calculated the similarity matrix through scikit-learn in Item-based_search.py
On the basis of predicted recommendations Recall is approx 0.29 on the given test holdout.

Thsese are my analysis and the insights for given dataset.
