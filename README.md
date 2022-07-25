# falabela_assignment

*run: pip install -r requirements.txt in your shell.*

**1. To find similar customer**

*run: similar_customer_search to find similar customer of each customer.*

(here we use cosine similarity to find similar customer on the basis of similar item purchase history as we only had purchase history in our dataset)

*check similar_customer_data.csv for the results.*

**2. To find next 3 Purchase:**

(To recommend purchase item we have content-based filtering and collaborative filtering approach,

The collaborative filtering method for recommender systems is a method that is solely based on the past interactions that have been recorded between users and items, in order to produce new recommendations.

The content-based approach uses additional information about users and/or items. This filtering method uses item features to recommend other items similar to what the user likes and also based on their previous actions or explicit feedback.

As per defination we have the data which showcase the past transaction of users, so we will use collaborative filtering to predict the next purchase)

*For that run: recommendation_systeam_analysis_and_evaluation_and_recall.py*

(In this file I have calculated recall using pupularity model, cosine similarity and pearson similarity.

with that I have also added purchase coloumn and scaled_purchase coloumn to analyse variation in the results.)



**Based on Recall(Calculated on the test set which is splitted from train set):**

  1. Popularity Model on Tran_qty:  0.0004737470885896125.

  2. Cosine Similarity on Tran_qty: 0.014189874927791538.
  
  3. Pearson Similarity on Tran_qty: 0.0004934406016595423.


  4. Popularity Model on Purchase Dummy: 0.00012868196178010942.

  5. Cosine Similarity on Purchase Dummy: 0.025122097347211856.

  6. Pearson Similarity on Purchase Dummy: 0.00023594210406964906.


  7. Popularity Model on Scaled Purchase Counts: 0.00034914047814751497. 

  8. Cosine Similarity on Scaled Purchase Counts: 0.02435555878348629.

  9. Pearson Similarity on Scaled Purchase Counts: 0.00029620917952794885.



Therefore By looking at the recall I choose Cosine Similarity on Purchase Dummy as the final model.


**I have calculated the recall on the test holdout which is approx 0.11**

*check next_3_recommendation.csv for results*

these model were created using turicreate python library.

(Note.: Recall using this library was very low so I traied another way using scikit-learn and same insights.)

Using same Insight's I have calculated the similarity matrix through scikit-learn in recomender_cosine_similarity_purchase.py

*run : recomender_cosine_similarity_purchase.py*

**On the basis of predicted recommendations Recall is approx 0.29 on the given test holdout.**

*check 3_purchase_recommendation_scikit.xlsx for the reults*

Thsese are my analysis and the insights for given dataset.
