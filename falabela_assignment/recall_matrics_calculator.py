import pandas as pd

predicted_df=pd.read_excel(r"ExportCustomerName-Itemname.xlsx")
test_df=pd.read_csv(r"reco_assignment_holdout.csv")

tp=0
tot=0

for idx,cus in enumerate(predicted_df["customer_no"].to_list()):
    if test_df.index[test_df["Product_num"].str.contains(predicted_df["top1"].iloc[idx]) & test_df["Customer_num"].str.contains(cus)].to_list():
        tp+=1
        tot+=1
    else:
        tot += 1
    if test_df.index[test_df["Product_num"].str.contains(predicted_df["top2"].iloc[idx]) & test_df["Customer_num"].str.contains(cus)].to_list():
        tp+=1
        tot+=1
    else:
        tot += 1
    if test_df.index[test_df["Product_num"].str.contains(predicted_df["top3"].iloc[idx]) & test_df["Customer_num"].str.contains(cus)].to_list():
        tp+=1
        tot+=1
    else:
        tot += 1

recall=tp/tot
print("recall :", recall)