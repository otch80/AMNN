import numpy as np
import os
import pandas as pd

root_path='./dataset/preprocessed_data'
target_path=os.path.join(root_path,"target_captions.txt")
pred_path=os.path.join(root_path,"predicted_hashtags.txt")
target=open(target_path,"r") 
test_y=[]
pred_y=[]

with open(pred_path,"r") as file:
    predicted_labels=file.readlines()
    for label in predicted_labels:
        target_label=target.readline()
        target_arr=target_label.strip().split(' ')
        label_arr_ori=label.strip().split(' ')
        label_arr = list(set(label_arr_ori)) 
        label_arr.sort(key=label_arr_ori.index)
        test_y.append(target_arr)
        pred_y.append(label_arr)
target.close()

def precision_score(test_y, pred_y, k=1):
    p_score = []
  
    for i in range(len(test_y)):
        count = 0
        for j in range(0,k):
            if(pred_y[i][j] in test_y[i]):
                count+=1
        p_score.append(float(count) / float(k))
    return np.mean(p_score)

def recall_score(test_y, pred_y, k=1):
    r_score = []
    for i in range(len(test_y)):
        count = 0
        end=min(k,len(pred_y[i]))
        for j in range(0,end):
            if(pred_y[i][j] in test_y[i]):
                count+=1
        r_score.append(float(count) / float(len(test_y[i])))

    return np.mean(r_score)

def hits_score(test_y, pred_y, k=1):
    h_score = []
    for i in range(len(test_y)):
        count = 0
        end=min(k,len(pred_y[i]))
        for j in range(0,end):
            if(pred_y[i][j] in test_y[i]):
                count+=1
        h_score.append(1 if count > 0 else 0)

    return np.mean(h_score)

precisions=[]
recalls=[]
hits_rates=[]
num=5
names=[]

for i in range(num*3):
    if(i<num):
        names.append("precision@%d" %(i%num+1))
    elif(i<num*2):
        names.append("recall@%d" %(i%num+1))
    else:
        names.append("accuracy@%d" %(i%num+1))


    
for i in range(num):
    topk=i+1
    precision=precision_score(test_y,pred_y,topk)
    recall=recall_score(test_y,pred_y,topk)
    hits_rate=hits_score(test_y,pred_y,topk)
    print("precision@%d:%f" %(topk,precision))
    print("recall@%d:%f" %(topk,recall))
    print("accuracy@%d:%f" %(topk,hits_rate))
    precisions.append([precision])
    recalls.append([recall])
    hits_rates.append([hits_rate])
list_data=[]
list_data.extend(precisions)
list_data.extend(recalls)
list_data.extend(hits_rates)
print(list_data)
dict_data={}
for i in range(num*3):
    dict_data[names[i]]=list_data[i]

print(names)
result_at_topk=pd.DataFrame(data=dict_data)
result_at_topk.to_csv("result.csv")