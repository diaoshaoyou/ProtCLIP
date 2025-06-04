import csv
import pdb
import sys
import json
import math
import random

csv.field_size_limit(sys.maxsize)

# trEMBL=251131639    SwissProt=570420
def clean(name='trEMBL'):
    # in our original data (from UniProt)
    # row[1]=entry name, row[2]=protein name, row[32]=function, row[69]=location, row[96]=similarity, row[19]=seq, row[45]=existence
    select_data=[]
    num_before={1:{1:0, 2:0, 3:0, 4:0}, 
                2:{1:0, 2:0, 3:0, 4:0}, 
                3:{1:0, 2:0, 3:0, 4:0}, 
                4:{1:0, 2:0, 3:0, 4:0}, 
                5:{1:0, 2:0, 3:0, 4:0}}
    num_after={1:{3:0, 4:0},
               2:{3:0, 4:0},
               3:{3:0, 4:0}}
    with open(f'/root/DATA/{name}.tsv', 'r', encoding='utf-8', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        for i, row in enumerate(reader): 
            if i==0:
                continue
            else:
                if row[45]=='Predicted':
                    existence=4
                elif row[45]=='Uncertain':
                    existence=5
                elif row[45]=='Inferred from homology':
                    existence=3
                elif row[45]=='Evidence at transcript level':
                    existence=2
                elif row[45]=='Evidence at protein level':
                    existence=1
                occupy=1
                item={}
                item['seq']=row[19]
                text='PROTEIN NAME: '+row[2]+'.'
                if row[32] != '':
                    text+=' '+row[32]
                    occupy+=1
                if row[69] != '':
                    text+=' '+row[69]
                    occupy+=1
                if row[96] != '':
                    text+=' '+row[96]
                    occupy+=1
                num_before[existence][occupy]+=1
                if occupy<3 or existence>3:
                    continue
                p_in=(math.sqrt(occupy/4)) / (existence*existence*existence)
                rn=random.random()
                if rn < p_in:
                    item['text']=text
                    select_data.append(item)
                    num_after[existence][occupy]+=1
            print(i)

    with open('/root/DATA/trEMBL_new.json', 'w') as file:
        for idx, item in enumerate(select_data):
            json.dump(item, file)
            file.write('\n')
            print(idx)
    print(num_before)
    print(num_after)
    

clean() # obtain trEMBL_new.json

# merge train.json (i.e. ProtAnno-S) and trEMBL_new.json to obtain train_new.json (i.e. ProtAnno-D)
# train.json is directly from the training data in ProtST (https://github.com/DeepGraphLearning/ProtST)

total=[]
with open('/root/DATA/datasets/ProtAnno/train.json', 'r') as f1:
    for idx, line in enumerate(f1):
        total.append(json.loads(line.strip()))
        print(len(total))
with open('/root/DATA/trEMBL_new.json', 'r') as f2:
    for idx, line in enumerate(f2):
        total.append(json.loads(line.strip()))
        print(len(total))

with open('/root/DATA/datasets/ProtAnno/train_new.json', 'w') as f3:
    for idx, item in enumerate(total):
        json.dump(item, f3)
        f3.write('\n')
        print('write', idx)
        