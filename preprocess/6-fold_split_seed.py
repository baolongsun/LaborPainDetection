from collections import defaultdict
import random
import os

seed = 8109
random.seed(seed)
patient_maps = defaultdict(list)
with open('./tags.txt','r',encoding='utf-8') as f:
    data = f.readlines()
for d in data:
    patient = d.split(' ')[0].split('\\')[-2]
    patient_maps[patient].append(d)
patients_list = list(patient_maps.keys())
patients_list.sort()
random.shuffle(patients_list)
os.makedirs(f'exp/{seed}_6fold',exist_ok=True)

tags = int(180/6) #180个patient，6组
for idx in range(6):
    split = int(0.7*len(patients_list))
    test_patients = patients_list[tags*idx:tags*(idx+1)]
    train_patients = list(set(patients_list) - set(test_patients))

    train_count = defaultdict(int)
    with open(f"exp/{seed}_6fold/train_{idx}.txt",'w',encoding='utf-8') as f:
        for train_patient in train_patients:
            for pp in patient_maps[train_patient]:
                train_count[pp.strip().split(' ')[1]] += 1
                f.write(pp)
    print('train_distribute:',sorted(train_count.items(),key=lambda x:x[0]))


    test_count = defaultdict(int)
    with open(f"exp/{seed}_6fold/test_{idx}.txt",'w',encoding='utf-8') as f:
        for test_patient in test_patients:
            for pp in patient_maps[test_patient]:
                test_count[pp.strip().split(' ')[1]] += 1
                f.write(pp)
    print('test_distribute:',sorted(test_count.items(),key=lambda x:x[0]))

    train_count = sorted(train_count.items(),key=lambda x:x[0])
    test_count = sorted(test_count.items(),key=lambda x:x[0])
    with open(f'exp/{seed}_6fold/dataInfo_{idx}.txt','w',encoding='utf-8') as f:
        f.write('train_info:\n')
        f.write(str(train_patients)+'\n')
        f.write(str(train_count)+'\n')
        f.write('test_info:\n')
        f.write(str(test_patients)+'\n')
        f.write(str(test_count))
