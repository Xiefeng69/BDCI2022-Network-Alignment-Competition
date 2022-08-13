import numpy as np
np.set_printoptions(suppress=True)

anchor = np.loadtxt('anchor.txt', delimiter=' ')
anchor_num = len(anchor)

# add anchor noise
anchor_1_ratio = 0.2
anchor_1_num = int(anchor_num * anchor_1_ratio)
anchor_1 = anchor[0:anchor_1_num]
anchor_1_left = anchor_1[:,0]
anchor_1_left = np.expand_dims(anchor_1_left, axis=1)
anchor_1_right = np.random.randint(low=0, high=1034, size=(len(anchor_1_left)))
anchor_1_right = np.expand_dims(anchor_1_right, axis=1)
anchor_1 = np.hstack((anchor_1_left, anchor_1_right))
print(f'anchor_1 number: {anchor_1_num}\n')
anchor_1withgt = np.concatenate((anchor_1, anchor[anchor_1_num:])) # concat with ground truth
with open(f'data/anchor/anchor_{anchor_1_ratio}.txt', 'w', encoding='utf-8') as f:
    for i in anchor_1withgt:
        f.write(f"{int(i[0])} {int(i[1])}\n")

anchor_2_ratio = 0.4
anchor_2_num = int(anchor_num * anchor_2_ratio)
anchor_2 = anchor[anchor_1_num: anchor_2_num]
anchor_2_left = anchor_2[:,0]
anchor_2_left = np.expand_dims(anchor_2_left, axis=1)
anchor_2_right = np.random.randint(low=0, high=1034, size=(len(anchor_2_left)))
anchor_2_right = np.expand_dims(anchor_2_right, axis=1)
anchor_2 = np.hstack((anchor_2_left, anchor_2_right))
anchor_2 = np.concatenate((anchor_1, anchor_2))
print(f'anchor_2 number: {anchor_2_num}\n')
anchor_2withgt = np.concatenate((anchor_2, anchor[anchor_2_num:])) # concat with ground truth
with open(f'data/anchor/anchor_{anchor_2_ratio}.txt', 'w', encoding='utf-8') as f:
    for i in anchor_2withgt:
        f.write(f"{int(i[0])} {int(i[1])}\n")

anchor_3_ratio = 0.6
anchor_3_num = int(anchor_num * anchor_3_ratio)
anchor_3 = anchor[anchor_2_num: anchor_3_num]
anchor_3_left = anchor_3[:,0]
anchor_3_left = np.expand_dims(anchor_3_left, axis=1)
anchor_3_right = np.random.randint(low=0, high=1034, size=(len(anchor_3_left)))
anchor_3_right = np.expand_dims(anchor_3_right, axis=1)
anchor_3 = np.hstack((anchor_3_left, anchor_3_right))
anchor_3 = np.concatenate((anchor_2, anchor_3))
print(f'anchor_3 number: {anchor_3_num}\n')
anchor_3withgt = np.concatenate((anchor_3, anchor[anchor_3_num:])) # concat with ground truth
with open(f'data/anchor/anchor_{anchor_3_ratio}.txt', 'w', encoding='utf-8') as f:
    for i in anchor_3withgt:
        f.write(f"{int(i[0])} {int(i[1])}\n")