import os
import pandas as pd
import numpy as np

if not os.path.exists('data_lists_upper'):
    os.mkdir('data_lists_upper')

all_data = pd.read_csv('data_lists/TIMIT_all.scp', header=None, names=['path'])
all_data['path'] = all_data.path.str.upper()
all_data.to_csv('data_lists_upper/TIMIT_all.scp', header=False, index=False)

train_data = pd.read_csv('data_lists/TIMIT_train.scp', header=None, names=['path'])
train_data['path'] = train_data.path.str.upper()
train_data.to_csv('data_lists_upper/TIMIT_train.scp', header=False, index=False)

test_data = pd.read_csv('data_lists/TIMIT_test.scp', header=None, names=['path'])
test_data['path'] = test_data.path.str.upper()
test_data.to_csv('data_lists_upper/TIMIT_test.scp', header=False, index=False)

labels = np.load('data_lists/TIMIT_labels.npy', allow_pickle=True).item()
labels = pd.DataFrame(list(labels.items()), columns=['path', 'label'])
labels['path'] = labels.path.str.upper()
label_dict = dict(labels.values.tolist())
np.save('data_lists_upper/TIMIT_labels.npy', label_dict)
