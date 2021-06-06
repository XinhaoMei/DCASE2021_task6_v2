import os
import subprocess
import glob
import pickle

def load_pickle(file_name):
    
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj

# move all validation wav files to development
out = subprocess.call('mv validation/* development/', shell=True)

# move old val files in development to validation
val_file = 'validation_file_names.p'
val_file_names = load_pickle(val_file)
for f_name in val_file_names:
    f_name = 'development/'+ f_name
    out = subprocess.call('mv "%s" validation/'%f_name, shell=True)
print('move done.')
dev_files = glob.glob('development/*.wav')
val_files = glob.glob('validation/*.wav')

print('Total {} dev files'.format(len(dev_files)))
print('Total {} val files'.format(len(val_files)))
