import ocr_preprocessed_data as pp
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd

picture_path = pp.get_picture_path()
denoised_path = '/home/advai-admin-root/projects/ocr_lib/denoised/'

all_denoised_files = pp.get_all_filenames(denoised_path)
all_denoised_files = [filename.split("/")[-1] for filename in all_denoised_files]

original_filenames = [name.split("-")[-1] for name in all_denoised_files]
original_data = [pp.extract_data_from_filename(name) for name in original_filenames]
no_noise_data = [data.copy() for data in original_data]
for data in no_noise_data:
    data['bt'] = 1

no_noise_filenames = [pp.create_filename_dataskewfix(**data) for data in no_noise_data]

#for idx in range(len(no_noise_filenames[:5])):
#    print('Original filename: ', original_filenames[idx])
#    pp.display(picture_path + original_filenames[idx])
#
#    print('Denoised filename: ', all_denoised_files[idx])
#    pp.display(denoised_path + all_denoised_files[idx])
#
#    print('No noise filename: ', no_noise_filenames[idx])
#    pp.display(picture_path + no_noise_filenames[idx])
#

denoise_nonoise_dif_metrics = []
for idx in tqdm(range(len(no_noise_filenames))):
    no_noise = picture_path + no_noise_filenames[idx]
    original = picture_path + original_filenames[idx]
    denoised = denoised_path + all_denoised_files[idx]
    # denoise to no noise comparison
    denoise_nonoise_dif_metrics += [pp.compare_image_difference(denoised, no_noise)]

dd = {'abs_err': [], 'rel_err': []}
for metric in denoise_nonoise_dif_metrics:
    dd['abs_err'] += [metric[0]]
    dd['rel_err'] += [metric[1]]

df = pd.DataFrame(dd)
print(df.describe())
df.rel_err.hist()
plt.show()

min_rel = df.rel_err.min()
min_rel_idx = df.rel_err.idxmin()
max_rel = df.rel_err.max()
max_rel_idx = df.rel_err.idxmax()

#print('The smallest relative error is ', min_rel * 100, '% error.')
#print('The no noise filename is: ', no_noise_filenames[min_rel_idx])
#pp.display(picture_path + no_noise_filenames[min_rel_idx])
#print('The noisy filename is: ', original_filenames[min_rel_idx])
#pp.display(picture_path + original_filenames[min_rel_idx])
#print('The denoised filename is: ', all_denoised_files[min_rel_idx])
#pp.display(denoised_path + all_denoised_files[min_rel_idx])
#
#print('The largest relative error is ', max_rel * 100, '% error.')
#print('The no noise filename is: ', no_noise_filenames[max_rel_idx])
#pp.display(picture_path + no_noise_filenames[max_rel_idx])
#print('The noisy filename is: ', original_filenames[max_rel_idx])
#pp.display(picture_path + original_filenames[max_rel_idx])
#print('The denoised filename is: ', all_denoised_files[max_rel_idx])
#pp.display(denoised_path + all_denoised_files[max_rel_idx])
