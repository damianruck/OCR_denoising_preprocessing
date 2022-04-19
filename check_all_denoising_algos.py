"""
This is a script that tests all denoising algorithms.
"""

#################
### LIBRARIES ###
#################

from tqdm import tqdm
import cv2
import algo_test as at


################################################
### DENOISING ALGORITHM FUNCTION DEFINITIONS ###
################################################

def noise_removal1(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

##############
### SCRIPT ###
##############

### GET ALL FILENAMES
picture_path = at.get_picture_path()
denoised_path = '/home/advai-admin-root/projects/ocr_lib/denoised/'

all_files = at.get_all_filenames(picture_path)
all_filenames = [filename.split("/")[-1] for filename in all_files]

### FILTER FILENAMES
noisy_filenames_dict = {'b': 0, 'bt': 2, 's': -1, 'dt': -1, 'do': -1, 'string_idx': -1, 'f': -1}
noisy_filenames = at.filter_filenames(all_filenames, noisy_filenames_dict)

original_filenames_dict = {'b': 0, 'bt': 1, 's': -1, 'dt': -1, 'do': -1, 'string_idx': -1, 'f': -1}
original_filenames = at.filter_filenames(all_filenames, original_filenames_dict)

denoise_funcs = {'noise_removal1': noise_removal1} # dict of denoising functions

results = {}
for algo_name, algo in tqdm(denoise_funcs):
    results[algo_name] = []
    for filename_idx in tqdm(range(len(noisy_filenames))):
        img_o = at.callum_binarise(original_filenames[filename_idx])
        img_a = at.callum_binarise(noisy_filenames[filename_idx])
        results[algo_name] += [at.get_error_breakdown(img_o, img_a, algo)]
