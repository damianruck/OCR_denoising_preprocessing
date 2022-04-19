import ocr_preprocessed_data as pp
import cv2
from tqdm import tqdm

required_filenames = {'b': 0, 'bt': 2, 's': -1, 'dt': -1, 'do': -1, 'string_idx': -1, 'f': -1}
picture_path = pp.get_picture_path()

all_files = pp.get_all_filenames(picture_path)
all_filenames = [filename.split("/")[-1] for filename in all_files]

filtered_names = pp.filter_filenames(all_filenames, required_filenames)
print(len(all_filenames))
print(len(filtered_names))
print(filtered_names[0])

for name in tqdm(filtered_names):
    img = cv2.imread(picture_path + name)

    no_noise = pp.noise_removal(img)
    cv2.imwrite("denoised/no_noise-" + name, no_noise)
    #pp.display("denoised/no_noise-" + name)
