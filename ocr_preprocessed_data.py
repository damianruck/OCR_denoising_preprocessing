import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

### GLOBAL VARIABLES (I know, I know!)

def get_picture_path():
    return "/mnt/advai-ukrandd-share/research/OCR Preprocess Data/data_skew_fix/"

def get_string_index_dict():
    return { "The quick brown fox jumps over the lazy dog!": 0, "90689249058624193404268466437211": 1, """i~"Lm--rM.TYm1=PIs_ztMvvO1T.z:v;""": 2 }

filename_ranges = {'b': tuple(range(4)), 'bt': tuple(range(4)), 's': tuple(range(5)), 'dt': tuple(range(4)), 'do': tuple(range(3)), 'string_idx': tuple(range(3)), 'f': tuple(range(3))}

### FUNCTIONS

# IMAGE FILE FUNCTIONS

def filter_filenames(filenames, filter_dict):
    filtered_names = []
    for name in filenames:
        data = extract_data_from_filename(name)
        correct_name = True
        for label, value in filter_dict.items():
            if value != -1:
                if data[label] != str(value):
                    correct_name = False

        if correct_name == True:
            filtered_names += [name]

    return filtered_names

def create_filename_dataskewfix(b,bt,s,dt,do,string_idx,f):
    return 'b,' + str(b) + ',bt,' + str(bt) + ',s,' + str(s) + ',dt,' + str(dt) + ',do,' + str(do) + ',s,' + str(string_idx) + ',f' + str(f) + '.png'

def extract_data_from_filename(filename):
    name_list = filename.split(",")
    last = name_list.pop()
    last = last.split(".")
    last = last[0]
    even_nos = [2*n for n in range(int(len(name_list)/2))]
    data = {}
    for n in even_nos:
        if name_list[n] == 's' and 's' in data:
            data['string_idx'] = name_list[n+1]
        else:
            data[name_list[n]] = name_list[n+1]

    data[last[0]] = last[1]

    return data

def get_all_filenames(path):
    if path[-1] == '/':
        return [f for f in glob.glob(path + "*.png")]
    else:
        return [f for f in glob.glob(path + "/*.png")]

# IMAGE FUNCTIONS

def callum_binarise(filename):
    # read image
    img = cv2.imread(filename)
    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray =cv2.bitwise_not(gray)
    # binarise
    ret, thresh = cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return thresh

def compare_image_difference(filename1, filename2):
    bin1 = callum_binarise(filename1)
    bin2 = callum_binarise(filename2)
    # change type to signed rather than unsigned int
    bin1 = bin1.astype(np.int16)
    bin2 = bin2.astype(np.int16)
    # calculate absolute difference
    abs_dif = np.abs(bin1 - bin2)
    abs_err = np.sum(abs_dif)
    perc_err = abs_err / float(abs_dif.shape[0] * abs_dif.shape[1])

    return (abs_err, perc_err)

def display(im_path):
    dpi = 600
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

    return fig

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# DENOISING FUNCTIONS

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

### SCRIPT
if __name__ == "__main__":

    test = {'b': 0, 'bt': 0, 's': 0, 'dt': 0, 'do': 0, 'string_idx': 0, 'f': 0}

    name = create_filename_dataskewfix(**test)
    data = extract_data_from_filename(name)
    print(test)
    print(name)
    print(data)

    picture_path = get_picture_path()
    all_filenames = get_all_filenames(picture_path)
    tmp_fn = all_filenames[0]
    tmp_n = tmp_fn.split("/")
    tmp_n = tmp_n[-1]
    #tmp_n = tmp_n.split(".")
    #tmp_n = tmp_n[0]
    print('tmp_n = ', tmp_n)
    display(tmp_fn)

    img = cv2.imread(tmp_fn)

    gray_image = grayscale(img)
    cv2.imwrite("temp/gray.png", gray_image)
    display("temp/gray.jpg")

    thresh, im_bw = cv2.threshold(gray_image, 160, 230, cv2.THRESH_BINARY)
    cv2.imwrite("temp/bw_image.png", im_bw)
    display("temp/bw_image.png")

    no_noise = noise_removal(im_bw)
    cv2.imwrite("denoised/no_noise-" + tmp_n, no_noise)
    display("denoised/no_noise-" + tmp_n)
