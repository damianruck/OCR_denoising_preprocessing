"""
This module contains functions neccessary to analyse the effect of
how preprocessing affects of images.

It relies on the user having three related images.

    - The original image (img_o),
    - The affected image (img_a i.e. the original image with some kind of
    undersired affect applied to it)
    - The undone image (img_u i.e. the affected image with some kind of
    algorithm applied to it that was designed to undo the undesired effect).
"""

#################
### LIBRARIES ###
#################

import glob
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageFilter
import copy

#################
### FUNCTIONS ###
#################

### CALCULATING ERRORS

def get_difference(img1, img2):
    """Calculate the absolute difference of two binarised images where:

    The absolute difference is np.abs(img1 - img2).

    img1 (2D np.array): A binarised image where the values are 0 and 1 ( not 0
    and 255) and the dtype is a signed integer (np.int16) rather than unsigned
    integer (np.uint).

    img2 (2D np.array): A binarised image where the values are 0 and 1 ( not 0
    and 255) and the dtype is a signed integer (np.int16) rather than unsigned
    integer (np.uint).

    return (2D np.array): A 2D numpy array which is the absoolute difference
    between img1 and img2.

    ValueError raised if the dimensions of img1 and img2 are not both equal to
    2.
    """

    if len(img2.shape) != 2 or len(img1.shape) != 2:
        raise ValueError('Dimensions of img1 and img2 must both be equal to two.')

    tmp1 = copy.deepcopy(img1)
    tmp1 = tmp1.astype(np.int16)
    tmp1 = tmp1 / 255
    tmp2 = copy.deepcopy(img2)
    tmp2 = tmp2.astype(np.int16)
    tmp2 = tmp2 / 255

    ans = np.abs(tmp1 - tmp2)
    ans = ans.astype(np.uint8)

    return ans * 255
 
def get_error(img):
    """Calculate the absolute and relative error of an image where:

    The absolute error (abs_err) is the sum of all the pixels.

    The relative error (rel_err) is the absolute error divided by the total
    number possible pixels.

    img (2D np.array): Is the "difference" between two images (i.e.
    get_difference(img1, img2)). It is a binarised image where the values are
    0 and 1 (not 0 and 255) and the dtype is a signed integer (np.int16)
    rather than unsigned integer (np.uint).

    return (abs_err, rel_err): A tuple where the first is the absolute error
    and the the second is the relative error.

    ValueError raised if the dimensions of img are not equal to 2.
    """

    if len(img.shape) != 2:
        raise ValueError('Dimensions of img must be equal to two.')

    tmp = img.copy()
    tmp = tmp.astype(np.int16)
    tmp = tmp / 255
    total_pixels = tmp.shape[0] * tmp.shape[-1]
    abs_error = np.sum(tmp)
    rel_error = abs_error / float(total_pixels)

    return abs_error, rel_error

def get_error_breakdown(img_o, img_a, undoing_function, function_parameters_dict):
    """Calculates the total undo error, pure undo error, the
    letter -> affect affect and the affect -> letter affect.

    - The total affected error is the error between the original image and the
    undone affected image (img_o - img_u).
    - The pure affected error is the pixels remaining after undoing the pure
    affect (i.e. all leters removed from the affected image: img_n - img_o).
     - The letter to undo affect is the difference between the total and the
     pure undone images (i.e. pure affected image minus the total affected
     image minus the original image.
    - The undo to letter affect is the difference in the lettering between
    the original and the undone image (i.e img_u minus the letter to undo
    affect minus img_o)

    img_o (2D np.array): A binarised original image where the values are 0 and
    1 (not 0 and 255) and the dtype is a signed integer (np.int16) rather than
    unsigned integer (np.uint).

    img_a (2D np.array): A binarised affected image where the values are 0 and
    1 (not 0 and 255) and the dtype is a signed integer (np.int16) rather than
    unsigned integer (np.uint).

    undoing_function (Function(2D np.array)): A Python function that takes a
    binarised image and tries to undo some undesired affect.

    return total_affected_error, pure_affected_error, letter_to_algo_affect,
    lettering_affect

    ValueError raised if the dimensions of img_o and img_n are not both equal
    to 2.
    """

    # check images are correct dimensions
    if len(img_o.shape) != 2 or len(img_a.shape) != 2:
        raise ValueError('Dimensions of img_o,img_a and img_u must both be equal to two.')

    if function_parameters_dict:
        img_u = undoing_function(img_a, **function_parameters_dict)
    else:
        img_u = undoing_function(img_a)

    # img_u has been turned from black and white to gray and so needs to be re-binarised
    img_u = post_undoing_binarisation(img_u)

    ### TOTAL AFFECTED IMAGE/ERROR
    total_affected_undone = get_total_affected_image(img_o, img_u)
    total_affected_undone_error = get_error(total_affected_undone)

    ### PURE AFFECTED IMAGE/ERROR
    pa = get_pure_affected_image(img_a, img_o, undoing_function, function_parameters_dict)
    pure_affect = pa[1]
    pure_affect_undone = pa[0]
    undoing_error = get_error(pure_affect_undone)
    pure_affect_error = get_error(pure_affect)

    ### THE LETTER TO UNDO AFFECT IMAGE/ERROR: pai - tai - img_o
    lua_s2, lua_s1 = get_letter_to_undo_affect_image(total_affected_undone, pure_affect_undone, img_o)
    luae_s2 = get_error(lua_s2)
    luae_s1 = get_error(lua_s1)

    ### THE UNDO TO LETTER AFFECT: img_u minus the letter to undo affect minus img_o
    ula_s2, ula_s1 = get_undo_to_letter_affect(img_u, lua_s2, img_o)
    ulae_s2 = get_error(ula_s2)
    ulae_s1 = get_error(ula_s1)

    return {'input': {'img_o': img_o, 'img_a': img_a, 'img_u': img_u}, 'total': {'total_undoing_error': total_affected_undone, 'absolute_undoing_error': total_affected_undone_error[0], 'pct_undoing_error': total_affected_undone_error[1]}, 'pure': (pure_affect, pure_affect_error, pure_affect_undone, undoing_error), 'ltu': (lua_s2, luae_s2, lua_s1, luae_s1), 'utl': (ula_s2, ulae_s2, ula_s1, ulae_s1)}

def get_total_affected_image(img_o, img_u):
    """The difference between the original image and the undone affected image.

    """
    return get_difference(img_o, img_u)

def get_pure_affected_image(img_a, img_o, undoing_function, function_parameters_dict):
    """How well the undo algorithm performs without the letters.

    """

    pure_noise_img = get_difference(img_a, img_o)
    if function_parameters_dict:
        img = undoing_function(pure_noise_img, **function_parameters_dict)
    else:
        img = undoing_function(pure_noise_img)

    img = post_undoing_binarisation(img)

    return img, pure_noise_img

def get_letter_to_undo_affect_image(total_affected_image, pure_affected_image, img_o):
    """The affect that the letters have on the undoing algorithm.

    """
    stage1 = get_difference(total_affected_image, pure_affected_image) # this is the letters and affect remaining that can't be explained by undoing the pure affect
    stage2 = get_difference(stage1, img_o) # this takes away the letters leaving on the affect that can't be explained by undoing the pure affect

    return stage2, stage1

def get_undo_to_letter_affect(img_u, letter_to_undo_affect_image, img_o):
    """The affect that the undoing algorithm has on the letters.

    """
    stage1 = get_difference(img_u, letter_to_undo_affect_image)
    stage2 = get_difference(stage1, img_o)

    return stage2, stage1

### BINARISATION

def post_undoing_binarisation(img):
    """
    Applying things like denoising algorithm can turn the black and white images into gray images and so we want to perform callum binarisation on it again but the callum_binarise function won't work on it in this form so this function was created.
    """
    gray = img.copy()
    #gray =cv2.bitwise_not(img)
    # binarise
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return thresh

def callum_binarise(filename):
    # read image
    img = cv2.imread(filename)
    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray =cv2.bitwise_not(gray)
    # binarise
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return thresh

### FILE FUNCTIONS

def display_file(im_path):
    dpi = 600
    im_data = plt.imread(im_path)

    height, width = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    #figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    #fig = plt.figure(figsize=figsize)
    fig = plt.figure(figsize=(height, width))
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

    return fig

def display_image(im_data):
#    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    #figsize = width / float(dpi), height / float(dpi)
    figsize = 12, 8

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
#    fig = plt.figure(figsize=(height, width))
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')
#    plt.imshow(im_data, cmap='gray')

    plt.show()

#    return fig
    return

def create_filename(b,bt,s,dt,do,string_idx,f):
    return 'b,' + str(b) + ',bt,' + str(bt) + ',s,' + str(s) + ',dt,' + str(dt) + ',do,' + str(do) + ',s,' + str(string_idx) + ',f' + str(f) + '.png'

def get_picture_path():
    return "/mnt/advai-ukrandd-share/research/OCR Preprocess Data/data_skew_fix/"

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

############################
### DENOISING ALGORITHMS ###
############################

### IMPORTANT!!!!!!!!     all denoising functions MUST have the image passed as the first parameter


def callum_noise_removal(
        image, dilate_kernel=np.ones((1, 1), np.uint8),
        dilate_iterations=1, erode_kernel=np.ones((1, 1), np.uint8),
        erode_iterations=1, morphology_type=cv2.MORPH_CLOSE, 
        morphology_kernel=np.ones((1, 1), np.uint8), aperture_linear_size=3):

    image = cv2.dilate(image, dilate_kernel, iterations=dilate_iterations)
    image = cv2.erode(image, erode_kernel, iterations=erode_iterations)
    image = cv2.morphologyEx(image, morphology_type, morphology_kernel)
    image = cv2.medianBlur(image, aperture_linear_size)

    return image

def mean_filter(img, kernel_size1, kernel_size2):
    return cv2.blur(img,(kernel_size1, kernel_size2))

def gaussian_filter(img, kernel_size, sigma_x, sigma_y):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), int(sigma_x))

def median_filter(img, kernel_size):
    return cv2.medianBlur(img, kernel_size)

def conservative_filter(img, filter_size):

    temp = []
    indexer = filter_size // 2
    new_image = img.copy()
    nrow, ncol = img.shape
    
    for i in range(nrow):
        for j in range(ncol):
            for k in range(i-indexer, i+indexer+1):
                for m in range(j-indexer, j+indexer+1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            
                            temp.append(img[k,m])
                            
            temp.remove(img[i,j])
            max_value = max(temp)
            min_value = min(temp)
            
            if img[i,j] > max_value:
                new_image[i,j] = max_value
            
            elif img[i,j] < min_value:
                new_image[i,j] = min_value
            
            temp =[]
    
    return new_image.copy()

def laplacian_filter(img):
    return cv2.Laplacian(img,cv2.CV_64F)

def laplacian_of_gaussian_filtering(img, kernel_size, sigma_x, sigma_y):
    gauss = gaussian_filter(img, kernel_size, sigma_x, sigma_y)
    lp = laplacian_filter(gauss)
    return gauss + lp

def low_pass_dft(img):
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    # shift the zero-frequncy component to the center of the spectrum
    dft_shift = np.fft.fftshift(dft)
    # save image of the image in the fourier domain.
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    # apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    return img_back, magnitude_spectrum

def crimmins_speckle_removal(data):
    new_image = data.copy()
    nrow = len(data)
    ncol = len(data[0])
    
    # Dark pixel adjustment
    
    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[i-1,j] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol-1):
            if data[i,j+1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[i-1,j-1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    #NE-SW
    for i in range(1, nrow):
        for j in range(ncol-1):
            if data[i-1,j+1] >= (data[i,j] + 2):
                new_image[i,j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow-1):
        for j in range(ncol):
            if (data[i-1,j] > data[i,j]) and (data[i,j] <= data[i+1,j]):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j+1] > data[i,j]) and (data[i,j] <= data[i,j-1]):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i-1,j-1] > data[i,j]) and (data[i,j] <= data[i+1,j+1]):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i-1,j+1] > data[i,j]) and (data[i,j] <= data[i+1,j-1]):
                new_image[i,j] += 1
    data = new_image
    #Third Step
    # N-S
    for i in range(1, nrow-1):
        for j in range(ncol):
            if (data[i+1,j] > data[i,j]) and (data[i,j] <= data[i-1,j]):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j-1] > data[i,j]) and (data[i,j] <= data[i,j+1]):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i+1,j+1] > data[i,j]) and (data[i,j] <= data[i-1,j-1]):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if (data[i+1,j-1] > data[i,j]) and (data[i,j] <= data[i-1,j+1]):
                new_image[i,j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow-1):
        for j in range(ncol):
            if (data[i+1,j] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol):
            if (data[i,j-1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # NW-SE
    for i in range(nrow-1):
        for j in range(ncol-1):
            if (data[i+1,j+1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image
    # NE-SW
    for i in range(nrow-1):
        for j in range(1,ncol):
            if (data[i+1,j-1] >= (data[i,j]+2)):
                new_image[i,j] += 1
    data = new_image

    # Light pixel adjustment
    
    # First Step
    # N-S
    for i in range(1,nrow):
        for j in range(ncol):
            if (data[i-1,j] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol-1):
            if (data[i,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow):
        for j in range(1,ncol):
            if (data[i-1,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow):
        for j in range(ncol-1):
            if (data[i-1,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1,nrow-1):
        for j in range(ncol):
            if (data[i-1,j] < data[i,j]) and (data[i,j] >= data[i+1,j]):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol-1):
            if (data[i,j+1] < data[i,j]) and (data[i,j] >= data[i,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i-1,j-1] < data[i,j]) and (data[i,j] >= data[i+1,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i-1,j+1] < data[i,j]) and (data[i,j] >= data[i+1,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1,nrow-1):
        for j in range(ncol):
            if (data[i+1,j] < data[i,j]) and (data[i,j] >= data[i-1,j]):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol-1):
            if (data[i,j-1] < data[i,j]) and (data[i,j] >= data[i,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i+1,j+1] < data[i,j]) and (data[i,j] >= data[i-1,j-1]):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1,nrow-1):
        for j in range(1,ncol-1):
            if (data[i+1,j-1] < data[i,j]) and (data[i,j] >= data[i-1,j+1]):
                new_image[i,j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow-1):
        for j in range(ncol):
            if (data[i+1,j] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1,ncol):
            if (data[i,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NW-SE
    for i in range(nrow-1):
        for j in range(ncol-1):
            if (data[i+1,j+1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    # NE-SW
    for i in range(nrow-1):
        for j in range(1,ncol):
            if (data[i+1,j-1] <= (data[i,j]-2)):
                new_image[i,j] -= 1
    data = new_image
    return new_image.copy()

def unsharp_filter(img, radius=2, percent=150):
    img = Image.fromarray(img.astype('uint8'))
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent))
