import numpy as np
# import cv2

# def normalize(a, axis=-1, order=2):
#     l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
#     l2[l2==0] = 1
#     return a / np.expand_dims(l2, axis)

def normalize(x):
    from sklearn import preprocessing as pp
    x_scaled = pp.scale(x)
    
    return x_scaled

def minmax(x):
    x = np.array(x)
    from sklearn import preprocessing as pp
    x = x.reshape(-1, 1)
    scaler = pp.MinMaxScaler().fit(x)
    x_scaled = scaler.transform(x)
    # 
    return x_scaled

# print(minmax(np.array([1, 2, 3, 4, 5])))
# print(minmax(np.array([1, 2, 3, 4, 5])).shape)

def minmax_inverse(x, x_scaled):
    x = np.array(x)
    x_scaled = np.array(x_scaled)
    from sklearn import preprocessing as pp
    x = x.reshape(-1, 1)
    x_scaled = x_scaled.reshape(-1, 1)
    scaler = pp.MinMaxScaler().fit(x)
    x_restone = scaler.inverse_transform(x_scaled)
    
    return x_restone

# print(minmax_inverse(np.array([1, 2, 3, 4, 5]), np.array([0, 0.25, 0.5, 0.75, 1])))

def threshold(img_gray):
    max_g = 0
    suitable_th = []
    for threshold in range(0, 255):
            bin_img = img_gray > threshold
            bin_img_inv = img_gray <= threshold
            fore_pix = np.sum(bin_img)#前景
            back_pix = np.sum(bin_img_inv)#後景
            if 0 == fore_pix:
                break
            if 0 == back_pix:
                continue
        
            w0 = float(fore_pix) / img_gray.size
            u0 = float(np.sum(img_gray * bin_img)) / fore_pix
            w1 = float(back_pix) / img_gray.size
            u1 = float(np.sum(img_gray * bin_img_inv)) / back_pix
            # intra-class variance
            g = w0 * w1 * (u0 - u1) * (u0 - u1)
            if g > max_g:
                max_g = g
                suitable_th=threshold
            
    
    #print(suitable_th)
    return suitable_th

def otsu(image):
    t = threshold(image)
    # print("threshold =", t)
    if t == [] or 0:
        t = 128
    return np.where(image > t, 1, 0)

def otsu_f(image):
    t = threshold(image)
    lst = image.flatten()
    
    fore = []
    for pix in lst:
        if pix > t:
            fore.append(pix)
    if len(fore) != 0:
        tfore = np.median(fore)
    else:
        tfore = 192
    return np.where(image > tfore, 1, 0)

def otsu_b(image):
    t = threshold(image)
    lst = image.flatten()
    
    back = []
    for pix in lst:
        if pix < t:
            back.append(pix)
    
    if len(back) != 0:
        tback = np.median(back)
    else:
        tback = 64
    return np.where(image < tback, 1, 0)

from PIL import Image, ImageFilter
test = np.array(Image.open('./LIVE3DIQD_origin/Phase1/3d_IQA_database/jpeg/im10_1_l.bmp').convert("L"))
# test = Image.fromarray(test)
# test.show()
# test.save("fig6use.png")
result = otsu_b(test)
result = Image.fromarray(result*255).convert("L")
result.show()
result.save("fig6use_b.png")




# import cv2
# from matplotlib import pyplot as plt
# stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
# #stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
# limage = cv2.imread('./LIVE3DIQD_origin/Phase1/3d_IQA_database/jpeg/im10_1_l.bmp')
# rimage = cv2.imread('./LIVE3DIQD_origin/Phase1/3d_IQA_database/jpeg/im10_1_r.bmp')
# lgrayscale = cv2.cvtColor(limage, cv2.COLOR_BGR2GRAY)
# rgrayscale = cv2.cvtColor(rimage, cv2.COLOR_BGR2GRAY)

# lnor = normalize(lgrayscale)
# rnor = normalize(rgrayscale)

# disparity = stereo.compute(lgrayscale, rgrayscale)
# # darr = np.array(disparity)
# plt.imshow(disparity,'gray')
# ax = plt.gca()
# ax.axes.xaxis.set_ticks([])
# ax.axes.yaxis.set_ticks([])
# plt.show()