import numpy as np
from PIL import Image

import torch
from torch import fft

def draw_cicle(shape, diamiter):
    '''
    Input:
    shape   : tuple (height, width)
    diameter  : scalar
    
    Output:
    np.array of shape  that says True within a circle with diamiter =  around center 
    '''
    assert len(shape) == 2
    TF = np.zeros(shape, dtype=np.bool)
    center = np.array(TF.shape) / 2.0

    for iy in range(shape[0]):
        for ix in range(shape[1]):
            TF[iy, ix] = (iy - center[0])**2 + (ix - center[1])**2 < diamiter**2
    return(TF)

def filter_circle(TFcircleIN, fft_img_channel):
    temp = np.zeros(fft_img_channel.shape[:2], dtype=complex)
    temp[TFcircleIN] = fft_img_channel[TFcircleIN]
    return(temp)

def imshow_fft(absfft):
    magnitude_spectrum = 20*np.log(absfft)
    return(ax.imshow(magnitude_spectrum,cmap="gray"))

def inv_FFT_all_channel(fft_img):
    img_reco = []
    for ichannel in range(fft_img.shape[2]):
        img_reco.append(np.fft.ifft2(np.fft.ifftshift(fft_img[:,:,ichannel])))
    img_reco = np.array(img_reco)
    img_reco = np.transpose(img_reco,(1,2,0))
    return(img_reco)

def pass_filter(img, filter=0):
    if filter == 0:
        return img

    numpy_img = np.asarray(img, dtype=float)
    numpy_img /= 255
    
    TFcircleIN   = draw_cicle(shape=numpy_img.shape[:2], diamiter=5)
    TFcircleOUT  = ~TFcircleIN

    fft_img = np.zeros_like(numpy_img, dtype=complex)
    for ichannel in range(fft_img.shape[2]):
        fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(numpy_img[:, :, ichannel]))
    # print(fft_img.shape)

    fft_img_filtered_IN = []
    fft_img_filtered_OUT = []
    ## for each channel, pass filter
    for ichannel in range(fft_img.shape[2]):
        fft_img_channel = fft_img[:,:,ichannel]
        ## circle IN
        temp = filter_circle(TFcircleIN, fft_img_channel)
        fft_img_filtered_IN.append(temp)
        ## circle OUT
        temp = filter_circle(TFcircleOUT, fft_img_channel)
        fft_img_filtered_OUT.append(temp)

    fft_img_filtered_IN = np.array(fft_img_filtered_IN)
    fft_img_filtered_IN = np.transpose(fft_img_filtered_IN,(1,2,0))
    fft_img_filtered_OUT = np.array(fft_img_filtered_OUT)
    fft_img_filtered_OUT = np.transpose(fft_img_filtered_OUT,(1,2,0))

    abs_fft_img = np.abs(fft_img)
    abs_fft_img_filtered_IN = np.abs(fft_img_filtered_IN)
    abs_fft_img_filtered_OUT = np.abs(fft_img_filtered_OUT)

    img_reco = inv_FFT_all_channel(fft_img)
    img_reco_filtered_IN = inv_FFT_all_channel(fft_img_filtered_IN)
    img_reco_filtered_OUT = inv_FFT_all_channel(fft_img_filtered_OUT)

    ### low pass filter image
    if filter == 1:
        # print(img_reco_filtered_IN.shape)
        # print(np.amin(np.abs(img_reco_filtered_IN)), np.amax(np.abs(img_reco_filtered_IN)))
        img_reco_filtered_IN = np.abs(img_reco_filtered_IN) * 255
        img_reco_filtered_IN = Image.fromarray(img_reco_filtered_IN.astype('uint8'))
        return img_reco_filtered_IN

    ### high pass filtered image
    elif filter == 2:
        # print(img_reco_filtered_OUT.shape)
        # print(np.amin(np.abs(img_reco_filtered_OUT)), np.amax(np.abs(img_reco_filtered_OUT)))
        img_reco_filtered_OUT = np.abs(img_reco_filtered_OUT) * 255
        img_reco_filtered_OUT = Image.fromarray(img_reco_filtered_OUT.astype('uint8'))
        return img_reco_filtered_OUT
        

# val_transforms_list = [
#         transforms.Resize((512, 512)),
#         transforms.ToTensor()
#         # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ]

# val_transforms = Compose(val_transforms_list)

# test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=val_transforms, download=True)

# num = 3
# n = 0

# for image, target in test_dataset:
#     # image.show()
#     # print(image)
#     # numpy_img = np.array(image, dtype=np.float16)
#     numpy_img = image.numpy()
#     numpy_img = np.transpose(numpy_img, (1, 2, 0))
#     # numpy_img /= 256
#     plt.axis('off')
#     plt.imshow(numpy_img)
#     plt.show()
#     # print(numpy_img.shape)

#     TFcircleIN   = draw_cicle(shape=numpy_img.shape[:2], diamiter=5)
#     TFcircleOUT  = ~TFcircleIN

#     fig = plt.figure(figsize=(30,10))
#     ax  = fig.add_subplot(1,2,1)
#     im  = ax.imshow(TFcircleIN,cmap="gray")
#     plt.colorbar(im)
#     ax  = fig.add_subplot(1,2,2)
#     im  = ax.imshow(TFcircleOUT,cmap="gray")
#     plt.colorbar(im)
#     plt.show()

#     fft_img = np.zeros_like(numpy_img, dtype=complex)
#     for ichannel in range(fft_img.shape[2]):
#         fft_img[:, :, ichannel] = np.fft.fftshift(np.fft.fft2(numpy_img[:, :, ichannel]))
#     # print(fft_img.shape)

#     fft_img_filtered_IN = []
#     fft_img_filtered_OUT = []
#     ## for each channel, pass filter
#     for ichannel in range(fft_img.shape[2]):
#         fft_img_channel = fft_img[:,:,ichannel]
#         ## circle IN
#         temp = filter_circle(TFcircleIN, fft_img_channel)
#         fft_img_filtered_IN.append(temp)
#         ## circle OUT
#         temp = filter_circle(TFcircleOUT, fft_img_channel)
#         fft_img_filtered_OUT.append(temp)

#     fft_img_filtered_IN = np.array(fft_img_filtered_IN)
#     fft_img_filtered_IN = np.transpose(fft_img_filtered_IN,(1,2,0))
#     fft_img_filtered_OUT = np.array(fft_img_filtered_OUT)
#     fft_img_filtered_OUT = np.transpose(fft_img_filtered_OUT,(1,2,0))

#     abs_fft_img = np.abs(fft_img)
#     abs_fft_img_filtered_IN = np.abs(fft_img_filtered_IN)
#     abs_fft_img_filtered_OUT = np.abs(fft_img_filtered_OUT)

#     fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15,10))
#     fontsize = 15 
#     for ichannel, color in enumerate(["R","G","B"]):
#         ax = axs[0,ichannel]
#         ax.set_title(color)
#         im = imshow_fft(abs_fft_img[:,:,ichannel])
#         ax.axis("off")
#         if ichannel == 0:
#             ax.set_ylabel("original DFT",fontsize=fontsize)
#         fig.colorbar(im,ax=ax)
        
        
#         ax = axs[1,ichannel]
#         im = imshow_fft(abs_fft_img_filtered_IN[:,:,ichannel])
#         ax.axis("off")
#         if ichannel == 0:
#             ax.set_ylabel("DFT + low pass filter",fontsize=fontsize)
#         fig.colorbar(im,ax=ax)
        
#         ax = axs[2,ichannel]
#         im = imshow_fft(abs_fft_img_filtered_OUT[:,:,ichannel])
#         ax.axis("off")
#         if ichannel == 0:
#             ax.set_ylabel("DFT + high pass filter",fontsize=fontsize)   
#         fig.colorbar(im,ax=ax)
#     plt.show()

#     img_reco = inv_FFT_all_channel(fft_img)
#     img_reco_filtered_IN = inv_FFT_all_channel(fft_img_filtered_IN)
#     img_reco_filtered_OUT = inv_FFT_all_channel(fft_img_filtered_OUT)

#     fig = plt.figure(figsize=(25,18))
#     ax  = fig.add_subplot(1,3,1)
#     ax.imshow(np.abs(img_reco))
#     ax.set_title("original image")

#     ax  = fig.add_subplot(1,3,2)
#     ax.imshow(np.abs(img_reco_filtered_IN))
#     ax.set_title("low pass filter image")


#     ax  = fig.add_subplot(1,3,3)
#     ax.imshow(np.abs(img_reco_filtered_OUT))
#     ax.set_title("high pass filtered image")
#     plt.show()

#     n += 1
#     if n >= 3:
#         break
