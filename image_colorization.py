import cv2
import numpy as np


protoFile = './models/colorization_deploy_v2.prototxt'
weightsFile = './models/colorization_release_v2.caffemodel'

frame = cv2.imread('./greyscaleImage.png')

width = 224
height = 224

net = cv2.dnn.readNetFromCaffe(protoFile,weightsFile)

pts_in_hull = np.load('./pts_in_hull.npy')
 
# populate cluster centers as 1x1 convolution kernel
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

img_rgb = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
img_l = img_lab[:,:,0] # pull out L channel

img_l_rs = cv2.resize(img_l, (width, height)) # resize image to network input size
img_l_rs -= 50 # subtract 50 for mean-centering

net.setInput(cv2.dnn.blobFromImage(img_l_rs))
ab_dec = net.forward()[0,:,:,:].transpose((1,2,0)) # this is our result
 
(H_orig,W_orig) = img_rgb.shape[:2] # original image size
ab_dec_us = cv2.resize(ab_dec, (W_orig, H_orig))
img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)
 
#cv2.imwrite('/home/krutika/Documents/Image_Colorization/dog_colorized.png', cv2.resize(img_bgr_out*255, imshowSize))
#outputFile = args.input[:-4]+'_colorized.png'
cv2.imwrite('/home/krutika/Documents/Image_Colorization/dog_colorized.png', (img_bgr_out*255).astype(np.uint8))