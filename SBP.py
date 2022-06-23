# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:39:32 2021

@author: danie
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import img_as_ubyte, img_as_float
from skimage.feature import canny
from skimage import io, util
from skimage import filters, morphology, restoration, segmentation 
from skimage import measure
import fnmatch


#%%
SBPbeads = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/"
og = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/0.Keyence JPG/" 
test = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/1.test group/"
greyscale = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/2.greyscale/" 
cropped = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/3.cropped/" 
tcropped = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/3.testcropped/" 
denoise = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/5.denoised/" 
nlmfill = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/5.nlm fill/" 
binary = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/6.binary/"
nosmall = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/6.nosmall/" 
cancan = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/6.canny/" 
invert = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/6.5.invert/"
linedance = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/7.contours/"
erodils = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/8.erodils/"
fhfront = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/8.0-fillhofront/"
fhback = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/8.1-fillhoback/"
fefront = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/8.0-fillexfront/"
feback = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/8.1-fillexback/"
toutput = "C:/Users/danie/Desktop/PhDan/SBP/Bead pics/50.testoutput/"

#%%
#######USING REAL IMAGES
#####save images as greyscale
for pics in os.listdir(og):
    bead = io.imread(og + pics, as_gray=True)  
    grey = img_as_ubyte(bead)
    cv2.imwrite(greyscale + pics, grey)    
    
###DONE    
    
#%%    
#####saving grey images with cropped labels  
for pics in os.listdir(greyscale):
    grey = cv2.imread(greyscale + pics)
    cbead = grey[0:2020, 0:2880]    

#print(grey.shape) #(2160, 2880)

    #plt.imshow(cbead)
    crop = img_as_ubyte(cbead)
    cv2.imwrite(cropped + pics, crop)

###DONE
    
#%%
######invert images with white background
blkbead1 = io.imread(cropped + "384-0.jpg")
blkbead2 = io.imread(cropped + "384-1.jpg")
blkbead3 = io.imread(cropped + "385-0.jpg")
blkbead4 = io.imread(cropped + "385-1.jpg")

blkback1 = util.invert(blkbead1)
bb1 = img_as_ubyte(blkback1)
cv2.imwrite(cropped + "i384-0.jpg", bb1)

blkback2 = util.invert(blkbead2)
bb2 = img_as_ubyte(blkback2)
cv2.imwrite(cropped + "i384-1.jpg", bb2)

blkback3 = util.invert(blkbead3)
bb3 = img_as_ubyte(blkback3)
cv2.imwrite(cropped + "i385-0.jpg", bb3)

blkback4 = util.invert(blkbead4)
bb4 = img_as_ubyte(blkback4)
cv2.imwrite(cropped + "i385-1.jpg", bb4)


#%%
###Inverted images have halo from shadow. Sharpen inverted images
for pics in os.listdir(cropped):
     if fnmatch.fnmatch(pics, "i*.jpg"):
         bbead = io.imread(toutput + pics, as_gray=True)
         sbead = filters.unsharp_mask(bbead, radius=6, amount=3)
         sharp = img_as_ubyte(sbead)
         cv2.imwrite(toutput + pics, sharp)

#put in cropped as s*.jpg

#%%
###darkening the background of inverted images 
for pics in os.listdir(cropped):
    if fnmatch.fnmatch(pics, "s*.jpg"):
        bead = io.imread(cropped + pics, as_gray=True)
        fill = morphology.flood_fill(bead, (150, 150), 0, tolerance=60)
        fbead = img_as_ubyte(fill)
        cv2.imwrite(toutput + pics, fbead)

#put in cropped as db*.jpg

#%%
###compare denoising methods
crop = cv2.imread(tcropped + "db385-1.jpg")
bbead = cv2.imread(binary + "db385-1.jpg")
fill = cv2.imread(SBPbeads + "6.fill binary/db385-1.jpg")
nlm = cv2.imread(SBPbeads + "6.nlm binary/db385-1.jpg") 
fillnlm = cv2.imread(SBPbeads + "6.fillnlm binary/db385-1.jpg")
nlmfill = cv2.imread(SBPbeads + "6.nlmfill binary/db385-1.jpg")

fig = plt.figure(figsize=(10,7))
rows=3
columns=3

fig.suptitle('Comparing Denoising Methods db385-1')

fig.add_subplot(rows, columns, 1)
plt.imshow(crop)
plt.title('Original')
plt.axis('off')

fig.add_subplot(rows, columns, 2)
plt.imshow(bbead)
plt.title('Yen')
plt.axis('off')

fig.add_subplot(rows, columns, 3)
plt.imshow(fill)
plt.title('Fill cracks')
plt.axis('off')

fig.add_subplot(rows, columns, 4)
plt.imshow(nlm)
plt.title('Non-local means')
plt.axis('off')

fig.add_subplot(rows, columns, 5)
plt.imshow(fillnlm)
plt.title('Fill + Non-local means')
plt.axis('off')

fig.add_subplot(rows, columns, 6)
plt.imshow(nlmfill)
plt.title('Non-local means + Fill')
plt.axis('off')


#%%
#####denoise cropped images
for pics in os.listdir(cropped):
    bead = img_as_float(io.imread(cropped + pics))
    sigma_est = np.mean(restoration.estimate_sigma(bead, multichannel=True))
    nlm_img = restoration.denoise_nl_means(bead, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6, multichannel=True)
    nlm = img_as_ubyte(nlm_img)
    cv2.imwrite(denoise + pics, nlm)


#%%
###fill cracks 
for pics in os.listdir(denoise):
    bead = io.imread(denoise + pics)
    darkerbead = morphology.opening(bead)
    bllibead = morphology.closing(darkerbead)
    ocbead = img_as_ubyte(bllibead)
    cv2.imwrite(nlmfill + pics, ocbead)


#%%
###test thresholding 
bbead4 = io.imread(toutput + "008-1.jpg", as_gray=True)
fig, ax = filters.try_all_threshold(bbead4, figsize=(10, 8), verbose=False)
plt.show()

#yen

#%%
##### saving image as binary using yen thresholding, label, fill holes, remove small objects
for pics in os.listdir(nlmfill):
    bead = cv2.imread(nlmfill + pics, 0)
    thresh = filters.threshold_yen(bead)
    binb = thresh <= bead
    label_bead = measure.label(binb, connectivity = None)
    nosmallo = morphology.remove_small_objects(label_bead, min_size=25000)
    nosmalloh = morphology.remove_small_holes(nosmallo, 15000)
    cleanbead = img_as_ubyte(nosmalloh)
    cv2.imwrite(nosmall + pics, cleanbead)

###omg it WOOOORKS

#%%
#####BATCH CANNY IT WORKS
for pics in os.listdir(nosmall):
    bead = cv2.imread(nosmall + pics, 0)
    canbead = canny(bead, sigma=5)
    cannybead = img_as_ubyte(canbead)
    cv2.imwrite(cancan + pics, cannybead)

##scatterplot the 3-5 to see .

#%%
##### converting canny to bead for smoother edge
for pics in os.listdir(cancan):
    beadcol = cv2.imread(cancan + pics)
    bead = cv2.imread(cancan + pics, 0)
    thresh = filters.threshold_otsu(bead) #because saving them as images doesn't preserve binary
    binb = thresh <= bead
    ahhh = binb.astype(np.uint8)
    binbead = 255 * ahhh
    contours, heirarchy = cv2.findContours(binbead, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #lines = cv2.fillPoly(binbead, contours, (255, 255, 255))
    lines = cv2.drawContours(beadcol, contours, -1, (255, 255, 255), 15)
    kernel = np.ones((3,3),np.uint8)
    nosmallo = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    connect = cv2.morphologyEx(nosmallo, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(linedance + pics, connect)

#%%
#####Filling in the contours for external shape
for pics in os.listdir(linedance):
    beadcol = cv2.imread(linedance + pics)
    bead = cv2.imread(linedance + pics, 0)
    contours, heirarchy = cv2.findContours(bead, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = cv2.drawContours(beadcol, contours, -1, (255, 255, 255), cv2.FILLED) 
    kernel = np.ones((5,5),np.uint8)
    ero = cv2.erode(lines.astype(np.uint8), kernel, iterations=2)
    erodil = cv2.dilate(ero, kernel, iterations=2)
    cv2.imwrite(erodils + pics, erodil)

#%%
#####Save fronts in front folder
for pics in os.listdir(erodils):
    if fnmatch.fnmatch(pics, "*-0.jpg"):
        fronts = io.imread(erodils + pics)
        #print(fronts)
        front = img_as_ubyte(fronts)
        cv2.imwrite(fefront + pics, front)
        


#%%
#####Save backs in back folder
for pics in os.listdir(erodils):
    if fnmatch.fnmatch(pics, "*-1.jpg"):
        backs = io.imread(erodils + pics)
        #print(backs)
        back = img_as_ubyte(backs)
        cv2.imwrite(feback + pics, back)


#%%
#####invert all for hole measurement
for pics in os.listdir(nosmall):
    bead = cv2.imread(nosmall + pics, 0)
    thresh = filters.threshold_otsu(bead) #because saving them as images doesn't preserve binary
    binb = thresh <= bead
    blkbead = util.invert(binb)
    nbhole = segmentation.clear_border(blkbead)
    bb = img_as_ubyte(nbhole)
    cv2.imwrite(invert + pics, bb)


#%%
#rinse and repeat
for pics in os.listdir(invert):
    bead = cv2.imread(invert + pics, 0)
    canbead = canny(bead, sigma=5)
    cannybead = img_as_ubyte(canbead)
    cv2.imwrite(cancan + pics, cannybead)

#%%
for pics in os.listdir(cancan):
    beadcol = cv2.imread(cancan + pics)
    bead = cv2.imread(cancan + pics, 0)
    thresh = filters.threshold_otsu(bead) #because saving them as images doesn't preserve binary
    binb = thresh <= bead
    ahhh = binb.astype(np.uint8)
    binbead = 255 * ahhh
    contours, heirarchy = cv2.findContours(binbead, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lines = cv2.drawContours(beadcol, contours, -1, (255, 255, 255), 2)
    kernel = np.ones((3,3),np.uint8)
    nosmallo = cv2.morphologyEx(lines.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    connect = cv2.morphologyEx(nosmallo, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(linedance + pics, connect)

#%%
#####Filling in the contours for external shape
for pics in os.listdir(linedance):
    beadcol = cv2.imread(linedance + pics)
    bead = cv2.imread(linedance + pics, 0)
    contours, heirarchy = cv2.findContours(bead, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = cv2.drawContours(beadcol, contours, -1, (255, 255, 255), cv2.FILLED) 
    kernel = np.ones((5,5),np.uint8)
    ero = cv2.erode(lines.astype(np.uint8), kernel, iterations=2)
    erodil = cv2.dilate(ero, kernel, iterations=2)
    cv2.imwrite(erodils + pics, erodil)


#%%
#####Save fronts in front folder
for pics in os.listdir(erodils):
    if fnmatch.fnmatch(pics, "*-0.jpg"):
        fronts = io.imread(erodils + pics)
        front = img_as_ubyte(fronts)
        cv2.imwrite(fhfront + pics, front)
        

#%%
#####Save backs in back folder
for pics in os.listdir(erodils):
    if fnmatch.fnmatch(pics, "*-1.jpg"):
        backs = io.imread(erodils + pics)
        back = img_as_ubyte(backs)
        cv2.imwrite(fhback + pics, back)


#%%
#MEASUREMENTS
pxl_mm = 1/184 #1mm = 184pxls

proplist = ['area',             
            'eccentricity',
            'centroid', 
            'major_axis_length', 
            'minor_axis_length',
            'perimeter'
            ]

outputf = open(SBPbeads + 'Batch 1 1-cont5-2fh.csv', 'w')
outputf.write('FileName' + "," + 'Region#' + "," + ",".join(proplist) + '\n')

for pics in os.listdir(fhback):
    bead = cv2.imread(fhback + pics, 0)
    thresh = filters.threshold_otsu(bead)
    binb = thresh <= bead
    labels = measure.label(binb, connectivity = None)
    
    props = measure.regionprops(labels)

    region_number = 1
    for beadprops in props:
        outputf.write(pics+",")
        outputf.write(str(beadprops['Label']))
        for i,prop in enumerate(proplist):
            if(prop == 'area'): 
                to_print = beadprops[prop]*pxl_mm**2   #Convert pixel to metric
            elif(prop == 'major_axis_length'): 
                to_print = beadprops[prop]*pxl_mm
            elif(prop == 'minor_axis_length'): 
                 to_print = beadprops[prop]*pxl_mm
            elif(prop == 'equivalent_diameter'): 
                 to_print = beadprops[prop]*pxl_mm
            elif(prop == 'perimeter'): 
                 to_print = beadprops[prop]*pxl_mm    
            else: 
                to_print = beadprops[prop]     
            outputf.write(',' + str(to_print))
        outputf.write('\n')
        region_number += 1

outputf.close()   #Closes the file, otherwise it would be read only.

