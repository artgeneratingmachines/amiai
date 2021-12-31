#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 10:16:13 2021

@author: halidziya
"""
import numpy as np
import tqdm
import json
import cv2
from PIL import Image

from tensorflow.keras.datasets import mnist
data = mnist.load_data()
data = np.concatenate([data[0][0],data[1][0]])
intensities = np.mean(np.mean(data, axis=1), axis=1)
intensities[intensities>80] = 80
normalized = 255*(intensities-np.min(intensities))/(np.max(intensities)-np.min(intensities))
indices = [np.argmin(np.abs(normalized-i)) for i in range(255)]
mapper = data[indices]




path = 'amiai.mp4'
cap = cv2.VideoCapture(path)

frames = []
ret = True
while ret:
    ret, img = cap.read()
    if not ret:
        break
    frame = np.mean(img, axis=2).astype(int)
    newframe = np.zeros(np.array(frame.shape), dtype=np.uint8)
    for i in tqdm.tqdm(range(frame.shape[0]//28)):
        for j in range(frame.shape[1]//28):
            newframe[(i*28):((i+1)*28),(j*28):((j+1)*28)] = \
            mapper[int(np.round(np.mean(frame[(i*28):((i+1)*28),(j*28):((j+1)*28)])))]
    frames.append(newframe)
    
im = Image.fromarray(newframe)
im.save('amiai.png')



size = newframe.shape
duration = 2
fps = 30
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
for frame in frames:
    out.write(frame)
out.release()

with open('mapper.json', 'w') as fil:
    json.dump(mapper.tolist(), fil, indent=2)
    
"Video by Pavel Danilyuk from Pexels"