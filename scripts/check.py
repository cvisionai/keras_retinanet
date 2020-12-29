#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Tool to check annotations")
    parser.add_argument("work_csv")
    args = parser.parse_args()

    training_dir = os.path.dirname(args.work_csv)
    images_dir = os.path.join(training_dir, 'images')
    data=pd.read_csv(args.work_csv, header=None, names=['vid_path', 'x1','y1', 'x2','y2', 'species'])
    for idx,row in data.iterrows():
        image_path = os.path.join(images_dir, row.vid_path)
        image = plt.imread(image_path)
        plt.imshow(image)
        x=np.array([row.x1, row.x2])
        y=np.array([row.y1, row.y2])
        plt.plot(x,y, 'r')
        plt.title(row.vid_path)
        plt.show()
