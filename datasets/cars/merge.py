import os
import numpy as np
from scipy.io import loadmat
import pandas as pd


def main():

    data = loadmat('cars_annos.mat')
    annotations = data['annotations']
    class_names = data['class_names']
    filename = []
    labels = []

    # for i in range(annotations.shape[1]):
    #     name = str(annotations[0,i][0])[2:-2]
    #     cls = int(annotations[0,i][5])-1
    #     filename.append(name)
    #     labels.append(cls)

    # df = pd.DataFrame({"filename":filename,"label":labels})
    # df.to_csv("all.csv",index=False, sep=',')
    inds = []
    cls_name = []
    for i in range(class_names.shape[1]):
        inds.append(i)
        cls_name.append(str(class_names[0,i][0]))
    df1 = pd.DataFrame({"ids":inds,"class":cls_name})
    df1.to_csv("class.cvs",index=False)


if __name__ == '__main__':
    main()
