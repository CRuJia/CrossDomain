from PIL import Image
import os
import pandas as pd

DATA_DIR = "/home/lemon/Datasets"

def judge(img_path):
    img = Image.open(img_path).convert('RGB')
    try:
        assert len(img.split())==3
    except:
        print(img_path)

def main():
    files = ["CUB_200_2011/all.csv",""]
    for f in files:
        print(f)
        df = pd.read_csv(os.path.join(DATA_DIR, f))
        images_path = list(df['filename'])
        for img in images_path:
            judge(img)

if __name__ == '__main__':
    main()