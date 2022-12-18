from glob import glob
import os.path as osp
from pathlib import Path
import shutil
import sys

def extract_images(source_dir:str, output_dir:str):
    pattern = osp.join(source_dir,'*','*','Selfie*.jpg')
    files = glob(pattern)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for img in files:
        dir_name = osp.dirname(img)
        id_, _, age, _, name = str(Path(dir_name).stem).split("_")
        _, idx = str(Path(img).stem).split("_")
        file_name = f'Selfie_{name}_age_{age}_{idx}.jpg'
        dst = osp.join(output_dir, file_name)
        shutil.copyfile(img, dst)


if __name__ == "__main__":
    output_dir = 'data'
    source_dir = 'raw_data'
    if len(sys.argv) >2:
        output_dir = sys.argv[2]
        source_dir = sys.argv[1]

    extract_images(source_dir, output_dir)