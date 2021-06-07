import cv2
import numpy as np
from path import Path
import tqdm
from skimage.metrics import structural_similarity as ssim
import click

def trash(files, out_dir, ssth):
    files = sorted(files)
    out_dir.makedirs_p()
    im0 = cv2.imread(files[0],0)
    for fi in tqdm.tqdm(files[1:]):
        img = cv2.imread(fi,0)
        ss = ssim(im0, img, data_range=img.max() - img.min())
        im0 = img.copy()

        if ss > ssth:
            continue
        else:
            fi.copy(out_dir)



@click.command()
@click.option('--frames_path', type=str, required=True)
@click.option('--ssim_th', type=float, default=0.90)
def main(frames_path, ssim_th):
    frames_path = Path(frames_path)
    frames_list = frames_path.files('*.jpg')
    class_dir = frames_path.parent / f"out_ssimth_{str(ssim_th).replace('.','')}"

    trash(frames_list, class_dir, ssim_th)


if __name__ == '__main__':
    main()
