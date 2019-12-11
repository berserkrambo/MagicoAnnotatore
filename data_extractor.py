import cv2
import numpy as np
import os
from path import Path
import subprocess
import click


def extractor(path, crop):
    path_out = path / 'frames'
    path_out.makedirs_p()

    if any([i == -1 for i in crop]):
        process = subprocess.Popen(['ffmpeg', '-i', f'{path}.avi', '-q:v', '1', f'{path_out / path.name}%05d.jpg'])
    else:
        w,h,x,y = crop[0], crop[1], crop[2], crop[3]
        process = subprocess.Popen(['ffmpeg', '-i', f'{path}.avi', '-vf', f'crop={w}:{h}:{x}:{y}', '-q:v', '1', f'{path_out/path.name}%05d.jpg'])
    process.wait()

    return path_out


def divider(path):
    tmp_nmap = Path("tmp_nmap")
    imgs_list = sorted(path.files())

    tm = cv2.imread(imgs_list[0])
    b,h,w,c = imgs_list.__len__(), tm.shape[0], tm.shape[1], tm.shape[2]
    imgs = np.memmap("tmp_nmap", dtype='uint8', mode='w+', shape=(b,h,w,c))

    anomaly_path = path.parent / f'{path.parent.name}_anomaly'
    normal_path = path.parent / f'{path.parent.name}_normal'

    anomaly_list = [0 for i in range(0, imgs_list.__len__())]
    i = 0
    while i < imgs_list.__len__():
        imgs[i] = cv2.imread(imgs_list[i])
        img = imgs[i].copy()
        cv2.imshow(f"{path.name}, img", img)
        k = cv2.waitKey()

        if k == 83:  # RightKey
            if i == len(imgs) - 1:
                continue    # gli faccio premere per forza 'a' o 'n'
            i = i + 1
        elif k == 81:  # LeftKey
            i = i - 1 if i > 0 else 0
        elif k == ord('a'):
            anomaly_list[i] = 1
            i = i + 1
        elif k == ord('n'):
            anomaly_list[i] = 0
            i = i + 1

    cv2.destroyAllWindows()
    print("writing frames")

    ima = [ii for ii in anomaly_list if ii == 1]
    imn = [ii for ii in anomaly_list if ii == 0]

    np.savez_compressed(normal_path,data=np.take(imgs, imn, axis=0), shape=(b,h,w,c))
    np.savez_compressed(anomaly_path,data=np.take(imgs, ima, axis=0), shape=(b,h,w,c))

    tmp_nmap.remove()
    del tmp_nmap

    print("exited")


@click.command()
@click.option('--video_path', type=str)
@click.option('--crop', nargs=4, type=int, default=[-1,-1,-1,-1])
@click.option('--extract_npz', is_flag=True, default=False)
def main(video_path, crop, extract_npz):
    video_path = Path(video_path)
    done_videos = Path(video_path / 'done')
    done_videos.makedirs_p()
    for i in Path(video_path).files('*.avi'):
        path = Path(i.replace(".avi", ""))
        path.makedirs_p()
        path_frames = extractor(path, crop)
        divider(path_frames)
        i.move(done_videos / i.name)

    
if __name__ == '__main__':

    main()
