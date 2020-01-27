import cv2
import numpy as np
import os
from path import Path
import subprocess
import click

ascii_list = ['1','2','3','4','5','6','7','8','9','0','q','w','e','r','t','y','u','i','o','p'] # add simbols for more classes

def class_divider(frame_list, class_dir):
    frame_list = sorted(frame_list)

    i = 0
    while i < frame_list.__len__():
        img = cv2.imread(frame_list[i])

        cv2.imshow("", img)
        k = cv2.waitKey()

        if k == ord('n'):  # NextFrame
            i = i + 1
        elif k == ord('b'):  # PrevFrame
            i = i - 1 if i > 0 else 0
        for a in ascii_list:
            if k==ord(a):
                (class_dir / a).makedirs_p()
                frame_list[i].copy(class_dir / a)
                i = i + 1
                break
        print(f"\r{i/frame_list.__len__()*100}", end='')

    cv2.destroyAllWindows()
    print("exited")


@click.command()
@click.option('--frames_path', type=str)
@click.option('--nclasses', type=int)
def main(frames_path, nclasses):
    assert nclasses <= len(ascii_list), 'add simbols to ascii_list list'

    frames_path = Path(frames_path)
    frames_list = frames_path.files('*.jpg')
    class_dir = frames_path.parent / "frames_by_classes"

    class_divider(frames_list, class_dir)


if __name__ == '__main__':
    main()
