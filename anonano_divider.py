import cv2
import numpy as np
import os
from path import Path
import subprocess
import click


def class_divider(frame_list, class_dir):
    frame_list = sorted(frame_list)
    to_move = []
    i = 0
    while i < frame_list.__len__():
        img = cv2.imread(frame_list[i])

        cv2.imshow("", img)
        k = cv2.waitKey()

        if k == ord('n'):  # NextFrame
            i = i + 1
        elif k == ord('b'):  # PrevFrame
            i = i - 1 if i > 0 else 0
        elif k == ord('a'):
            to_move.append(frame_list[i])
            i = i + 1

        print(f"\r{i / frame_list.__len__() * 100}", end='')

    to_move = set(to_move)
    for fr in to_move:
        fr.move(class_dir)
        fr.move(class_dir)

    cv2.destroyAllWindows()
    print("exited")


@click.command()
@click.option('--frames_path', type=str, required=True)
def main(frames_path):
    frames_path = Path(frames_path)
    frames_list = frames_path.files('*.png')
    class_dir = frames_path.parent / "ano"

    class_divider(frames_list, class_dir)


if __name__ == '__main__':
    main()
