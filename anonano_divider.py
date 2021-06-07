import cv2
from path import Path

import click


def class_divider(frame_list, class_dir, image_crop):
    crop_x, crop_y, crop_w, crop_h = [int(i) for i in image_crop.split(',')]

    anomaly_dir = class_dir / "anomaly"
    normal_dir = class_dir / "normal"
    anomaly_dir.makedirs_p()
    normal_dir.makedirs_p()

    frame_list = sorted(frame_list)
    anomaly_list = [0 for i in range(0, frame_list.__len__())]

    i = 0
    while i < frame_list.__len__():
        img = cv2.imread(frame_list[i])
        
        to_view = img.copy()
        cv2.rectangle(to_view,(crop_x,crop_y),(crop_x+crop_w,crop_y+crop_h),[0,0,255], 1)
        cv2.imshow("", to_view)
        k = cv2.waitKey()

        if k == 83:  # RightKey
            if i == frame_list.__len__() - 1:
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
        elif k == ord('q'):
            print("exiting")
            quit()

        print(f"\r{i / frame_list.__len__() * 100}", end='')

    cv2.destroyAllWindows()
    print("writing frames")

    for i, vi in enumerate(anomaly_list):
        nm = frame_list[i].name
        if vi == 0:
            nm = nm.replace(".jpg","_normal_.jpg")
            frame_list[i].copy(normal_dir/nm)
        elif vi == 1:
            nm = nm.replace(".jpg", "_anomaly_.jpg")
            frame_list[i].copy(anomaly_dir / nm)

    print("exited")


@click.command()
@click.option('--frames_path', type=str, required=True)
@click.option('--image_crop', type=str, default='-1,-1,-1,-1')
def main(frames_path, image_crop):
    frames_path = Path(frames_path)
    frames_list = frames_path.files('*.jpg')
    class_dir = frames_path.parent

    class_divider(frames_list, class_dir, image_crop)


if __name__ == '__main__':
    main()
