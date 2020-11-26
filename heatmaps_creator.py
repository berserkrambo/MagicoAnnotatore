from path import Path
import cv2
from tqdm import tqdm
import numpy as np
import math
from heatmap_utils import draw_gaussian
import random


def create_heatmaps_in_sequences():
    # path = Path("/media/rgasparini/Volume/RFI_DATA/BOSON")
    path = Path("/media/rgasparini/Volume/RFI_DATA/new_basler_3")
    path_dataset = path / "annotazioni"
    path_frames = path / "frames"
    path_output = path / "heatmaps"
    path_output_crops = path / "crops"
    # path_normal_frames = path / "dataset_A/train/normal"
    # path_normal_frames_files = path_normal_frames.files("*.png")

    path_output.makedirs_p()
    path_output_crops.makedirs_p()

    files = sorted(path_dataset.files("*.txt"))
    anomaly_file_list = []
    anomaly_ids = []
    crops = [[0, []] for i in range(14)]
    normal_file_list = []

    for dir in path_frames.dirs():
        normal_file_list.extend(dir.files('*.png'))
    print(f"total images found {len(normal_file_list)}")
    print("Found {} annotated files...".format(len(files)))

    for cls in range(0, 13):
        (path_output / "classi" / str(cls)).makedirs_p()
    (path_output / "classi" / "normal").makedirs_p()

    for file in tqdm(files):
        with open(file, "r") as f:
            _ = f.readline()
            lines = f.readlines()

            # dir_out_hmap = path_output / file.name[:-4] / "heatmaps"
            # dir_out_hmap.makedirs_p()
            #
            # dir_out_frames = path_output / file.name[:-4] / "frames"
            # dir_out_frames.makedirs_p()
            #
            # dir_out_demos = path_output / file.name[:-4] / "demos"
            # dir_out_demos.makedirs_p()

            i = 0

            for index, line in enumerate(lines):
                line = line.split()
                frame_id = line[0]
                x1 = int(line[1])
                y1 = int(line[2])
                x2 = int(line[3])
                y2 = int(line[4])
                obj = line[5]

                m_x = int((x2 + x1) / 2)
                m_y = int((y2 + y1) / 2)

                if index < len(lines) - 1:
                    line_next = lines[index + 1].split()
                    frame_id_next = line_next[0]
                else:
                    frame_id_next = -1

                if index > 0:
                    line_old = lines[index - 1].split()
                    frame_id_old = line_old[0]
                else:
                    frame_id_old = -1

                if file.find("BOSON") >= 0:
                    real_id = (int(frame_id) // 2)
                elif file.find("BASLER_3") >= 0:
                    real_id = int(frame_id)

                frame = cv2.imread(path_frames / file.name[:-4] / str(real_id).zfill(4) + ".png", 0)

                detection = [x1, y1, x2, y2]

                # crops[int(obj)][0] += 1
                # crops[int(obj)][1].append({'name': f"{file.name[:-4] +'_'+ str(real_id).zfill(4)}", 'img':frame[y1:y2,x1:x2]})
                # if random.randint(0,13) == 13:
                #     found = True
                #     counter = 0
                #     idn = random.randint(0, real_id)
                #     while idn in anomaly_ids:
                #         idn = random.randint(0, real_id)
                #         counter += 1
                #         if counter >= 1000:
                #             found = False
                #             break
                #     if found:
                #         imn = cv2.imread(path_frames / file.name[:-4] / str(idn).zfill(4) + ".png", 0)
                #         crops[13][0] += 1
                #         crops[13][1].append({'name': f"normal_crop_{crops[13][0]}.png", 'img':imn[y1:y2,x1:x2]})


                # tmp = cv2.circle(frame, (m_x, m_y), 5, (0, 0, 255), -1)
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

                # DEBUG
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                # cv2.imshow("frame", tmp)
                # cv2.waitKey(25)

                # new heatmap only if new detection is on another frame
                if frame_id != frame_id_old:
                    hmap = np.zeros((frame.shape[0], frame.shape[1]))
                    # f = open(path_output / "classi" / str(obj) / file.name[:-4] + "_" + str(i).zfill(4) + ".txt", "w")
                # elif frame_id == frame_id_old:
                #     print(frame_id, frame_id_old)

                if frame_id != frame_id_next:
                    i += 1

                if x2-x1 <0 or y2-y1 <0:
                    print(f"skipped file {file} annotation {str(real_id).zfill(4)}")
                    continue
                draw_gaussian(hmap, (m_x, m_y), sigma=int(math.sqrt(((x2 - x1) * (y2 - y1))) * 0.25))

                if frame_id != frame_id_next:
                    # classi
                    # f.close()
                    cv2.imwrite(path_output / "classi" / str(obj) / file.name[:-4] + "_" + str(i).zfill(4) + "_heatmap_" + ".png", (hmap * 255.).astype(np.uint8))
                    cv2.imwrite(path_output / "classi" / str(obj) / file.name[:-4] + "_" + str(i).zfill(4) + '_anomaly_' + ".png", frame)
                    anomaly_file_list.append(path_frames / file.name[:-4] / str(real_id).zfill(4) + ".png")
                    anomaly_ids.append(real_id)

    normal_file_list = list(set(normal_file_list) - set(anomaly_file_list))
    print(f"total normal file {len(normal_file_list)}, total anomaly file {len(anomaly_file_list)} ")
    print("saving normal frames")
    for img in tqdm(normal_file_list):
        img.copy(path_output / "classi" / "normal" / img.parent.name + "_" + img.name.replace(".png", "_normal_.png"))

    # empty = 0
    # for cri, cr in enumerate(crops):
    #     (path_output_crops / str(cri)).makedirs_p()
    #     for ci, c in enumerate(cr[1]):
    #         try:
    #             cv2.imwrite(path_output_crops / str(cri) / f"{c['name']}.png", c['img'])
    #         except:
    #             empty += 1
    # print(f"{empty} empty images")


if __name__ == "__main__":
    # extract_annotations()
    # extract_patches()
    # extract_patches_no_anomaly()
    # extract_frames(path_dataset)
    # create_heatmaps()
    create_heatmaps_in_sequences()
    # view_heatmaps()
