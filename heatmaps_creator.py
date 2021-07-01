from path import Path
import cv2
from tqdm import tqdm
import numpy as np
import math
from heatmap_utils import draw_gaussian
import random
import json
import click

class_dict = {
    "Isolatore TE" : 0,
    "Tanica" : 1,
    "Segnale riflettente" : 2,
    "Mazza" : 3,
    "Piccone" : 4,
    "Fermascambio di sicurezza" : 5,
    "Binda" : 6,
    "Semaforo luminoso" : 7,
    "Fioretto" : 8,
    "Bombola GPL" : 9,
    "Boa" : 10,
    "Oliatore" : 11
}

def convert_from_json(path_frames):
    parent_dir = path_frames.parent
    path_dataset = parent_dir / "annotazioni"

    json_files = sorted(path_dataset.files('*.json'))

    annotations = []
    annotations.append("frame_id\tx1\ty1\tx2\ty2\tclasse\n")
    for jj in json_files:
        f = open(jj)
        jf = json.load(f)
        f.close()
        out = jf['outputs']
        if len(out) == 0:
            continue
        objs = out['object']
        for o in objs:
            bb = o['bndbox']
            f_id = jj.name.split('.json')[0]
            annotations.append(f"{f_id}\t{bb['xmin']}\t{bb['ymin']}\t{bb['xmax']}\t{bb['ymax']}\t{class_dict[o['name']]}\n")

    with open(path_dataset / parent_dir.name + '.txt', 'w') as file:
        for line in annotations:
            file.write(line)


def create_heatmaps_in_sequences(path_frames):
    parent_dir = path_frames.parent

    path_dataset = parent_dir / "annotazioni"

    path_output = parent_dir / "heatmaps"
    path_output_crops = parent_dir / "crops"

    frames_list = path_frames.files('*.jpg')

    path_output.makedirs_p()
    path_output_crops.makedirs_p()

    file = path_dataset / f"{parent_dir.name}.txt"
    anomaly_file_list = []
    anomaly_ids = []
    crops = [[0, []] for i in range(14)]
    normal_file_list = []

    normal_file_list.extend(frames_list)
    print(f"total images found {len(normal_file_list)}")
    print("Found {} annotated files...".format(len(file) -1 ))

    (path_output / "normal").makedirs_p()
    (path_output / "anomaly").makedirs_p()

    with open(file, "r") as f:
        _ = f.readline()    # skip first line, header
        lines = f.readlines()

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

            real_id = int(frame_id)

            frame = cv2.imread(path_frames / file.name[:-4] + str(real_id).zfill(5) + ".jpg", 0)

            detection = [x1, y1, x2, y2]

            crops[int(obj)][0] += 1
            crops[int(obj)][1].append({'name': f"{file.name[:-4] +'_'+ str(real_id).zfill(5)}", 'img':frame[y1:y2,x1:x2]})
            if random.randint(0,13) == 13:
                found = True
                counter = 0
                idn = random.randint(0, real_id)
                while idn in anomaly_ids:
                    idn = random.randint(0, real_id)
                    counter += 1
                    if counter >= 1000:
                        found = False
                        break
                if found:
                    imn = cv2.imread(path_frames / file.name[:-4] + str(idn).zfill(5) + ".jpg", 0)
                    crops[13][0] += 1
                    crops[13][1].append({'name': f"normal_crop_{crops[13][0]}.png", 'img':imn[y1:y2,x1:x2]})

            # DEBUG
            # tmp = frame.copy()
            # cv2.circle(tmp, (m_x, m_y), 5, (0, 0, 255), -1)
            # cv2.rectangle(tmp, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # cv2.imshow("frame", tmp)
            # cv2.waitKey(25)
            ########

            # new heatmap only if new detection is on another frame
            if frame_id != frame_id_old:
                hmap = np.zeros((frame.shape[0], frame.shape[1]))

            if frame_id != frame_id_next:
                i += 1

            if x2-x1 <0 or y2-y1 <0:
                print(f"skipped file {file} annotation {str(real_id).zfill(4)}")
                continue
            draw_gaussian(hmap, (m_x, m_y), sigma=int(math.sqrt(((x2 - x1) * (y2 - y1))) * 0.25))

            if frame_id != frame_id_next:
                # classi
                cv2.imwrite(path_output / "anomaly" / file.name[:-4] + "_" + str(i).zfill(5) + "_heatmap_" + ".png", (hmap * 255.).astype(np.uint8))
                cv2.imwrite(path_output / "anomaly" / file.name[:-4] + "_" + str(i).zfill(5) + '_anomaly_' + ".png", frame)
                anomaly_file_list.append(path_frames / file.name[:-4] + str(real_id).zfill(5) + ".jpg")
                anomaly_ids.append(real_id)

    normal_file_list = list(set(normal_file_list) - set(anomaly_file_list))
    print(f"total normal file {len(normal_file_list)}, total anomaly file {len(anomaly_file_list)} ")
    print("saving normal frames")
    for img in tqdm(normal_file_list):
        img.copy(path_output / "normal" / img.parent.name + "_" + img.name.replace(".jpg", "_normal_.jpg"))

    empty = 0
    for cri, cr in enumerate(crops):
        (path_output_crops / str(cri)).makedirs_p()
        for ci, c in enumerate(cr[1]):
            try:
                cv2.imwrite(path_output_crops / str(cri) / f"{c['name']}.png", c['img'])
            except:
                empty += 1
    print(f"{empty} empty images")



@click.command()
@click.option('--frames_path', type=str, required=True)
def main(frames_path):
    frames_path = Path(frames_path)
    convert_from_json(frames_path)
    create_heatmaps_in_sequences(frames_path)

if __name__ == '__main__':
    main()
