from path import Path
import json

path = Path('/media/rgasparini/Volume/RFI_DATA/new_basler_3')
folder = path.dirs("JSON--BASLER_3-rgb*")

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

annotation_dir = path / "annotazioni"
annotation_dir.makedirs_p()
# frame_dir = path / "frames"
# frame_dir.makedirs_p()

for fold in sorted(folder)[1:]:
    print(f"converting folder {fold}")
    json_folder = fold / 'outputs'

    # current_fold = frame_dir / fold.name
    # current_fold.makedirs_p()

    if not json_folder.exists():
        continue
    json_files = sorted(json_folder.files('*.json'))\
    # for j in json_files:
    #     j.rename(j.parent / f"{int(j.split(' ')[1].split('.')[0]):04d}.json")
    # image_files = sorted(image_folder.files('*.png'))
    # for j in image_files:
    #     j.rename(j.parent / f"{int(j.split(' ')[1].split('.')[0]):04d}.png")

    # for im in image_files:
    #     im.copy(current_fold)

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

    with open(annotation_dir / fold.name + '.txt', 'w') as file:
        for line in annotations:
            file.write(line)