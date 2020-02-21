from path import Path
import json

folder = Path('/home/rgasparini/Documents/ann/asd').dirs("BASLER_3-rgb*")

for fold in sorted(folder)[1:]:
    print(f"converting folder {fold}")
    json_folder = fold / ('JSON--'+fold.name) / 'outputs'
    image_folder = fold  / 'JPG'
    if not json_folder.exists():
        continue
    json_files = sorted(json_folder.files('*.json'))
    # for j in json_files:
    #     j.rename(j.parent / f"{int(j.split(' ')[1].split('.')[0]):04d}.json")
    image_files = sorted(image_folder.files('*.png'))
    # for j in image_files:
    #     j.rename(j.parent / f"{int(j.split(' ')[1].split('.')[0]):04d}.png")

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
            annotations.append(f"{f_id}\t{bb['xmin']}\t{bb['ymin']}\t{bb['xmax']}\t{bb['ymax']}\t{o['name']}\n")

    with open(fold / "annotations.txt", 'w') as file:
        for line in annotations:
            file.write(line)