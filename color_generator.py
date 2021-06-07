import numpy as np

c = (0,100,50)
cc = {}
cc[c] = 0

with open("colors_hsl_track.yaml", 'w') as f:
    for i in range(1, 359):
        while c in cc:
            h = int(np.random.randint(0, 359, 1))
            c = (h,100,50)
        cc[c] = i

    for k1, v1 in cc.items():
        h,s,l = k1
        f.write(f"{v1}: {k1[0]},{k1[1]},{k1[2]}\n")

