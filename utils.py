import threading
import cv2
import time
from path import Path
import yaml

def readColors():
    color_class_dict = {}
    if not Path("colors_hsl.yaml").exists():
        raise FileNotFoundError("Unable to find colors_hsl.yaml, please run color_generator.py")
    with open("colors_hsl.yaml") as f:
        yml = yaml.load(f, yaml.Loader)
        for k,v in yml.items():
            v = tuple([int(v1) for v1 in v.split(',')])
            color_class_dict[k] = tuple(v)
    return color_class_dict

def hls2rgb(hsl):
    h, s ,l = hsl
    s /= 100
    l /= 100
    c = (1 - abs(2 * l - 1)) * s
    k = h / 60
    x = c * (1 - abs(k % 2 - 1))
    r1 = g1 = b1 = 0
    if (k >= 0 and k <= 1):
        r1=c
        g1=x
    if (k > 1 and k <= 2):
        r1=x
        g1=c
    if (k > 2 and k <= 3):
        g1=c
        b1=x
    if (k > 3 and k <= 4):
        g1=x
        b1=c
    if (k > 4 and k <= 5):
        r1=x
        b1=c
    if (k > 5 and k <= 6):
        r1=c
        b1=x
    m = l - c / 2

    return (int((r1 + m)*255), int((g1 + m)*255), int((b1 + m)*255))

class ClickPos:
    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.click = 0

    def getPos(self):
        """
        :return: Pos (x,y) or None if need a second click
        """
        self.click += 1


class Annotator:
    def __init__(self, tot_frames, file_name, scale):
        self.file_name = file_name
        self.file = open(file_name, 'w')
        self.file.write("frame_id\tx1\ty1\tx2\ty2\tclasse\n")
        self.total_frames = tot_frames
        self.current_frame = 0
        self.frame_objs = {}  # chiave: tupla (frane_number, x1,y1,x2,y2), valore: classe
        self.objs = {}
        self.scale_factor = scale

    def update_obj(self, pos, val):
        x1 = pos[1] / self.scale_factor
        y1 = pos[2] / self.scale_factor
        x2 = pos[3] / self.scale_factor
        y2 = pos[4] / self.scale_factor
        new_pos = (pos[0], x1, y1, x2, y2)
        self.frame_objs[new_pos] = val

    def remove_obj(self, pos):
        if pos in self.frame_objs:
            self.frame_objs.pop(pos)

    def reset(self):
        self.file = open(self.file_name, 'w')
        self.current_frame = 0
        self.frame_objs = {}  # chiave: tupla (frane_number, x1,y1,x2,y2), valore: classe
        self.objs = {}

    def save_current(self):
        for key, value in self.frame_objs.items():
            self.objs[key] = value
        self.frame_objs = {}

    def save_all(self):
        self.save_current()
        for obj, val in self.objs.items():
            self.file.write(f"{obj[0]}\t{obj[1]}\t{obj[2]}\t{obj[3]}\t{obj[4]}\t{val}\n")
            self.file.write("{val}\n")
        self.objs = {}
        self.file.close()


class VideoLoader(threading.Thread):
    def __init__(self, file, batch_size=300, max_width=1280):
        super(VideoLoader, self).__init__()
        self.file = file
        self.tot_frames = 0
        self.frame_loaded = 0
        self.batch_size = batch_size
        self.current_batch_frame = 0
        self.max_width = max_width
        self.batch_imgs = []
        self.scale_factor = None
        self.batch_loaded = False
        self.load()

    def load(self):
        self.reader = cv2.VideoCapture(self.file)
        self.tot_frames = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.reader.get(cv2.CAP_PROP_FPS))
        self.video_heigth = int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_width = int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(
            f"video {self.file}:\n {self.tot_frames} frames, {time.strftime('%H:%M:%S', time.gmtime(self.tot_frames // self.fps))}, {self.fps} fps, {self.video_width}x{self.video_heigth}")

        if self.max_width is not None:
            self.scale_factor = self.max_width / self.video_width

        self.get_next_batch()

    def get_next_img(self):
        if self.current_batch_frame >= len(self.batch_imgs) - 1:
            self.get_next_batch()
            if not self.batch_loaded or self.frame_loaded >= self.tot_frames - 1:
                return None
        img = self.batch_imgs[self.current_batch_frame]
        self.current_batch_frame += 1
        return img

    def get_next_batch(self):
        self.batch_imgs = []
        self.batch_loaded = False
        self.current_batch_frame = 0
        batch_frame_loaded = 0

        ret, frame = self.reader.read()
        if ret:
            frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
            batch_frame_loaded += 1
            self.frame_loaded += 1
            self.batch_imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        while ret and self.frame_loaded < self.tot_frames and batch_frame_loaded < self.batch_size:
            ret, frame = self.reader.read()
            if ret:
                frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
                batch_frame_loaded += 1
                self.frame_loaded += 1
                self.batch_imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.batch_loaded = True
