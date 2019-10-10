import threading
import cv2
import time


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
    def __init__(self, tot_frames, file_name):
        self.file_name = file_name
        self.file = open(file_name, 'w')
        self.total_frames = tot_frames
        self.current_frame = 0
        self.frame_objs = {}  # chiave: tupla (frane_number, x,y), valore: classe
        self.objs = []

    def update_obj(self, pos, val):
        self.frame_objs[pos] = val

    def remove_obj(self, pos):
        if pos in self.frame_objs:
            self.frame_objs.pop(pos)

    def reset(self):
        self.file = open(self.file_name, 'w')
        self.current_frame = 0
        self.frame_objs = {}  # chiave: tupla (frane_number, x,y), valore: classe
        self.objs = []

    def save_current(self):
        for obj in self.frame_objs:
            self.objs.append(obj)
        self.frame_objs = {}

    def save_all(self):
        self.save_current()
        for obj in self.objs:
            self.file.write(obj)
            self.file.write('\n')
        self.objs = []

    def __del__(self):
        self.save_all()
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
