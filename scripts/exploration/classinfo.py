from collections import defaultdict

class ClassInfo(object):
    def __init__(self):
        self.class_name = None
        self.class_counter = 0
        self.aspect_ratios = []
        self.areas = []
        self.imgs = []
        self.dict_iou_class = defaultdict(list)