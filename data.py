import os
import ctypes
import numpy


class TrainDataLoader(object):

    def __init__(self, 
                 in_path = "./",
                 tri_file = None,
                 ent_file = None,
                 rel_file = None,
                 batch_size = None,
                 nbatches = None,
                 threads = 8,
                 sampling_mode = "normal",
                 bren_flag = True,
                 filter_flag = True,
                 neg_ent = 1,
                 neg_rel = 0):

                 