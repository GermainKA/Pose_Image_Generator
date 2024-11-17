
import numpy as np
import cv2
import os
import pickle
import sys
import math
import time

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        #generator of image from skeleton 
        # compute the distance between the new skeleton and all the skeletons in the videoSkeTgt
        distances = []
        for i in range(len(self.videoSkeletonTarget.ske)):
            distances.append(ske.distance(self.videoSkeletonTarget.ske[i]))
        
        # find the index of the skeleton in videoSkeTgt that is closest to the new skeleton
        idx_nearest = np.argmin(distances)
        
        tgt_nearest = self.videoSkeletonTarget.readImage(idx_nearest)
        tgt_nearest = cv2.cvtColor(tgt_nearest, cv2.COLOR_BGR2RGB)
        return tgt_nearest

    def m_generate(self, ske):           
        #generator of image from skeleton 
        # compute the distance between the new skeleton and all the skeletons in the videoSkeTgt
        #start = time.time()
        distances = np.array([np.linalg.norm([np.array(obj.ske) - np.array(ske.ske)]) for obj in self.videoSkeletonTarget.ske])
        idx_nearest = np.argmin(distances)
        tgt_nearest = self.videoSkeletonTarget.readImage(idx_nearest)
        tgt_nearest = cv2.cvtColor(tgt_nearest, cv2.COLOR_BGR2RGB)
        #end = time.time()
        #print("Time to compute the nearest neighbor: ", (end - start))
        return tgt_nearest