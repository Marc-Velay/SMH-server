import os
import json
from pprint import pprint
import numpy as np
import time

TargetLabel = "RotG"
MVT_NAMES = ["Fire", "Kick", "No", "RotD", "RotG", "ZoomIn", "ZoomOut"]

def get_movements():
    movements = []
    fileList = os.listdir("data2/")
    for file in fileList:
        with open("data2/"+file, 'r') as f:
            movements.append(json.load(f))
    return movements



def parse_vector(string_vect):
    #Separe le string sous format "(a, b, c)" en liste [a, b, c]
    # on retire le premier et dernier char, les parenthèses
    # on split le string sur les virgules, et on parse en float
    return [float(coord) for coord in string_vect[1:-1].split(', ')]

def get_hand_vector(hand):
    vector = []
    pprint(hand)
    input()
    #Palm data
    vector.append(parse_vector(hand["PalmNormal"]))
    vector.append(parse_vector(hand["WristPosition"]))
    vector.append(parse_vector(hand["PalmPosition"]))
    vector.append(parse_vector(hand["PalmVelocity"]))
    vector.append(parse_vector(hand["Direction"]))

    #Arm data
    vector.append(parse_vector(hand["Arm"]["ElbowPosition"]))
    vector.append(parse_vector(hand["Arm"]["PrevJoint"]))
    vector.append(parse_vector(hand["Arm"]["NextJoint"]))
    vector.append(parse_vector(hand["Arm"]["Center"]))
    vector.append(parse_vector(hand["Arm"]["Direction"]))
    vector.append(parse_vector(hand["Arm"]["Rotation"]))

    #Finger data w/ 5 bones
    for finger in hand["fingers"]:
        for bone in finger["Bones"]:
            vector.append(parse_vector(bone["PrevJoint"]))
            vector.append(parse_vector(bone["NextJoint"]))
            vector.append(parse_vector(bone["Center"]))
            vector.append(parse_vector(bone["Direction"]))
            vector.append(parse_vector(bone["Rotation"]))
            #TODO add Bone_Type as vector from string

        vector.append(parse_vector(finger["Direction"]))
        vector.append(parse_vector(finger["TipPosition"]))
        #TODO add Finger_Type as vector from string
    #print(np.array(vector).shape)
    flat_list = [item for sublist in vector for item in sublist]
    #print(np.array(flat_list).shape)
    return flat_list


if __name__ == "__main__":

    mvts = []
    t0 = time.time()
    movement_seqs = get_movements()
    print('loaded in: %.3f seconds' %(time.time()-t0))
    print("nb sequences: ", len(movement_seqs))
    for mov in movement_seqs:
        hand_seq = []
        for frame in mov["frames"]:
            for hand in frame["hands"]:
                hand_seq.append(get_hand_vector(hand))
        #print(np.array(hand_seq).shape)
        mvts.append(hand_seq)
        #print(np.array(hand_seq).shape)
    print(np.array(mvts).shape)
