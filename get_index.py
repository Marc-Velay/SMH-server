import numpy as np
import os, io
import json
from pprint import pprint
import pickle


def parse_vector(string_vect):
		# Separe le string sous format "(a, b, c)" en liste [a, b, c]
		# on retire le premier et dernier char, les parenthèses
		# on split le string sur les virgules, et on parse en float
	if '(' in string_vect:
			return [float(coord) for coord in string_vect[1:-1].split(', ')]
	else:
		return [float(coord) for coord in string_vect.split(', ')]

def int2onehot(list_len, index):
	one_hot = np.zeros((1, list_len))
	one_hot[np.arange(1), index] = 1

	return list(one_hot[0])

def flatten(x):
		if isinstance(x, collections.Iterable):
				return [a for i in x for a in flatten(i)]
		else:
				return [x]

def get_hand_vector(hand):
    vector = []
    #Palm data
    for idx, val in enumerate(parse_vector(hand["PalmNormal"])):
        vector.append("PalmNormal"+str(idx))
    for idx, val in enumerate(parse_vector(hand["WristPosition"])):
        vector.append("WristPosition"+str(idx))
    for idx, val in enumerate(parse_vector(hand["PalmPosition"])):
        vector.append("PalmPosition"+str(idx))
    for idx, val in enumerate(parse_vector(hand["PalmVelocity"])):
        vector.append("PalmVelocity"+str(idx))
    for idx, val in enumerate(parse_vector(hand["Direction"])):
        vector.append("Direction"+str(idx))
    for idx, val in enumerate(parse_vector(hand["GrabAngle"])):
        vector.append("GrabAngle"+str(idx))
    for idx, val in enumerate(parse_vector(hand["GrabStrength"])):
        vector.append("GrabStrength"+str(idx))
    for idx, val in enumerate(parse_vector(hand["PalmWidth"])):
        vector.append("PalmWidth"+str(idx))
    for idx, val in enumerate(parse_vector(hand["PinchDistance"])):
        vector.append("PinchDistance"+str(idx))
    for idx, val in enumerate(parse_vector(hand["PinchStrength"])):
        vector.append("PinchStrength"+str(idx))
    for idx, val in enumerate(parse_vector(hand["StabilizedPalmPosition"])):
        vector.append("StabilizedPalmPosition"+str(idx))



    for idx, val in enumerate(parse_vector(hand["Arm"]["ElbowPosition"])):
        vector.append("StabilizedPalmPosition"+str(idx))
    for idx, val in enumerate(parse_vector(hand["Arm"]["PrevJoint"])):
        vector.append("Arm_PrevJoint"+str(idx))
    for idx, val in enumerate(parse_vector(hand["Arm"]["NextJoint"])):
        vector.append("Arm_NextJoint"+str(idx))
    for idx, val in enumerate(parse_vector(hand["Arm"]["Center"])):
        vector.append("Arm_Center"+str(idx))
    for idx, val in enumerate(parse_vector(hand["Arm"]["Direction"])):
        vector.append("Arm_Direction"+str(idx))
    for idx, val in enumerate(parse_vector(hand["Arm"]["Rotation"])):
        vector.append("Arm_Rotation"+str(idx))
    for idx, val in enumerate(parse_vector(hand["Arm"]["Length"])):
        vector.append("Arm_Length"+str(idx))
    for idx, val in enumerate(parse_vector(hand["Arm"]["Width"])):
        vector.append("Arm_Width"+str(idx))

	#Finger data w/ 5 bones
    for idx_f, finger in enumerate(hand["fingers"]):
        for idx_b, bone in enumerate(finger["Bones"]):
            for idx, val in enumerate(parse_vector(bone["PrevJoint"])):
                vector.append(str(finger["FingerType"])+"_"+str(bone["BoneType"])+"_"+"PrevJoint"+str(idx))
            for idx, val in enumerate(parse_vector(bone["NextJoint"])):
                vector.append(str(finger["FingerType"])+"_"+str(bone["BoneType"])+"_"+"NextJoint"+str(idx))
            for idx, val in enumerate(parse_vector(bone["Center"])):
                vector.append(str(finger["FingerType"])+"_"+str(bone["BoneType"])+"_"+"Center"+str(idx))
            for idx, val in enumerate(parse_vector(bone["Direction"])):
                vector.append(str(finger["FingerType"])+"_"+str(bone["BoneType"])+"_"+"Direction"+str(idx))
            for idx, val in enumerate(parse_vector(bone["Rotation"])):
                vector.append(str(finger["FingerType"])+"_"+str(bone["BoneType"])+"_"+"Rotation"+str(idx))
            for idx, val in enumerate(parse_vector(bone["Length"])):
                vector.append(str(finger["FingerType"])+"_"+str(bone["BoneType"])+"_"+"Length"+str(idx))
            for idx, val in enumerate(parse_vector(bone["Width"])):
                vector.append(str(finger["FingerType"])+"_"+str(bone["BoneType"])+"_"+"Width"+str(idx))
            for idx, val in enumerate(int2onehot(len(BONES), BONES.index(bone["BoneType"]))):
                vector.append(str(finger["FingerType"])+"_"+str(bone["BoneType"])+"_"+"BONETYPE"+str(idx))

        for idx, val in enumerate(parse_vector(finger["Direction"])):
            vector.append(str(finger["FingerType"])+"_"+"Direction"+str(idx))
        for idx, val in enumerate(parse_vector(finger["TipPosition"])):
            vector.append(str(finger["FingerType"])+"_"+"TipPosition"+str(idx))
        for idx, val in enumerate(parse_vector(finger["Length"])):
            vector.append(str(finger["FingerType"])+"_"+"Length"+str(idx))
        for idx, val in enumerate(parse_vector(finger["Width"])):
            vector.append(str(finger["FingerType"])+"_"+"Width"+str(idx))
        for idx, val in enumerate(int2onehot(len(FINGERS), FINGERS.index(finger["FingerType"]))):
            vector.append(str(finger["FingerType"])+"_"+"FINGERTYPE"+str(idx))

    return vector

MVT_NAMES = ["Fire", "Kick", "RotD", "RotG", "ZoomIn", "ZoomOut", "NOP"]
BONES = ["TYPE_METACARPAL", "TYPE_PROXIMAL", "TYPE_INTERMEDIATE", "TYPE_DISTAL"]
FINGERS = ["TYPE_THUMB", "TYPE_INDEX", "TYPE_MIDDLE", "TYPE_RING", "TYPE_PINKY"]

dim = 549

datadir="data2/"

fileList = os.listdir(datadir)
with open(datadir+fileList[0], 'r') as f:
	hand_data = json.load(f)

named_vars = []
for hand in hand_data["frames"][0]["hands"]:
	named_vars.append(get_hand_vector(hand))
named_vars = [item for sublist in named_vars for item in sublist]
print(named_vars[600])
