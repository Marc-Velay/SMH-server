import os
import json
from pprint import pprint


TargetLabel = "No"

def get_label_movements(label):
    movements = []
    fileList = os.listdir("data/"+TargetLabel+"/")
    for file in fileList:
        with open("data/"+label+"/"+file, 'r') as f:
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
    #Palm data
    vector.append(parse_vector(hand["PalmNormal"]))
    vector.append(parse_vector(hand["WristPosition"]))
    vector.append(parse_vector(hand["PalmPosition"]))
    vector.append(parse_vector(hand["PalmVelocity"]))
    vector.append(parse_vector(hand["Direction"]))
    vector.append(parse_vector(hand["Arm"]["ElbowPosition"]))
    vector.append(parse_vector(hand["Arm"]["PrevJoint"]))
    vector.append(parse_vector(hand["Arm"]["NextJoint"]))
    vector.append(parse_vector(hand["Arm"]["Center"]))
    vector.append(parse_vector(hand["Arm"]["Direction"]))
    vector.append(parse_vector(hand["Arm"]["Rotation"]))
    print(vector)
    for finger in hand["fingers"]:
        for bone in finger["Bones"]:
            #print(bone["BoneType"])
            #print(parse_vector(bone["Center"]))
            pass

if __name__ == "__main__":
    movements = get_label_movements(TargetLabel)
    print(len(movements))
    for mov in movements[:1]:
        for frame in mov["frames"][:1]:
            #print(frame)
            for hand in frame["hands"]:
                #print(hand["handId"])
                hand_vector = get_hand_vector(hand)


    #print(movements[0]["frames"][0]["hands"][0]["Arm"]["Center"])
    #print(parse_vector(movements[0]["frames"][0]["hands"][0]["Arm"]["Center"]))
