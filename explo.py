import os
import json

TargetLabel = "No"

def get_label_movements(label):
    movements = []
    fileList = os.listdir("data/"+TargetLabel+"/")
    for file in fileList:
        with open("data/"+label+"/"+file, 'r') as f:
            movements.append(json.load(f))
    return movements



def parse_vector(string_vect):
    return [float(coord) for coord in string_vect[1:-1].split(', ')]


if __name__ == "__main__":
    movements = get_label_movements(TargetLabel)
    print(len(movements))
    for mov in movements[:1]:
        for frame in mov["frames"]:
            for hand in frame["hands"]:
                #print(hand["handId"])
                for finger in hand["fingers"]:
                    #print(finger)
                    for bone in finger["Bones"]:
                        print(bone["BoneType"])
                        print(parse_vector(bone["Center"]))

    #print(movements[0]["frames"][0]["hands"][0]["Arm"]["Center"])
    #print(parse_vector(movements[0]["frames"][0]["hands"][0]["Arm"]["Center"]))
