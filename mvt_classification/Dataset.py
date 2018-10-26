import numpy as np
import os
import json
from pprint import pprint

class DataSet(object):
	def __init__(self, dirname, nbdata, L2normalize=False, batchSize=16):
		self.nbdata = nbdata
		# taille des images 48*48 pixels en niveau de gris
		self.dim = 768
		self.data = None
		self.label = None
		self.batchSize = batchSize
		self.curPos = 0
		self.MVT_NAMES = ["Fire", "Kick", "RotD", "RotG", "ZoomIn", "ZoomOut"]

		movements = []
		self.label = []
		fileList = os.listdir(dirname)
		for file in fileList:
			with open(dirname+file, 'r') as f:
				movements.append(json.load(f))
				self.label.append(file.split('_')[0])

		print("nb sequences: ", len(movements))

		self.data = []
		for mov in movements:
			hand_seq = []
			for frame in mov["frames"]:
				frame_seq = []
				for hand in frame["hands"]:
					frame_seq.append(self.get_hand_vector(hand))
				if(len(frame_seq) < 2):
					frame_seq.append(np.zeros((384,)))
				hand_seq.append(frame_seq)
			self.data.append(hand_seq)


		print(np.array(self.data).shape)

		self.data = np.array(self.data)
		self.label = np.array(self.label)
		tmpdata = np.empty([1, 60 , 2, self.dim], dtype=np.float32)
		tmplabel = np.empty([1, 1])#, dtype=np.float32)
		arr = np.arange(nbdata)
		np.random.shuffle(arr)
		tmpdata = self.data[arr[0],:]
		tmplabel = self.label[arr[0]]
		for i in range(nbdata-1):
			self.data[arr[i],:] = self.data[arr[i+1],:]
			self.label[arr[i]] = self.label[arr[i+1]]
		self.data[arr[nbdata-1],:] = tmpdata
		self.label[arr[nbdata-1]] = tmplabel
		print(self.label[:5])

		if L2normalize:
			self.data /= np.sqrt(np.expand_dims(np.square(self.data).sum(axis=1), 1))


	def NextTrainingBatch(self):
		if self.curPos + self.batchSize > self.nbdata:
			self.curPos = 0
		xs = self.data[self.curPos:self.curPos+self.batchSize,:]
		ys = self.label[self.curPos:self.curPos+self.batchSize,:]
		self.curPos += self.batchSize
		return xs,ys

	def parse_vector(self, string_vect):
	    # Separe le string sous format "(a, b, c)" en liste [a, b, c]
	    # on retire le premier et dernier char, les parenthèses
	    # on split le string sur les virgules, et on parse en float
	    return [float(coord) for coord in string_vect[1:-1].split(', ')]

	def get_hand_vector(self, hand):
		vector = []
		#Palm data
		vector.append(self.parse_vector(hand["PalmNormal"]))
		vector.append(self.parse_vector(hand["WristPosition"]))
		vector.append(self.parse_vector(hand["PalmPosition"]))
		vector.append(self.parse_vector(hand["PalmVelocity"]))
		vector.append(self.parse_vector(hand["Direction"]))

		#Arm data
		vector.append(self.parse_vector(hand["Arm"]["ElbowPosition"]))
		vector.append(self.parse_vector(hand["Arm"]["PrevJoint"]))
		vector.append(self.parse_vector(hand["Arm"]["NextJoint"]))
		vector.append(self.parse_vector(hand["Arm"]["Center"]))
		vector.append(self.parse_vector(hand["Arm"]["Direction"]))
		vector.append(self.parse_vector(hand["Arm"]["Rotation"]))

		#Finger data w/ 5 bones
		for finger in hand["fingers"]:
			for bone in finger["Bones"]:
				vector.append(self.parse_vector(bone["PrevJoint"]))
				vector.append(self.parse_vector(bone["NextJoint"]))
				vector.append(self.parse_vector(bone["Center"]))
				vector.append(self.parse_vector(bone["Direction"]))
				vector.append(self.parse_vector(bone["Rotation"]))
				#TODO add Bone_Type as vector from string

			vector.append(self.parse_vector(finger["Direction"]))
			vector.append(self.parse_vector(finger["TipPosition"]))
			#TODO add Finger_Type as vector from string
		#print(np.array(vector).shape)
		flat_list = [item for sublist in vector for item in sublist]

		return flat_list
