import numpy as np
import os
import json
from pprint import pprint
import pickle

class DataSet(object):

	def __init__(self, dirname, nbdata, L2normalize=False, batchSize=16, load=False, onehot=True):
		self.nbdata = nbdata
		# taille des images 48*48 pixels en niveau de gris
		self.dim = 549
		self.data = None
		self.label = None
		self.batchSize = batchSize
		self.curPos = 0
		self.MVT_NAMES = ["Fire", "Kick", "RotD", "RotG", "ZoomIn", "ZoomOut"]
		self.BONES = ["TYPE_METACARPAL", "TYPE_PROXIMAL", "TYPE_INTERMEDIATE", "TYPE_DISTAL"]
		self.FINGERS = ["TYPE_THUMB", "TYPE_INDEX", "TYPE_MIDDLE", "TYPE_RING", "TYPE_PINKY"]

		if os.path.isfile(dirname+"data.pkl") and os.path.isfile(dirname+"label.pkl") and load is True:
			#load the files
			print("Loaded data from files!")
			self.data = pickle.load(open(dirname+"data.pkl", "rb"))
			self.label = pickle.load(open(dirname+"label.pkl", "rb"))
		else:
			movements = []
			self.label = []
			fileList = os.listdir(dirname)
			for file in fileList:
				with open(dirname+file, 'r') as f:
					movements.append(json.load(f))
					if onehot is True:
						self.label.append([self.int2onehot(len(self.MVT_NAMES), self.MVT_NAMES.index(file.split('_')[0]))])#*60)
					else:
						self.label.append(self.MVT_NAMES.index(file.split('_')[0]))

			print("nb sequences loaded: ", len(movements))

			print("Vectorising sequences!")
			self.data = []
			for mov in movements:
				hand_seq = []
				for frame in mov["frames"]:
					frame_seq = []
					for hand in frame["hands"]:
						frame_seq.append(self.get_hand_vector(hand))
					if(len(frame_seq) < 2):
						frame_seq.append(np.zeros((self.dim,)))
					#hand_seq.append(frame_seq)
					hand_seq.append([item for sublist in frame_seq for item in sublist])
				self.data.append(hand_seq)


			print(np.array(self.data).shape)

			self.data = np.array(self.data)
			self.label = np.array(self.label)
			"""tmpdata = np.empty([1, 60 , 2*self.dim], dtype=np.float32)
			tmplabel = np.empty([1, 1])#, dtype=np.float32)
			arr = np.arange(nbdata)
			np.random.shuffle(arr)
			tmpdata = self.data[arr[0],:]
			tmplabel = self.label[arr[0]]
			for i in range(nbdata-1):
				self.data[arr[i],:] = self.data[arr[i+1],:]
				self.label[arr[i]] = self.label[arr[i+1]]
			self.data[arr[nbdata-1],:] = tmpdata
			self.label[arr[nbdata-1]] = tmplabel"""
			p = np.random.permutation(len(self.data))
			self.data=self.data[p]
			self.label=self.label[p]

			if onehot is True:
				# A single one hot vector for each sequence
				self.label = np.reshape(self.label, (len(self.label), len(self.MVT_NAMES)))

			pickle.dump(self.data, open(dirname+"data.pkl", 'wb'))
			pickle.dump(self.label, open(dirname+"label.pkl", 'wb'))



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
		if '(' in string_vect:
				return [float(coord) for coord in string_vect[1:-1].split(', ')]
		else:
			return [float(coord) for coord in string_vect.split(', ')]

	def int2onehot(self, list_len, index):
		one_hot = np.zeros((1, list_len))
		one_hot[np.arange(1), index] = 1

		return list(one_hot[0])

	def flatten(self, x):
			if isinstance(x, collections.Iterable):
					return [a for i in x for a in flatten(i)]
			else:
					return [x]

	def get_hand_vector(self, hand):
		vector = []
		#Palm data
		vector.append(self.parse_vector(hand["PalmNormal"]))
		vector.append(self.parse_vector(hand["WristPosition"]))
		vector.append(self.parse_vector(hand["PalmPosition"]))
		vector.append(self.parse_vector(hand["PalmVelocity"]))
		vector.append(self.parse_vector(hand["Direction"]))
		vector.append(self.parse_vector(hand["GrabAngle"]))
		vector.append(self.parse_vector(hand["GrabStrength"]))
		vector.append(self.parse_vector(hand["PalmWidth"]))
		vector.append(self.parse_vector(hand["PinchDistance"]))
		vector.append(self.parse_vector(hand["PinchStrength"]))
		vector.append(self.parse_vector(hand["StabilizedPalmPosition"]))

		#Arm data
		vector.append(self.parse_vector(hand["Arm"]["ElbowPosition"]))
		vector.append(self.parse_vector(hand["Arm"]["PrevJoint"]))
		vector.append(self.parse_vector(hand["Arm"]["NextJoint"]))
		vector.append(self.parse_vector(hand["Arm"]["Center"]))
		vector.append(self.parse_vector(hand["Arm"]["Direction"]))
		vector.append(self.parse_vector(hand["Arm"]["Rotation"]))
		vector.append(self.parse_vector(hand["Arm"]["Length"]))
		vector.append(self.parse_vector(hand["Arm"]["Width"]))

		#Finger data w/ 5 bones
		for finger in hand["fingers"]:
			for bone in finger["Bones"]:
				vector.append(self.parse_vector(bone["PrevJoint"]))
				vector.append(self.parse_vector(bone["NextJoint"]))
				vector.append(self.parse_vector(bone["Center"]))
				vector.append(self.parse_vector(bone["Direction"]))
				vector.append(self.parse_vector(bone["Rotation"]))
				vector.append(self.parse_vector(bone["Length"]))
				vector.append(self.parse_vector(bone["Width"]))
				vector.append(self.int2onehot(len(self.BONES), self.BONES.index(bone["BoneType"])))

			vector.append(self.parse_vector(finger["Direction"]))
			vector.append(self.parse_vector(finger["TipPosition"]))
			vector.append(self.parse_vector(finger["Length"]))
			vector.append(self.parse_vector(finger["Width"]))
			vector.append(self.int2onehot(len(self.FINGERS), self.FINGERS.index(finger["FingerType"])))
		#print(np.array(vector).shape)
		flat_list = [item for sublist in vector for item in sublist]

		return flat_list
