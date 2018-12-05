from mvt_classification import Layers
from mvt_classification import Dataset
import numpy as np
import itertools
import pickle

vector_length = 1098
nb_frames = 30
nb_classes = 7
X_shape = (1, nb_frames, vector_length)
y_shape = (1, nb_classes)
batchSize = 1

model,dump  = Layers.create_LSTM3(X_shape, y_shape, batchSize)

filepath="mvt_classification/weights/weights-lstm4.hdf5"
model.load_weights(filepath)
#print(model.get_layer(index=1))


#model = pickle.load(open("mvt_classification/weights/rf_model.pkl", "rb"))

def classify(self, frame):
    hand_seq = []
    frame_seq = []
    for hand in frame["hands"]:
        frame_seq.append(Dataset.get_hand_vector(hand))
    while(len(frame_seq) < 2):
        frame_seq.append(np.zeros((int(vector_length/2),)))
    # flatten sequence to (1, 1098) from (2,549)
    # couldnt use extend due to empty array not being extendable
    hand_seq.append([item for sublist in frame_seq for item in sublist])
    self.buffer.append(hand_seq)

    temp = np.reshape(list(itertools.islice(self.buffer, 0, self.nb_frames)), X_shape)
    #print("buffe shape", temp.shape)

    #For LSTM
    predictions = model.predict(temp, batch_size=batchSize)
    #print(predictions)

    #For random forest
    #predictions = model.predict(hand_seq)
    #print(predictions)

    #return "nop"
    return Dataset.MVT_NAMES[np.argmax(predictions[0])]
