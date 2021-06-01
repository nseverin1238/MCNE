#COM 496 Research Project
#LSTM to run OpenPose data landmarks 
#Niko Severino

 # lstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
import csv
import numpy as np
from random import randrange
from time import* 

# load a single file as a numpy array
def load_file(filepath):
    
    #Read in .txt
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)

	#print("DATAFRAME: " + str(dataframe.values))
	return dataframe.values

# load a list of files and return as a 3d numpy array
def load_group(filename, frames, prefix=''):
        
        data = load_file(filename)
        frameList = load_file(frames)

        samples_features = list()
   
        #A few time steps
        SAMPLES = 18
        
        #Loop through frames list
        for line in frameList:

                #start frame to end frame
                start = line[2]
                end = line[3]

                counter = 0
      
                # Index of all frames in one clip
                SIZE = len(data[start:end])

                #creating features dimension
                features = list()
                #Loop to choose the samples we need
                while counter < SAMPLES:
                        #Pick sample_size rows from the data for each video
                        index = start + (counter * round(SIZE/SAMPLES))
                        
                        #Checks if index is over the clips frame bound    
                        if(index == SIZE + start): #if at bound, append clip data beneath bound
                                features.append(data[start + SIZE - 1][3:-2])
                  
                        else:   #if beneath bound and a sample, append data at index.
                                features.append(data[index][3:-2])
       
                 
                        counter += 1
                samples_features.append(features)
              
        #Stack samples/features array to make batch dimension 
        samples_features = dstack(samples_features)

        return samples_features

#return ground truth values
def load_gt(filename):
        filename = load_file(filename)
        gt = []
        for line in filename:
            #the line containing the class/label
            gt.append([line[1]])
        gt = np.array(gt)
        return gt

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/'
	# load all 9 files as a single array
	filenamex = filepath + 'Normalized_data.txt'
	filenamey = filepath + 'Class_frames.txt'
	frames = filepath +'Class_frames.txt'

	# load input data
	X = load_group(filenamex, frames, filepath)
	y = load_gt(filenamey)
	return X,y

# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
        # load all train
        x, y = load_dataset_group('load', 'data/')

        #reshape to be [sample,time_step,features]
        x = x.reshape(x.shape[2],x.shape[0],x.shape[1])
       
        #Create test and train sets from entire dataset
        skip = 4  #choose every nth for test
        start_idx = randrange(0,skip)      
        testx = []
        testy = []
        trainy = []
        trainx = []
   
        for i in range(len(x)):
            #if divisible by 6 then its a 6th sample  
            if ((i + start_idx)%skip == 0):
                testx.append(x[i])
                testy.append(y[i])
            else:
                trainx.append(x[i])
                trainy.append(y[i])

        print('\n Loading Training set... \n')
        
        #convert to numpy array for LSTM
        testy = np.array(testy)
        testx = np.array(testx)
        trainy = np.array(trainy)
        trainx = np.array(trainx)
   
        trainy = to_categorical(trainy)
        testy = to_categorical(testy)
       
        return trainx,trainy, testx,testy

# fit and evaluate a model
def evaluate_model(trainx, trainy, testx, testy):
        print(trainx.shape, trainy.shape, testx.shape, testy.shape)
       
        #Adjust parameters here 

        verbose, epochs, batch_size = 0, 1000, 3
        
        n_timesteps, n_features, n_outputs = trainx.shape[1], trainx.shape[2] , trainy.shape[1]
        model = Sequential()
        model.add(LSTM(15, input_shape=(n_timesteps,n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit network


        model.fit(trainx, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

        ##plot loss

        #pyplot.plot(history.history['loss'])
        #pyplot.plot(history.history['val_loss'])
        #pyplot.title('model train vs validation loss')
        #pyplot.ylabel('loss')
        #pyplot.xlabel('epoch')
        #pyplot.legend(['train', 'validation'], loc='upper right')
        #pyplot.show()
	
        # evaluate model
        _, accuracy = model.evaluate(testx, testy, batch_size=batch_size, verbose=0)
        return accuracy
 
# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
 
# run an experiment
def run_experiment(repeats=10):
	# load data
	trainx,trainy, testx,testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainx,trainy, testx,testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r+1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)
 
# run the experiment
run_experiment()
