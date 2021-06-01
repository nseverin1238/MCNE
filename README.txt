COM 496 Research Project

Multimodal Communication in a Noisy Environment

Niko Severino

HOW TO RUN LSTM: 

Open CMD pointed at C:\MCNE\data\OP_landmarks

Within data, copy paste whichever set you'd like to 
test on into /load

"whole_hands_set" contains the set of hand landmarks from
OpenPose that have been scaled/normalized to the wrist. 

For the LSTM, the parameters that resulted in best results were

Line 40 : Samples = 2

Line 144: 0 verbose, 200 epochs, 75 batch size

Line 148: 15 layers 

"whole_face_set" contains the set of face landmarks from 
OpenPose that have been scaled/normalized to the bridg of the nose

For the LSTM, the parameters that resulted in best results were

Line 40 : Samples = 18

Line 144: 0 verbose, 1000 epochs, 3 batch size

Line 148: 15 layers


Run python open2lstm.py


*Extra_sets consists of sets that I tested on that didn't
have great results, such as combining face and hands, 
using the arms in the hands set, scaling the body as opposed 
to the face. Stuff like that. 



