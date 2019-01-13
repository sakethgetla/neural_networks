import numpy as np
import pickle

d = [ [1, 2, 3], [4,5,6] ]
afile = open(r'd.pkl', 'wb')
pickle.dump(d, afile)
afile.close()

#reload object from file
file2 = open(r'd.pkl', 'rb')
new_d = pickle.load(file2)
file2.close()

#print dictionary object loaded from file
print (new_d)

