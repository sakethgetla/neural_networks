import numpy as np
import pickle

#reload object from file
file2 = open(r'd.pkl', 'rb')
new_d = pickle.load(file2)
file2.close()

#print dictionary object loaded from file
print (new_d)

d = [ [3, 2, 8], [4,7,9] ]
afile = open(r'd.pkl', 'wb')
pickle.dump(d, afile)
afile.close()

#reload object from file
file2 = open(r'd.pkl', 'rb')
new_d = pickle.load(file2)
file2.close()

#print dictionary object loaded from file
print (new_d)

