import pickle
import numpy
with open('models/SmartAdapt_cost_20211009.pb','rb') as file:
    data = pickle.load(file)

print(data)