import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_location_names():
    return(__locations)


def load_saved_artifacts():
    print("loading saved artifacts ....")    #gets columns json from artifacts
    global __locations
    global __data_columns
    global __model

    with open(r'server/artifacts/columns.json','r') as f:
        __data_columns =json.load(f)['data_columns']  #json.load gives back a dict we take data columns key
        __locations = __data_columns[3:]   #3rd columns onwards are the locations

    with open(r'server/artifacts/bangalore_home_prices_model.pickle','rb') as f:
        __model = pickle.load(f)    
    
    print("finished loading artifacts")


def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index =-1

    location = location.strip()
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1    
    return round(__model.predict([x])[0],3)

load_saved_artifacts()
if __name__=='__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Indira Nagar',2000,2,2))

