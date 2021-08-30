import numpy as np


def read_array(array_file_name,existing_station_list=None) :

    if (existing_station_list is None) :
        existing_station_list = ['PV','AZ','SM','LM','AA','AP','SP','JC','GL','PB','KP','HA']
    
    stations = np.loadtxt(array_file_name,usecols=[0],dtype=str)
    locs = np.loadtxt(array_file_name,usecols=[1,2,3])
    sefd = np.loadtxt(array_file_name,usecols=[4,5,6,7,8])
    sefd_freq = np.array([86,230,345,480,690])
    
    statdict = {}
    for j in range(len(stations)) :
        if (stations[j] in existing_station_list) :
            statdict[stations[j]] = {'on':True,'loc':locs[j],'sefd':sefd[j],'sefd_freq':sefd_freq,'name':stations[j],'exists':True,'diameter':None}
        else :
            statdict[stations[j]] = {'on':True,'loc':locs[j],'sefd':sefd[j],'sefd_freq':sefd_freq,'name':stations[j],'exists':False,'diameter':6}

    return statdict

