import numpy as np


def read_array(array_file_name,existing_station_list=None) :

    if (existing_station_list is None) :
        existing_station_list = ['PV','AZ','SM','LM','AA','AP','SP','JC','GL','PB','KP','HA']
    
    stations = np.loadtxt(array_file_name,usecols=[0],dtype=str)
    locs = np.loadtxt(array_file_name,usecols=[1,2,3])
    sefd = np.loadtxt(array_file_name,usecols=[4,5,6,7,8])
    cost_factors = np.loadtxt(array_file_name,usecols=[9,10])
    sefd_freq = np.array([86,230,345,480,690])
    
    statdict = {}
    for j in range(len(stations)) :
        if (stations[j] in existing_station_list) :
            statdict[stations[j]] = {'on':True,'loc':locs[j],'sefd':sefd[j],'sefd_freq':sefd_freq,'name':stations[j],'exists':True,'diameter':None,'cost_factors':cost_factors[j]}
        else :
            statdict[stations[j]] = {'on':True,'loc':locs[j],'sefd':sefd[j],'sefd_freq':sefd_freq,'name':stations[j],'exists':False,'diameter':6,'cost_factors':cost_factors[j]}

    return statdict



def cost_model(statdict,ngeht_diameter) :

    # if (ngeht_diameter==3.5) :
    #     total_new_site_NRE = 2.655e6        
    # elif (ngeht_diameter==6) :
    #     total_new_site_NRE = 2.655e6 * 1.5      
    # elif (ngeht_diameter==10) :
    #     total_new_site_NRE = 2.655e6 * 2.0  
    # else :
    #     total_new_site_NRE = 2.655e6 * (ngeht_diameter/3.5)**0.67  
    total_new_site_NRE = 2.655 * (ngeht_diameter/3.5)**0.67  # Approximate diameter dependence that matches to 4%

    # print(total_new_site_NRE)
    
    capex = total_new_site_NRE
    print("tota_new_site_NRE %10.5g"%(capex))
    for s in statdict.keys() :
        if (statdict[s]['on']) :
            capex = capex + statdict[s]['cost_factors'][0] + statdict[s]['cost_factors'][1]*ngeht_diameter**2.7

            # print("%2s %10.4g %10.4g"%(s,capex,statdict[s]['cost_factors'][0] + statdict[s]['cost_factors'][1]*ngeht_diameter**2.7))
            
    return capex
            

