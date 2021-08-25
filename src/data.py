import numpy as np

#########
# To read in data to get the 
def read_data(v_file_name) :

    # Read in Themis-style data, which is simple and compact
    data = np.loadtxt(v_file_name,usecols=[5,6,7,8,9,10,3])
    baselines = np.loadtxt(v_file_name,usecols=[4],dtype=str)
    s1 = np.array([x[:2] for x in baselines])
    s2 = np.array([x[2:] for x in baselines])
    u = data[:,0]/1e3
    v = data[:,1]/1e3
    V = data[:,2] + 1.0j*data[:,4]
    err = data[:,3] + 1.0j*data[:,5]
    t = data[:,6]
    
    # Make conjugate points
    u = np.append(u,-u)
    v = np.append(v,-v)
    V = np.append(V,np.conj(V))
    err = np.append(err,err)
    t = np.append(t,t)
    s1d = np.append(s1,s2)
    s2d = np.append(s2,s1)
    
    return {'u':u,'v':v,'V':V,'s1':s1d,'s2':s2d,'t':t,'err':err}

#########
# Binlinear interpolation
def bilinear(x1d,y1d,f,X,Y,indexing='xy') :

    xfac = 1.0/(x1d[-1]-x1d[0])
    yfac = 1.0/(y1d[-1]-y1d[0])
    ifac = (len(x1d)-1)*xfac
    jfac = (len(y1d)-1)*yfac

    i = np.minimum(len(x1d)-2,np.maximum(0,((X-x1d[0])*ifac)).astype(int))
    j = np.minimum(len(y1d)-2,np.maximum(0,((Y-y1d[0])*jfac)).astype(int))

    wx = (X-x1d[i])/(x1d[1]-x1d[0])
    wy = (Y-y1d[j])/(y1d[1]-y1d[0])

    if (indexing=='ij') :
        F = (1.0-wx)*(1.0-wy)*f[i,j] + (1.0-wx)*wy*f[i,j+1] + wx*(1.0-wy)*f[i+1,j] + wx*wy*f[i+1,j+1]
    elif (indexing=='xy') :
        F = (1.0-wx)*(1.0-wy)*f[j,i] + (1.0-wx)*wy*f[j+1,i] + wx*(1.0-wy)*f[j,i+1] + wx*wy*f[j+1,i+1]
    else :
        print("ERROR: %s is not a valid indexing value."%(indexing))
        
    return F


#########
# To generate data from a station dictionary and image
def generate_data(freq,ra,dec,imgx,imgy,imgI,statdict,integration_time=10,scan_time=600,min_elev=15,bandwidth=8.0,day=100) :
    # Takes:
    #  freq in GHz
    #  ra in hr
    #  dec in deg
    #  x in uas
    #  y in uas
    #  I in Jy
    #  statdict as specified in ngeht_array.py
    #  integration_time in s
    #  scan_time in s
    #  minimum elevation in deg
    #  bandwidth in GHz
    #  day of year
    
    s1 = []
    s2 = []
    u = []
    v = []
    V = []
    err = []
    t = []

    one_over_lambda = freq*1e9 / 2.998e8 / 1e9

    thermal_error_factor = 1.0/np.sqrt( 0.8*2*bandwidth*1e9*integration_time ) * np.sqrt(integration_time/scan_time)

    min_cos_zenith = np.cos( (90-min_elev)*np.pi/180.0 )

    print("Minimum cos(zenith):",min_cos_zenith)
    
    ################################################################
    # Generate observation map
    #
    for obstime in np.arange(0,24.0,scan_time/3600.0) :

        csph = -np.cos((ra-obstime-(day-80)*24/365.25)*np.pi/12.)
        snph = -np.sin((ra-obstime-(day-80)*24/365.25)*np.pi/12.)
        csth = np.sin(dec*np.pi/180.0)
        snth = np.cos(dec*np.pi/180.0)
        
        X = csph*snth
        Y = snph*snth
        Z = csth
        for k,stat1 in enumerate(statdict.keys()) :
            x,y,z = statdict[stat1]['loc']
            csze1 = (x*X+y*Y+z*Z)/np.sqrt(x*x+y*y+z*z)
            if (csze1>=min_cos_zenith) :
                x1 = csth*(csph*x+snph*y) - snth*z
                y1 = -snph*x + csph*y
                # z1 = snth*(csph*x+snph*y) + csth*z
                for stat2 in list(statdict.keys())[(k+1):] :
                    x,y,z = statdict[stat2]['loc']
                    csze2 = (x*X+y*Y+z*Z)/np.sqrt(x*x+y*y+z*z)
                    if (csze2>=min_cos_zenith) :
                        x2 = csth*(csph*x+snph*y) - snth*z
                        y2 = -snph*x + csph*y
                        # z2 = snth*(csph*x+snph*y) + csth*z
                        
                        u.append( (y1-y2) * one_over_lambda )
                        v.append( -(x1-x2) * one_over_lambda )
                        t.append( obstime )

                        s1.append( stat1 )
                        s2.append( stat2 )

                        err.append( np.sqrt( statdict[stat1]['sefd']*statdict[stat2]['sefd'] ) * thermal_error_factor )

    u = np.array(u)
    v = np.array(v)
    t = np.array(t)
    s1 = np.array(s1)
    s2 = np.array(s2)
    err = np.array(err)

    ################################################################
    # Generate observation map
    #
    uas2rad = np.pi/180.0/3600e6
    V0 = np.fft.fftshift(np.fft.fft2(np.pad(imgI,pad_width=((0,imgI.shape[0]),(0,imgI.shape[1])))))
    u01d = -np.fft.fftshift(np.fft.fftfreq(2*imgx.shape[0],d=(imgx[1,1]-imgx[0,0])*uas2rad)/1e9)
    v01d = -np.fft.fftshift(np.fft.fftfreq(2*imgy.shape[1],d=(imgy[1,1]-imgy[0,0])*uas2rad)/1e9)

    # Phase center the image
    xc = 0.5*(imgx[-1,-1]-imgx[0,0])*uas2rad*1e9
    yc = 0.5*(imgy[-1,-1]-imgy[0,0])*uas2rad*1e9
    u0,v0 = np.meshgrid(u01d,v01d)
    V0 = V0*np.exp(-2.0j*np.pi*(u0*xc+v0*yc))

    # Interpolate to the data points
    V = bilinear(u01d,v01d,V0,u,v)

    # Make conjugate points
    u = np.append(u,-u)
    v = np.append(v,-v)
    V = np.append(V,np.conj(V))
    err = np.append(err,err)
    t = np.append(t,t)
    s1d = np.append(s1,s2)
    s2d = np.append(s2,s1)

    return {'u':u,'v':v,'V':V,'s1':s1d,'s2':s2d,'t':t,'err':(1.0+1.0j)*err}

                        
                        
                        
    

    
    
