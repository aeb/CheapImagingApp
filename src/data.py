import os
import numpy as np
import matplotlib.image as mi

__data_debug__ = True


#########
# To read in data to get the 
def read_themis_data_file(v_file_name) :

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

    if (__data_debug__) :
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

    if (__data_debug__) :
        print("Generated baseline map")

    ################################################################
    # Generate observation map
    #
    uas2rad = np.pi/180.0/3600e6
    V0 = np.fft.fftshift(np.fft.fft2(np.pad(imgI,pad_width=((0,imgI.shape[0]),(0,imgI.shape[1])))))
    # u01d = -np.fft.fftshift(np.fft.fftfreq(2*imgx.shape[0],d=(imgx[1,1]-imgx[0,0])*uas2rad)/1e9)
    # v01d = -np.fft.fftshift(np.fft.fftfreq(2*imgy.shape[1],d=(imgy[1,1]-imgy[0,0])*uas2rad)/1e9)
    u01d = -np.fft.fftshift(np.fft.fftfreq(2*imgx.shape[1],d=(imgx[1,1]-imgx[0,0])*uas2rad)/1e9)
    v01d = -np.fft.fftshift(np.fft.fftfreq(2*imgy.shape[0],d=(imgy[1,1]-imgy[0,0])*uas2rad)/1e9)

    if (__data_debug__) :
        print("Finished FFTs")

    
    # Phase center the image
    xc = 0.5*(imgx[-1,-1]-imgx[0,0])*uas2rad*1e9
    yc = 0.5*(imgy[-1,-1]-imgy[0,0])*uas2rad*1e9
    # u0,v0 = np.meshgrid(u01d,v01d,indexing='ij')
    # u0,v0 = np.meshgrid(v01d,u01d,indexing='xy')
    u0,v0 = np.meshgrid(u01d,v01d)
    V0 = V0*np.exp(-2.0j*np.pi*(u0*xc+v0*yc))

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(imgI)
    # plt.figure()
    # plt.imshow(V0.real)
    # plt.figure()
    # plt.imshow(np.abs(V0))
    # # plt.figure()
    # # plt.pcolor(u0,v0,np.abs(V0))
    # plt.show()

    
    if (__data_debug__) :
        print("Phase centered visibilities")
    
    # Interpolate to the data points
    V = bilinear(u01d,v01d,V0,u,v)

    if (__data_debug__) :
        print("Interpolated to the truth")

    # Make conjugate points
    u = np.append(u,-u)
    v = np.append(v,-v)
    V = np.append(V,np.conj(V))
    err = np.append(err,err)
    t = np.append(t,t)
    s1d = np.append(s1,s2)
    s2d = np.append(s2,s1)

    if (__data_debug__) :
        print("Made conjugates, all done!")

    
    return {'u':u,'v':v,'V':V,'s1':s1d,'s2':s2d,'t':t,'err':(1.0+1.0j)*err}

                        
                        
def generate_data_from_file(file_name,statdict,freq=230,ra=17.7611225,dec=-29.007810,scale=500.0,total_flux=None,**kwargs) :
    
    ext = os.path.splitext(file_name)[1]

    if (__data_debug__) :
        print("Started in generate_data_from_file, I think this is a %s file."%(ext))
    
    if ( ext=='.dat' ) :
        return read_themis_data_file(file_name)  # No kwargs

    elif ( ext=='.npy' ) :

        [img_total_flux,xdim,ydim,psize,drf,ii] = np.load(file_name,allow_pickle=True)
        I = 10**(ii.reshape(xdim,ydim)/256.0 * drf - drf)
        if (total_flux is None) :
            total_flux = img_total_flux
        I = I * total_flux/np.sum(I)
        I = np.flipud(np.fliplr(I)) 
        # I = np.transpose(np.flipud(np.fliplr(I)))
        # x,y = np.meshgrid(np.arange(0,I.shape[0]),np.arange(0,I.shape[1]),indexing='ij')
        # x,y = np.meshgrid(np.arange(0,I.shape[1]),np.arange(0,I.shape[0]),indexing='ij')
        x,y = np.meshgrid(np.arange(0,I.shape[1]),np.arange(0,I.shape[0]))
        x = (x-0.5*I.shape[1])*psize
        y = (y-0.5*I.shape[0])*psize

        # print("Shapes:",x.shape,y.shape,I.shape)
        # print("x:",x[:5,0],x[:,0].shape)
        # print("y:",y[0,:5],y[0,:].shape)
        # import matplotlib.pyplot as plt
        # plt.pcolor(x,y,np.log10(I/np.max(I)),vmax=0,vmin=-4,cmap='afmhot')
        # plt.show()

        if (__data_debug__) :
            print("Finished npy read:",x.shape,y.shape,I.shape,x[0,0],x[-1,-1],y[0,0],y[-1,-1],np.max(I))
            
        return generate_data(freq,ra,dec,x,y,I,statdict,**kwargs)
    
    elif ( ext.lower() in ['.jpg','.jpeg','.png','.gif'] ) :

        img = mi.imread(file_name)

        if (__data_debug__) :
            print("Read",file_name,img.shape)        
        
        I = np.sqrt( (img[:,:,0].astype(float))**2 + (img[:,:,1].astype(float))**2 + (img[:,:,2].astype(float))**2 ) * (img[:,:,3].astype(float))

        drf = 1
        I = 1 * I / np.max(I)

        #I = I**2

        if (scale<1000.0) :
            if (__data_debug__) :
                print("Plot too small in x-direction:",scale)
            dim = int(np.ceil(1000.0/scale * I.shape[1]))
            mpad = (dim-I.shape[1])//2
            ppad = (dim-I.shape[1])-mpad
            if (__data_debug__) :
                print("  new dim, mpad, ppad:",dim,mpad,ppad)
            scale = float(dim)/I.shape[1] * scale
            if (__data_debug__) :
                print("  new scale:",scale)
            I = np.pad(I,pad_width=((0,0),(mpad,ppad)))

        if (__data_debug__) :
            print("Shape after x-dim check:",I.shape)
            
        if (scale*I.shape[0]/I.shape[1]<1000.0) :
            if (__data_debug__) :
                print("Plot too small in y-direction:",scale)
            dim = int(np.ceil(1000.0/scale * I.shape[1]))
            mpad = (dim-I.shape[0])//2
            ppad = (dim-I.shape[0])-mpad
            if (__data_debug__) :
                print("  new dim, mpad, ppad:",dim,mpad,ppad)
            I = np.pad(I,pad_width=((mpad,ppad),(0,0)))

        if (__data_debug__) :
            print("Shape after y-dim check:",I.shape)

        # if (I.shape[0]<1024 or I.shape[1]<1024) :
        #     x_mpad = (1024-I.shape[0])//2
        #     x_ppad = 1024-I.shape[0]-x_mpad
        #     y_mpad = (1024-I.shape[1])//2
        #     y_ppad = 1024-I.shape[1]-y_mpad
        #     scale = 1024./I.shape[0] * scale
        #     I = np.pad(I,pad_width=((x_mpad,x_ppad),(y_mpad,y_ppad)))
        # print("Shape after size check:",I.shape)

        

        
        if (__data_debug__) :
            print("Set intensity array",I.shape,np.max(I),np.min(I))
        if (total_flux is None) :
            total_flux = 1.0
        I = I * total_flux / np.sum(I)
        I = np.flipud(np.fliplr(I))
        # x,y = np.meshgrid(np.arange(0,I.shape[0]),np.arange(0,I.shape[1]),indexing='ij')
        # x = (x-0.5*I.shape[0])*scale/I.shape[0]
        # y = (y-0.5*I.shape[1])*scale/I.shape[0]

        x,y = np.meshgrid(np.arange(0,I.shape[1]),np.arange(0,I.shape[0]))
        x = (x-0.5*I.shape[1])*scale/I.shape[1]
        y = (y-0.5*I.shape[0])*scale/I.shape[1]


        I = I + 0.1*np.max(I)*np.exp(-(x**2+y**2)/(2.0*200.0**2) )
        # I = 0.1*np.max(I)*np.exp(-(x**2+y**2)/(2.0*200.0**2) )


        # Crop image to 1024x1024 on -0.5 mas to 0.5 mas
        x2,y2 = np.meshgrid(-np.linspace(-500,500,1024),np.linspace(-500,500,1024))
        I2 = bilinear(x[0,:],y[:,0],I,x2,y2)

        
        if (__data_debug__) :
            print("Shapes:",x2.shape,y2.shape,I2.shape)
            print("x2:",x2[:5,0],x2[:,0].shape)
            print("y2:",y2[0,:5],y2[0,:].shape)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.pcolor(x,y,I,cmap='afmhot')
        # plt.figure()
        # plt.pcolor(x2,y2,I2,cmap='afmhot')
        # plt.show()
        
        if (__data_debug__) :
            print("Finished %s read."%(ext))        

        return generate_data(freq,ra,dec,x2,y2,I2,statdict,**kwargs)

        

        
        
    
    
