import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import copy

# Data dictionary: datadict has form {'u':u, 'v':v, 'V':V}
# Station dictionary: statdict has form {<station code>:{'on':<True/False>,'name':<name>,'loc':(x,y,z)}}


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


def read_array(array_file_name,existing_station_list=None) :

    if (existing_station_list is None) :
        existing_station_list = ['PV','AZ','SM','LM','AA','SP','JC','GL','PB','KP','HA']
    
    stations = np.loadtxt(array_file_name,usecols=[0],dtype=str)
    locs = np.loadtxt(array_file_name,usecols=[1,2,3])

    statdict = {}
    for j in range(len(stations)) :
        if (stations[j] in existing_station_list) :
            statdict[stations[j]] = {'on':True,'loc':locs[j],'name':stations[j], 'exists':True, 'diameter':None}
        else :
            statdict[stations[j]] = {'on':True,'loc':locs[j],'name':stations[j], 'exists':False, 'diameter':6}            
        
    return statdict


class BaselinePlots :

    def __init__(self) :

        self.ddict = None
        self.ddnew = None
    

    def plot_baselines(self,axs,datadict,statdict,time_range=None,snr_cut=None,ngeht_diameter=6,make_hermitian=False,limits=None) :
        
        # Set this to current axes for convenience, might be a minor performance hit?
        plt.sca(axs)

        # Exclude stations not in array
        # stations = list(np.unique(np.array(list(statdict.keys()))))
        stations = list(statdict.keys())
        keep = np.array([ (datadict['s1'][j] in stations) and (datadict['s2'][j] in stations) for j in range(len(datadict['s1'])) ])
        ddtmp = {}
        for key in ['u','v','V','s1','s2','t','err'] :
            ddtmp[key] = datadict[key][keep]

        self.ddict = ddtmp
        
        # Exclude stations that are "off"
        if (len(self.ddict['u'])>0) :
            keep = np.array([ statdict[ddtmp['s1'][j]]['on'] and statdict[ddtmp['s2'][j]]['on'] for j in range(len(ddtmp['s1'])) ])
            self.ddnew = {}
            for key in ['u','v','V','s1','s2','t','err'] :
                self.ddnew[key] = ddtmp[key][keep]
        else :
            self.ddnew = copy.deepcopy(self.ddict)
                
        # Exclude data points outside the specified time range
        if (len(self.ddnew['u'])>0) :
            if (not time_range is None) :
                keep = (self.ddnew['t']>=time_range[0])*(self.ddnew['t']<time_range[1])
                for key in ['u','v','V','s1','s2','t','err'] :
                    self.ddnew[key] = self.ddnew[key][keep]

                    
        # Cut points with S/N less than the specified minimum value
        if (not snr_cut is None)  and (snr_cut>0):
            if (len(self.ddnew['u'])>0) :
                # Get a list of error adjustments based on stations
                diameter_correction_factor = {}
                for s in stations :
                    if (statdict[s]['exists']) :
                        diameter_correction_factor[s] = 1.0
                    else :
                        diameter_correction_factor[s] = statdict[s]['diameter']/ngeht_diameter
                keep = np.array([ np.abs(self.ddnew['V'][j])/(self.ddnew['err'][j].real * diameter_correction_factor[self.ddnew['s1'][j]] * diameter_correction_factor[self.ddnew['s2'][j]]) > snr_cut for j in range(len(self.ddnew['s1'])) ])
                for key in ['u','v','V','s1','s2','t','err'] :
                    self.ddnew[key] = self.ddnew[key][keep]
                
        # Double up data to make V hemitian
        if (make_hermitian) :
            self.ddnew['u'] = np.append(self.ddnew['u'],-self.ddnew['u'])
            self.ddnew['v'] = np.append(self.ddnew['v'],-self.ddnew['v'])
            self.ddnew['V'] = np.append(self.ddnew['V'],np.conj(self.ddnew['V']))

            
        plt.plot(self.ddict['u'],self.ddict['v'],'.',color=[0.14,0.14,0.14])
        plt.plot(self.ddnew['u'],self.ddnew['v'],'.',color='cornflowerblue')

        uvmax = np.max(np.sqrt( self.ddict['u']**2 + self.ddict['v']**2 ))

        plt.gca().spines['left'].set_position('zero')
        plt.gca().spines['left'].set_color('w')
        plt.gca().spines['bottom'].set_position('zero')
        plt.gca().spines['bottom'].set_color('w')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().xaxis.set_tick_params(bottom='on',top='off',direction='inout',colors='w')
        plt.gca().yaxis.set_tick_params(left='on',right='off',direction='inout',colors='w')

        if (limits is None) :
            limits = [1.1*uvmax,-1.1*uvmax,-1.1*uvmax,1.1*uvmax]
        plt.xlim(limits[:2])
        plt.ylim(limits[2:])

        plt.grid(True,alpha=0.25,linewidth=2)

        xc = 0.5*(limits[0]+limits[1])
        dx = 0.5*(limits[1]-limits[0])
        yc = 0.5*(limits[2]+limits[3])
        dy = 0.5*(limits[3]-limits[2])
        plt.text( xc+0.85*dx, 0.05*dy, 'u (G$\lambda$)',color='w',ha='center',fontsize=14)
        plt.text( 0.05*dx, yc+0.85*dy, 'v (G$\lambda$)',color='w',va='center',fontsize=14)
        
        plt.gca().set_facecolor('k')


    def replot(self,axs,limits=None) :
        plt.sca(axs)

        plt.plot(self.ddict['u'],self.ddict['v'],'.',color=[0.14,0.14,0.14])
        plt.plot(self.ddnew['u'],self.ddnew['v'],'.',color='cornflowerblue')

        uvmax = np.max(np.sqrt( self.ddict['u']**2 + self.ddict['v']**2 ))

        plt.gca().spines['left'].set_position('zero')
        plt.gca().spines['left'].set_color('w')
        plt.gca().spines['bottom'].set_position('zero')
        plt.gca().spines['bottom'].set_color('w')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().xaxis.set_tick_params(bottom='on',top='off',direction='inout',colors='w')
        plt.gca().yaxis.set_tick_params(left='on',right='off',direction='inout',colors='w')

        if (limits is None) :
            limits = [1.1*uvmax,-1.1*uvmax,-1.1*uvmax,1.1*uvmax]
        plt.xlim(limits[:2])
        plt.ylim(limits[2:])

        plt.grid(True,alpha=0.25,linewidth=2)

        xc = 0.5*(limits[0]+limits[1])
        dx = 0.5*(limits[1]-limits[0])
        yc = 0.5*(limits[2]+limits[3])
        dy = 0.5*(limits[3]-limits[2])
        plt.text( xc+0.85*dx, 0.05*dy, 'u (G$\lambda$)',color='w',ha='center',fontsize=14)
        plt.text( 0.05*dx, yc+0.85*dy, 'v (G$\lambda$)',color='w',va='center',fontsize=14)
        
        plt.gca().set_facecolor('k')


        

class CheapImageReconstruction :

    def __init__(self) :
        pass

    x=None
    y=None
    I=None
    
    ##########
    # Low-level image reconstruction function
    def reconstruct_image(self,datadict,statdict,time_range=None,snr_cut=None,ngeht_diameter=6,f=2,method='cubic',make_hermitian=False) :

        # Useful constant
        uas2rad = np.pi/180.0/3600e6

        # Exclude stations not in array
        stations = list(np.unique(np.array(list(statdict.keys()))))
        keep = np.array([ (datadict['s1'][j] in stations) and (datadict['s2'][j] in stations) for j in range(len(datadict['s1'])) ])
        ddtmp = {}
        for key in ['u','v','V','s1','s2','t','err'] :
            ddtmp[key] = datadict[key][keep]

        if (len(ddtmp['u'])==0) :
            return None,None,None

        # Exclude stations that are "off"
        keep = np.array([ statdict[ddtmp['s1'][j]]['on'] and statdict[ddtmp['s2'][j]]['on'] for j in range(len(ddtmp['s1'])) ])
        ddnew = {}
        for key in ['u','v','V','s1','s2','t','err'] :
            ddnew[key] = ddtmp[key][keep]

        if (len(ddnew['u'])==0) :
            return None,None,None

        # Exclude data points outside the specified time range
        if (not time_range is None) :
            keep = (ddnew['t']>=time_range[0])*(ddnew['t']<time_range[1])
            for key in ['u','v','V','s1','s2','t','err'] :
                ddnew[key] = ddnew[key][keep]

        if (len(ddnew['u'])==0) :
            return None,None,None
                
        # Cut points with S/N less than the specified minimum value
        if (not snr_cut is None) and snr_cut>0:
            # Get a list of error adjustments based on stations
            diameter_correction_factor = {}
            for s in stations :
                if (statdict[s]['exists']) :
                    diameter_correction_factor[s] = 1.0
                else :
                    diameter_correction_factor[s] = statdict[s]['diameter']/ngeht_diameter
            keep = np.array([ np.abs(ddnew['V'][j])/(ddnew['err'][j].real * diameter_correction_factor[ddnew['s1'][j]] * diameter_correction_factor[ddnew['s2'][j]]) > snr_cut for j in range(len(ddnew['s1'])) ])
            for key in ['u','v','V','s1','s2','t','err'] :
                ddnew[key] = ddnew[key][keep]

        if (len(ddnew['u'])==0) :
            return None,None,None

        # Double up data to make V hemitian
        if (make_hermitian) :
            ddnew['u'] = np.append(ddnew['u'],-ddnew['u'])
            ddnew['v'] = np.append(ddnew['v'],-ddnew['v'])
            ddnew['V'] = np.append(ddnew['V'],np.conj(ddnew['V']))

        if (len(ddnew['u'])<=2) :
            return None,None,None
            
        # Get the region on which to compute gridded visibilities
        umax = np.max(ddnew['u'])
        vmax = np.max(ddnew['v'])
        u2,v2 = np.meshgrid(np.linspace(-f*umax,f*umax,256),np.linspace(-f*vmax,f*vmax,256))
        pts = np.array([ddnew['u'],ddnew['v']]).T
        V2r = si.griddata(pts,np.real(ddnew['V']),(u2,v2),method=method,fill_value=0.0)
        V2i = si.griddata(pts,np.imag(ddnew['V']),(u2,v2),method=method,fill_value=0.0)
        V2 = V2r + 1.0j*V2i

        # Filter to smooth at edges
        V2 = V2 * np.cos(u2/umax*0.5*np.pi) * np.cos(v2/vmax*0.5*np.pi)

        # Generate the x,y grid on which to image
        x1d = np.fft.fftshift(np.fft.fftfreq(u2.shape[0],d=(u2[1,1]-u2[0,0])*1e9)/uas2rad)
        y1d = np.fft.fftshift(np.fft.fftfreq(v2.shape[1],d=(v2[1,1]-v2[0,0])*1e9)/uas2rad)
        x,y = np.meshgrid(-x1d,-y1d)

        # Compute image estimate via FFT
        #I = np.maximum(0.0,np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(V2)))))
        I = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(V2))))
        
        # Return
        return x,y,I


    ############
    # High-level plot generation
    def plot_image_reconstruction(self,axs,datadict,statdict,time_range=None,snr_cut=None,ngeht_diameter=6,limits=None,show_map=True,show_contours=True) :

        # Reconstruct image
        self.x,self.y,self.I=self.reconstruct_image(datadict,statdict,time_range=time_range,snr_cut=snr_cut,ngeht_diameter=ngeht_diameter)

        self.replot_image_reconstruction(axs,time_range=time_range,limits=limits,show_map=show_map,show_contours=show_contours)
        

    ############
    # High-level plot generation
    def replot_image_reconstruction(self,axs,time_range=None,limits=None,show_map=True,show_contours=True) :

        plt.sca(axs)
        
        if (self.I is None) :
            plt.text(0.5,0.5,"Insufficient Data!",color='w',fontsize=24,ha='center',va='center')
            plt.gca().set_facecolor('k')
            plt.gcf().set_facecolor('k')
            return


        # Plot linear image
        if (show_map) :
            #plt.pcolormesh(self.x,self.y,self.I,cmap='afmhot')
            plt.imshow(self.I,origin='lower',extent=[self.x[0,0],self.x[0,-1],self.y[0,0],self.y[-1,0]],cmap='afmhot',vmin=0,interpolation='spline16')
            
        # Plot the log contours
        if (show_contours) :
            # lI = np.log10(self.I/np.max(self.I)+1e-20)
            # lmI = np.log10(-self.I/np.max(self.I)+1e-20)
            lI = np.log10(np.maximum(0.0,self.I)/np.max(self.I)+1e-20)
            lmI = np.log10(np.maximum(0.0,-self.I)/np.max(self.I)+1e-20)
            
            lev10lo = max(np.min(lI[self.I>0]),-4)
            lev10 = np.sort( -np.arange(0,lev10lo,-1) )
            plt.contour(self.x,self.y,-lI,levels=lev10,colors='cornflowerblue',alpha=0.5)
            #plt.contour(self.x,self.y,-lmI,levels=lev10,colors='green',alpha=0.5)            
            lev1 = []
            for l10 in -lev10[1:] :
                lev1.extend( np.log10(np.array([2,3,4,5,6,7,8,9])) + l10 )
            lev1 = np.sort(-np.array(lev1))
            plt.contour(self.x,self.y,-lI,levels=lev1,colors='cornflowerblue',alpha=0.5,linewidths=0.5)
            plt.contour(self.x,self.y,-lmI,levels=lev1[-10:],colors='green',alpha=0.5,linewidths=0.5)

        # Fix the limits
        if (not limits is None) :
            plt.xlim((limits[0],limits[1]))
            plt.ylim((limits[2],limits[3]))
        else :
            xmin = min(np.min(x[lI>-2]),np.min(y[lI>-2]))
            xmax = max(np.max(x[lI>-2]),np.max(y[lI>-2]))
            plt.xlim((xmax,xmin))
            plt.ylim((xmin,xmax))

        plt.gca().set_facecolor('k')




# dd = read_data('./data/V_M87_ngeht_ref1_230_thnoise_scanavg_tygtd.dat')
# sd = read_array('./arrays/ngeht_ref1_230_ehtim.txt')

# plt.figure()

# bp = BaselinePlots()

# bp.plot_baselines(dd,sd)

# plt.show()
