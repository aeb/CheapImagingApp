import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi
# import scipy.interpolate as si
import matplotlib.tri as tri
import copy
from os import path

from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Ellipse, Color, Rectangle, Line
from kivy.metrics import dp


from kivy.core.window import Window

from mpl_texture import InteractiveWorldMapOverlayWidget, InteractivePlotWidget, InteractiveWorldMapWidget
# from mpl_texture import InteractivePlotWidget, InteractiveWorldMapWidget

#from PIL import Image

# Data dictionary: datadict has form {'u':u, 'v':v, 'V':V}
# Station dictionary: statdict has form {<station code>:{'on':<True/False>,'name':<name>,'loc':(x,y,z)}}

__mydebug__ = True


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
        existing_station_list = ['PV','AZ','SM','LM','AA','AP','SP','JC','GL','PB','KP','HA']
    
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


                

class InteractiveBaselinePlot(InteractivePlotWidget) :
    
    def __init__(self,**kwargs) :

        self.ddict = {}
        self.ddnew = {}
        self.sdict = {}

        super().__init__(**kwargs)

    def generate_mpl_plot(self,fig,ax,**kwargs) :
        # This is where we insert a Matplotlib figure.  Must use ax. and fig. child commands.
        # You probably want, but do not require, the following in your over-lay
        self.plot_baselines(ax,self.ddict,self.sdict,**kwargs)
        ax.set_facecolor((0,0,0,0))
        fig.set_facecolor((0,0,0,0))


    def update(self,datadict,statdict,**kwargs) :

        self.sdict = statdict
        self.ddict = datadict

        self.update_mpl(**kwargs)


    def replot(self,datadict,statdict,**kwargs) :

        self.sdict = statdict
        self.ddict = datadict

        self.update_mpl(**kwargs)
        
        
    def plot_baselines(self,axs,datadict,statdict,time_range=None,snr_cut=None,ngeht_diameter=6,make_hermitian=False,limits=None) :

        if (len(statdict.keys())==0) :
            return

        
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

            
        axs.plot(self.ddict['u'],self.ddict['v'],'.',color=[0.14,0.14,0.14])
        axs.plot(self.ddnew['u'],self.ddnew['v'],'.',color='cornflowerblue')

        uvmax = np.max(np.sqrt( self.ddict['u']**2 + self.ddict['v']**2 ))

        axs.spines['left'].set_position('zero')
        axs.spines['left'].set_color('w')
        axs.spines['bottom'].set_position('zero')
        axs.spines['bottom'].set_color('w')
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.xaxis.set_tick_params(bottom='on',top='off',direction='inout',colors='w')
        axs.yaxis.set_tick_params(left='on',right='off',direction='inout',colors='w')

        if (limits is None) :
            limits = [1.1*uvmax,-1.1*uvmax,-1.1*uvmax,1.1*uvmax]
        axs.set_xlim(limits[:2])
        axs.set_ylim(limits[2:])

        axs.grid(color='k',alpha=0.25,linewidth=1)

        xc = 0.5*(limits[0]+limits[1])
        dx = 0.5*(limits[1]-limits[0])
        yc = 0.5*(limits[2]+limits[3])
        dy = 0.5*(limits[3]-limits[2])
        axs.text( xc+0.25*dx, 0.05*dy, 'u (G$\lambda$)',color='w',ha='center',fontsize=14)
        axs.text( 0.05*dx, yc+0.25*dy, 'v (G$\lambda$)',color='w',va='center',fontsize=14)
        




        

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

        # SciPy
        # pts = np.array([ddnew['u'],ddnew['v']]).T
        # V2r = si.griddata(pts,np.real(ddnew['V']),(u2,v2),method=method,fill_value=0.0)
        # V2i = si.griddata(pts,np.imag(ddnew['V']),(u2,v2),method=method,fill_value=0.0)

        # Maptlotlib
        triang = tri.Triangulation(ddnew['u'], ddnew['v'])
        if (method=='linear') :
            V2r = np.array(np.ma.fix_invalid(tri.LinearTriInterpolator(triang, np.real(ddnew['V']))(u2,v2),fill_value=0.0))
            V2i = np.array(np.ma.fix_invalid(tri.LinearTriInterpolator(triang, np.imag(ddnew['V']))(u2,v2),fill_value=0.0))
        elif (method=='cubic') :
            V2r = np.array(np.ma.fix_invalid(tri.CubicTriInterpolator(triang, np.real(ddnew['V']),kind='geom')(u2,v2),fill_value=0.0))
            V2i = np.array(np.ma.fix_invalid(tri.CubicTriInterpolator(triang, np.imag(ddnew['V']),kind='geom')(u2,v2),fill_value=0.0))
        else :
            print("ERROR: method %s not implemented"%(method))
        
        V2 = V2r + 1.0j*V2i

        # Filter to smooth at edges
        V2 = V2 * np.cos(u2/umax*0.5*np.pi) * np.cos(v2/vmax*0.5*np.pi)

        # Generate the x,y grid on which to image
        x1d = np.fft.fftshift(np.fft.fftfreq(u2.shape[0],d=(u2[1,1]-u2[0,0])*1e9)/uas2rad)
        y1d = np.fft.fftshift(np.fft.fftfreq(v2.shape[1],d=(v2[1,1]-v2[0,0])*1e9)/uas2rad)
        x,y = np.meshgrid(-x1d,-y1d)

        # Compute image estimate via FFT
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




class InteractiveImageReconstructionPlot(InteractivePlotWidget) :
    
    def __init__(self,**kwargs) :

        self.xarr = 0
        self.yarr = 0
        self.Iarr = 1

        self.ddict = {}
        self.sdict = {}
        
        super().__init__(**kwargs)


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

        # SciPy
        # pts = np.array([ddnew['u'],ddnew['v']]).T
        # V2r = si.griddata(pts,np.real(ddnew['V']),(u2,v2),method=method,fill_value=0.0)
        # V2i = si.griddata(pts,np.imag(ddnew['V']),(u2,v2),method=method,fill_value=0.0)

        # Maptlotlib
        triang = tri.Triangulation(ddnew['u'], ddnew['v'])
        if (method=='linear') :
            V2r = np.array(np.ma.fix_invalid(tri.LinearTriInterpolator(triang, np.real(ddnew['V']))(u2,v2),fill_value=0.0))
            V2i = np.array(np.ma.fix_invalid(tri.LinearTriInterpolator(triang, np.imag(ddnew['V']))(u2,v2),fill_value=0.0))
        elif (method=='cubic') :
            V2r = np.array(np.ma.fix_invalid(tri.CubicTriInterpolator(triang, np.real(ddnew['V']),kind='geom')(u2,v2),fill_value=0.0))
            V2i = np.array(np.ma.fix_invalid(tri.CubicTriInterpolator(triang, np.imag(ddnew['V']),kind='geom')(u2,v2),fill_value=0.0))
        else :
            print("ERROR: method %s not implemented"%(method))
        
        V2 = V2r + 1.0j*V2i

        # Filter to smooth at edges
        V2 = V2 * np.cos(u2/umax*0.5*np.pi) * np.cos(v2/vmax*0.5*np.pi)

        # Generate the x,y grid on which to image
        x1d = np.fft.fftshift(np.fft.fftfreq(u2.shape[0],d=(u2[1,1]-u2[0,0])*1e9)/uas2rad)
        y1d = np.fft.fftshift(np.fft.fftfreq(v2.shape[1],d=(v2[1,1]-v2[0,0])*1e9)/uas2rad)
        xarr,yarr = np.meshgrid(-x1d,-y1d)

        # Compute image estimate via FFT
        Iarr = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(V2))))
        
        # Return
        return xarr,yarr,Iarr

    
        
    def generate_mpl_plot(self,fig,ax,**kwargs) :
        # This is where we insert a Matplotlib figure.  Must use ax. and fig. child commands.
        # You probably want, but do not require, the following in your over-lay
        self.plot_image_reconstruction(ax,self.ddict,self.sdict,**kwargs)
        ax.set_facecolor((0,0,0,1))
        fig.set_facecolor((0,0,0,1))


    def update(self,datadict,statdict,**kwargs) :

        self.sdict = statdict
        self.ddict = datadict

        self.update_mpl(**kwargs)


    def replot(self,datadict,statdict,**kwargs) :

        self.sdict = statdict
        self.ddict = datadict

        self.update_mpl(**kwargs)

    def check_boundaries(self,tex_coords) :
        return tex_coords

    def check_size(self,size) :
        if (size[0]<self.width) :
            size = (self.width, size[1]/size[0] * self.width)
        elif  (size[1]<self.height) :
            size = (size[0]/size[1] * self.height, self.height)
        return size

    ############
    # High-level plot generation
    def plot_image_reconstruction(self,axs,datadict,statdict,time_range=None,snr_cut=None,ngeht_diameter=6,limits=None,show_map=True,show_contours=True) :

        if (len(statdict.keys())==0) :
            return

        
        # Reconstruct image
        self.xarr,self.yarr,self.Iarr=self.reconstruct_image(datadict,statdict,time_range=time_range,snr_cut=snr_cut,ngeht_diameter=ngeht_diameter)

        self.replot_image_reconstruction(axs,time_range=time_range,limits=limits,show_map=show_map,show_contours=show_contours)
        

    ############
    # High-level plot generation
    def replot_image_reconstruction(self,axs,time_range=None,limits=None,show_map=True,show_contours=True) :

        if (self.Iarr is None) :
            axs.text(0.5,0.5,"Insufficient Data!",color='w',fontsize=24,ha='center',va='center')
            return


        # Plot linear image
        if (show_map) :
            axs.imshow(self.Iarr,origin='lower',extent=[self.xarr[0,0],self.xarr[0,-1],self.yarr[0,0],self.yarr[-1,0]],cmap='afmhot',vmin=0,interpolation='spline16')
            
        # Plot the log contours
        if (show_contours) :
            lI = np.log10(np.maximum(0.0,self.Iarr)/np.max(self.Iarr)+1e-20)
            lmI = np.log10(np.maximum(0.0,-self.Iarr)/np.max(self.Iarr)+1e-20)
            
            lev10lo = max(np.min(lI[self.Iarr>0]),-4)
            lev10 = np.sort( -np.arange(0,lev10lo,-1) )
            axs.contour(self.xarr,self.yarr,-lI,levels=lev10,colors='cornflowerblue',alpha=0.5)
            #plt.contour(self.x,self.y,-lmI,levels=lev10,colors='green',alpha=0.5)            
            lev1 = []
            for l10 in -lev10[1:] :
                lev1.extend( np.log10(np.array([2,3,4,5,6,7,8,9])) + l10 )
            lev1 = np.sort(-np.array(lev1))
            axs.contour(self.xarr,self.yarr,-lI,levels=lev1,colors='cornflowerblue',alpha=0.5,linewidths=0.5)
            axs.contour(self.xarr,self.yarr,-lmI,levels=lev1[-10:],colors='green',alpha=0.5,linewidths=0.5)

        # Fix the limits
        if (not limits is None) :
            axs.set_xlim((limits[0],limits[1]))
            axs.set_ylim((limits[2],limits[3]))
        else :
            xmin = min(np.min(self.xarr[lI>-2]),np.min(self.yarr[lI>-2]))
            xmax = max(np.max(self.xarr[lI>-2]),np.max(self.yarr[lI>-2]))
            axs.set_xlim((xmax,xmin))
            axs.set_ylim((xmin,xmax))





        

        


class InteractiveBaselineMapPlot(InteractiveWorldMapOverlayWidget):

    statdict = {}
    gcdict = {}
    lldict = {}
    
    def __init__(self,**kwargs) :

        self.statdict = {}

        super().__init__(**kwargs)
            
        self.off_color = (0.5,0,0)
        self.on_color = (1,0.75,0.25)
        
        self.gcdict = {}
        self.lldict = {}

    def generate_mpl_plot(self,fig,ax,**kwargs) :
        # This is where we insert a Matplotlib figure.  Must use ax. and fig. child commands.
        # You probably want, but do not require, the following in your over-lay
        self.plot_map(ax,self.statdict)
        ax.set_facecolor((0,0,0,0))
        fig.set_facecolor((0,0,0,0))
        
    def update(self,datadict,statdict,**kwargs) :

        self.statdict = statdict
        if (__mydebug__):
            print("InteractiveBaselineMapPlot.update:",self.statdict.keys())

        if (list(self.lldict.keys()) != list(self.statdict.keys())) :
            if (__mydebug__):
                print("InteractiveBaselineMapPlot.update: remaking circles")
            lims=[-180,180,-90,90]
            self.generate_all_station_latlon(statdict)
            if ('SP' in self.statdict.keys()) :
                self.lldict['SP']=[-89.0, 0.5*(lims[0]+lims[1])]
            self.generate_all_great_circles(self.lldict, lims)
            
        self.update_mpl(**kwargs)

    def replot(self,datadict,statdict,**kwargs) :

        if (__mydebug__):
            print("InteractiveBaselineMapPlot.replot:",self.statdict.keys())

        self.update(datadict,statdict,**kwargs)
        
    # limits is a list that has in degrees the min longitude, max longitude, min latitude, max latitude to be plotted.
    def plot_map(self,axs,statdict) :
        if (__mydebug__):
            print("InteractiveBaselineMapPlot.plot_map:",statdict.keys())
        lims=[-180,180,-90,90]
        for i in self.gcdict.keys() :
            if (self.statdict[self.gcdict[i]['s1']]['on']==False or self.statdict[self.gcdict[i]['s2']]['on']==False) :
                axs.plot(self.gcdict[i]['x'],self.gcdict[i]['y'],'-',color=self.off_color,alpha=0.5)
                axs.plot(self.gcdict[i]['x']-360,self.gcdict[i]['y'],'-',color=self.off_color,alpha=0.5)
        for i in self.gcdict.keys() :
            if (self.statdict[self.gcdict[i]['s1']]['on']==True and self.statdict[self.gcdict[i]['s2']]['on']==True) :
                axs.plot(self.gcdict[i]['x'],self.gcdict[i]['y'],'-',color=self.on_color,alpha=0.5)
                axs.plot(self.gcdict[i]['x']-360,self.gcdict[i]['y'],'-',color=self.on_color,alpha=0.5)
        for s in self.statdict.keys() :
            if (self.statdict[s]['on']==False) :
                axs.plot(self.lldict[s][1], self.lldict[s][0], 'o', color = self.off_color)
        for s in self.statdict.keys() :
            if (self.statdict[s]['on']==True) :
                axs.plot(self.lldict[s][1], self.lldict[s][0], 'o', color = self.on_color)
        # Set limits
        axs.set_xlim((lims[:2]))
        axs.set_ylim((lims[2:]))
        # Eliminate axes
        for sdir in ['left','right','top','bottom'] :
            axs.spines[sdir].set_visible(False)
        axs.xaxis.set_tick_params(bottom='off',top='off')
        axs.yaxis.set_tick_params(left='off',right='off')
        
    def generate_all_station_latlon(self, statdict) :
        self.lldict = {}
        for s in statdict.keys():
            self.lldict[s] = self.xyz_to_latlon(statdict[s]['loc'])
        return statdict

    def generate_all_great_circles(self,lldict,limits,N=64) :
        self.gcdict = {}
        i = 0
        for k,s1 in enumerate(list(lldict.keys())) :
            for s2 in list(lldict.keys())[(k+1):] :
                ll1 = lldict[s1]
                ll2 = lldict[s2]
                llgA = self.great_circle(ll1,ll2,N=N)
                lonc = 0.5*(limits[0]+limits[1])
                y = llgA[0]
                x = llgA[1] - (llgA[1][0]-lonc) + (llgA[1][0]-lonc)%360 
                x,y = self.resample_by_length(x,y,N=32)
                self.gcdict[i] = {'s1':s1,'s2':s2,'x':x,'y':y}
                i += 1

    def resample_by_length(self,x0,y0,N=None) :
        ds = np.sqrt( (x0[1:]-x0[:-1])**2 + (y0[1:]-y0[:-1])**2 )
        s = np.cumsum(ds)
        s = np.append([0],s/s[-1])
        if (N is None) :
            N = len(x0)
        t = np.linspace(0,1,N)
        x = np.interp(t,s,x0)
        y = np.interp(t,s,y0)
        return x,y
                
    def great_circle(self,pos1,pos2,N=32) :

        lat1, lon1 = pos1
        lat2, lon2 = pos2

        # First, rotate about z so that latlon1 is in the x-z plane
        ll1 = [lat1, 0]
        ll2 = [lat2, lon2-lon1]
    
        # Second, rotate about y so that ll1 is at the north pole
        ll1 = self.xyz_to_latlon(self.RotateY(self.latlon_to_xyz(ll1),angle=-(90-lat1)))
        ll2 = self.xyz_to_latlon(self.RotateY(self.latlon_to_xyz(ll2),angle=-(90-lat1)))
    
        # Third, generate a great circle that goes through the pole (easy) and ll2 (not hard)
        latA = np.linspace(ll2[0],90.0,N)
        lonA = 0*latA + ll2[1]
        llgA = np.array([latA,lonA])

        # Fourth, unrotate about y
        llgA = self.xyz_to_latlon(self.RotateY(self.latlon_to_xyz(llgA),angle=(90-lat1)))
        llgA[1] = llgA[1] + lon1

        return llgA

    def latlon_to_xyz(self,latlon,radius=1) :

        lat_rad = latlon[0]*np.pi/180.
        lon_rad = latlon[1]*np.pi/180.

        x = radius*np.cos(lat_rad)*np.cos(lon_rad)
        y = radius*np.cos(lat_rad)*np.sin(lon_rad)
        z = radius*np.sin(lat_rad)

        return np.array([x,y,z])

    def xyz_to_latlon(self,xyz) :
        lat = np.arcsin( xyz[2]/np.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2) ) * 180./np.pi
        lon = np.arctan2( xyz[1], xyz[0] ) * 180.0/np.pi
        return np.array([lat,lon])

    def RotateY(self,xyz,angle=0) :

        angle = angle*np.pi/180.0
        xyz2 = 0*xyz
        xyz2[0] = xyz[0]*np.cos(angle) + xyz[2]*np.sin(angle)
        xyz2[1] = xyz[1]
        xyz2[2] = xyz[2]*np.cos(angle) - xyz[0]*np.sin(angle)

        return xyz2

    def check_size(self,size) :
        if (size[0]==0 or size[1]==0) :
            return size
        if (size[0]<self.width and size[1]<self.height) :
            if (self.width/size[0] < self.height/size[1]) :
                size = (self.width, size[1]/size[0] * self.width)
            else :
                size = (size[0]/size[1] * self.height, self.height)
        return size


    
class InteractiveBaselineMapPlot_kivygraph(InteractiveWorldMapWidget):

    statdict = {}
    gcdict = {}
    lldict = {}
    
    def __init__(self,**kwargs) :

        self.statdict = {}

        super().__init__(**kwargs)
            
        self.off_color = (0.5,0,0)
        self.on_color = (1,0.75,0.25)
        
        self.gcdict = {}
        self.lldict = {}

    def generate_mpl_plot(self,fig,ax,**kwargs) :
        # This is where we insert a Matplotlib figure.  Must use ax. and fig. child commands.
        # You probably want, but do not require, the following in your over-lay
        self.plot_map(ax,self.statdict)
        ax.set_facecolor((0,0,0,0))
        fig.set_facecolor((0,0,0,0))
        
    def update(self,datadict,statdict,**kwargs) :

        self.statdict = statdict
        if (__mydebug__):
            print("InteractiveBaselineMapPlot.update:",self.statdict.keys())

        if (list(self.lldict.keys()) != list(self.statdict.keys())) :
            if (__mydebug__):
                print("InteractiveBaselineMapPlot.update: remaking circles")
            lims=[-180,180,-90,90]
            self.generate_all_station_latlon(statdict)
            if ('SP' in self.statdict.keys()) :
                self.lldict['SP']=[-85.0, 0.5*(lims[0]+lims[1])]
            self.generate_all_great_circles(self.lldict, lims)

        # self.bmc.plot_stations(self.statdict,self.lldict,self.gcdict,self.rect.size)
        # self.update_mpl(**kwargs)

    def replot(self,datadict,statdict,**kwargs) :

        if (__mydebug__):
            print("InteractiveBaselineMapPlot.replot:",self.statdict.keys())

        self.update(datadict,statdict,**kwargs)


                    
    # limits is a list that has in degrees the min longitude, max longitude, min latitude, max latitude to be plotted.
    def plot_map(self,axs,statdict) :
        if (__mydebug__):
            print("InteractiveBaselineMapPlot.plot_map:",statdict.keys())
        lims=[-180,180,-90,90]
        for i in self.gcdict.keys() :
            if (self.statdict[self.gcdict[i]['s1']]['on']==False or self.statdict[self.gcdict[i]['s2']]['on']==False) :
                axs.plot(self.gcdict[i]['x'],self.gcdict[i]['y'],'-',color=self.off_color,alpha=0.5)
                axs.plot(self.gcdict[i]['x']-360,self.gcdict[i]['y'],'-',color=self.off_color,alpha=0.5)
        for i in self.gcdict.keys() :
            if (self.statdict[self.gcdict[i]['s1']]['on']==True and self.statdict[self.gcdict[i]['s2']]['on']==True) :
                axs.plot(self.gcdict[i]['x'],self.gcdict[i]['y'],'-',color=self.on_color,alpha=0.5)
                axs.plot(self.gcdict[i]['x']-360,self.gcdict[i]['y'],'-',color=self.on_color,alpha=0.5)
        for s in self.statdict.keys() :
            if (self.statdict[s]['on']==False) :
                axs.plot(self.lldict[s][1], self.lldict[s][0], 'o', color = self.off_color)
        for s in self.statdict.keys() :
            if (self.statdict[s]['on']==True) :
                axs.plot(self.lldict[s][1], self.lldict[s][0], 'o', color = self.on_color)
        # Set limits
        axs.set_xlim((lims[:2]))
        axs.set_ylim((lims[2:]))
        # Eliminate axes
        for sdir in ['left','right','top','bottom'] :
            axs.spines[sdir].set_visible(False)
        axs.xaxis.set_tick_params(bottom='off',top='off')
        axs.yaxis.set_tick_params(left='off',right='off')
        
    def generate_all_station_latlon(self, statdict) :
        self.lldict = {}
        for s in statdict.keys():
            self.lldict[s] = self.xyz_to_latlon(statdict[s]['loc'])
        return statdict

    def generate_all_great_circles(self,lldict,limits,N=128) :
        self.gcdict = {}
        i = 0
        for k,s1 in enumerate(list(lldict.keys())) :
            for s2 in list(lldict.keys())[(k+1):] :
                ll1 = lldict[s1]
                ll2 = lldict[s2]
                llgA = self.great_circle(ll1,ll2,N=N)
                lonc = 0.5*(limits[0]+limits[1])
                y = llgA[0]
                x = llgA[1] - (llgA[1][0]-lonc) + (llgA[1][0]-lonc)%360 
                x,y = self.resample_by_length(x,y,N=32)
                self.gcdict[i] = {'s1':s1,'s2':s2,'x':x,'y':y}
                i += 1

    def resample_by_length(self,x0,y0,N=None) :
        ds = np.sqrt( (x0[1:]-x0[:-1])**2 + (y0[1:]-y0[:-1])**2 )
        s = np.cumsum(ds)
        s = np.append([0],s/s[-1])
        if (N is None) :
            N = len(x0)
        t = np.linspace(0,1,N)
        x = np.interp(t,s,x0)
        y = np.interp(t,s,y0)
        
        return x,y

                
    def great_circle(self,pos1,pos2,N=32) :

        lat1, lon1 = pos1
        lat2, lon2 = pos2

        # First, rotate about z so that latlon1 is in the x-z plane
        ll1 = [lat1, 0]
        ll2 = [lat2, lon2-lon1]
    
        # Second, rotate about y so that ll1 is at the north pole
        ll1 = self.xyz_to_latlon(self.RotateY(self.latlon_to_xyz(ll1),angle=-(90-lat1)))
        ll2 = self.xyz_to_latlon(self.RotateY(self.latlon_to_xyz(ll2),angle=-(90-lat1)))
    
        # Third, generate a great circle that goes through the pole (easy) and ll2 (not hard)
        latA = np.linspace(ll2[0],90.0,N)
        lonA = 0*latA + ll2[1]
        llgA = np.array([latA,lonA])

        # Fourth, unrotate about y
        llgA = self.xyz_to_latlon(self.RotateY(self.latlon_to_xyz(llgA),angle=(90-lat1)))
        llgA[1] = llgA[1] + lon1

        return llgA

    def latlon_to_xyz(self,latlon,radius=1) :

        lat_rad = latlon[0]*np.pi/180.
        lon_rad = latlon[1]*np.pi/180.

        x = radius*np.cos(lat_rad)*np.cos(lon_rad)
        y = radius*np.cos(lat_rad)*np.sin(lon_rad)
        z = radius*np.sin(lat_rad)

        return np.array([x,y,z])

    def xyz_to_latlon(self,xyz) :
        lat = np.arcsin( xyz[2]/np.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2) ) * 180./np.pi
        lon = np.arctan2( xyz[1], xyz[0] ) * 180.0/np.pi
        return np.array([lat,lon])

    def RotateY(self,xyz,angle=0) :

        angle = angle*np.pi/180.0
        xyz2 = 0*xyz
        xyz2[0] = xyz[0]*np.cos(angle) + xyz[2]*np.sin(angle)
        xyz2[1] = xyz[1]
        xyz2[2] = xyz[2]*np.cos(angle) - xyz[0]*np.sin(angle)

        return xyz2

    def check_size(self,size) :
        if (size[0]==0 or size[1]==0) :
            return size

        # if (size[0]<self.width and size[1]<self.height) :
        #     if (self.width/size[0] < self.height/size[1]) :
        #         size = (self.width, size[1]/size[0] * self.width)
        #     else :
        #         size = (size[0]/size[1] * self.height, self.height)

        if (size[0]<Window.width and size[1]<Window.height) :
            if (Window.width/size[0] < Window.height/size[1]) :
                size = (Window.width, size[1]/size[0] * Window.width)
            else :
                size = (size[0]/size[1] * Window.height, Window.height)
                
        if __mydebug__ :
            print("InteractiveBaselineMapPlot_kivygraph.check_size",self.width,self.height,size,Window.width,Window.height)
        return size

    

    
class BaselineMapCanvas(FloatLayout) :

    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
        self.off_color = (0.5,0,0)
        self.on_color = (1,0.75,0.25)
        
    def plot_stations(self,statdict,lldict,gcdict,rect) :
        if (__mydebug__):
            print("BaselineMapCanvas.plot_stations:",statdict.keys())

        if (rect.size[0]==0  or rect.size[1]==0) :
            return
            
        lims=[-180,180,-90,90]
        lon_to_xpx_scale = rect.size[0]/(lims[1]-lims[0]) 
        lon_to_xpx_offset = lon_to_xpx_scale*(-lims[0]) + rect.pos[0] - rect.tex_coords[0]*rect.size[0]/(rect.tex_coords[2]-rect.tex_coords[0])
        lat_to_ypx_scale = rect.size[1]/(lims[3]-lims[2])
        lat_to_ypx_offset = lat_to_ypx_scale*(-lims[2]) + rect.pos[1] - rect.tex_coords[5]*rect.size[1]/(rect.tex_coords[5]-rect.tex_coords[1])

        
        # Index manipulation stuff
        j = np.arange(len(gcdict[0]['x']))
        j2 = 2*j
        j2p1 = 2*j+1
        points = np.arange(2*len(j))
        
        linewidth = 2

        # Get the current limits.
        self.canvas.clear()        
        with self.canvas :

            igc = 0
            Color(self.off_color[0],self.off_color[1],self.off_color[2],0.5)
            for k,s1 in enumerate(list(statdict.keys())) :
                for s2 in list(statdict.keys())[(k+1):] :
                    if (statdict[s1]['on']==False or statdict[s2]['on']==False) :
                        total_x_shift = ( (lon_to_xpx_scale*gcdict[igc]['x'][0] + lon_to_xpx_offset)//rect.size[0]  )*rect.size[0] - 0.5*linewidth
                        
                        points[j2] = lon_to_xpx_scale*gcdict[igc]['x'] + lon_to_xpx_offset - total_x_shift
                        points[j2p1] = lat_to_ypx_scale*gcdict[igc]['y'] + lat_to_ypx_offset
                        Line(points=list(points),width=linewidth)
                        if (points[0]<0 or points[-2]<0) :
                            points[j2] = points[j2]+lon_to_xpx_scale*360 
                            Line(points=list(points),width=linewidth)
                            points[j2] = points[j2]-lon_to_xpx_scale*360
                        if (points[0]>self.width or points[-2]>self.width) :
                            points[j2] = points[j2]-lon_to_xpx_scale*360
                            Line(points=list(points),width=linewidth)
                    igc += 1

            igc = 0
            Color(self.on_color[0],self.on_color[1],self.on_color[2],0.5)
            for k,s1 in enumerate(list(statdict.keys())) :
                for s2 in list(statdict.keys())[(k+1):] :
                    if (statdict[s1]['on']==True and statdict[s2]['on']==True) :
                        total_x_shift = ( (lon_to_xpx_scale*gcdict[igc]['x'][0] + lon_to_xpx_offset)//rect.size[0]  )*rect.size[0] - 0.5*linewidth
                        
                        points[j2] = lon_to_xpx_scale*gcdict[igc]['x'] + lon_to_xpx_offset - total_x_shift
                        points[j2p1] = lat_to_ypx_scale*gcdict[igc]['y'] + lat_to_ypx_offset
                        Line(points=list(points),width=linewidth)
                        if (points[0]<0 or points[-2]<0) :
                            points[j2] = points[j2]+lon_to_xpx_scale*360 
                            Line(points=list(points),width=linewidth)
                            points[j2] = points[j2]-lon_to_xpx_scale*360
                        if (points[0]>self.width or points[-2]>self.width) :
                            points[j2] = points[j2]-lon_to_xpx_scale*360
                            Line(points=list(points),width=linewidth)
                    igc += 1


            for s in statdict.keys() :
                if (statdict[s]['on']==False) :
                    xpx = (lon_to_xpx_scale*lldict[s][1] + lon_to_xpx_offset)%(rect.size[0])
                    ypx = lat_to_ypx_scale*lldict[s][0] + lat_to_ypx_offset
                    Color(0,0,0,0.1)
                    Ellipse(pos=(xpx-dp(7),ypx-dp(7)),size=(dp(14),dp(14)))
                    Ellipse(pos=(xpx-dp(6),ypx-dp(6)),size=(dp(12),dp(12)))
                    Color(self.off_color[0],self.off_color[1],self.off_color[2])
                    Ellipse(pos=(xpx-dp(5),ypx-dp(5)),size=(dp(10),dp(10)))
                    # if (__mydebug__) :
                    #     print("Adding OFF circle for",s,xpx,ypx,self.on_color,self.height,rect.size,rect.pos)
                    
            for s in statdict.keys() :
                if (statdict[s]['on']==True) :
                    xpx = (lon_to_xpx_scale*lldict[s][1] + lon_to_xpx_offset)%(rect.size[0])
                    ypx = lat_to_ypx_scale*lldict[s][0] + lat_to_ypx_offset
                    Color(0,0,0,0.1)
                    Ellipse(pos=(xpx-dp(7),ypx-dp(7)),size=(dp(14),dp(14)))
                    Ellipse(pos=(xpx-dp(6),ypx-dp(6)),size=(dp(12),dp(12)))
                    Color(self.on_color[0],self.on_color[1],self.on_color[2])
                    Ellipse(pos=(xpx-dp(5),ypx-dp(5)),size=(dp(10),dp(10)))
                    # if (__mydebug__) :
                    #     print("Adding ON circle for",s,xpx,ypx,self.on_color,self.height,rect.size,rect.pos)

    

# dd = read_data('./data/V_M87_ngeht_ref1_230_thnoise_scanavg_tygtd.dat')
# sd = read_array('./arrays/ngeht_ref1_230_ehtim.txt')

# plt.figure()


# rp = CheapImageReconstruction()
# x,y,I = reconstruct_image(dd,sd)


# bp = BaselinePlots()

# bp.plot_baselines(dd,sd)

# plt.show()

# mp = MapPlots()
# plt.show()
