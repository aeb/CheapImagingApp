import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi
import matplotlib.tri as tri
import copy
from os import path

# from kivy.uix.floatlayout import FloatLayout
# from kivy.graphics import Ellipse, Color, Line, Point
# from kivy.metrics import dp, sp
# from kivy.uix.label import Label
# from kivy.core.window import Window

#from mpl_texture import InteractivePlotWidget, InteractiveWorldMapWidget
from mpl_texture import InteractivePlotWidget

import hashlib


__cheap_image_debug__ = True


class InteractiveImageReconstructionPlot(InteractivePlotWidget) :
    
    def __init__(self,**kwargs) :

        self.xarr = 0
        self.yarr = 0
        self.Iarr = 1

        self.ddict = {}
        self.sdict = {}

        self.argument_hash = None
        
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

        if (__cheap_image_debug__) :
            print("InteractiveImageReconstructionPlot.generate_mpl_plot: start")
        
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
