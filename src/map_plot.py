import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi
import matplotlib.tri as tri
import copy
from os import path

from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Ellipse, Color, Line, Point
from kivy.metrics import dp, sp
from kivy.uix.label import Label
from kivy.core.window import Window

from mpl_texture import InteractivePlotWidget, InteractiveWorldMapWidget

import hashlib


__map_plot_debug__ = True


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
        if (__map_plot_debug__):
            print("InteractiveBaselineMapPlot.update:",self.statdict.keys())

        if (list(self.lldict.keys()) != list(self.statdict.keys())) :
            if (__map_plot_debug__):
                print("InteractiveBaselineMapPlot.update: remaking circles")
            lims=[-180,180,-90,90]
            self.generate_all_station_latlon(statdict)
            if ('SP' in self.statdict.keys()) :
                self.lldict['SP']=[-85.0, 0.5*(lims[0]+lims[1])]
            self.generate_all_great_circles(self.lldict, lims)

        # self.bmc.plot_stations(self.statdict,self.lldict,self.gcdict,self.rect.size)
        # self.update_mpl(**kwargs)

    def replot(self,datadict,statdict,**kwargs) :

        if (__map_plot_debug__):
            print("InteractiveBaselineMapPlot.replot:",self.statdict.keys())

        self.update(datadict,statdict,**kwargs)


                    
    # limits is a list that has in degrees the min longitude, max longitude, min latitude, max latitude to be plotted.
    def plot_map(self,axs,statdict) :
        if (__map_plot_debug__):
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
                
        if (__map_plot_debug__) :
            print("InteractiveBaselineMapPlot_kivygraph.check_size",self.width,self.height,size,Window.width,Window.height)
        return size

    

    
class BaselineMapCanvas(FloatLayout) :

    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
        self.off_color = (0.5,0,0)
        self.on_color = (1,0.75,0.25)
        
    def plot_stations(self,statdict,lldict,gcdict,rect) :
        if (__map_plot_debug__):
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
                    # if (__map_plot_debug__) :
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
                    # if (__map_plot_debug__) :
                    #     print("Adding ON circle for",s,xpx,ypx,self.on_color,self.height,rect.size,rect.pos)

