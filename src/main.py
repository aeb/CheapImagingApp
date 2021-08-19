__version__ = "0.5"

__mydebug__ = False

from kivy.app import App
from kivymd.app import MDApp
from kivy.lang import Builder

from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.properties import StringProperty, NumericProperty, ObjectProperty, VariableListProperty, BooleanProperty, ListProperty
from kivy.clock import Clock

from kivy.uix.screenmanager import Screen, ScreenManager, FadeTransition, SlideTransition

from kivy.metrics import dp
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.behaviors.touchripple import TouchRippleBehavior


from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.toolbar import MDBottomAppBar
from kivymd.uix.behaviors import CircularRippleBehavior

import numpy as np
import matplotlib.pyplot as plt
#from kivy.garden.matplotlib import FigureCanvasKivyAgg
from backend_kivyagg import FigureCanvasKivyAgg

from kivy.core.window import Window

from fancy_mdslider import FancyMDSlider

from os import path

####################
# TESTING
# Window.size = (300,500)
##################



import copy

import cheap_image as ci

_on_color = (1,0.75,0.25,1)
_off_color = (0.5,0,0,1)

_time_range = [0,24]
_ngeht_diameter = 6
_snr_cut = 0
_ngeht_diameter_setting = 6
_snr_cut_setting = 0


_existing_arrays = ['EHT 2017','EHT 2022']
_existing_station_list = ['PV','AZ','SM','LM','AA','SP','JC','GL','PB','KP','HA']

_stationdicts={}
# _stationdicts['ngEHT ref1']=ci.read_array('./arrays/ngeht_ref1_230_ehtim.txt',existing_station_list=_existing_station_list)
# _stationdicts['EHT 2017']=ci.read_array('./arrays/eht2017_230_ehtim.txt',existing_station_list=_existing_station_list)
# _stationdicts['EHT 2022']=ci.read_array('./arrays/eht2022_230_ehtim.txt',existing_station_list=_existing_station_list)

_stationdicts={}
_stationdicts['ngEHT ref1']=ci.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/ngeht_ref1_230_ehtim.txt')), existing_station_list=_existing_station_list)
_stationdicts['EHT 2017']=ci.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/eht2017_230_ehtim.txt')),existing_station_list=_existing_station_list)
_stationdicts['EHT 2022']=ci.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/eht2022_230_ehtim.txt')),existing_station_list=_existing_station_list)


_array = list(_stationdicts.keys())[0]
_array_index = 0

_statdict=copy.deepcopy(_stationdicts['ngEHT ref1'])
_datadict=ci.read_data(path.abspath(path.join(path.dirname(__file__),'data/V_M87_ngeht_ref1_230_perfect_scanavg_tygtd.dat')))


class ReconstructionPlot(BoxLayout) :

    plot_maxsize = 500.0
    plot_center = np.array([0.0,0.0])

    img = ci.CheapImageReconstruction()
    menu_id = ObjectProperty(None)

    def __init__(self,**kwargs) :
        super().__init__(**kwargs)

        self.fig=plt.figure(figsize=(6,6))
        self.axs=plt.axes([0,0,1,1])

        self.time_range = _time_range
        self.ngeht_diameter = _ngeht_diameter
        self.snr_cut = _snr_cut
        
        limits = np.array([1,-1,-1,1])*self.plot_maxsize
        limits[:2] = limits[:2] + self.plot_center[0]
        limits[2:] = limits[2:] + self.plot_center[1]
        
        self.img.plot_image_reconstruction(self.axs,_datadict,_statdict,time_range=self.time_range,snr_cut=self.snr_cut,ngeht_diameter=self.ngeht_diameter,limits=limits)
        #self.add_widget(FigureCanvasKivyAgg(self.fig))

        self.ddict = _datadict
        self.sdict = _statdict

        self.bind(height=self.resize)
        self.bind(width=self.resize)

        self.freeze_plot = False

        if __mydebug__ :
            print("rp.__init__: finished")
        

    def update(self,datadict,statdict) :
        global _datadict, _statdict
        _datadict = datadict
        _statdict = statdict
        self.ddict = _datadict
        self.sdict = _statdict

        if __mydebug__ :
            print("rp.update:",self.sdict.keys(),self.fig,self.size)
            print("         :",_statdict.keys(),self.fig,self.size)
        
        self.replot()
        
        
    def replot(self) :
        global _datadict, _statdict
        self.ddict = _datadict
        self.sdict = _statdict
        
        if __mydebug__ :
            print("rp.replot:",self.sdict.keys(),self.fig,self.size)
            print("         :",_statdict.keys(),self.fig,self.size)

        if (self.width==0 or self.height==0) :
            self.clear()
            return

        if (self.fig is None) :
            self.fig=plt.figure()
            self.axs=plt.axes([0,0,1,1])

        limits = np.array([1,-1,-1,1])*self.plot_maxsize
        if (self.width==0 or self.height==0) :
            pass
        elif (self.width>self.height) :
            limits[:2] = limits[:2] + self.plot_center[0]
            limits[2:] = float(self.height)/float(self.width)*limits[2:] + self.plot_center[1]
        else :
            limits[:2] = float(self.width)/float(self.height)*limits[:2] + self.plot_center[0]
            limits[2:] = limits[2:] + self.plot_center[1]

        self.axs.cla()
        self.clear_widgets()
        self.img.plot_image_reconstruction(self.axs,self.ddict,self.sdict,time_range=self.time_range,snr_cut=self.snr_cut,ngeht_diameter=self.ngeht_diameter,limits=limits)
        self.add_widget(FigureCanvasKivyAgg(self.fig))

    def refresh(self) :
        if __mydebug__ :
            print("rp.refresh:",self.sdict.keys(),self.fig,self.size)
            print("          :",_statdict.keys(),self.fig,self.size)
        
        if (self.width==0 or self.height==0) :
            self.clear()
            return

        if (self.fig is None) :
            self.fig=plt.figure()
            self.axs=plt.axes([0,0,1,1])
            
        limits = np.array([1,-1,-1,1])*self.plot_maxsize
        if (self.width==0 or self.height==0) :
            pass
        elif (self.width>self.height) :
            limits[:2] = limits[:2] + self.plot_center[0]
            limits[2:] = float(self.height)/float(self.width)*limits[2:] + self.plot_center[1]
        else :
            limits[:2] = float(self.width)/float(self.height)*limits[:2] + self.plot_center[0]
            limits[2:] = limits[2:] + self.plot_center[1]

        self.axs.cla()
        self.clear_widgets()
        self.img.replot_image_reconstruction(self.axs,time_range=self.time_range,limits=limits)
        self.add_widget(FigureCanvasKivyAgg(self.fig))

        
    def resize(self,widget,newsize) :
        if __mydebug__ :
            print("rp.resize:",newsize)
        # self.refresh()
        self.replot()
            
    def on_touch_move(self,touch) :
        if (not self.freeze_plot) :
            screen_to_plot = 2.0*float(self.plot_maxsize)/float(max(self.height,self.width))
            self.plot_center[0] = self.plot_center[0] + touch.dpos[0]*screen_to_plot
            self.plot_center[1] = self.plot_center[1] - touch.dpos[1]*screen_to_plot
            self.refresh()

    def on_touch_down(self,touch) :
        if (touch.is_double_tap) :
            self.plot_maxsize = 500.0
            self.plot_center = np.array([0,0])
            self.refresh()

    # def on_touch_up(self,touch) :
    #     self.freeze_plot = False

    def zoom_out(self) :
        self.plot_maxsize = self.plot_maxsize * 1.414
        self.refresh()

    def zoom_in(self) :
        self.plot_maxsize = self.plot_maxsize * 0.707
        self.refresh()

    def clear(self) :
        if (not self.fig is None) :
            plt.close(self.fig)
            self.fig=None
        self.clear_widgets()

    def set_start_time(self,val) :
        self.time_range[1] = self.time_range[1]-self.time_range[0]+val
        self.time_range[0] = val
        self.replot()
        
    def set_obs_time(self,val) :
        self.time_range[1] = self.time_range[0] + val
        self.replot()

    def set_ngeht_diameter(self,val) :
        self.ngeht_diameter = val
        _ngeht_diameter = self.ngeht_diameter
        self.replot()

    def set_snr_cut(self,val) :
        self.snr_cut = val
        if (val is None) :
            self.snr_cut = 0
        _snr_cut = self.snr_cut
        self.replot()

        
class BaselinesPlot(BoxLayout) :

    plot_maxsize = 10.0
    plot_center = np.array([0.0,0.0])

    blp = ci.BaselinePlots()
    menu_id = ObjectProperty(None)

    def __init__(self,**kwargs) :
        super().__init__(**kwargs)

        self.fig=plt.figure(figsize=(6,6))
        self.axs=plt.axes([0,0,1,1])

        self.time_range = _time_range
        self.ngeht_diameter = _ngeht_diameter
        self.snr_cut = _snr_cut
        
        limits = np.array([1,-1,-1,1])*self.plot_maxsize
        limits[:2] = limits[:2] + self.plot_center[0]
        limits[2:] = limits[2:] + self.plot_center[1]
        
        self.blp.plot_baselines(self.axs,_datadict,_statdict,time_range=self.time_range,limits=limits,snr_cut=self.snr_cut,ngeht_diameter=self.ngeht_diameter)
        self.add_widget(FigureCanvasKivyAgg(self.fig))

        self.ddict = _datadict
        self.sdict = _statdict

        self.bind(height=self.resize)
        self.bind(width=self.resize)

        self.freeze_plot = False

        if __mydebug__ :
            print("bp.__init__: finished")

        
    def update(self,datadict,statdict) :
        global _datadict, _statdict
        _datadict = datadict
        _statdict = statdict
        self.ddict = _datadict
        self.sdict = _statdict
        
        if __mydebug__ :
            print("bp.update:",self.sdict.keys(),self.fig,self.size)
            print("         :",_statdict.keys(),self.fig,self.size)
        
        self.replot()
        
    def replot(self) :
        global _datadict, _statdict
        self.ddict = _datadict
        self.sdict = _statdict
        
        # Why is this not updating the self.sdict properly until we swap twice?
        if __mydebug__ :
            print("bp.replot:",self.sdict.keys(),self.fig,self.size)
            print("         :",_statdict.keys(),self.fig,self.size)
        
        if (self.width==0 or self.height==0) :
            self.clear()
            return

        if (self.fig is None) :
            self.fig=plt.figure()
            self.axs=plt.axes([0,0,1,1])

        limits = np.array([1,-1,-1,1])*self.plot_maxsize
        if (self.width==0 or self.height==0) :
            pass
        elif (self.width>self.height) :
            limits[:2] = limits[:2] + self.plot_center[0]
            limits[2:] = float(self.height)/float(self.width)*limits[2:] + self.plot_center[1]
        else :
            limits[:2] = float(self.width)/float(self.height)*limits[:2] + self.plot_center[0]
            limits[2:] = limits[2:] + self.plot_center[1]

            
        self.axs.cla()
        self.clear_widgets()
        self.blp.plot_baselines(self.axs,self.ddict,self.sdict,time_range=self.time_range,limits=limits,snr_cut=self.snr_cut,ngeht_diameter=self.ngeht_diameter)
        self.add_widget(FigureCanvasKivyAgg(self.fig))
        
    def refresh(self):
        if __mydebug__ :
            print("bp.refresh:",self.sdict.keys(),self.fig,self.size)
            print("          :",_statdict.keys(),self.fig,self.size)

        if (self.width==0 or self.height==0) :
            self.clear()
            return

        if (self.fig is None) :
            self.fig=plt.figure()
            self.axs=plt.axes([0,0,1,1])

        limits = np.array([1,-1,-1,1])*self.plot_maxsize
        if (self.width==0 or self.height==0) :
            pass
        elif (self.width>self.height) :
            limits[:2] = limits[:2] + self.plot_center[0]
            limits[2:] = float(self.height)/float(self.width)*limits[2:] + self.plot_center[1]
        else :
            limits[:2] = float(self.width)/float(self.height)*limits[:2] + self.plot_center[0]
            limits[2:] = limits[2:] + self.plot_center[1]

            
        self.axs.cla()
        self.clear_widgets()
        self.blp.replot(self.axs,limits=limits)
        # self.blp.plot_baselines(self.axs,self.ddict,self.sdict,time_range=self.time_range,limits=limits)
        self.add_widget(FigureCanvasKivyAgg(self.fig))
        
    def resize(self,widget,newsize) :
        if __mydebug__ :
            print("bp.resize:",newsize)
        # self.refresh()
        self.replot()
            
    def on_touch_move(self,touch) :
        if (not self.freeze_plot) :
            screen_to_plot = 2.0*float(self.plot_maxsize)/float(max(self.height,self.width))
            self.plot_center[0] = self.plot_center[0] + touch.dpos[0]*screen_to_plot
            self.plot_center[1] = self.plot_center[1] - touch.dpos[1]*screen_to_plot
            self.refresh()

    def on_touch_down(self,touch) :
        if (touch.is_double_tap) :
            self.plot_maxsize = 10.0
            self.plot_center = np.array([0,0])
            self.refresh()

    # def on_touch_up(self,touch) :
    #     self.freeze_plot = False
            
    def zoom_out(self) :
        self.plot_maxsize = self.plot_maxsize * 1.414
        self.refresh()

    def zoom_in(self) :
        self.plot_maxsize = self.plot_maxsize * 0.707
        self.refresh()

    def clear(self) :
        if (not self.fig is None) :
            plt.close(self.fig)
            self.fig=None
        self.clear_widgets()

    def set_start_time(self,val) :
        self.time_range[1] = self.time_range[1]-self.time_range[0]+val
        self.time_range[0] = val
        self.replot()
        
    def set_obs_time(self,val) :
        self.time_range[1] = self.time_range[0] + val
        self.replot()

    def set_ngeht_diameter(self,val) :
        global _ngeht_diameter
        self.ngeht_diameter = val
        _ngeht_diameter = self.ngeht_diameter
        self.replot()

    def set_snr_cut(self,val) :
        global _snr_cut
        self.snr_cut = val
        if (val is None) :
            self.snr_cut = 0
        _snr_cut = self.snr_cut
        self.replot()



class MapsPlot(BoxLayout) :

    plot_vmaxsize = 90
    plot_hmaxsize = 90
    plot_center = np.array([0.0,0.0])

    mp = ci.MapPlots()
    menu_id = ObjectProperty(None)

    def __init__(self,**kwargs) :
        super().__init__(**kwargs)

        self.time_range = _time_range
        self.ngeht_diameter = _ngeht_diameter
        self.snr_cut = _snr_cut

        
        self.fig=plt.figure(figsize=(6,6))
        self.axs=plt.axes([0,0,1,1])

        limits = np.array([-1,1,-1,1])*self.plot_vmaxsize
        limits[:2] = limits[:2] + self.plot_center[0]
        limits[2:] = limits[2:] + self.plot_center[1]
        
        self.mp.plot_map(self.axs,_statdict,limits=limits)
        self.add_widget(FigureCanvasKivyAgg(self.fig))

        self.sdict = _statdict

        self.bind(height=self.resize)
        self.bind(width=self.resize)

        self.freeze_plot = False

        if __mydebug__ :
            print("mp.__init__: finished")
        

    def update(self,datadict,statdict) :
        global _statdict
        _statdict = statdict
        self.sdict = _statdict

        if __mydebug__ :
            print("mp.update:",self.sdict.keys(),self.fig,self.size)
            print("         :",_statdict.keys(),self.fig,self.size)
        
        self.replot()
        
    def replot(self) :
        global _statdict
        self.sdict = _statdict

        if __mydebug__ :
            print("mp.replot:",self.sdict.keys(),self.fig,self.size)
            print("         :",_statdict.keys(),self.fig,self.size)


        if (self.width==0 or self.height==0) :
            self.clear()
            return

        if (self.fig is None) :
            self.fig=plt.figure()
            self.axs=plt.axes([0,0,1,1])

        limits = np.array([-self.plot_hmaxsize,self.plot_hmaxsize,-self.plot_vmaxsize,self.plot_vmaxsize])
        if (self.width==0 or self.height==0) :
            pass
        elif (self.width/self.height>self.plot_vmaxsize/self.plot_hmaxsize) :
            limits[:2] = limits[:2] + self.plot_center[0]
            limits[2:] = float(self.height)/float(self.width)*limits[:2] + self.plot_center[1]
        else :
            limits[:2] = float(self.width)/float(self.height)*limits[2:] + self.plot_center[0]
            limits[2:] = limits[2:] + self.plot_center[1]

        if (limits[1]-limits[0]>360.0) :
            old_hsize = limits[1]-limits[0]
            old_ratio = (limits[3]-limits[2])/(limits[1]-limits[0])
            limits[0] = self.plot_center[0] - 180.0
            limits[1] = self.plot_center[0] + 180.0
            limits[2] = - old_ratio * 180.0 + self.plot_center[1]
            limits[3] = old_ratio * 180.0 + self.plot_center[1]
            self.plot_hmaxsize = 180.0
            self.plot_vmaxsize = old_ratio*180.0
            
        self.axs.cla()
        self.clear_widgets()
        self.mp.plot_map(self.axs,self.sdict,limits=limits,window_size=[self.width,self.height])
        self.add_widget(FigureCanvasKivyAgg(self.fig))
        
    def refresh(self):
        if __mydebug__ :
            print("mp.refresh:",self.sdict.keys())
            print("          :",_statdict.keys())

        if (self.width==0 or self.height==0) :
            self.clear()
            return

        if (self.fig is None) :
            self.fig=plt.figure()
            self.axs=plt.axes([0,0,1,1])

        limits = np.array([-self.plot_hmaxsize,self.plot_hmaxsize,-self.plot_vmaxsize,self.plot_vmaxsize])
        if (self.width==0 or self.height==0) :
            pass
        elif (self.width/self.height>self.plot_vmaxsize/self.plot_hmaxsize) :
            limits[:2] = limits[:2] + self.plot_center[0]
            limits[2:] = float(self.height)/float(self.width)*limits[:2] + self.plot_center[1]
        else :
            limits[:2] = float(self.width)/float(self.height)*limits[2:] + self.plot_center[0]
            limits[2:] = limits[2:] + self.plot_center[1]

        if (limits[1]-limits[0]>360.0) :
            old_hsize = limits[1]-limits[0]
            old_ratio = (limits[3]-limits[2])/(limits[1]-limits[0])
            limits[0] = self.plot_center[0] - 180.0
            limits[1] = self.plot_center[0] + 180.0
            limits[2] = - old_ratio * 180.0 + self.plot_center[1]
            limits[3] = old_ratio * 180.0 + self.plot_center[1]
            self.plot_hmaxsize = 180.0
            self.plot_vmaxsize = old_ratio*180.0
            
        self.axs.cla()
        self.clear_widgets()
        self.mp.replot(self.axs,limits=limits,window_size=[self.width,self.height])
        self.add_widget(FigureCanvasKivyAgg(self.fig))
        
    def resize(self,widget,newsize) :
        if __mydebug__ :
            print("mp.resize:",newsize)
        #self.refresh()
        self.replot()
            
    def on_touch_move(self,touch) :
        if (not self.freeze_plot) :
            self.plot_center[0] = self.plot_center[0] - touch.dpos[0]*2.0*float(self.plot_hmaxsize)/float(self.width)
            self.plot_center[1] = self.plot_center[1] - touch.dpos[1]*2.0*float(self.plot_vmaxsize)/float(self.height)
            self.refresh()

    def on_touch_down(self,touch) :
        if (touch.is_double_tap) :
            self.plot_vmaxsize = 90.0
            self.plot_hmaxsize = self.width/self.height * self.plot_vmaxsize
            # print("imp:",self.plot_vmaxsize,self.plot_hmaxsize,self.width,self.height)
            self.plot_center = np.array([0,0])
            self.refresh()

    def zoom_out(self) :
        self.plot_vmaxsize = self.plot_vmaxsize * 1.414
        self.plot_hmaxsize = self.plot_hmaxsize * 1.414
        self.refresh()

    def zoom_in(self) :
        self.plot_vmaxsize = self.plot_vmaxsize * 0.707
        self.plot_hmaxsize = self.plot_hmaxsize * 0.707
        self.refresh()

    def clear(self) :
        if (not self.fig is None) :
            plt.close(self.fig)
            self.fig=None
        self.clear_widgets()

    def set_start_time(self,val) :
        self.time_range[1] = self.time_range[1]-self.time_range[0]+val
        self.time_range[0] = val
        # self.replot()
        
    def set_obs_time(self,val) :
        self.time_range[1] = self.time_range[0] + val
        # self.replot()

    def set_ngeht_diameter(self,val) :
        global _ngeht_diameter
        self.ngeht_diameter = val
        _ngeht_diameter = self.ngeht_diameter
        # self.replot()

    def set_snr_cut(self,val) :
        global _snr_cut
        self.snr_cut = val
        if (val is None) :
            self.snr_cut = 0
        _snr_cut = self.snr_cut
        # self.replot()

        

class DynamicBoxLayout(BoxLayout):

    is_open = BooleanProperty(False)

    opened_height = NumericProperty(None)
    closed_height = NumericProperty(None)

    expand_time = NumericProperty(0.5)
    fps = NumericProperty(30)

    tab_width = NumericProperty(1)
    tab_pos_x = NumericProperty(0)
    
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
        self.is_open = False
        self.animation = 'cubic'
        self.current_opened_height = self.opened_height

        
    def expand_model(self, x) :
        if (self.animation=='cubic') :
            return 6.0*(0.25*x - 0.3333333333*(x-0.5)**3 - 0.041666666667)
    
        else :
            raise ValueError("expand model animation type not defined!")


    def set_open_height(self,dt) :
        self.height = self.closed_height + self.expand_model(dt/self.expand_time)*(self.opened_height-self.closed_height)

    def set_close_height(self,dt) :
        self.height = self.closed_height + self.expand_model(1.0-dt/self.expand_time)*(self.current_opened_height-self.closed_height)
        
    def open_box(self) :
        self.current_opened_height = copy.copy(self.opened_height)
        for dt in np.linspace(0,self.expand_time,int(self.expand_time*self.fps)) :
            Clock.schedule_once( self.set_open_height , dt)
        
    def close_box(self) :
        for dt in np.linspace(0,self.expand_time,int(self.expand_time*self.fps)) :
            Clock.schedule_once( self.set_close_height , dt)

    def set_is_open(self,val) :
        self.is_open = val
            
    def toggle_state(self) :
        if (self.is_open) :
            self.close_box()
            Clock.schedule_once(lambda x : self.set_is_open(False), self.expand_time)
        else :
            self.open_box()
            Clock.schedule_once(lambda x : self.set_is_open(True), self.expand_time)

    def reset_state(self) :
        if (self.is_open) :
            self.toggle_state()
            Clock.schedule_once( lambda dt: self.toggle_state(), self.expand_time)
            
        
class VariableToggleList(StackLayout) :

    
    rpp = ObjectProperty(None)
    nstations = NumericProperty(0)
    button_size = ("30dp","30dp")

    bkgnd_color = [0,0,0,0]
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.sdict = _stationdicts[_array]
        _statdict = self.sdict
        
        self.nstations = len(self.sdict.keys())

        self.bs = []
        for s in np.sort(list(self.sdict.keys())) :
            b = ToggleButton(text=s,size_hint=(None,None),size=self.button_size,color=_on_color,background_color=self.bkgnd_color)
            b.bind(on_press=self.on_toggle)
            self.add_widget(b)
            self.bs.append(b)
            self.sdict[s]['on']=True

    def remake(self,sdict) :
        self.sdict = sdict
        self.nstations = len(self.sdict.keys())

        self.clear_widgets()
        self.bs = []
        for s in np.sort(list(self.sdict.keys())) :
            if (self.sdict[s]['on']) :
                b = ToggleButton(text=s,size_hint=(None,None),size=self.button_size,color=_on_color,background_color=self.bkgnd_color)
            else :
                b = ToggleButton(text=s,size_hint=(None,None),size=self.button_size,color=_off_color,background_color=self.bkgnd_color)
            b.bind(on_press=self.on_toggle)
            self.add_widget(b)
            self.bs.append(b)
            #self.sdict[s]['on']=True

        self.rpp.update(_datadict,self.sdict)
        
    def refresh(self,sdict) :
        self.sdict = sdict        
        self.nstations = len(self.sdict.keys())

        self.clear_widgets()
        self.bs = []
        for s in np.sort(list(self.sdict.keys())) :
            if ( self.sdict[s]['on'] ) :
                b = ToggleButton(text=s,size_hint=(None,None),size=self.button_size,color=_on_color,background_color=self.bkgnd_color,state="normal")
            else :
                b = ToggleButton(text=s,size_hint=(None,None),size=self.button_size,color=_off_color,background_color=self.bkgnd_color,state="down")
            b.bind(on_press=self.on_toggle)
            self.add_widget(b)
            self.bs.append(b)

        self.rpp.update(_datadict,self.sdict)
        
    def on_toggle(self,val) :

        if __mydebug__ :
            print("VariableToggleList.on_toggle:",self.rpp,self.sdict)

        for b in self.bs :
            if b.state == "normal" :
                b.color = _on_color
                self.sdict[b.text]['on']=True
            else :
                b.color = _off_color
                self.sdict[b.text]['on']=False
                
        self.rpp.update(_datadict,self.sdict)        

        if __mydebug__ :
            print("                            :",self.rpp,self.sdict)

        
    def turn_all_stations_on(self) :
        for b in self.bs:
            b.color = _on_color
            b.state = 'normal'
            self.sdict[b.text]['on']=True

        self.rpp.update(_datadict,self.sdict)        
            
        
    def turn_all_stations_off(self) :
        for b in self.bs:
            b.color = _off_color
            b.state = 'down'
            self.sdict[b.text]['on']=False

        self.rpp.update(_datadict,self.sdict)        
            
        
        
class StationMenu(DynamicBoxLayout) :

    _array = list(_stationdicts.keys())[_array_index]
    array_name = StringProperty(_array)

    rpp = ObjectProperty(None)
    menu_id = ObjectProperty(None)
    submenu_id = ObjectProperty(None)
    ddm_id = ObjectProperty(None)
    
    array_list = list(_stationdicts.keys())

    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
    
    def cycle_array_backward(self) :
        global _array_index
        _array_index = (_array_index-1+len(_stationdicts.keys()))%len(_stationdicts.keys())
        self.array_name = list(_stationdicts.keys())[_array_index]
        self.submenu_id.remake(_stationdicts[self.array_name])
        self.reset_state()

    def cycle_array_forward(self) :
        global _array_index
        _array_index = (_array_index+1)%len(_stationdicts.keys())
        self.array_name = list(_stationdicts.keys())[_array_index]
        self.submenu_id.remake(_stationdicts[self.array_name])
        self.reset_state()

    def select_array(self,array_index) :

        if __mydebug__ :
            print("StationMenu.select_array:",self.rpp,array_index)
        
        global _array_index,_statdict
        _array_index = array_index
        self.array_name = list(_stationdicts.keys())[_array_index]
        #_statdict = _stationdicts[self.array_name]
        self.submenu_id.remake(_stationdicts[self.array_name])
        self.reset_state()

        if __mydebug__ :
            print("                        :",self.rpp,array_index)
        
        
    def refresh(self) :

        if __mydebug__ :
            print("StationMenu.refresh",self.rpp)
        
        self.array_name = list(_stationdicts.keys())[_array_index]
        self.submenu_id.refresh(_stationdicts[self.array_name])
        self.reset_state()


class SMESpinnerOption(SpinnerOption):

    def __init__(self, **kwargs):
        super(SMESpinnerOption,self).__init__(**kwargs)
        self.background_normal = ''
        #self.background_down = ''
        # self.background_color = [0.14,0.14,0.14, 0.75]    # blue colour
        self.background_color = [0.77,0.55,0.17,0.7]    # blue colour        
        self.color = [1, 1, 1, 1]
        self.height = dp(30)

class SMESpinner(Spinner):

    sme_id = ObjectProperty(None)
    ddm_id = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super(SMESpinner,self).__init__(**kwargs)

        self.option_cls = SMESpinnerOption

        self.array_index_dict = {}
        for i,a in enumerate(list(_stationdicts.keys())) :
            self.array_index_dict[a] = i

        self.values = list(_stationdicts.keys())

        self.text = self.values[0]

            
    def on_selection(self,text) :
        
        if __mydebug__ :
            print("SMESpinner.on_selection:",self.text,text)
            
        self.sme_id.select_array(self.array_index_dict[text])
        self.text = self.sme_id.array_name

        if (self.text in _existing_arrays) :
            # print("SME: Disabling slider",self.ddm_id.ddm_id)
            self.ddm_id.ddm_id.dms.disabled = True
        else :
            # print("SME: Enabling slider")
            self.ddm_id.ddm_id.dms.disabled = False


class ObsTimeMenu(DynamicBoxLayout) :
    plot = ObjectProperty(None)
    ots_id = ObjectProperty(None)
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)

    def refresh(self) :

        if __mydebug__ :
            print("ObsTimeMenu.refresh",self.plot)
        
        self.ots_id.refresh()
        
    
class ObsTimeSliders(BoxLayout) :
    plot = ObjectProperty(None)
    top_menu = ObjectProperty(None)
    is_open = BooleanProperty(False)
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        self.sts_box = BoxLayout()
        self.sts_box.orientation='horizontal'
        self.sts_label = Label(text='Obs Start:',color=(1,1,1,0.75),size_hint=(0.5,1))
        self.sts_box.add_widget(self.sts_label)
        
        self.sts = StartTimeMDSlider()
        self.sts.background_color=(0,0,0,0)
        self.sts.color=(1,1,1,0.75)
        self.sts.orientation='horizontal'
        self.sts.size_hint=(1,1)
        self.sts.step=0.5
        self.sts.bind(value=self.adjust_start_time)
        self.sts.bind(active=self.on_active)
        self.sts_box.add_widget(self.sts)
        
        self.sts_label2 = Label(text="%5.1f UT"%(self.sts.value),color=(1,1,1,0.75),size_hint=(0.5,1))
        self.sts_box.add_widget(self.sts_label2)

        
        self.ots_box = BoxLayout()
        self.ots_box.orientation='horizontal'
        self.ots_label = Label(text='Duration:',color=(1,1,1,0.75),size_hint=(0.5,1))
        self.ots_box.add_widget(self.ots_label)
        
        self.ots = ObsTimeMDSlider()
        self.ots.background_color=(0,0,0,0)
        self.ots.color=(1,1,1,0.75)
        self.ots.orientation='horizontal'
        self.ots.size_hint=(1,1)
        self.ots.step=0.5
        self.ots.bind(value=self.adjust_obs_time)
        self.ots.bind(active=self.on_active)
        self.ots_box.add_widget(self.ots)

        self.ots_label2 = Label(text="%5.1f h"%(self.ots.value),color=(1,1,1,0.75),size_hint=(0.5,1))
        self.ots_box.add_widget(self.ots_label2)

    
    def toggle_state(self) :
        if (self.is_open) :
            self.is_open = False
            self.clear_widgets()
            self.top_menu.toggle_state()
        else :
            self.is_open = True
            self.top_menu.toggle_state()
            Clock.schedule_once(lambda x: self.add_widget(self.sts_box), self.top_menu.expand_time)
            Clock.schedule_once(lambda x: self.add_widget(self.ots_box), self.top_menu.expand_time)

    def refresh(self) :
        self.sts.value = _time_range[0]
        self.ots.value = _time_range[1]-_time_range[0]

    def on_active(self,widget,active) :
        if active :
            self.plot.freeze_plot = True
        else :
            self.plot.freeze_plot = False
            
    def adjust_start_time(self,widget,val) :
        self.plot.set_start_time(val)
        self.sts_label2.text = "%5.1f UT"%(val)
        
    def adjust_obs_time(self,widget,val) :
        self.plot.set_obs_time(val)
        self.ots_label2.text = "%5.1f h"%(val)


class StartTimeMDSlider(FancyMDSlider):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.min = 0
        self.max = 24
        self.value = 0
        self.show_off = False

    def hint_box_text(self,value) :
        return "%5.1f UT"%(value)

    def hint_box_size(self) :
        return (dp(50),dp(28))

    
class ObsTimeMDSlider(FancyMDSlider):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.min = 0.5
        self.max = 24
        self.value = 24
        self.show_off = False

    def hint_box_text(self,value) :
        return "%5.1f h"%(value)

    def hint_box_size(self) :
        return (dp(50),dp(28))



class DiameterMenu(DynamicBoxLayout) :
    plot = ObjectProperty(None)
    ddm_id = ObjectProperty(None)
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)

    def refresh(self) :

        if __mydebug__ :
            print("DiameterMenu.refresh",self.plot)
        
        self.ddm_id.refresh()
        
    
class DiameterSliders(BoxLayout) :
    plot = ObjectProperty(None)
    top_menu = ObjectProperty(None)
    is_open = BooleanProperty(False)
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        self.dms_box = BoxLayout()
        self.dms_box.orientation='horizontal'
        self.dms_label = Label(text='Diameter:',color=(1,1,1,0.75),size_hint=(0.5,1))
        self.dms_box.add_widget(self.dms_label)
        
        self.dms = DiameterMDSlider()
        self.dms.background_color=(0,0,0,0)
        self.dms.color=(1,1,1,0.75)
        self.dms.orientation='horizontal'
        self.dms.size_hint=(0.8,1)
        self.dms.step=0.5
        self.dms.bind(value=self.adjust_diameter)
        self.dms.bind(active=self.on_active)
        self.dms_box.add_widget(self.dms)
        
        self.dms_label2 = Label(text="%5.1f m"%(self.dms.value),color=(1,1,1,0.75),size_hint=(0.5,1))
        self.dms_box.add_widget(self.dms_label2)
        
        self.sns_box = BoxLayout()
        self.sns_box.orientation='horizontal'
        self.sns_label = Label(text='S/N Limit:',color=(1,1,1,0.75),size_hint=(0.5,1))
        self.sns_box.add_widget(self.sns_label)
        
        self.sns = SNRMDSlider()
        self.sns.background_color=(0,0,0,0)
        self.sns.color=(1,1,1,0.75)
        self.sns.orientation='horizontal'
        self.sns.size_hint=(0.8,1)
        self.sns.step=0.25
        self.sns.bind(value=self.adjust_snrcut)
        self.sns.bind(active=self.on_active)
        self.sns_box.add_widget(self.sns)
        
        self.sns_label2 = Label(text="%5.1f"%(10**self.sns.value),color=(1,1,1,0.75),size_hint=(0.5,1))
        self.sns_box.add_widget(self.sns_label2)

    
    def toggle_state(self) :
        if (self.is_open) :
            self.is_open = False
            self.clear_widgets()
            self.top_menu.toggle_state()
        else :
            self.is_open = True
            self.top_menu.toggle_state()
            Clock.schedule_once(lambda x: self.add_widget(self.dms_box), self.top_menu.expand_time)
            Clock.schedule_once(lambda x: self.add_widget(self.sns_box), self.top_menu.expand_time)

    def refresh(self) :
        global _ngeht_diameter_setting, _snr_cut_setting
        self.dms.value = _ngeht_diameter_setting
        self.sns.value = _snr_cut_setting

    def on_active(self,widget,active) :
        if active :
            self.plot.freeze_plot = True
        else :
            self.plot.freeze_plot = False
            
    def adjust_diameter(self,widget,val) :
        global _ngeht_diameter_setting
        _ngeht_diameter_setting = val
        self.plot.set_ngeht_diameter(val)
        self.dms_label2.text = "%5.1f m"%(val)
        
    def adjust_snrcut(self,widget,val) :
        global _snr_cut_setting
        _snr_cut_setting = val
        if self.sns._is_off :
            self.plot.set_snr_cut(None)
            self.sns_label2.text = "None"
        else :
            self.plot.set_snr_cut(10**val)
            self.sns_label2.text = "%5.1f"%(10**val)
        

class DiameterMDSlider(FancyMDSlider):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.min = 3
        self.max = 30
        self.value = 6
        self.show_off = False

    def hint_box_text(self,value) :
        return "%5.1f m"%(value)

    def hint_box_size(self) :
        return (dp(50),dp(28))

    
class SNRMDSlider(FancyMDSlider):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

        self.min = 0
        self.max = 3
        self.value = 2
        self.show_off = True

    def hint_box_text(self,value) :
        return "%5.1f"%(10**value)

    def hint_box_size(self) :
        return (dp(50),dp(28))


    
class SimpleDataSetSelection(Spinner) :

    global _datadict
    datadict = _datadict
    
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
        
        self.datasets = {}
        self.datasets['M87 230 GHz']={'file':'data/V_M87_ngeht_ref1_230_perfect_scanavg_tygtd.dat'}
        self.datasets['Sgr A* 230 GHz']={'file':'data/V_SGRA_ngeht_ref1_230_perfect_scanavg_tygtd.dat'}    
        self.datasets['M87 345 GHz']={'file':'data/V_M87_ngeht_ref1_345_perfect_scanavg_tygtd.dat'}
        self.datasets['Sgr A* 345 GHz']={'file':'data/V_SGRA_ngeht_ref1_345_perfect_scanavg_tygtd.dat'}

        # Set values
        self.values = []
        for ds in self.datasets.keys() :
            self.values.append(ds)

        # Choose key
        self.text = list(self.datasets.keys())[0]

        # Set default data
        global _datadict
        _datadict = ci.read_data(path.abspath(path.join(path.dirname(__file__),self.datasets[self.text]['file'])))

    def select_dataset(self) :
        print("Reading data set from",self.datasets[self.text]['file'])
        global _datadict
        _datadict = ci.read_data(path.abspath(path.join(path.dirname(__file__),self.datasets[self.text]['file'])))


        
class CircularRippleButton(CircularRippleBehavior, ButtonBehavior, Image):
    def __init__(self, **kwargs):
        self.ripple_scale = 0.85
        super().__init__(**kwargs)

    def delayed_switch_to_imaging(self,delay=0) :
        Clock.schedule_once(self.switch_to_imaging, delay)
        
    def switch_to_imaging(self,val):
        sm = ngEHTApp.get_running_app().root
        sm.transition = FadeTransition()
        sm.current = "screen0"
        sm.transition = SlideTransition()

        
        
class MovieSplashScreen(BoxLayout) :
    img_id = ObjectProperty(None)
        
class DataSetSelectionScreen(BoxLayout) :
    pass

class InteractiveReconstructionPlot(FloatLayout) :
    ddm_id = ObjectProperty(None)
    otm_id = ObjectProperty(None)
    menu_id = ObjectProperty(None)
    plot_id = ObjectProperty(None)

class InteractiveBaselinesPlot(FloatLayout) :
    ddm_id = ObjectProperty(None)
    otm_id = ObjectProperty(None)
    menu_id = ObjectProperty(None)
    plot_id = ObjectProperty(None)

class InteractiveMapsPlot(FloatLayout) :
    ddm_id = ObjectProperty(None)
    otm_id = ObjectProperty(None)
    menu_id = ObjectProperty(None)
    plot_id = ObjectProperty(None)
    
class ReconstructionScreen(BoxLayout) :
    ddm_id = ObjectProperty(None)
    otm_id = ObjectProperty(None)
    menu_id = ObjectProperty(None)
    plot_id = ObjectProperty(None)
    
class BaselinesScreen(BoxLayout) :
    ddm_id = ObjectProperty(None)
    otm_id = ObjectProperty(None)
    menu_id = ObjectProperty(None)
    plot_id = ObjectProperty(None)

class MapsScreen(BoxLayout) :
    ddm_id = ObjectProperty(None)
    otm_id = ObjectProperty(None)
    menu_id = ObjectProperty(None)
    plot_id = ObjectProperty(None)

    
class TopBanner(MDBoxLayout) :
    
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)

        menu_list = ["Splash Screen","Reconstructions","News","ngEHT","About"]
        
        menu_items = [{ 'viewclass':'OneLineListItem', 'text':f"{menu_list[i]}", "height": dp(40), "on_release": lambda x=f"{menu_list[i]}": self.menu_callback(x), 'text_color':(1,1,1,1), 'theme_text_color':"Custom",} for i in range(len(menu_list)) ]

        self.menu = MDDropdownMenu(items=menu_items,width_mult=2.5,background_color=(0.7,0.7,0.7,0.5))

        
    def callback(self,button) :
        self.menu.caller = button
        self.menu.open()

    def menu_callback(self,text) :
        self.menu.dismiss()

        if (text=="Splash Screen") :
            self.set_splash_screen()
        elif (text=="Reconstructions") :
            self.set_target_screen()
        elif (text=="News") :
            self.set_news_screen()
        elif (text=="ngEHT") :
            import webbrowser
            webbrowser.open("http://www.ngeht.org/science")
        elif (text=="About") :
            self.set_about_screen()
        else :
            print("WTF BBQ!?!?!")
            
    def set_splash_screen(self) :
        sm = ngEHTApp.get_running_app().root
        sm.transition = FadeTransition()
        sm.current = "home"
        sm.transition = SlideTransition()

    def set_target_screen(self) :
        sm = ngEHTApp.get_running_app().root
        sm.transition = FadeTransition()
        sm.current = "screen0"
        sm.transition = SlideTransition()
    
    def set_news_screen(self) :
        sm = ngEHTApp.get_running_app().root
        sm.transition = FadeTransition()
        sm.current = "news"
        sm.transition = SlideTransition()

    def set_about_screen(self) :
        sm = ngEHTApp.get_running_app().root
        sm.transition = FadeTransition()
        sm.current = "about"
        sm.transition = SlideTransition()

        
class BottomBanner(MDBottomAppBar) :
    menu_id = ObjectProperty(None)
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)


class ngEHTApp(MDApp):
    # pass
    # Window.size = (300,500)

    def twitter_follow(self) :
        import webbrowser
        webbrowser.open("http://www.twitter.com")
        print("Insert twitter follow link")
    
    def facebook_follow(self) :
        import webbrowser
        webbrowser.open("http://www.facebook.com")
        print("Insert facebook follow link")

    def instagram_follow(self) :
        import webbrowser
        webbrowser.open("http://www.instagram.com")
        print("Insert instagram follow link")

    def youtube_follow(self) :
        import webbrowser
        webbrowser.open("https://www.youtube.com/channel/UCJeDtgEqIM6DCS-4lDpMnLw/featured")
        print("Insert YouTube follow link")

    def website_link(self) :
        import webbrowser
        webbrowser.open("http://www.ngeht.org/science")
        print("Insert twitter follow link")

    def null_func(self) :
        pass
        
        
if __name__ == '__main__' :
    ngEHTApp().run()

    
