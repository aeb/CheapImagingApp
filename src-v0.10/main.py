__version__ = "0.11"

__main_debug__ = True

from kivy.app import App
from kivymd.app import MDApp
from kivy.lang import Builder

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.properties import StringProperty, NumericProperty, ObjectProperty, BooleanProperty, ListProperty
from kivy.clock import Clock
from kivy.uix.screenmanager import FadeTransition, SlideTransition
from kivy.metrics import dp,sp
from kivy.uix.image import Image
from kivy.uix.behaviors import ButtonBehavior
from kivy.graphics import Color, Line, Rectangle

from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.behaviors import CircularRippleBehavior
from kivymd.uix.filemanager import MDFileManager

import numpy as np

from fancy_mdslider import FancyMDSlider

from os import path

from pathlib import Path as plP

import hashlib

from kivy.core.window import Window
import time

####################
# TESTING
# import pickle
# 
# Window.size = (300,500)
##################


try :
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.READ_EXTERNAL_STORAGE])
except:
    print("Could not load android permissions stuff, pobably not on android?")


import copy

import ngeht_array
import data
import baseline_plot
import cheap_image
import map_plot
import skymap_plot



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

_stationdicts={}
_stationdicts['ngEHT+']=ngeht_array.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/ngeht_ref1.txt')), existing_station_list=_existing_station_list)
_stationdicts['ngEHT']=ngeht_array.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/ngeht_ref1.txt')), existing_station_list=_existing_station_list)
_stationdicts['EHT 2017']=ngeht_array.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/eht2017.txt')),existing_station_list=_existing_station_list)
_stationdicts['EHT 2022']=ngeht_array.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/eht2022.txt')),existing_station_list=_existing_station_list)

# _stationdicts['ngEHT+']=ngeht_array.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/ngeht_ref1_230_ehtim.txt')), existing_station_list=_existing_station_list)
# _stationdicts['ngEHT']=ngeht_array.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/ngeht_ref1_230_ehtim.txt')), existing_station_list=_existing_station_list)
# _stationdicts['EHT 2017']=ngeht_array.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/eht2017_230_ehtim.txt')),existing_station_list=_existing_station_list)
# _stationdicts['EHT 2022']=ngeht_array.read_array(path.abspath(path.join(path.dirname(__file__),'arrays/eht2022_230_ehtim.txt')),existing_station_list=_existing_station_list)


_array_index = 1
_array = list(_stationdicts.keys())[_array_index]

_statdict_maximum=_stationdicts['ngEHT+']
_statdict=_stationdicts[_array]
# _statdict_maximum=copy.deepcopy(_stationdicts['ngEHT ref1'])
# _statdict=copy.deepcopy(_stationdicts['ngEHT ref1'])
# _datadict=data.read_data(path.abspath(path.join(path.dirname(__file__),'data/V_M87_ngeht_ref1_230_perfect_scanavg_tygtd.dat')))
_datadict=data.read_themis_data_file(path.abspath(path.join(path.dirname(__file__),'data/V_M87_ngeht_ref1_230_perfect_scanavg_tygtd.dat')))

_source_RA = 17.7611225
_source_Dec = -29.007810

class MenuedReconstructionPlot(BoxLayout) :

    plot_maxsize = 750.0
    plot_center = np.array([0.0,0.0])

    irp = cheap_image.InteractiveImageReconstructionPlot()
    menu_id = ObjectProperty(None)
    
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)

        self.time_range = _time_range
        self.ngeht_diameter = _ngeht_diameter
        self.snr_cut = _snr_cut

        self.sdict = _statdict
        self.ddict = _datadict

        self.plot_frozen = False

        self.limits = np.array([1,-1,-1,1])*self.plot_maxsize
        self.limits[:2] = self.limits[:2] + self.plot_center[0]
        self.limits[2:] = self.limits[2:] + self.plot_center[1]

        self.argument_hash = None
        
        self.update(self.ddict,self.sdict,time_range=self.time_range,snr_cut=self.snr_cut,ngeht_diameter=self.ngeht_diameter,limits=self.limits)

        self.add_widget(self.irp)

        
        if __main_debug__ :
            print("mrp.__init__: finished")
        

    def update(self,datadict,statdict,**kwargs) :
        kwargs['time_range']=self.time_range
        kwargs['limits']=self.limits
        kwargs['snr_cut']=self.snr_cut
        kwargs['ngeht_diameter']=self.ngeht_diameter
        new_argument_hash = hashlib.md5(bytes(str(datadict)+str(statdict)+str(kwargs),'utf-8')).hexdigest()
        if (__main_debug__) :
            print("update kwargs:",kwargs)
            print("update New image md5 hash:",new_argument_hash)
            print("update Old image md5 hash:",self.argument_hash)
        if ( new_argument_hash == self.argument_hash ) :
            return
        self.argument_hash = new_argument_hash
        self.irp.update(datadict,statdict,**kwargs)
        if __main_debug__ :
            print("mrp.update:",self.sdict.keys(),self.size)

    def replot(self,**kwargs) :
        global _datadict, _statdict
        self.ddict = _datadict
        self.sdict = _statdict
        kwargs['time_range']=self.time_range
        kwargs['limits']=self.limits
        kwargs['snr_cut']=self.snr_cut
        kwargs['ngeht_diameter']=self.ngeht_diameter
        new_argument_hash = hashlib.md5(bytes(str(self.ddict)+str(self.sdict)+str(kwargs),'utf-8')).hexdigest()
        if (__main_debug__):
            print("replot New image md5 hash:",new_argument_hash)
            print("replot Old image md5 hash:",self.argument_hash)
        if ( new_argument_hash == self.argument_hash ) :
            return
        self.argument_hash = new_argument_hash
        self.irp.replot(self.ddict,self.sdict,**kwargs)
        if __main_debug__ :
            print("mrp.replot:",self.sdict.keys(),self.size)

    def refresh(self,**kwargs) :
        kwargs['time_range']=self.time_range
        kwargs['limits']=self.limits
        kwargs['snr_cut']=self.snr_cut
        kwargs['ngeht_diameter']=self.ngeht_diameter
        new_argument_hash = hashlib.md5(bytes(str(self.ddict)+str(self.sdict)+str(kwargs),'utf-8')).hexdigest()
        if (__main_debug__):
            print("refresh New image md5 hash:",new_argument_hash)
            print("refresh Old image md5 hash:",self.argument_hash)
        if ( new_argument_hash == self.argument_hash ) :
            return
        self.argument_hash = new_argument_hash
        self.irp.replot(self.ddict,self.sdict,**kwargs)
        if __main_debug__ :
            print("mrp.refresh:",self.sdict.keys(),self.size)
            
    def set_start_time(self,val) :
        if __main_debug__ :
            print("mrp.set_start_time:",val)
        self.time_range[1] = self.time_range[1]-self.time_range[0]+val
        self.time_range[0] = val
        self.update(self.ddict,self.sdict)
        
    def set_obs_time(self,val) :
        self.time_range[1] = self.time_range[0] + val
        self.update(self.ddict,self.sdict)

    def set_ngeht_diameter(self,val) :
        global _ngeht_diameter
        self.ngeht_diameter = val
        _ngeht_diameter = self.ngeht_diameter
        self.update(self.ddict,self.sdict)

    def set_snr_cut(self,val) :
        global _snr_cut
        self.snr_cut = val
        if (val is None) :
            self.snr_cut = 0
        _snr_cut = self.snr_cut
        self.update(self.ddict,self.sdict)

    def freeze_plot(self) :
        self.irp.plot_frozen = True

    def unfreeze_plot(self) :
        self.irp.plot_frozen = False


class MenuedBaselinePlot(BoxLayout) :

    ibp = baseline_plot.InteractiveBaselinePlot_kivygraph()
    menu_id = ObjectProperty(None)

    def __init__(self,**kwargs) :
        super().__init__(**kwargs)

        self.time_range = _time_range
        self.ngeht_diameter = _ngeht_diameter
        self.snr_cut = _snr_cut

        self.sdict = _statdict
        self.ddict = _datadict

        self.plot_frozen = False

        self.limits = [-20,20,20,-20]

        self.update(self.ddict,self.sdict,limits=self.limits)

        self.add_widget(self.ibp)
        
        if __main_debug__ :
            print("mp.__init__: finished")
        

    def update(self,datadict,statdict,**kwargs) :

        kwargs['time_range']=self.time_range
        kwargs['limits']=self.limits
        kwargs['snr_cut']=self.snr_cut
        kwargs['ngeht_diameter']=self.ngeht_diameter

        global _datadict, _statdict
        _datadict = datadict
        _statdict = statdict
        self.ddict = _datadict
        self.sdict = _statdict
        
        self.ibp.update(datadict,statdict,**kwargs)
                    
        if __main_debug__ :
            print("bp.update:",self.sdict.keys(),self.size)

    def replot(self,**kwargs) :
        kwargs['time_range']=self.time_range
        kwargs['limits']=self.limits
        kwargs['snr_cut']=self.snr_cut
        kwargs['ngeht_diameter']=self.ngeht_diameter

        global _datadict, _statdict
        self.ddict = _datadict
        self.sdict = _statdict

        self.ibp.replot(self.ddict,self.sdict,**kwargs)
        
        if __main_debug__ :
            print("mp.replot:",self.sdict.keys(),self.size)

    def refresh(self,**kwargs) :
        kwargs['time_range']=self.time_range
        kwargs['limits']=self.limits
        kwargs['snr_cut']=self.snr_cut
        kwargs['ngeht_diameter']=self.ngeht_diameter
        self.ibp.replot(self.ddict,self.sdict,**kwargs)
        
        if __main_debug__ :
            print("mp.refresh:",self.sdict.keys(),self.size)
            
    def set_start_time(self,val) :
        self.time_range[1] = self.time_range[1]-self.time_range[0]+val
        self.time_range[0] = val
        self.refresh()
        
    def set_obs_time(self,val) :
        self.time_range[1] = self.time_range[0] + val
        self.refresh()
        if __main_debug__ :
            print("MenuedBaselinePlot.set_obs_time: set the time")
        
    def set_ngeht_diameter(self,val) :
        global _ngeht_diameter
        self.ngeht_diameter = val
        _ngeht_diameter = self.ngeht_diameter
        self.refresh()

    def set_snr_cut(self,val) :
        global _snr_cut
        self.snr_cut = val
        if (val is None) :
            self.snr_cut = 0
        _snr_cut = self.snr_cut

    def freeze_plot(self) :
        self.ibp.plot_frozen = True

    def unfreeze_plot(self) :
        self.ibp.plot_frozen = False

    def zoom_in(self) :
        self.ibp.zoom_in()

    def zoom_out(self) :
        self.ibp.zoom_out()

    
            
class MenuedBaselineMapPlot_kivygraph(BoxLayout) :

    bmc = map_plot.BaselineMapCanvas()
    mp = map_plot.InteractiveBaselineMapPlot_kivygraph()
    menu_id = ObjectProperty(None)
    
    def __init__(self,**kwargs) :
        global _datadict, _statdict
        
        self.sdict = _statdict
        self.ddict = _datadict

        super().__init__(**kwargs)

        self.time_range = _time_range
        self.ngeht_diameter = _ngeht_diameter
        self.snr_cut = _snr_cut

        self.plot_frozen = False

        self.add_widget(self.mp)
        self.add_widget(self.bmc)

        self.pixel_offset = (0,0)

        # Generate some default resizing behaviors
        self.bind(height=self.resize)
        self.bind(width=self.resize)

        # Generate first set of baselines
        self.mp.replot(self.ddict,self.sdict)
        
        if __main_debug__ :
            print("mp.__init__: finished")

    def update(self,datadict,statdict) :
        global _datadict, _statdict
        self.mp.update(datadict,statdict)
        self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
        if __main_debug__ :
            print("MenuedBaselineMapPlot_kivygraph.update:",self.sdict.keys(),self.size)
            print("         :",_statdict.keys(),self.size)
            print("         :",statdict.keys(),self.size)

    def replot(self) :
        global _datadict, _statdict
        self.ddict = _datadict
        self.sdict = _statdict
        self.mp.replot(self.ddict,self.sdict)
        self.bmc.plot_stations(self.sdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
        if __main_debug__ :
            print("MenuedBaselineMapPlot_kivygraph.replot:",self.sdict.keys(),self.size)
            print("         :",_statdict.keys(),self.size)

    def set_start_time(self,val) :
        self.time_range[1] = self.time_range[1]-self.time_range[0]+val
        self.time_range[0] = val
        
    def set_obs_time(self,val) :
        self.time_range[1] = self.time_range[0] + val

    def set_ngeht_diameter(self,val) :
        global _ngeht_diameter
        self.ngeht_diameter = val
        _ngeht_diameter = self.ngeht_diameter

    def set_snr_cut(self,val) :
        global _snr_cut
        self.snr_cut = val
        if (val is None) :
            self.snr_cut = 0
        _snr_cut = self.snr_cut

    def freeze_plot(self) :
        self.plot_froze = True
        self.mp.plot_frozen = True

    def unfreeze_plot(self) :
        self.plot_froze = False
        self.mp.plot_frozen = False

    def on_touch_move(self,touch) :
        if (not self.plot_frozen) :
            self.pixel_offset = ( self.pixel_offset[0] + touch.dpos[0], self.pixel_offset[1] + touch.dpos[1] )
        self.mp.on_touch_move(touch)
        self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
        if __main_debug__ :
            print("MenuedBaselineMapPlot_kivygraph.on_touch_move(): replotting")

    def on_touch_down(self,touch) :
        self.mp.on_touch_down(touch)
        if (touch.is_double_tap) :
            self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
        if (touch.is_touch) :
            snap_source = None
            # for s in _statdict.keys() :
            for s in self.mp.statdict.keys() :
                xpx_src,ypx_src = self.bmc.coords_to_px(self.mp.lldict[s][0],self.mp.lldict[s][1],self.mp.rect)
                dxpx = (touch.pos[0] - xpx_src + 0.5*self.mp.rect.size[0])%self.mp.rect.size[0] - 0.5*self.mp.rect.size[0]
                dypx = (touch.pos[1] - ypx_src)
                if ( dxpx**2 + dypx**2 <= dp(15)**2 ) :
                    snap_source = s
            if (snap_source is None) :
                self.bmc.cursor_lat,self.bmc.cursor_lon = self.bmc.px_to_coords(touch.pos[0],touch.pos[1],self.mp.rect)
                self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
            else :
                self.bmc.cursor_lat,self.bmc.cursor_lon = self.mp.lldict[snap_source]
                self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
                
                
        if __main_debug__ :
            print("MenuedBaselineMapPlot_kivygraph.on_touch_down(): replotting",self.size,self.mp.rect.size)
            
    def zoom_in(self) :
        self.mp.zoom_in()
        self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
        if __main_debug__ :
            print("MenuedBaselineMapPlot_kivygraph.zoom_in(): replotting")

    def zoom_out(self) :
        self.mp.zoom_out()
        self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
        if __main_debug__ :
            print("MenuedBaselineMapPlot_kivygraph.zoom_out(): replotting")

    def resize(self,widget,newsize) :
        self.mp.resize(widget,newsize)
        # print("MBLMP_kg.resize(): after mp.resize --",self.mp.rect.size,self.mp.rect.pos)
        self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
        # Hack to fix the plot resize on initialization
        # if (self.mp.rect.size[0]==0 or self.mp.rect.size[1]==0) :
        #     Clock.schedule_once(lambda x : self.replot(), 0.1)
        Clock.schedule_once(lambda x : self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect), 0.1)
        
        if __main_debug__ :
            print("MenuedBaselineMapPlot_kivygraph.resize(): replotting with",newsize,self.size,self.mp.rect.size)
            # Clock.schedule_once(self.delayed_report,0.1)
            
    # def delayed_report(self,dt) :
    #     print("MenuedBaselineMapPlot_kivygraph.delayed_report(): replotting with",self.size,self.mp.rect.size)

    def cursor_on(self) :
        self.bmc.cursor_on(self.mp.rect)
        self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)

    def cursor_off(self) :
        lat,lon = self.bmc.cursor_off(self.mp.rect)
        self.bmc.plot_stations(self.mp.statdict,self.mp.lldict,self.mp.gcdict,self.mp.rect)
        return lat,lon

    
        
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
                b = ToggleButton(text=s,size_hint=(None,None),size=self.button_size,color=_on_color,background_color=self.bkgnd_color,state="normal")
            else :
                b = ToggleButton(text=s,size_hint=(None,None),size=self.button_size,color=_off_color,background_color=self.bkgnd_color,state="down")
            b.bind(on_press=self.on_toggle)
            self.add_widget(b)
            self.bs.append(b)
            #self.sdict[s]['on']=True

        if (__main_debug__) :
            print("VariableToggleList.remake: updating plot")
            
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

        if (__main_debug__) :
            print("VariableToggleList.refresh: updating plot")
            
        self.rpp.update(_datadict,self.sdict)
        
    def on_toggle(self,val) :

        if __main_debug__ :
            print("VariableToggleList.on_toggle:",self.rpp,self.sdict)

        for b in self.bs :
            if b.state == "normal" :
                b.color = _on_color
                self.sdict[b.text]['on']=True
            else :
                b.color = _off_color
                self.sdict[b.text]['on']=False
                
        self.rpp.update(_datadict,self.sdict)        

        if __main_debug__ :
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

        if __main_debug__ :
            print("StationMenu.select_array:",self.rpp,array_index)
        
        global _array_index,_statdict
        _array_index = array_index
        self.array_name = list(_stationdicts.keys())[_array_index]
        #_statdict = _stationdicts[self.array_name]
        self.submenu_id.remake(_stationdicts[self.array_name])
        self.reset_state()

        if __main_debug__ :
            print("                        :",self.rpp,array_index)
        

    def refresh(self) :

        if __main_debug__ :
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
        
        if __main_debug__ :
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

        if __main_debug__ :
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
        
        self.sts_label2 = Label(text="%5.1f GST"%(self.sts.value),color=(1,1,1,0.75),size_hint=(0.5,1))
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
            self.plot.freeze_plot()
        else :
            self.plot.unfreeze_plot()
            
    def adjust_start_time(self,widget,val) :
        self.plot.set_start_time(val)
        self.sts_label2.text = "%5.1f GST"%(val)
        
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
        return "%5.1f GST"%(value)

    def hint_box_size(self) :
        return (dp(60),dp(28))

    
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

        if __main_debug__ :
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
            self.plot.freeze_plot()
        else :
            self.plot.unfreeze_plot()
            
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

        self.datasets['Jet 230 GHz']={'file':'source_images/GRRT_IMAGE_data1400_freq230.npy'}
        self.datasets['Jet 345 GHz']={'file':'source_images/GRRT_IMAGE_data1400_freq345.npy'}
        self.datasets['RIAF 230 GHz']={'file':'source_images/fromm230_scat.npy'}
        self.datasets['RIAF 345 GHz']={'file':'source_images/fromm345_scat.npy'}

        self.datasets['First contact']={'file':'source_images/toy_story_aliens.png'}        
        
        # self.datasets['datagen']={'file':'datagen.pkl'}
        
        # Set values
        self.values = []
        for ds in self.datasets.keys() :
            self.values.append(ds)

        # Choose key
        self.text = list(self.datasets.keys())[0]

        # Set default data
        global _datadict, _statdict
        # _datadict = data.read_themis_data_file(path.abspath(path.join(path.dirname(__file__),self.datasets[self.text]['file'])))
        _datadict = data.generate_data_from_file(path.abspath(path.join(path.dirname(__file__),self.datasets[self.text]['file'])),_statdict)

        
    def select_dataset(self) :
        if __main_debug__ :
            print("Reading data set from",self.datasets[self.text]['file'])
        global _datadict, _statdict

        if (self.text=='datagen') :
            with open("datagen.pkl","rb") as f :
                _datadict = pickle.load(f)
        else :
            # _datadict = data.read_data(path.abspath(path.join(path.dirname(__file__),self.datasets[self.text]['file'])))
            _datadict = data.generate_data_from_file(path.abspath(path.join(path.dirname(__file__),self.datasets[self.text]['file'])),_statdict)

        print("Read:",_datadict)



class SimpleTargetSelection(Spinner) :

    RA_label = StringProperty(None)
    Dec_label = StringProperty(None)

    
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
        
        self.targets = {}
        self.targets['Sgr A*']={'RA':self.RA_hr(17,45,40.049),'Dec':self.Dec_deg(-29,0,28.118)}
        self.targets['M87']={'RA':self.RA_hr(12,30,49.42338),'Dec':self.Dec_deg(12,23,28.0439)}
        self.targets['M31']={'RA':self.RA_hr(0,42,44.3),'Dec':self.Dec_deg(41,16,9)}
        self.targets['Cen A']={'RA':self.RA_hr(13,25,27.6),'Dec':self.Dec_deg(-43,1,9)}
        self.targets['OJ 287']={'RA':self.RA_hr(8,54,48.9),'Dec':self.Dec_deg(20,6,31)}
        self.targets['3C 279']={'RA':self.RA_hr(12,56,11.1),'Dec':self.Dec_deg(-5,47,22)}
        self.targets['Mkn 421']={'RA':self.RA_hr(11,4,27.314),'Dec':self.Dec_deg(38,12,31.80)}
        self.targets['BL Lac']={'RA':self.RA_hr(22,2,43.3),'Dec':self.Dec_deg(42,16,40)}
        self.targets['M81']={'RA':self.RA_hr(9,55,33.2),'Dec':self.Dec_deg(69,3,55)}

        
        # Set values
        self.values = []
        for ds in self.targets.keys() :
            self.values.append(ds)

        # Choose key
        self.text = list(self.targets.keys())[0]

        
        
        # Set default RA/DEC
        self.select_target(self,self.text)
        
        # global _source_RA, _source_Dec
        # _source_RA = self.targets[self.text]['RA']
        # _source_Dec = self.targets[self.text]['Dec']


        
        self.bind(text=self.select_target)

        
        print("RA:",self.hr_to_str(_source_RA))
        print("Dec:",self.deg_to_str(_source_RA))
        
        
    def RA_hr(self,hh,mm,ss) :
        return hh+mm/60.0+ss/3600.
        
    def Dec_deg(self,deg,arcmin,arcsec) :
        return (np.sign(deg)*(np.abs(deg)+arcmin/60.0+arcsec/3600.))

    def hr_to_str(self,RA) :
        hh = int(RA)
        mm = int((RA-hh)*60.0)
        ss = ((RA-hh)*60.0-mm)*60.0
        #return ("%2i\u02B0 %2i\u1D50 %4.1f\u02E2"%(hh,mm,ss))
        return ("%02ih %02im %02.0fs"%(hh,mm,ss))
        
    def deg_to_str(self,Dec) :
        if (Dec<0) :
            ns = '-'
        else :
            ns = '+'
        Dec = np.abs(Dec)
        dg = int(Dec)
        mm = int((Dec-dg)*60.0)
        ss = ((Dec-dg)*60.0-mm)*60.0
        return ("%1s%02i\u00B0 %02i\' %02.0f\""%(ns,dg,mm,ss))
    
    def select_target(self,widget,value) :
        if (__main_debug__) :
            print("Selecting target:",widget,value,self.text)
        global _source_RA, _source_Dec
        _source_RA = self.targets[self.text]['RA']
        _source_Dec = self.targets[self.text]['Dec']

        self.RA_label = self.hr_to_str(_source_RA)
        self.Dec_label = self.deg_to_str(_source_Dec)
        

            
class TargetSelectionMap(BoxLayout) :


    smc = skymap_plot.StarMapCanvas()
    ismp = skymap_plot.InteractiveSkyMapPlot()
    # tss = ObjectProperty(None)

    fps = NumericProperty(30)
    
    def __init__(self,**kwargs) :
        global _source_RA, source_Dec
        
        super().__init__(**kwargs)

        self.add_widget(self.ismp)
        self.add_widget(self.smc)


        self.tbox = BoxLayout(orientation='vertical',size_hint=(None,None),width=dp(150),height=sp(100)) #,dp(200))) #,pos=(dp(100),dp(100)))
        # self.tbox = BoxLayout(orientation='vertical',size_hint=(None,None),size=(dp(50),dp(90)))
        
        self.targets = {}
        self.targets['Sgr A*']={'RA':self.RA_hr(17,45,40.049),'Dec':self.Dec_deg(-29,0,28.118)}
        self.targets['M87']={'RA':self.RA_hr(12,30,49.42338),'Dec':self.Dec_deg(12,23,28.0439)}
        self.targets['M31']={'RA':self.RA_hr(0,42,44.3),'Dec':self.Dec_deg(41,16,9)}
        self.targets['Cen A']={'RA':self.RA_hr(13,25,27.6),'Dec':self.Dec_deg(-43,1,9)}
        self.targets['OJ 287']={'RA':self.RA_hr(8,54,48.9),'Dec':self.Dec_deg(20,6,31)}
        self.targets['3C 279']={'RA':self.RA_hr(12,56,11.1),'Dec':self.Dec_deg(-5,47,22)}
        # self.targets['Mkn 421']={'RA':self.RA_hr(11,4,27.314),'Dec':self.Dec_deg(38,12,31.80)}
        # self.targets['BL Lac']={'RA':self.RA_hr(22,2,43.3),'Dec':self.Dec_deg(42,16,40)}
        # self.targets['M81']={'RA':self.RA_hr(9,55,33.2),'Dec':self.Dec_deg(69,3,55)}
        # self.targets['LMC']={'RA':self.RA_hr(5,23,34.5),'Dec':self.Dec_deg(-69,45,22)}
        # self.targets['SMC']={'RA':self.RA_hr(0,52,44.8),'Dec':self.Dec_deg(-72,49,43)}

        self.targets['--- Select ---']={'RA':None,'Dec':None}
        
        if (__main_debug__) :
            for s in self.targets.keys() :
                if (s!='--- Select ---') :
                    print("%10s %15.8g %15.8g"%(s,self.targets[s]['RA'],self.targets[s]['Dec']))
        
        self.tss = skymap_plot.TargetSelectionSpinner(self.targets)
        self.tss.size_hint = (1,1)
        self.tss.background_color = (1,1,1,0.1)
        self.tss.color = (1,0.75,0.25,1)
        self.tss.bind(text=self.select_target)
        self.tbox.add_widget(self.tss)
        self.ra_label = Label(text=" RA: ",size_hint=(1,1),color=(1,1,1))
        self.dec_label = Label(text="Dec: ",size_hint=(1,1),color=(1,1,1))
        self.tbox.add_widget(self.ra_label)
        self.tbox.add_widget(self.dec_label)

        self.add_widget(self.tbox)
        
        self.pixel_offset = (0,0)

        # Generate some default resizing behaviors
        self.bind(height=self.resize)
        self.bind(width=self.resize)

        self.animation_RA_start = 0
        self.animation_Dec_start = 0
        self.animation_total_time = 1.0
        self.animation_type = 'cubic'


        # Select a target
        self.select_target(self,list(self.targets.keys())[0])
        self.set_map_center(_source_RA,_source_Dec)

        
        if __main_debug__ :
            print("mp.__init__: finished")


    def select_target(self,widget,value) :
        if (__main_debug__) :
            print("Selecting target:",widget,value,self.tss.text)
        global _source_RA, _source_Dec
        if (self.tss.text!="--- Select ---") :
            # print("====== Setting to",self.tss.text)
            _source_RA = self.targets[self.tss.text]['RA']
            _source_Dec = self.targets[self.tss.text]['Dec']
            # print("====== RA,Dec",_source_RA,_source_Dec)
        self.ra_label.text = " RA: "+self.hr_to_str(_source_RA)
        self.dec_label.text = "Dec: "+self.deg_to_str(_source_Dec)
        # print("====== RA,Dec 2",_source_RA,_source_Dec)
        # print("====== RA,Dec lbls",self.ra_label.text,self.dec_label.text)
        self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)
        if (self.tss.text!="--- Select ---") :
            self.animate_to_target(0.5)


    def set_target(self,widget,value) :
        if (__main_debug__) :
            print("Selecting target:",widget,value,self.tss.text)
        global _source_RA, _source_Dec
        if (self.tss.text!="--- Select ---") :
            # print("====== Setting to",self.tss.text)
            _source_RA = self.targets[self.tss.text]['RA']
            _source_Dec = self.targets[self.tss.text]['Dec']
            # print("====== RA,Dec",_source_RA,_source_Dec)
        self.ra_label.text = " RA: "+self.hr_to_str(_source_RA)
        self.dec_label.text = "Dec: "+self.deg_to_str(_source_Dec)
        # print("====== RA,Dec 2",_source_RA,_source_Dec)
        # print("====== RA,Dec lbls",self.ra_label.text,self.dec_label.text)
        self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)
        self.set_map_center(_source_RA,_source_Dec)
        #self.animate_to_target(0.5)
        
        
    def RA_hr(self,hh,mm,ss) :
        return hh+mm/60.0+ss/3600.
        
    def Dec_deg(self,deg,arcmin,arcsec) :
        return (np.sign(deg)*(np.abs(deg)+arcmin/60.0+arcsec/3600.))

    def hr_to_str(self,RA) :
        hh = int(RA)
        mm = int((RA-hh)*60.0)
        ss = ((RA-hh)*60.0-mm)*60.0
        return ("%02ih %02im %02.0fs"%(hh,mm,ss))
        
    def deg_to_str(self,Dec) :
        if (Dec<0) :
            ns = '-'
        else :
            ns = '+'
        Dec = np.abs(Dec)
        dg = int(Dec)
        mm = int((Dec-dg)*60.0)
        ss = ((Dec-dg)*60.0-mm)*60.0
        return ("%1s%02i\u00B0 %02i\' %02.0f\""%(ns,dg,mm,ss))

            
    def update(self,datadict,statdict) :
        self.ismp.update()
        self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)

    def replot(self) :
        self.ismp.replot()
        self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)

    def on_touch_move(self,touch) :
        #if (not self.plot_frozen) :
        self.pixel_offset = ( self.pixel_offset[0] + touch.dpos[0], self.pixel_offset[1] + touch.dpos[1] )
        self.ismp.on_touch_move(touch)
        self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)
        if __main_debug__ :
            print("TargetSelectionMap.on_touch_move(): replotting")

    def on_touch_down(self,touch) :

        global _source_RA, _source_Dec

        # print("touch coords:",self.smc.px_to_coords(touch.pos[0],touch.pos[1],self.ismp.rect))

        # Do the normal stuff for the map, whatever that is
        self.ismp.on_touch_down(touch)

        # Catch the map centering
        if (touch.is_double_tap) :
            self.set_map_center(_source_RA,_source_Dec)

        # Pass to the spinner menu to choose a source
        self.tss.on_touch_down(touch)

        # Make a selection/set the target
        if (touch.is_touch) :
            if (self.tss.text=="--- Select ---") :

                snap_source = None
                for s in self.targets.keys() :
                    if (s!="--- Select ---") :
                        xpx_src,ypx_src = self.smc.coords_to_px(self.targets[s]['RA'],self.targets[s]['Dec'],self.ismp.rect)
                        dxpx = (touch.pos[0] - xpx_src + 0.5*self.ismp.rect.size[0])%self.ismp.rect.size[0] - 0.5*self.ismp.rect.size[0]
                        dypx = (touch.pos[1] - ypx_src)
                        if ( dxpx**2 + dypx**2 <= dp(15)**2 ) :
                            snap_source = s
                            
                if (snap_source is None) :
                    RA,Dec = self.smc.px_to_coords(touch.pos[0],touch.pos[1],self.ismp.rect)
                    _source_RA = RA
                    _source_Dec = Dec
                
                    self.ra_label.text = " RA: "+self.hr_to_str(_source_RA)
                    self.dec_label.text = "Dec: "+self.deg_to_str(_source_Dec)
                    self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)
                else :
                    self.tss.text = snap_source
                    self.select_target(self,snap_source)
        
        
    def animate_to_target(self,total_time) :
        self.animation_RA_start, self.animation_Dec_start = self.ismp.get_coord_center()
        if __main_debug__ :
            print("TargetSelectionMap.animate_to_target: 1 --",self.animation_RA_start, self.animation_Dec_start)
        # Get closest branch to new RA
        self.animation_RA_start = (self.animation_RA_start-_source_RA + 12)%24 - 12 + _source_RA
        if __main_debug__ :
            print("TargetSelectionMap.animate_to_target: 2 --",self.animation_RA_start, self.animation_Dec_start)
        self.animation_total_time = total_time        
        for dt in np.linspace(0,total_time,int(total_time*self.fps)) :
            Clock.schedule_once( self.animate_map_center , dt)

    def animate_map_center(self,dt) :
        ds = self.animation_model(dt/self.animation_total_time)
        RA = ds*_source_RA + (1.0-ds)*self.animation_RA_start
        Dec = ds*_source_Dec + (1.0-ds)*self.animation_Dec_start
        self.set_map_center(RA,Dec)

    def animation_model(self, x) :
        if (self.animation_type=='cubic') :
            return 6.0*(0.25*x - 0.3333333333*(x-0.5)**3 - 0.041666666667)
    
        else :
            raise ValueError("animation model type not defined!")
        
    def set_map_center(self,RA,Dec) :
        if (self.size[0]==0 or self.size[1]==0) :
            return
        self.ismp.set_coord_center(RA,Dec)
        self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)

    
        
    def zoom_in(self) :
        self.ismp.zoom_in()
        self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)

    def zoom_out(self) :
        self.ismp.zoom_out()
        self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)

    def resize(self,widget,newsize) :
        self.ismp.resize(widget,newsize)
        self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec)

        # Hack to fix the plot resize on initialization
        # if (self.mp.rect.size[0]==0 or self.mp.rect.size[1]==0) :
        #     Clock.schedule_once(lambda x : self.replot(), 0.1)
        # Clock.schedule_once(lambda x : self.smc.plot_targets(self.targets,self.ismp.rect,_source_RA,_source_Dec), 0.1)
        Clock.schedule_once(lambda x : self.set_target(self,self.tss.text), 0.1)


        

        
        
class CircularRippleButton(CircularRippleBehavior, ButtonBehavior, Image):
    def __init__(self, **kwargs):
        self.ripple_scale = 0.85
        super().__init__(**kwargs)

    def delayed_switch_to_imaging(self,delay=0) :
        Clock.schedule_once(self.switch_to_imaging, delay)
        
    def switch_to_imaging(self,val):
        sm = ngEHTApp.get_running_app().root
        sm.transition = FadeTransition()
        # sm.current = "screen0"
        sm.current = "targets"
        sm.transition = SlideTransition()



class DataSetSelectionPage(BoxLayout) :

    # path_info = StringProperty("")
    
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
        
        self.orientation = "vertical"
        
        self.ic = data.ImageCarousel()

        self.ic.add_image(path.abspath(path.join(path.dirname(__file__),"source_images/M87_230.png")),
                          path.abspath(path.join(path.dirname(__file__),"source_images/GRRT_IMAGE_data1400_freq230.npy")),
                          "Simulated jet at 230 GHz.",
                          False)
        self.ic.add_image(path.abspath(path.join(path.dirname(__file__),"source_images/M87_345.png")),
                          path.abspath(path.join(path.dirname(__file__),"source_images/GRRT_IMAGE_data1400_freq345.npy")),
                          "Simulated jet at 345 GHz.",
                          False)
        self.ic.add_image(path.abspath(path.join(path.dirname(__file__),"source_images/SGRA_230.png")),
                          path.abspath(path.join(path.dirname(__file__),"source_images/fromm230_scat.npy")),
                          "Simulated RIAF at 230 GHz.",
                          False)
        self.ic.add_image(path.abspath(path.join(path.dirname(__file__),"source_images/SGRA_345.png")),
                          path.abspath(path.join(path.dirname(__file__),"source_images/fromm345_scat.npy")),
                          "Simulated RIAF at 345 GHz.",
                          False)
        self.ic.add_image(path.abspath(path.join(path.dirname(__file__),"source_images/toy_story_aliens.png")),
                          path.abspath(path.join(path.dirname(__file__),"source_images/toy_story_aliens.png")),
                          "First contact!",
                          True)
        # self.ic.add_image(path.abspath(path.join(path.dirname(__file__),"images/image_file_icon.png")),
        #                   None,
        #                   "Choose a file of your own!")

        self.add_widget(self.ic)

        self.dss = data.DataSelectionSliders()
        self.dss.size_hint = 1,0.5
        self.dss.its.active=True
        self.dss.its.disabled = True
        
        self.add_widget(self.dss)

        self.argument_hash = None
        self.ic.index = 1
        self.produce_selected_data_set()

        self.file_manager_obj = MDFileManager(
            select_path=self.select_path,
            exit_manager=self.exit_manager,
            preview=True,
            ext=['png','jpg','jpeg','gif']
        )
        self.file_manager_obj.md_bg_color = (0.25,0.25,0.25,1)
        # self.file_manager_obj.toolbar.specific_text_color = (0.77,0.55,0.17,1)

        self.ic.add_btn.bind(on_release=self.open_file_manager)
        
    def select_path(self,path) :
        self.ic.add_image(path,path,path,True)
        self.dss.its.disabled = False
        self.exit_manager(0)
        
    def open_file_manager(self,widget) :

        home = str(plP.home())

        # self.path_info = home
        
        if (home!='/data') :
            topdir = home
        else :
            topdir = '/'

        topdir = '/'
            
        self.file_manager_obj.show(topdir)
        

    def exit_manager(self,value) :
        if (value==1) : # a valid file wasn't selected, return to screen
            self.ic.index = 0
        else :
            self.ic.index = -1 # Set to value just added
            
        self.file_manager_obj.close()

    def on_touch_move(self,touch) :
        self.ic.on_touch_move(touch)
        self.dss.on_touch_move(touch)
        Clock.schedule_once(lambda x: self.on_selection(), self.ic.anim_move_duration+0.1)
        
    def on_selection(self) :
        if (__main_debug__) :
            print("Setting taper switch to active?",self.ic.taperable_list[self.ic.index])
        self.dss.its.disabled = not self.ic.taperable_list[self.ic.index]
        if (__main_debug__) :
            print("Setting taper switch to active?",self.ic.taperable_list[self.ic.index])

    def selection_check(self) :
        if (self.ic.index==0) :
            if (__main_debug__) :
                print("Bad selection!  Setting to index 1.")
            # 
            self.ic.load_slide(self.ic.slides[1])
            #self.ic.index = 1

            return False
        return True
            
    def produce_selected_data_set(self) :
        if (__main_debug__) :
            print("DSSP.produce_selected_data_set:",self.ic.selected_data_file(),self.dss.observation_frequency,_source_RA,_source_Dec,self.dss.source_size,self.dss.source_flux)

        new_argument_hash = hashlib.md5(bytes(str(self.ic.selected_data_file())+str(_statdict_maximum) + str(self.dss.observation_frequency) + str(_source_RA) + str(_source_Dec) + str(self.dss.source_size) + str(self.dss.source_flux) + str(self.dss.its.active and not self.dss.its.disabled),'utf-8')).hexdigest()
        if (__main_debug__) :
            print("New data md5 hash:",new_argument_hash)
            print("Old data md5 hash:",self.argument_hash)
        if ( new_argument_hash == self.argument_hash ) :
            return
        self.argument_hash = new_argument_hash
            
        global _datadict
        _datadict = data.generate_data_from_file( self.ic.selected_data_file(), \
                                                  _statdict_maximum, \
                                                  freq=self.dss.observation_frequency, \
                                                  ra=_source_RA,dec=_source_Dec, \
                                                  scale=self.dss.source_size, \
                                                  total_flux=self.dss.source_flux, \
                                                  taper_image=(self.dss.its.active and not self.dss.its.disabled))
            

class LogoBackground(FloatLayout) :

    background_color = ListProperty(None)
    highlight_color = ListProperty(None)
    logo_color = ListProperty(None)
    logo_size = NumericProperty(None)
    logo_offset = ListProperty(None,size=2)
    highlight_offset = ListProperty(None,size=2)
    
    def __init__(self,**kwargs) :
        super().__init__(**kwargs)
        self.bind(height=self.resize)
        self.bind(width=self.resize)

        # Generate the circle details
        self.radius_list = np.array([1.0, 0.80, 0.69, 0.53, 0.41, 0.24])
        self.phi0_list = np.array([0, 90, 180, 240, 30, 180, 180])
        self.total_phi0_list = (self.phi0_list[1:]-self.phi0_list[:-1]+360.0)%360.0 + 360.0
        self.dx_list = np.zeros(len(self.radius_list))
        self.dy_list = np.zeros(len(self.radius_list))
        for j in range(1,len(self.radius_list)) :
            self.dx_list[j] = (self.radius_list[j]-self.radius_list[j-1]) * np.sin(self.phi0_list[j]*np.pi/180.0) + self.dx_list[j-1]
            self.dy_list[j] = (self.radius_list[j]-self.radius_list[j-1]) * np.cos(self.phi0_list[j]*np.pi/180.0) + self.dy_list[j-1]

        self.dx_list = -self.dx_list
        self.dy_list = -self.dy_list
            
        self.background_color = (0.25,0.25,0.25,1)
        self.highlight_color = (0.35,0.35,0.35,1)
        self.logo_color = (0.14,0.14,0.14,1)
        self.logo_offset = (75,45)
        self.logo_size = 75
        self.logo_thickness = dp(6)
        self.highlight_offset = (-0.2*dp(6),0.2*dp(6))
        
    
    def redraw_background(self) :

        self.canvas.clear()
        
        with self.canvas.before :
            
            Color(self.background_color[0],self.background_color[1],self.background_color[2],self.background_color[3])
            Rectangle(size=self.size)
            
            # circ_scale = 0.5*max(self.height,self.width)
            # Xc = 0.75*self.logo_size+0.25*self.width
            # Yc = 0.45*self.height
            Xc = self.logo_offset[0] + self.highlight_offset[0]
            Yc = self.logo_offset[1] + self.highlight_offset[1]
            Color(self.highlight_color[0],self.highlight_color[1],self.highlight_color[2],self.highlight_color[3])
            for j in range(len(self.radius_list)) :
                xc = self.dx_list[j]*self.logo_size + Xc
                yc = self.dy_list[j]*self.logo_size + Yc
                rc = self.radius_list[j]*self.logo_size
                Line(circle=(xc,yc,rc),close=True,width=self.logo_thickness)

            # Xc = Xc + 0.2*self.logo_thickness
            # Yc = Yc - 0.2*self.logo_thickness
            Xc = Xc - self.highlight_offset[0]
            Yc = Yc - self.highlight_offset[1]
            Color(self.logo_color[0],self.logo_color[1],self.logo_color[2],self.logo_color[3])
            for j in range(len(self.radius_list)) :
                xc = self.dx_list[j]*self.logo_size + Xc
                yc = self.dy_list[j]*self.logo_size + Yc
                rc = self.radius_list[j]*self.logo_size
                Line(circle=(xc,yc,rc),close=True,width=self.logo_thickness)
                

    def resize(self,widget,newsize) :
        self.redraw_background()




        
class TargetSelectionScreen(BoxLayout) :
    pass

class DataSetSelectionScreen(BoxLayout) :
    dssp_id = ObjectProperty(None)

class MovieSplashScreen(BoxLayout) :
    img_id = ObjectProperty(None)

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

    def __init__(self,**kwargs) :
        super().__init__(**kwargs)

        # New station stuff
        self.add_station_btn = Button(text="Add",font_size=sp(16),color=_on_color,background_color=(1,1,1,0.2))
        self.del_station_btn = Button(text="Del",font_size=sp(16),color=_on_color,background_color=(1,1,1,0.2))
        self.add_station_btn.bind(on_release=self.add_station)
        self.del_station_btn.bind(on_release=self.del_station)
        #self.new_station_name_list = ['.LU', '.XE', '.XT', '.ER', '.MI', '.NO']
        self.new_station_name_list = []
        for j in range(20) :
            self.new_station_name_list.append('%02i'%j)
        self.prototype_station = 'BA'
        self.number_new_stations = 0
        self.editing_mode = False
        
    def add_stn_buttons(self) :
        if (__main_debug__) :
            print("InteractiveMapsPlot.add_stn_buttons called:",len(self.ids['ad_stn_box'].children))
        if ( len(self.ids['ad_stn_box'].children)==0 ) :
            self.ids['ad_stn_box'].add_widget(self.add_station_btn)
            self.ids['ad_stn_box'].add_widget(self.del_station_btn)

    def remove_stn_buttons(self) :
        if (__main_debug__) :
            print("InteractiveMapsPlot.remove_stn_buttons called")
        self.ids['ad_stn_box'].clear_widgets()

    def update(self,ddict,sdict) :
        if (__main_debug__) :
            print("InteractiveMapsPlot.update pass through called",_array)
        self.plot_id.update(ddict,sdict)
        if (_array_index==0) :
            self.add_stn_buttons()
        else :
            self.remove_stn_buttons()
            if (self.editing_mode) :
                self.editing_mode = False
                self.plot_id.cursor_off()

    def add_station(self,widget) :
        global _statdict, _datadict
        if (_array_index==0) :
            if (self.number_new_stations<len(self.new_station_name_list)) :
                if (self.editing_mode==False) :
                    self.editing_mode = True
                    self.plot_id.cursor_on()
                    self.add_station_btn.text = '+'+self.new_station_name_list[self.number_new_stations]
                    self.add_station_btn.color = _off_color
                    self.del_station_btn.text = 'Del'
                    self.del_station_btn.color = _on_color
                else :
                    self.editing_mode = False
                    latlon = self.plot_id.cursor_off()
                    nn = self.new_station_name_list[self.number_new_stations]
                    _stationdicts['ngEHT+'][nn] = copy.deepcopy(_statdict[self.prototype_station])
                    _stationdicts['ngEHT+'][nn]['on'] = True
                    _stationdicts['ngEHT+'][nn]['loc'] = self.plot_id.mp.latlon_to_xyz(latlon,radius=6.371e6)
                    _stationdicts['ngEHT+'][nn]['name'] = nn
                    _statdict = _stationdicts['ngEHT+']
                    self.number_new_stations += 1
                    self.add_station_btn.text = 'Add'
                    self.add_station_btn.color = _on_color

                    self.plot_id.update(_datadict,_statdict)
                    self.menu_id.refresh()
                    
        
    def del_station(self,widget) :
        global _statdict, _datadict
        if (_array_index==0) :
            if (self.number_new_stations>0) :
                if (self.editing_mode==False) :
                    self.editing_mode = True
                    self.plot_id.cursor_on()
                    self.del_station_btn.text = '-'+self.new_station_name_list[self.number_new_stations-1]
                    self.del_station_btn.color = _off_color
                    self.add_station_btn.text = 'Add'
                    self.add_station_btn.color = _on_color                    
                else :
                    self.editing_mode = False
                    lat,lon = self.plot_id.cursor_off()
                    snap_source = None
                    for s in _statdict.keys() :
                        xpx_src,ypx_src = self.plot_id.bmc.coords_to_px(self.plot_id.mp.lldict[s][0],self.plot_id.mp.lldict[s][1],self.plot_id.mp.rect)
                        xpx_sel,ypx_sel = self.plot_id.bmc.coords_to_px(lat,lon,self.plot_id.mp.rect)                        
                        dxpx = (xpx_sel - xpx_src + 0.5*self.plot_id.mp.rect.size[0])%self.plot_id.mp.rect.size[0] - 0.5*self.plot_id.mp.rect.size[0]
                        dypx = (ypx_sel - ypx_src)
                        if ( dxpx**2 + dypx**2 <= dp(15)**2 ) :
                            snap_source = s
                    if (snap_source in self.new_station_name_list) :
                        del _stationdicts['ngEHT+'][snap_source]
                        self.number_new_stations -= 1
                        _statdict = _stationdicts['ngEHT+']
                    self.del_station_btn.text = 'Del'
                    self.del_station_btn.color = _on_color

                    self.plot_id.update(_datadict,_statdict)
                    self.menu_id.refresh()

    
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


class ngEHTApp(MDApp):
    # pass
    
    # def __init__(self,**kwargs) :
    #     Window.bind(on_keyboard=self.key_input)


    # def build(self):
    #     pass
        
    # def build(self):
    #     Window.bind(on_keyboard=self.key_input)
    #     return ScreenManager # your root widget here as normal


    def build(self):
        Window.bind(on_keyboard=self.key_input)
        return None
    
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


    def key_input(self, window, key, scancode, codepoint, modifier):
        if key == 27:
            return True  # override the default behaviour
        else:           # the key now does nothing
            return False

     
        
if __name__ == '__main__' :
    ngEHTApp().run()

    
