from kivy.clock import Clock
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.graphics import RenderContext, Color, Rectangle, BindTexture
from kivy.graphics.texture import Texture
from kivy.properties import ObjectProperty, ListProperty, StringProperty
from kivy.core.image import Image

from array import array
import numpy as np

import copy

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

__mydebug__ = True


class InteractivePlotWidget(Widget):

    tex_coords = ListProperty([0, 1, 1, 1, 1, 0, 0, 0])
    
    def __init__(self, **kwargs):
        self.canvas = RenderContext()
        # setting shader.fs to new source code automatically compiles it.
        # self.canvas.shader.fs = fs_multitexture

        self.nx = 1024
        self.ny = self.nx

        print("On init:",self.nx,self.ny)
        
        with self.canvas:
            Color(1, 1, 1)
            self.texture = Texture.create(size=(self.nx,self.ny))
            self.buf = [0,0,0,255]*(self.nx*self.ny)
            self.arr = array('B',self.buf)
            self.update_mpl()
            self.texture.blit_buffer(self.arr, colorfmt='rgba', bufferfmt='ubyte')
            BindTexture(texture=self.texture, index=0)
            self.texture.wrap = 'clamp_to_edge'
            
            # create a rectangle on which to plot texture (will be at index 0)
            Color(1,1,1)
            self.rect = Rectangle(size=(self.nx,self.ny),texture=self.texture)
            self.rect.tex_coords = self.tex_coords
            

        self.plot_frozen = False
        
        # call the constructor of parent
        # if they are any graphics objects, they will be added on our new
        # canvas
        super(InteractivePlotWidget, self).__init__(**kwargs)

        # We'll update our glsl variables in a clock
        # Clock.schedule_interval(self.update_glsl, 0)        
        Clock.schedule_interval(self.texture_init, 0)

        # Generate some default resizing behaviors
        self.bind(height=self.resize)
        self.bind(width=self.resize)
        
    def update_glsl(self, *largs):
        # This is needed for the default vertex shader.
        self.canvas['projection_mat'] = Window.render_context['projection_mat']
        self.canvas['modelview_mat'] = Window.render_context['modelview_mat']

    def texture_init(self, *args):
        self.texture = self.canvas.children[-1].texture
        self.update_glsl()

    def on_touch_move(self,touch) :
        if (not self.plot_frozen) :
            x_shift = - touch.dpos[0]/float(self.rect.size[0])
            y_shift = touch.dpos[1]/float(self.rect.size[1])
            for i in range(0,8,2) :
                self.tex_coords[i] = self.tex_coords[i] + x_shift
                self.tex_coords[i+1] = self.tex_coords[i+1] + y_shift
            self.tex_coords = self.check_boundaries(self.tex_coords)
            self.rect.tex_coords = self.tex_coords

    def on_touch_down(self,touch) :
        if (touch.is_double_tap) :
            self.tex_coords = [0, 1, 1, 1, 1, 0, 0, 0]
            self.rect.tex_coords = self.tex_coords
            self.rect.size = (self.nx,self.ny)

    def zoom_in(self) :
        self.rect.size = (self.rect.size[0]*1.414,self.rect.size[1]*1.414)

    def zoom_out(self) :
        self.rect.size = (self.rect.size[0]*0.707,self.rect.size[1]*0.707)
        self.tex_coords = self.check_boundaries(self.tex_coords)
        self.rect.tex_coords = self.tex_coords

    def resize(self,widget,newsize) :
        self.set_zoom_factor(2)

    def set_zoom_factor(self,value) :
        self.rect.size = (self.nx*value,self.ny*value)
        x_shift = -0.5*(self.width-self.rect.size[0])/float(self.rect.size[0])
        y_shift = 0.5*(self.height-self.rect.size[1])/float(self.rect.size[1])
        self.tex_coords = [0, 1, 1, 1, 1, 0, 0, 0]        
        for i in range(0,8,2) :
            self.tex_coords[i] = self.tex_coords[i] + x_shift
            self.tex_coords[i+1] = self.tex_coords[i+1] + y_shift
        self.tex_coords = self.check_boundaries(self.tex_coords)
        self.rect.tex_coords = self.tex_coords
        
    def check_boundaries(self,tex_coords) :
        new_tex_coords = [0]*len(tex_coords)
        max_x_shift = max((self.rect.size[0]-self.width)/self.rect.size[0],0)
        new_tex_coords[0] = max(min(tex_coords[0],max_x_shift),0)
        new_tex_coords[2] = max(min(tex_coords[2],1+max_x_shift),1)
        new_tex_coords[4] = max(min(tex_coords[4],1+max_x_shift),1)
        new_tex_coords[6] = max(min(tex_coords[6],max_x_shift),0)
        max_y_shift = max((self.rect.size[1]-self.height)/self.rect.size[1],0)
        new_tex_coords[1] = min(max(tex_coords[1],1-max_y_shift),1)
        new_tex_coords[3] = min(max(tex_coords[3],1-max_y_shift),1)
        new_tex_coords[5] = min(max(tex_coords[5],-max_y_shift),0)
        new_tex_coords[7] = min(max(tex_coords[7],-max_y_shift),0)
        return new_tex_coords

    def update_mpl(self) :
        fig = Figure(figsize=(self.nx/128,self.ny/128),dpi=128)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111,position=[0,0,1,1])
        self.generate_mpl_plot(fig,ax)
        canvas.draw()
        self.buf = np.asarray(canvas.buffer_rgba()).ravel()
        self.arr = array('B', self.buf)
        self.texture.blit_buffer(self.arr, colorfmt='rgba', bufferfmt='ubyte')

    def generate_mpl_plot(self,fig,ax) :
        # This is where we insert a Matplotlib figure.  Must use ax. and fig. child commands.
        pass



fs_multitexture = '''
$HEADER$

// New uniform that will receive texture at index 1
uniform sampler2D texture1;

void main(void) {

    // multiple current color with both texture (0 and 1).
    // currently, both will use exactly the same texture coordinates.
    //gl_FragColor = frag_color * \
    //    texture2D(texture0, tex_coord0) * \
    //    texture2D(texture1, tex_coord0);
    vec4 c0 = texture2D(texture0, tex_coord0);
    vec4 c1 = texture2D(texture1, tex_coord0);
    //gl_FragColor = vec4 ((c0.r*c0.a+c1.r*c1.a)/(c0.a+c1.a),(c0.g*c0.a+c1.g*c1.a)/(c0.a+c1.a),(c0.b*c0.a+c1.b*c1.a)/(c0.a+c1.a),1.0);
    
    //gl_FragColor = (1.0/(c0.a+10.0*c1.a)) * (c0.a*c0 + 10.0*c1.a*c1) ;
    gl_FragColor = (1.0-c1.a)*c0 + c1.a*c1;
}
'''

class InteractiveWorldMapOverlayWidget(Widget):

    tex_coords = ListProperty([0, 1, 1, 1, 1, 0, 0, 0])
    texture_wrap = StringProperty('repeat')
    
    def __init__(self, **kwargs):
        self.canvas = RenderContext()
        self.canvas.shader.fs = fs_multitexture

        self.nx = 1024
        self.ny = self.nx//2

        print("On init:",self.nx,self.ny)
        
        with self.canvas:
            # Overlay texture
            self.texture1 = Texture.create(size=(self.nx,self.ny))
            self.buf = [255,255,255,0]*(self.nx*self.ny)
            self.arr = array('B',self.buf)
            self.update_mpl()
            self.texture1.blit_buffer(self.arr, colorfmt='rgba', bufferfmt='ubyte')
            BindTexture(texture=self.texture1, index=1)
            self.texture1.wrap = self.texture_wrap

            # Background texture
            self.texture2 = Image('./images/world_spherical.jpg').texture
            self.texture2.wrap = self.texture_wrap
            self.rect = Rectangle(size=(self.nx,self.ny),texture=self.texture2)
            self.rect.tex_coords = self.tex_coords

        if (__mydebug__) :
            print("InteractiveWorldMapOverlayWidget._init__ rect.size:",self.rect.size)
            
        # set the texture1 to use texture index 1
        self.canvas['texture1'] = 1

        # Don't restrict zooming at start
        self.plot_frozen = False
        
        # call the constructor of parent
        # if they are any graphics objects, they will be added on our new
        # canvas
        super(InteractiveWorldMapOverlayWidget, self).__init__(**kwargs)

        # We'll update our glsl variables in a clock
        # Clock.schedule_interval(self.update_glsl, 0)        
        Clock.schedule_interval(self.texture_init, 0)

        # Generate some default resizing behaviors
        self.bind(height=self.resize)
        self.bind(width=self.resize)
        
    def update_glsl(self, *largs):
        # This is needed for the default vertex shader.
        self.canvas['projection_mat'] = Window.render_context['projection_mat']
        self.canvas['modelview_mat'] = Window.render_context['modelview_mat']

    def texture_init(self, *args):
        self.texture = self.canvas.children[-1].texture
        self.update_glsl()

    def on_touch_move(self,touch) :
        if (not self.plot_frozen) :
            x_shift = - touch.dpos[0]/float(self.rect.size[0])
            y_shift = touch.dpos[1]/float(self.rect.size[1])
            
            for i in range(0,8,2) :
                self.tex_coords[i] = self.tex_coords[i] + x_shift
                self.tex_coords[i+1] = self.tex_coords[i+1] + y_shift

            if (__mydebug__) :
                print("InteractiveWorldMapOverlayWidget.on_touch_move:")
                print("   tex_coords before :",self.tex_coords)
                print("   size/pos/width/height :",self.rect.size,self.rect.pos,self.width,self.height)
                
            self.tex_coords = self.check_boundaries(self.tex_coords)
            
            if (__mydebug__) :
                print("InteractiveWorldMapOverlayWidget.on_touch_move:")
                print("   tex_coords  after :",self.tex_coords)
                print("   size/pos/width/height :",self.rect.size,self.rect.pos,self.width,self.height)
            
            self.rect.tex_coords = self.tex_coords

    def on_touch_down(self,touch) :
        if (touch.is_double_tap) :
            self.tex_coords = [0, 1, 1, 1, 1, 0, 0, 0]
            self.rect.tex_coords = self.tex_coords
            self.rect.size = (self.nx*self.height/self.ny,self.height)
            self.rect.pos = (max(0,0.5*(self.width-self.rect.size[0])),(self.height-self.rect.size[1]))

    def zoom_in(self) :
        self.rect.size = (self.rect.size[0]*1.414,self.rect.size[1]*1.414)
        self.rect.pos = (max(0,0.5*(self.width-self.rect.size[0])),(self.height-self.rect.size[1]))
        if (__mydebug__) :
            print("InteractiveWorldMapOverlayWidget.zoom_in",self.rect.size,self.rect.pos)
            
    def zoom_out(self) :
        self.rect.size = (self.rect.size[0]*0.707,self.rect.size[1]*0.707)
        self.rect.pos = (max(0,0.5*(self.width-self.rect.size[0])),(self.height-self.rect.size[1]))
        self.tex_coords = self.check_boundaries(self.tex_coords)
        self.rect.tex_coords = self.tex_coords
        if (__mydebug__) :
            print("InteractiveWorldMapOverlayWidget.zoom_in:",self.rect.size,self.rect.pos)

    def resize(self,widget,newsize) :
        self.tex_coords = [0, 1, 1, 1, 1, 0, 0, 0]
        self.rect.tex_coords = self.tex_coords
        self.rect.size = (self.nx*self.height/self.ny,self.height)
        self.rect.pos = (max(0,0.5*(self.width-self.rect.size[0])),(self.height-self.rect.size[1]))

    def set_zoom_factor(self,value) :
        self.rect.size = (self.nx*value,self.ny*value)
        x_shift = -0.5*(self.width-self.rect.size[0])/float(self.rect.size[0])
        y_shift = 0.5*(self.height-self.rect.size[1])/float(self.rect.size[1])
        self.tex_coords = [0, 1, 1, 1, 1, 0, 0, 0]        
        for i in range(0,8,2) :
            self.tex_coords[i] = self.tex_coords[i] + x_shift
            self.tex_coords[i+1] = self.tex_coords[i+1] + y_shift
        self.tex_coords = self.check_boundaries(self.tex_coords)
        self.rect.tex_coords = self.tex_coords
        self.rect.pos = (max(0,0.5*(self.width-self.rect.size[0])),(self.height-self.rect.size[1]))

    def check_boundaries(self,tex_coords) :
        new_tex_coords = copy.copy(tex_coords)
        max_y_shift = max((self.rect.size[1]-self.height)/self.rect.size[1],0)
        new_tex_coords[1] = max(min(tex_coords[1],1+max_y_shift),1)
        new_tex_coords[3] = max(min(tex_coords[3],1+max_y_shift),1)
        new_tex_coords[5] = max(min(tex_coords[5],max_y_shift),0)
        new_tex_coords[7] = max(min(tex_coords[7],max_y_shift),0)
        return new_tex_coords

    def update_mpl(self) :
        fig = Figure(figsize=(self.nx/128,self.ny/128),dpi=128)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111,position=[0,0,1,1])
        self.generate_mpl_plot(fig,ax)
        canvas.draw()
        self.buf = np.asarray(canvas.buffer_rgba()).ravel()
        self.arr = array('B', self.buf)
        self.texture1.blit_buffer(self.arr, colorfmt='rgba', bufferfmt='ubyte')

    def generate_mpl_plot(self,fig,ax) :
        # This is where we insert a Matplotlib figure.  Must use ax. and fig. child commands.
        # You probably want, but do not require, the following in your over-lay
        ax.set_facecolor((0,0,0,0))
        fig.set_facecolor((0,0,0,0))

