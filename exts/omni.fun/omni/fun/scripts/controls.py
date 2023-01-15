import carb
import omni.ui
import omni.usd
import omni.kit.app
from pxr import Usd, Sdf
import time
from .sim import Sim

class ControlsWindow:
    def __init__(self):
        self._window = None
        
        stage = omni.usd.get_context().get_stage()
        self.sim = Sim(stage)
        self.paused = True
        self.init_button = None

        stage = omni.usd.get_context().get_stage()
        self.sim = Sim(stage)        
        

    def __bool__(self):
        return self._window is not None


    def create_window(self, visibility_changed_fn):
        window_flags = omni.ui.WINDOW_FLAGS_NO_SCROLLBAR
        self._window = omni.ui.Window("Fun Controls", flags=window_flags, width=400, height=400, dockPreference=omni.ui.DockPreference.RIGHT_TOP)
        self._window.set_visibility_changed_fn(visibility_changed_fn)
        self.build_ui()


    def show_window(self):
        self._window.visible = True


    def hide_window(self):
        self._window.visible = False


    def destroy_window(self):
        if self._window:
            self._window.visible = False
            self._window.destroy()
            self._window = None


    def init_sim(self):
        stage = omni.usd.get_context().get_stage()
        self.sim = Sim(stage)
        self.init_button.text = "Reset"
        

    def reset_sim(self):
        self.sim = None
        self.init_button.text = "Init"


    def init_pressed(self):
        if self.sim is not None:
            self.reset_sim()
        else:
            self.init_sim()


    def play_pressed(self):
        pass
        

    def set_parameter(self, param_name, val):
        if param_name == "gravity":
            self.gravity = val


    def build_ui(self):
        ui = omni.ui
        row_height = 20
        v_spacing = 10
        h_spacing = 20

        if self._window and self._window.visible:
            with self._window.frame:

                with ui.VStack(spacing=v_spacing, padding=50):

                    with ui.HStack(spacing=h_spacing, height=row_height):

                        self.init_button = ui.Button("Init", width=100, height=15, margin=10, clicked_fn=self.init_pressed)
                        self.start_button = ui.Button("Start", width=100, height=15, margin=10, clicked_fn=self.start_pressed)

                    with ui.HStack(spacing=h_spacing, height=row_height):

                        ui.Label("Gravity", width=ui.Percent(50), height=10, name="Gravity")
                        slider = ui.FloatSlider(min=0.0,max=10.0, width=ui.Percent(50))                                        
                        slider.model.add_value_changed_fn(lambda val, param_name="gravity": self.set_parameter(param_name, val.get_value_as_float()))

 


