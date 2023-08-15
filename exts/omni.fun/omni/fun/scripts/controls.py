import carb
import omni.ui
import omni.usd
import omni.kit.app
from pxr import Usd, Sdf
from .sim import Sim


class ControlsWindow:
    def __init__(self, init_callback=None, play_callback=None):
        self._window = None
        self.buttons = [
            [None, init_callback, False, "Init", "Reset"], 
            [None, play_callback, False, "Play", "Pause"]]

    def __bool__(self):
        return self._window is not None

    def create_window(self, visibility_changed_fn):
        window_flags = omni.ui.WINDOW_FLAGS_NO_SCROLLBAR
        self._window = omni.ui.Window("Fun Controls", flags=window_flags, width=400, height=400, dockPreference=omni.ui.DockPreference.RIGHT_TOP)
        self._window.set_visibility_changed_fn(visibility_changed_fn)
        self.rebuild_ui()


    def show_window(self):
        self._window.visible = True


    def hide_window(self):
        self._window.visible = False


    def destroy_window(self):
        if self._window:
            self._window.visible = False
            self._window.destroy()
            self._window = None        


    def button_pressed(self, button):
        state = not button[2]
        button[2] = state
        button[0].text = button[4] if state else button[3]
        button[1](state)


    def set_parameter(self, param_name, val):
        if param_name == "gravity":
            sim.gravity = val


    def rebuild_ui(self):
        ui = omni.ui
        row_height = 20
        v_spacing = 10
        h_spacing = 20

        if self._window and self._window.visible:
            with self._window.frame:

                with ui.VStack(spacing=v_spacing, padding=50):

                    with ui.HStack(spacing=h_spacing, height=row_height):

                        for button in self.buttons:

                            button[0] = ui.Button(
                                button[3], width=100, height=15, margin=10, 
                                clicked_fn=lambda button=button: self.button_pressed(button))

                    with ui.HStack(spacing=h_spacing, height=row_height):

                        ui.Label("Gravity", width=ui.Percent(50), height=10, name="Gravity")
                        slider = ui.FloatSlider(min=0.0,max=10.0, width=ui.Percent(50))                                        
                        slider.model.add_value_changed_fn(
                            lambda val, param_name="gravity": self.set_parameter(param_name, val.get_value_as_float()))

 


