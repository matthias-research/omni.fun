# Copyright 2022 Matthias Müller - Ten Minute Physics, 
# https://www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import omni.ext
import os
import omni.usd
from omni import ui
from pxr import Usd
from .controls import ControlsWindow
from .sim import Sim

EXAMPLES_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scenes"))

class OmniFunExtension(omni.ext.IExt):

    def on_startup(self, ext_id):

        print("fun on_startup")

        setattr(self, "controls", None)
        setattr(self, "sim", None)

        stage = omni.usd.get_context().get_stage()
        self.sim = Sim(stage)
        self.sim.init()
        
        editor_menu = omni.kit.ui.get_editor_menu()
        self.menu_items = []

        if editor_menu:

            self.controls_menu = editor_menu.add_item(
                f"Window/Fun/Controls", 
                lambda _, value: self.show_controls(value), 
                toggle=True, value=False
            )            

            self.menu_items.append(editor_menu.add_item(
                f"Window/Fun/SimpleScene", 
                lambda _, value: self.load_example("simple.usd"),
                toggle=False, value=False
            ))

        # self.show_controls(True)

        # set callbacks

        self.update_event_stream = omni.kit.app.get_app_interface().get_update_event_stream()
        self.stage_event_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self.on_event)


    def on_shutdown(self):

        print("fun on_shutdown")
        self.menu_items = None
        self.update_event_stream = None
        self.stage_event_sub = None
        if self.sim:
            self.sim.reset()
        self.show_controls(False)


    def init_callback(self, state):
        if state:
            stage = omni.usd.get_context().get_stage()
            if self.sim:
                self.sim = Sim(stage)        
            self.update_event_sub = self.update_event_stream.create_subscription_to_pop(self.on_update)
        else:
            if self.sim:
                self.sim.reset()
            self.sim = None


    def play_callback(self, state):
        if self.sim:
            self.sim.paused = not state


    def on_update(self, dt):
        if self.sim:
            self.sim.simulate()


    def set_controls_menu(self, visible):
        omni.kit.ui.get_editor_menu().set_value(f"Window/Fun/Controls", visible)


    def show_controls(self, is_visible):
        if is_visible: 
            if not hasattr(self, "controls"):
                setattr(self, "controls", None)
            if self.controls is None:
                self.controls = ControlsWindow(
                    init_callback=self.init_callback, 
                    play_callback=self.play_callback)
                self.controls.create_window(lambda visible: self.set_controls_menu(visible))
                self.controls.show_window()
            else:
                self.controls.show_window()
        elif self.controls:
            self.controls.destroy_window()
            self.controls = None


    def on_event(self, event):

        if event.type == int(omni.usd.StageEventType.CLOSED):
            if self.sim:
                self.sim.reset()

        if event.type == int(omni.usd.StageEventType.OPENED):
            if self.sim:
                self.sim.init()


    def load_example(self, scene_name):
        
        def new_stage():
            stage_path = os.path.normpath(os.path.join(EXAMPLES_PATH, scene_name))
            omni.usd.get_context().open_stage(stage_path)

            if self.sim:
                self.sim.init()
    
        omni.kit.window.file.prompt_if_unsaved_stage(new_stage)


