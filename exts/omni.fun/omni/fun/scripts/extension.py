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
import sim

EXAMPLES_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scenes"))

class OmniPlayExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        
        editor_menu = omni.kit.ui.get_editor_menu()
        self.menu_items = []

        if editor_menu:

            self.menu_items.append(editor_menu.add_item(
                f"Window/Fun/Start", 
                lambda _, value: self.start(),
                toggle=False, value=False
            ))
            self.menu_items.append(editor_menu.add_item(
                f"Window/Fun/Basics example", 
                lambda _, value: self.load_example("basics.usd"),
                toggle=False, value=False
            ))

        self.sim = None
 
        # set callbacks

        self.update_event_stream = omni.kit.app.get_app_interface().get_update_event_stream()
        self.stage_event_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self.on_event)


    def on_shutdown(self):

        self.menu_items = None
        self.update_event_stream = None
        self.stage_event_sub = None
        self.sim = None


    def init_sim(self):

        stage = omni.usd.get_context().get_stage()
        self.sim = sim.Sim(stage)
        self.update_event_sub = self.update_event_stream.create_subscription_to_pop(self.on_update)


    def on_update(self, event):

        if self.sim:
            self.sim.on_update()

    def on_event(self, event):

        if event.type == int(omni.usd.StageEventType.CLOSED):
            self.sim = None

        if event.type == int(omni.usd.StageEventType.OPENED):
            pass


    def load_example(self, scene_name):
        
        def new_stage():
            stage_path = os.path.normpath(os.path.join(EXAMPLES_PATH, scene_name))
            omni.usd.get_context().open_stage(stage_path)

            self.init_sim()
    
        omni.kit.window.file.prompt_if_unsaved_stage(new_stage)


