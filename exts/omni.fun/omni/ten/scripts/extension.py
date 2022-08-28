# Copyright 2022 Matthias Müller - Ten Minute Physics, 
# https://www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import omni.ext
import os
import carb
import omni.usd
from omni import ui
import imp
from pxr import Usd, UsdGeom, Gf
import numpy as np
import sim

EXAMPLES_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data/scenes"))

class OmniPlayExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        carb.log_info("OmniTenExtension startup")
        
        editor_menu = omni.kit.ui.get_editor_menu()
        self.menu_items = []

        if editor_menu:

            self.menu_items.append(editor_menu.add_item(
                f"Window/TenMinPhysics/Basics example", 
                lambda _, value: self.load_example("basics.usd"),
                toggle=False, value=False
            ))
            self.menu_items.append(editor_menu.add_item(
                f"Window/TenMinPhysic/Init", 
                lambda _, value: self.init(),
                toggle=False, value=False
            ))

        self.sim = None
        self.viewProjMtxInv = None
 
        # set callbacks

        self.update_event_stream = omni.kit.app.get_app_interface().get_update_event_stream()
        self.stage_event_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(self.on_event)

        input = carb.input.acquire_input_interface()
        input.subscribe_to_input_events(self.on_input_event, order=0)


    def on_shutdown(self):
        carb.log_info("OmniTenExtension shutdown")

        self.menu_items = None
        self.update_event_stream = None
        self.stage_event_sub = None
        self.sim = None


    def init_sim(self):

        stage = omni.usd.get_context().get_stage()
        self.sim = sim.Sim(stage)
        self.update_event_sub = self.update_event_stream.create_subscription_to_pop(self.on_update)


    def on_update(self, event):

        timeline = omni.timeline.get_timeline_interface()
        # if timeline.is_playing():
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


    def on_input_event(self, event, *_):
        if event.deviceType == carb.input.DeviceType.MOUSE:
            if event.event.modifiers in [0, carb.input.KEYBOARD_MODIFIER_FLAG_SHIFT]:
                self.on_mouse_event(event.event)
        elif event.deviceType == carb.input.DeviceType.KEYBOARD:
            self.on_keyboard_event(event.event)
        else:
            return True


    def on_mouse_event(self, event, *_):

        if self.play:
            if event.modifiers in [0, carb.input.KEYBOARD_MODIFIER_FLAG_SHIFT]:
                proj_x = -1.0 + 2.0 * event.normalized_coords.x
                proj_y = 1.0 - 2.0 * event.normalized_coords.y

                viewport_api = omni.kit.viewport.utility.get_active_viewport_window().viewport_api
                view_proj_mat = viewport_api.view * viewport_api.projection
                inv_view_proj_mat = view_proj_mat.GetInverse()

                ray_origin = inv_view_proj_mat.Transform(Gf.Vec3d(proj_x, proj_y, 0.0))
                forward = inv_view_proj_mat.Transform(Gf.Vec3d(proj_x, proj_y, 1.0))
                ray_dir = (forward - ray_origin).GetNormalized()

                if event.type == carb.input.MouseEventType.LEFT_BUTTON_DOWN:
                    self.sim.on_mouse_button(ray_origin, ray_dir, True, 0)
                elif event.type == carb.input.MouseEventType.LEFT_BUTTON_UP:
                    self.sim.on_mouse_button(ray_origin, ray_dir, False, 0)
                elif event.type == carb.input.MouseEventType.RIGHT_BUTTON_DOWN:
                    self.sim.on_mouse_button(ray_origin, ray_dir, True, 1)
                elif event.type == carb.input.MouseEventType.RIGHT_BUTTON_UP:
                    self.sim.on_mouse_button(ray_origin, ray_dir, False, 1)
                elif event.type == carb.input.MouseEventType.MOVE:
                    self.sim.on_mouse_motion(ray_origin, ray_dir)


    def on_keyboard_event(self, event, *args, **kwargs):
        if self.sim:
            down = event.type == carb.input.KeyboardEventType.KEY_PRESS

            if event.input == carb.input.KeyboardInput.A:
                self.sim.on_key(b'A', down)
            elif event.input == carb.input.KeyboardInput.B:
                self.sim.on_key(b'B', down)
            elif event.input == carb.input.KeyboardInput.C:
                self.sim.on_key(b'C', down)
            elif event.input == carb.input.KeyboardInput.D:
                self.sim.on_key(b'D', down)
            elif event.input == carb.input.KeyboardInput.E:
                self.sim.on_key(b'E', down)
            elif event.input == carb.input.KeyboardInput.F:
                self.sim.on_key(b'F', down)
            elif event.input == carb.input.KeyboardInput.G:
                self.sim.on_key(b'G', down)
            elif event.input == carb.input.KeyboardInput.H:
                self.sim.on_key(b'H', down)
            elif event.input == carb.input.KeyboardInput.I:
                self.sim.on_key(b'I', down)
            elif event.input == carb.input.KeyboardInput.J:
                self.sim.on_key(b'J', down)
            elif event.input == carb.input.KeyboardInput.K:
                self.sim.on_key(b'K', down)
            elif event.input == carb.input.KeyboardInput.L:
                self.sim.on_key(b'L', down)
            elif event.input == carb.input.KeyboardInput.M:
                self.sim.on_key(b'M', down)
            elif event.input == carb.input.KeyboardInput.N:
                self.sim.on_key(b'N', down)
            elif event.input == carb.input.KeyboardInput.O:
                self.sim.on_key(b'O', down)
            elif event.input == carb.input.KeyboardInput.P:
                self.sim.on_key(b'P', down)
            elif event.input == carb.input.KeyboardInput.Q:
                self.sim.on_key(b'Q', down)
            elif event.input == carb.input.KeyboardInput.R:
                self.sim.on_key(b'R', down)
            elif event.input == carb.input.KeyboardInput.S:
                self.sim.on_key(b'S', down)
            elif event.input == carb.input.KeyboardInput.T:
                self.sim.on_key(b'T', down)
            elif event.input == carb.input.KeyboardInput.U:
                self.sim.on_key(b'U', down)
            elif event.input == carb.input.KeyboardInput.V:
                self.sim.on_key(b'V', down)
            elif event.input == carb.input.KeyboardInput.W:
                self.sim.on_key(b'W', down)
            elif event.input == carb.input.KeyboardInput.X:
                self.sim.on_key(b'X', down)
            elif event.input == carb.input.KeyboardInput.Y:
                self.sim.on_key(b'Y', down)
            elif event.input == carb.input.KeyboardInput.Z:
                self.sim.on_key(b'Z', down)
