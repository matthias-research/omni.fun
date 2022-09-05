# Copyright 2022 Matthias Müller - Ten Minute Physics, 
# https://www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import math
import warp as wp

from pxr import Usd, UsdGeom, Gf, Sdf
from gpu import *


class Sim():

    def __init__(self, stage, controls):
        self.stage = stage
        self.device = 'cuda'
        self.controls = controls
        self.collision_mesh = None
        self.spheres = []

        self.dev_data = DeviceData
        self.host_pos = None

        self.time_step = 1.0 / 30.0
        self.num_substeps = 5
        self.gravity = wp.vec3(0.0, 0.0, -self.controls.gravity)


    class Sphere():

        def __init__(self, prim):
            self.xform_op = None
            self.radius = 0.0
            self.pos = wp.vec3(0.0, 0.0, 0.0)

            if prim.GetTypeName() == "Sphere":
                sphere = UsdGeom.Sphere(prim)
                self.radius = sphere.GetRadiusAttr().Get()

                parent = prim.GetParent()
                if parent.GetTypeName() == "Xform":
                    xform = UsdGeom.Xform(parent)
                    ops = xform.GetOrderedXformOps()
                    for i in range(len(ops)):
                        if ops[i].GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            self.xform_op = ops[i]
                            pos = self.xform_op.Get()
                            self.pos = wp.vec3(pos[0], pos[1], pos[2])

        def set_position(self, pos: wp.vec3):
            self.pos = pos
            if self.xform_op:
                self.xform_op.Set(Gf.Vec3d(pos[0], pos[1], pos[2]))


    def init_sim(self):

        if not self.stage:
            return

        # collect all sphere shapes

        self.spheres = []
        spheres_pos = []
        spheres_radius = []
        spheres_mass = []

        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Sphere":
                sphere = self.Sphere(prim)
                self.spheres.append(sphere)
                spheres_pos.append(sphere.pos)
                r = sphere.radius
                spheres_radius.append(r)
                spheres_mass.append(4.0 * math.pi / 3.0 * r*r*r)

        num_spheres = len(self.spheres)

        # create on big triangle mesh for all  meshes in the scene

        tri_ids = []
        verts = []

        prim_cache = UsdGeom.XformCache()
        prim_cache.SetTime(0.0)

        for prim in self.stage.Traverse():

            if prim.GetTypeName() == "Mesh":

                mesh = UsdGeom.Mesh(prim)
                mesh_points = np.array(mesh.GetPointsAttr().Get(0.0))

                m = prim_cache.GetLocalToWorldTransform(prim)
                trans_mat = np.array([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]) 
                trans_t = np.array([m[3][0], m[3][1], m[3][2]])

                verts = mesh_points @ trans_mat + trans_t

                face_ids = mesh.GetFaceVertexIndicesAttr().Get(0.0)
                face_sizes = mesh.GetFaceVertexCountsAttr().Get(0.0)

                first_id = 0
                for size in face_sizes:
                    for i in range(1, size-1):
                        tri_ids.append(face_ids[first_id])
                        tri_ids.append(face_ids[first_id + i])
                        tri_ids.append(face_ids[first_id + i + 1])


        # create device buffers

        if len(verts) > 0:
            self.dev_data.spheres_pos = wp.array(spheres_pos, dtype=wp.vec3, device=self.device)
            self.dev_data.spheres_prev_pos = wp.array(verts, dtype=wp.vec3, device=self.device)
            self.dev_data.spheres_prev_pos = wp.zeros(shape=(num_spheres), dtype=wp.vec3, device=self.device)
            self.dev_data.spheres_vel: wp.zeros(shape=(num_spheres), dtype=wp.vec3, device=self.device)            
            self.dev_data.spheres_radius: wp.array(spheres_radius, dtype=float)
            self.dev_data.spheres_mass: wp.array(spheres_mass, dtype=float)

            dev_verts = wp.array(verts, dtype = wp.vec3, device=self.device)
            dev_tri_ids = wp.array(tri_ids, dtype = int, device=self.device)
            dev_mesh = wp.Mesh(dev_verts, dev_tri_ids)
            self.dev_data.mesh_id = dev_mesh.id

            self.host_pos = wp.array(spheres_pos, dtype=wp.vec3, device="cpu")


    def update(self):

        if not self.stage:
            return
            
        num_spheres = len(self.spheres)    
        if num_spheres == 0:
            return

        dt = self.time_step / self.num_substeps

        # send positions to gpu

        

        # simulate

        for i in range(self.num_substeps):

            integrate_spheres(num_spheres, dt, self.gravity, self.dev_data, self.device)
            solve_spheres_collisions(num_spheres, self.dev_data)
            update_spheres(num_spheres, dt, self.dev_data, self.device)

        # read back pos

        wp.copy(self.host_pos, self.dev_data.sphere_pos)
        np_pos = self.host_pos.numpy()

        for i in range(num_spheres):
            self.spheres[i].set_position(np_pos[i])

