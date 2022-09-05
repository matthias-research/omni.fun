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


@wp.kernel
def solve_collisions(
        dt: float,
        mesh_id: wp.uint64,
        pos_array: wp.array(dtype=wp.vec3),
        vel_array: wp.array(dtype=wp.vec3),
        mass_array: wp.array(dtype=float)),
        radius_array: wp.array(dtype=float)):
    
    bounds_lower = wp.vec3(
        wp.min(wp.min(p0[0], p1[0]), p2[0]) - boundary,
        wp.min(wp.min(p0[1], p1[1]), p2[1]) - boundary,
        wp.min(wp.min(p0[2], p1[2]), p2[2]) - boundary)

    bounds_upper = wp.vec3(
        wp.max(wp.max(p0[0], p1[0]), p2[0]) + boundary,
        wp.max(wp.max(p0[1], p1[1]), p2[1]) + boundary,
        wp.max(wp.max(p0[2], p1[2]), p2[2]) + boundary)



    att_nr = wp.tid()

    first_param = att_nr * ATTACHMENT_PARAMS_SIZE
    if attachment.params[first_param + ATTACHMENT_ENABLED] == 0.0:
        return

    type = attachment.type[att_nr]
    particle_nr = attachment.particle_nr[att_nr]
    if particle_nr < 0:
        return


class Sim():

    def __init__(self, stage, controls):
        self.stage = stage
        self.device = 'cuda'
        self.controls = controls
        self.gravity = wp.vec3(0.0, 0.0, -self.controls.gravity)
        self.collision_mesh = None
        self.spheres = []

        self.pos_array = None
        self.gpu_pos_array = None
        self.vel_array = None
        self.gpu_vel_array = None

    class Sphere():

        def __init__(self, prim):
            self.xform_op = None
            self.radius = 0.0
            self.pos = wp.vec3(0.0, 0.0, 0.0)
            self.vel = wp.vec3(0.0, 0.0, 0.0)

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

        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Sphere":
                self.spheres.append(self.Sphere(prim))

        num_spheres = len(self.spheres)
        if num_spheres > 0:
            self.pos_array = np.zeros(num_spheres, dtype = wp.vec3)
            self.vel_array = np.zeros(num_spheres, dtype = wp.vec3)

            for i in range(num_spheres):
                self.pos_array[i] = self.spheres[i].pos

            self.gpu_pos_array = wp.zeros(shape=(num_spheres,1), dtype = wp.vec3)
            self.gpu_vel_array = wp.zeros(shape=(num_spheres,1), dtype = wp.vec3)
            wp.copy(self.gpu_pos_array, self.pos_array)
            wp.copy(self.gpu_vel_array, self.vel_array)


        # create one big triangle mesh from all meshes in the scene

        prim_cache = UsdGeom.XformCache()
        prim_cache.SetTime(0.0)

        tri_ids = []
        verts = []

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

        if len(verts) > 0:
            dev_verts = wp.array(verts, dtype = wp.vec3, device=self.device)
            dev_tri_ids = wp.array(verts, dtype = int, device=self.device)
            self.collision_mesh = wp.Mesh(dev_verts, dev_tri_ids)







    def update(self):

        if not self.stage:
            return

        num_spheres = len(self.spheres)
    
        if num_spheres > 0:

    


        

 
 