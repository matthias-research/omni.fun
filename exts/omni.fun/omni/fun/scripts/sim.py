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
from .gpu import *

@wp.struct
class SimData:
    sphere_radius: wp.array(dtype=float)
    sphere_pos: wp.array(dtype=wp.vec3)
    sphere_quat: wp.array(dtype=wp.quat)
    sphere_vel: wp.array(dtype=wp.vec3)
    sphere_omega: wp.array(dtype=wp.vec3)
    sphere_lower_bounds: wp.array(dtype=wp.vec3)
    sphere_upper_bounds: wp.array(dtype=wp.vec3)

    scene_mesh_id: wp.uint64
    scene_sphere_bvh_id: wp.uint64


@wp.kernel
def dev_integrate(
        dt: float,
        gravity: wp.vec3,
        sim: SimData):

    sphere_nr = wp.tid()

    vel = sim.sphere_vel[sphere_nr]
    vel = vel + gravity * dt
    sim.sphere_vel[sphere_nr] = vel

    pos = sim.sphere_pos[sphere_nr]
    pos = pos + vel * dt
    sim.sphere_pos[sphere_nr] = pos

    radius = sim.sphere_radius[sphere_nr]

    sim.sphere_lower_bounds[sphere_nr] = pos - wp.vec3(radius, radius, radius)
    sim.sphere_upper_bounds[sphere_nr] = pos + wp.vec3(radius, radius, radius)

 

class Sim():

    def __init__(self, stage):
        self.stage = stage
        self.device = 'cuda'
        self.controls = None

        self.dev_sim_data = SimData()
        self.host_sim_data = SimData()
        self.scene_mesh = None
        self.sphere_bvh = None

        self.sphere_usd_transforms = []

        self.initialized = False

        self.time_step = 1.0 / 30.0
        self.num_substeps = 5
        self.gravity = wp.vec3(0.0, 0.0, -self.controls.gravity)
        self.jacobi_scale = 0.25
        self.paused = True
        self.num_spheres = 0


    def init_sim(self):

        if not self.stage:
            return

        scene_points = []
        scene_tri_indices = []

        sphere_pos = []
        sphere_radius = []
        sphere_inv_mass = []
        self.sphere_usd_transforms = []

        s = 4.0 / 3.0 * 3.141592

        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Xform":
                xform = UsdGeom.Xform(prim)

                for child in prim.GetChildren():
                    if child.GetTypeName() == "Mesh":

                        trans = xform.MakeMatrixXform()
                        mat = trans.GetOpTransform(0.0)

                        mesh = UsdGeom.Mesh(child)
                        points = mesh.GetPointsAttr().Get(0.0)
                        name = mesh.GetName()

                        if name.find("sphere") != 0 or name.find("Sphere") != 0:

                            # create a sphere

                            radius = 0.0
                            for point in points:
                                radius = max(radius, mat.TransformDir(point).GetLength())

                            sphere_radius.append(radius)
                            sphere_pos.append([*mat.ExtractTranslation()])
                            mass = s * radius * radius * radius
                            sphere_inv_mass.append(1.0 / mass)
                            self.sphere_usd_transforms.append(trans)

                        else:

                            # create obstacle triangles

                            mesh_poly_indices = mesh.GetFaceVertexIndicesAttr().Get(0.0)
                            mesh_face_sizes = mesh.GetFaceVertexCountsAttr().Get(0.0)
                            mesh_points = np.array(points)

                            first_point = len(scene_points)

                            for i in range(len(mesh_points)):
                                scene_points.append(mesh_face_sizes[i])

                            first_index = 0

                            for i in range(len(mesh_face_sizes)):
                                face_size = mesh_face_sizes[i]
                                for j in range(1, face_size-1):
                                    scene_tri_indices.append(first_point + mesh_poly_indices[first_index])
                                    scene_tri_indices.append(first_point + mesh_poly_indices[first_index + j])
                                    scene_tri_indices.append(first_point + mesh_poly_indices[first_index + j + 1])
                                first_local = first_local + face_size

                        break
        
        # create scene warp buffers

        if len(scene_points) > 0:

            dev_points = wp.array(scene_points, dtype=wp.vec3, device=self.device)
            dev_tri_indices = wp.array(scene_tri_indices, dtype=int, device=self.device)
            self.scene_mesh = wp.Mesh(dev_points, dev_tri_indices)
            self.dev_sim_data.scene_mesh_id = self.scene_mesh.id

        # create sphere warp buffers

        self.num_spheres = len(sphere_pos)

        if self.num_spheres > 0:

            self.dev_sim_data.sphere_radius = wp.array(sphere_radius, dtype=float, device=self.device)
            self.dev_sim_data.sphere_pos = wp.array(sphere_pos, dtype=wp.vec3, device=self.device)
            self.dev_sim_data.sphere_quat = wp.zeros(shape=(self.num_spheres), dtype=wp.quat, device=self.device)
            self.dev_sim_data.sphere_vel = wp.zeros(shape=(self.num_spheres), dtype=wp.vec3, device=self.device)
            self.dev_sim_data.sphere_omega = wp.zeros(shape=(self.num_spheres), dtype=wp.vec3, device=self.device)
            self.dev_sim_data.sphere_lower_bounds = wp.zeros(shape=(self.num_spheres), dtype=wp.vec3, device=self.device)
            self.dev_sim_data.sphere_upper_bounds = wp.zeros(shape=(self.num_spheres), dtype=wp.vec3, device=self.device)

            self.host_sim_data.sphere_pos = wp.array(sphere_pos, dtype=wp.vec3, device="cpu")
            self.host_sim_data.sphere_quat = wp.zeros(shape=(self.num_spheres), dtype=wp.quat, device="cpu")

            # zero time step to initialize sphere bounds

            wp.launch(kernel = self.dev_integrate, 
                inputs = [0.0, self.gravity, self.dev_sim_data],
                dim = self.num_spheres, device=self.device)

            self.sphere_bvh = wp.Bvh(self.dev_sim_data.sphere_lower_bounds, self.dev_sim_data.sphere_upper_bounds)
            self.dev_sim_data.sphere_bvh_id = self.sphere_bvh.id


    def simulate(self):

        wp.launch(kernel = self.dev_integrate, 
            inputs = [self.time_step, self.gravity, self.dev_sim_data],
            dim = self.num_spheres, device=self.device)

        self.sphere_bvh.refit()


    def update_stage(self):

        wp.copy(self.host_sim_data.sphere_pos, self.dev_sim_data.sphere_pos)
        wp.copy(self.host_sim_data.sphere_quat, self.dev_sim_data.sphere_quat)

        pos = self.host_sim_data.numpy()
        quat = self.host_sim_data.numpy()

        for i in range(self.num_spheres):

            mat = Gf.Matrix4d(Gf.Rotation(Gf.Quatd(*quat[i])), Gf.Vec3d(*pos[i]))
            self.sphere_usd_transforms[i] .Set(mat)

 


    def reset(self):

        for sphere in self.spheres:
            sphere.set_position(sphere.pos)

        self.paused = True




