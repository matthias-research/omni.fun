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
import usdutils

@wp.struct
class SimData:
    mesh_points: wp.array(dtype=wp.vec3)
    mesh_prev_points: wp.array(dtype=wp.vec3)
    mesh_tri_ids: wp.array(dtype=int)

    sphere_radius: wp.array(dtype=float)
    sphere_mass: wp.array(dtype=float)

    sphere_pos: wp.array(dtype=wp.vec3)
    sphere_rot: wp.array(dtype=wp.quat)
    sphere_lin_vel: wp.array(dtype=wp.vec3)
    sphere_ang_vel: wp.array(dtype=wp.vec3)

    sphere_pos_corr: wp.array(dtype=wp.vec3)
    sphere_lin_corr: wp.array(dtype=wp.vec3)
    sphere_ang_corr: wp.array(dtype=wp.vec3)
    sphere_num_corr: wp.array(dtype=int)

    sphere_lower_bounds: wp.array(dtype=wp.vec3)
    sphere_upper_bounds: wp.array(dtype=wp.vec3)

    scene_mesh_id: wp.uint64
    scene_sphere_bvh_id: wp.uint64


@wp.kernel
def dev_integrate(
        dt: float,
        gravity: wp.vec3,
        bounds_margin: float,
        sim: SimData):

    sphere_nr = wp.tid()

    pos = sim.sphere_pos[sphere_nr]
    lin_vel = sim.sphere_lin_vel[sphere_nr]
    rot = sim.sphere_rot[sphere_nr]
    ang_vel = sim.sphere_ang_vel[sphere_nr]

    # move state forward in time

    lin_vel = lin_vel + gravity * dt
    pos = pos + lin_vel * dt
    qt = wp.quat(ang_vel[0], ang_vel[1], ang_vel[2], 0.0) * (dt * 0.5)
    rot = wp.normalize(rot + qt * rot)

    sim.sphere_pos[sphere_nr] = pos
    sim.sphere_lin_vel[sphere_nr] = lin_vel
    sim.sphere_rot[sphere_nr] = rot
    
    # compute bounding box for bvh

    pred_pos = pos + lin_vel * dt
    lower = wp.vec3(wp.min(pos[0], pred_pos[0]), wp.min(pos[1], pred_pos[1]), wp.min(pos[2], pred_pos[2]))
    upper = wp.vec3(wp.max(pos[0], pred_pos[0]), wp.max(pos[1], pred_pos[1]), wp.max(pos[2], pred_pos[2]))

    m = bounds_margin + sim.sphere_radius[sphere_nr]
    sim.sphere_lower_bounds[sphere_nr] = lower - wp.vec3(m, m, m)
    sim.sphere_upper_bounds[sphere_nr] = upper + wp.vec3(m, m, m)

 
@wp.kernel
def dev_handle_sphere_sphere_collisions(
        restitution: float,
        sim: SimData):

    sphere0 = wp.tid()
    eps = 0.00001

    pos0 = sim.sphere_pos[sphere0]
    radius0 = sim.sphere_radius[sphere0]
    m0 = sim.sphere_mass[sphere0]
    w0 = 1.0 / (m0 + eps)
    vel0 = sim.lin_vel[sphere0]
    ang0 = sim.ang_vel[sphere0]

    lower = sim.sphere_lower_bounds[sphere0]
    upper = sim.sphere_upper_bounds[sphere0]

    query = wp.bvh_query_aabb(sim.scene_sphere_bvh_id, lower, upper)
    sphere1 = int(0)

    while (wp.bvh_query_next(query, sphere1)):
        if sphere1 < sphere0:   # handle each pair only once!

            pos1 = sim.sphere_pos[sphere1]
            radius1 = sim.sphere_radius[sphere1]
            m1 = sim.sphere_mass[sphere1]
            w1 = 1.0 / (m1 + eps)
            vel1 = sim.lin_vel[sphere1]
            ang1 = sim.ang_vel[sphere1]

            min_dist = radius0 + radius1
            pos_normal = wp.normalize(pos1 - pos0)
            dist = wp.dot(pos_normal, pos1 - pos0)

            if dist < min_dist:

                # bounce

                wp.atomic_add(sim.sphere_num_corr, sphere0, 1)
                wp.atomic_add(sim.sphere_num_corr, sphere1, 1)

                pos_corr = pos_normal / (w0 + w1) * (min_dist - dist + eps)
                wp.atomic_add(sim.pos_corr, sphere0, -w0 * pos_corr)
                wp.atomic_add(sim.pos_corr, sphere1, +w1 * pos_corr)

                vn0 = wp.dot(vel0, pos_normal)
                vn1 = wp.dot(vel1, pos_normal)

                new_vn0 = (m0 * vn0 + m1 * vn1 - m1 * (vn0 - vn1) * restitution) / (m0 + m1)
                new_vn1 = (m0 * vn0 + m1 * vn1 - m0 * (vn1 - vn0) * restitution) / (m0 + m1)
                new_vn0 = wp.min(0.0, new_vn0)
                new_vn1 = wp.max(0.0, new_vn1)
                lin_corr0 = pos_normal * (new_vn0 - vn0)
                lin_corr1 = pos_normal * (new_vn1 - vn1)

                wp.atomic_add(sim.sphere_lin_corr, sphere0, lin_corr0)
                wp.atomic_add(sim.sphere_lin_corr, sphere1, lin_corr1)
                vel0 = vel0 + lin_corr0
                vel1 = vel1 + lin_corr1

                # friction

                ang_normal = wp.normalize(ang0 * m0 + ang1 * m1)
                ang_normal = wp.nomralize(ang_normal - pos_normal * wp.dot(pos_normal, ang_normal))

                vt0 = wp.dot(vel0, wp.cross(ang_normal, pos_normal))
                vt1 = wp.dot(vel1, wp.cross(ang_normal, pos_normal))
                omega0 = wp.dot(ang0, ang_normal)
                omega1 = wp.dot(ang1, ang_normal)

                # v0 + (o0 - do*w0) * r0 = v1 + (o1 + do*w1) * r1
 
                domega = (vt1 + omega1 * radius1 - vt0 - omega0 * radius0) / (radius0 * w0 + radius1 * w1)
                ang_corr0 = ang_normal * (omega0 - domega * w0) - ang0
                ang_corr1 = ang_normal * (omega1 + domega * w1) - ang1
                ang0 = ang0 + ang_corr0
                ang1 = ang1 + ang_corr1
                wp.atomic_add(sim.sphere_ang_corr, sphere0, ang_corr0)
                wp.atomic_add(sim.sphere_ang_corr, sphere1, ang_corr1)


@wp.kernel
def dev_handle_sphere_scene_collisions(
        dt: float,
        restitution: float,
        sim: SimData):

    sphere_nr = wp.tid()

    pos = sim.sphere_pos[sphere_nr]
    radius = sim.sphere_radius[sphere_nr]
    m = sim.sphere_mass[sphere_nr]
    vel = sim.lin_vel[sphere_nr]
    ang = sim.ang_vel[sphere_nr]

    inside = float(0.0)
    face_nr = int(0)
    u = float(0.0)
    v = float(0.0)

    found = wp.mesh_query_point(sim.scene_mesh_id, pos, radius, inside, face_nr, u, v)

    if not found:
        return

    id0 = sim.mesh_tri_ids[3 * face_nr]
    id1 = sim.mesh_tri_ids[3 * face_nr + 1]
    id2 = sim.mesh_tri_ids[3 * face_nr + 2]

    p0 = sim.mesh_points[id0]
    p1 = sim.mesh_points[id1]
    p2 = sim.mesh_points[id2]
    closest = u * p0 + v * p1 + (1.0 - u - v) * p2

    pos_normal = wp.normalize(pos - closest)
    dist = wp.dot(pos_normal, pos - closest)

    if dist >= radius:
        return

    sim.sphere_pos[sphere_nr] = pos - pos_normal * (radius - dist)

    v0 = (p0 - sim.mesh_prev_points[id0]) / dt
    v1 = (p1 - sim.mesh_prev_points[id1]) / dt
    v2 = (p2 - sim.mesh_prev_points[id2]) / dt

    v_mesh = v0 + u * (v1 - v0) + v * (v2 - v0)
    v_mesh = u * v0 + v * v1 + (1.0 - u - v) * v2

    vn_sphere = wp.dot(sim.sphere_lin_vel[sphere_nr], pos_normal)
    vn_mesh = wp.dot(v_mesh, pos_normal)
    new_vn = wp.min(vn_mesh - (vn_sphere - vn_mesh) * restitution, 0.0)
    sim.sphere_lin_vel[sphere_nr] = v + pos_normal * (new_vn - vn_sphere)

    # friction

    ang_normal = wp.normalize(ang)
    ang_normal = wp.nomralize(ang - pos_normal * wp.dot(pos_normal, ang_normal))

    vt = wp.dot(vel, wp.cross(ang_normal, pos_normal))
    omega = wp.dot(ang, ang_normal)

    # vel + (omega + do) * r = v_mesh

    domega = (vt + omega * radius - v_mesh) / radius
    ang = ang + ang_normal * (omega - domega) 
    sim.sphere_ang_vel[sphere_nr] = ang

@wp.kernel
def dev_apply_corrections(
        sim: SimData):

    sphere_nr = wp.tid()

    num = sim.sphere_num_corr[sphere_nr]
    if num > 0:
        s = 1.0 / float(num)
        sim.sphere_pos[sphere_nr] += sim.sphere_pos_corr[sphere_nr] * s
        sim.sphere_lin_vel[sphere_nr] += sim.sphere_lin_corr[sphere_nr] * s
        sim.sphere_ang_vel[sphere_nr] += sim.sphere_ang_corr[sphere_nr] * s


class Sim():

    def __init__(self, stage):
        self.stage = stage
        self.device = 'cuda'
        self.controls = None
        self.prim_cache = UsdGeom.XformCache()

        self.dev_sim_data = SimData()
        self.host_sim_data = SimData()
        self.scene_mesh = None
        self.sphere_bvh = None

        self.sphere_usd_meshes = []
        self.sphere_usd_transforms = []
        self.object_usd_meshes = []
        self.object_usd_transforms = []

        self.initialized = False

        self.time_step = 1.0 / 30.0
        self.num_substeps = 5
        self.gravity = wp.vec3(0.0, 0.0, -self.controls.gravity)
        self.restitution = 0.1
        self.jacobi_scale = 0.25
        self.paused = True
        self.num_spheres = 0


    def init_sim(self):

        if not self.stage:
            return

        scene_points = []
        scene_point_obj = []
        scene_tri_indices = []

        sphere_pos = []
        sphere_radius = []
        sphere_inv_mass = []
        self.sphere_usd_meshes = []
        self.sphere_usd_transforms = []

        s = 4.0 / 3.0 * 3.141592

        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                trans_mat, trans_t = usdutils.get_global_transform(prim, 0.0)

                mesh = UsdGeom.Mesh(prim)
                name = mesh.GetName()
                points = mesh.GetPointsAttr().Get(0.0)

                if name.find("sphere") != 0 or name.find("Sphere") != 0:

                    # create a sphere
                    trans_points = points @ trans_mat
                    min = np.min(trans_points, axis = 0)
                    max = np.max(trans_points, axis = 0)
                    radius = np.max(max - min) * 0.5

                    sphere_radius.append(radius)
                    sphere_pos.append(trans_t)
                    mass = s * radius * radius * radius
                    sphere_inv_mass.append(1.0 / mass)

                    clone = usdutils.clone_prim(self.stage, prim)
                    self.sphere_usd_meshes.append(UsdGeom.Mesh(clone))
                    self.sphere_usd_transforms.append(clone.Get)
 
                else:

                    obj_nr = len(self.object_usd_meshes)
                    self.object_usd_meshes.append(mesh)

                    # create obstacle points

                    first_point = len(scene_points)

                    for i in range(len(mesh_points)):
                        p = mesh_points[i]
                        scene_points.append(wp.vec3(*p))
                        scene_point_obj.append(obj_nr)

                    # create obstacle triangles

                    mesh_poly_indices = mesh.GetFaceVertexIndicesAttr().Get(0.0)
                    mesh_face_sizes = mesh.GetFaceVertexCountsAttr().Get(0.0)
                    mesh_points = np.array(points)

                    first_index = 0

                    for i in range(len(mesh_face_sizes)):
                        face_size = mesh_face_sizes[i]
                        for j in range(1, face_size-1):
                            scene_tri_indices.append(first_point + mesh_poly_indices[first_index])
                            scene_tri_indices.append(first_point + mesh_poly_indices[first_index + j])
                            scene_tri_indices.append(first_point + mesh_poly_indices[first_index + j + 1])
                        first_index += face_size

        
        # create scene warp buffers

        if len(scene_points) > 0:

            self.dev_sim_data.mesh_points = wp.array(scene_points, dtype=wp.vec3, device=self.device)
            self.dev_sim_data.mesh_prev_points = wp.array(scene_points, dtype=wp.vec3, device=self.device)
            self.dev_sim_data.mesh_tri_indices = wp.array(scene_tri_indices, dtype=int, device=self.device)
            self.scene_mesh = wp.Mesh(self.dev_sim_data.mesh_points, self.dev_sim_data.mesh_tri_indices)
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

        wp.launch(kernel = dev_integrate, 
            inputs = [self.time_step, self.gravity, self.dev_sim_data],
            dim = self.num_spheres, device=self.device)

        self.sphere_bvh.refit()

        self.dev_sim_data.sphere_pos_corr.zero_()
        self.dev_sim_data.sphere_lin_corr.zero_()
        self.dev_sim_data.sphere_ang_corr.zero_()
        self.dev_sim_data.sphere_num_corr.zero_()

        wp.launch(kernel = dev_handle_sphere_sphere_collisions, 
            inputs = [self.restitution, self.dev_sim_data],
            dim = self.num_spheres, device=self.device)

        wp.launch(kernel = dev_apply_corrections, 
            inputs = [self.dev_sim_data],
            dim = self.num_spheres, device=self.device)

        wp.launch(kernel = dev_handle_sphere_scene_collisions, 
            inputs = [self.time_step, self.restitution, self.dev_sim_data],
            dim = self.num_spheres, device=self.device)


    def update_stage(self):

        wp.copy(self.host_sim_data.sphere_pos, self.dev_sim_data.sphere_pos)
        wp.copy(self.host_sim_data.sphere_quat, self.dev_sim_data.sphere_quat)

        pos = self.host_sim_data.numpy()
        quat = self.host_sim_data.numpy()

        for i in range(self.num_spheres):

            mat = Gf.Matrix4d(Gf.Rotation(Gf.Quatd(*quat[i])), Gf.Vec3d(*pos[i]))
            self.sphere_usd_transforms[i].Set(mat)

 


    def reset(self):

        for sphere in self.spheres:
            sphere.set_position(sphere.pos)

        self.paused = True




