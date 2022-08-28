# Copyright 2022 Matthias Müller - Ten Minute Physics, 
# https://www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import math

from pxr import Usd, UsdGeom, Gf, UsdShade

class Sim():

    def __init__(self, stage):

        self.stage = stage


    def free(self):

        self.stage = None
        self.paused = True


    def on_mouse_button(self, ray_orig, ray_dir, down: bool, button_nr: int):

        pass


    def on_mouse_motion(self, ray_orig: wp.vec3, ray_dir: wp.vec3):

        pass


    def on_key(self, key, down: bool):
        if down:
            if key == b'P' or key == b'p':
                self.paused = not self.paused


    def on_update(self):

        stage = self.stage
        if not stage:
            return

        
        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                attributes = prim.GetAttributes()

                mesh = UsdGeom.Mesh(prim)
                mesh_points = np.array(mesh.GetPointsAttr().Get(0.0))
                mesh_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(0.0))
                mesh_face_sizes = np.array(mesh.GetFaceVertexCountsAttr().Get(0.0))
                mesh_tri_ids = self.engine.triangulate_poly_faces(mesh_indices, mesh_face_sizes)

                spacing = 0.01
                density = 1.0
                compliance = 0.01
                compression_compliance = 0.0
                bending_compliance = 0.0
                torsion_compliance = 0.0
                damping = 1.0
                visible = True
                static_friction_coeff = 0.0
                dynamic_friction_coeff = 0.0
                restitution = 0.0
                collision_groups = 0
                attachment_groups = 0

                for att in attributes:
                    att_name = att.GetName()
                    if att_name.find("spacing") >= 0:
                        spacing = att.Get(0)
                    elif att_name.find("density") >= 0:
                        density = att.Get(0)
                    elif att_name.find("compliance") >= 0:
                        compliance = att.Get(0)
                    elif att_name.find("volCompliance") >= 0 or att_name.find("compressionCompliance") > 0:
                        compression_compliance = att.Get(0)
                    elif att_name.find("bendingCompliance") >= 0:
                        bending_compliance = att.Get(0)
                    elif att_name.find("torsionCompliance") >= 0:
                        torsion_compliance = att.Get(0)
                    elif att_name.find("damping") >= 0:
                        damping = att.Get(0)
                    elif att_name.find("staticFrictionCoefficient") >= 0:
                        static_friction_coeff = att.Get(0)
                    elif att_name.find("dynamicFrictionCoefficient") >= 0:
                        dynamic_friction_coeff = att.Get(0)
                    elif att_name.find("restitution") >= 0:
                        restitution = att.Get(0)
                    elif att_name.find("collisionGroups") >= 0:
                        collision_groups = att.Get(0)
                    elif att_name.find("attachmentGroups") >= 0:
                        attachment_groups = att.Get(0)
                    elif att_name.find("visible") >= 0:
                        visible = att.Get(0)

                    elif att_name.find("numIters") >= 0:
                        self.engine.solver.num_iters = int(att.Get(0))
                    elif att_name.find("numSubsteps") >= 0:
                        self.engine.solver.num_substeps = int(att.Get(0))
                    elif att_name.find("gravity") >= 0:
                        self.engine.solver.gravity = wp.vec3(0.0, att.Get(0), 0.0)

                trans_mat, trans_t = self.get_global_transform(prim, 0.0, True)
                trans_points = mesh_points @ trans_mat + trans_t

                for att in attributes:
                    att_name = att.GetName()
                    if att_name.find("objectType") >= 0:
                        type = att.Get(0)
                        obj_created = False

                        if type.find("dynamic") >= 0 or type.find("static") >= 0 or type.find("plane") >= 0:

                            transform = Gf.Matrix4d()

                            if type.find("dynamic") >= 0:
                                transform = self.prim_cache.GetLocalToWorldTransform(prim)
                                xform = UsdGeom.Xform(mesh)
                                xform.ClearXformOpOrder()
                                xform.AddXformOp(UsdGeom.XformOp.TypeTransform)

                            print("creating rigid body ", prim.GetName())

                            b = np.amax(mesh_points, 0) - np.amin(mesh_points, 0)
                            a0 = trans_mat[0, :]
                            a1 = trans_mat[1, :]
                            a2 = trans_mat[2, :]
                            l0 = np.linalg.norm(a0)
                            l1 = np.linalg.norm(a1)
                            l2 = np.linalg.norm(a2)
                            b[0] *= l0
                            b[1] *= l1
                            b[2] *= l2
                            radius = 0.5 * b[1]
                            height = max([b[0] - 2 * radius, 0.0])

                            A = wp.mat33(
                                a0[0]/l0, a1[0]/l1, a2[0]/l2,
                                a0[1]/l0, a1[1]/l1, a2[1]/l2,
                                a0[2]/l0, a1[2]/l1, a2[2]/l2)
                            
                            pose = wp.transform(trans_t, wp.quat_from_matrix(A))

                            shape_nr = -1
                            is_static = type.find("static") >= 0 or type.find("plane") >= 0

                            mat_nr = self.engine.add_material(static_friction_coeff, dynamic_friction_coeff, restitution,
                                compliance, compression_compliance, bending_compliance, torsion_compliance)

                            if type == "dynamicBox" or type == "staticBox":
                                shape_nr = self.engine.add_box_shape(pose, 
                                    wp.vec3(b[0], b[1], b[2]), mat_nr)

                            elif type == "dynamicSphere" or type == "staticSphere":
                                shape_nr = self.engine.add_sphere_shape(trans_t, radius, collision_groups, attachment_groups)

                            elif type == "dynamicCapsule" or type == "staticCapsule":
                                shape_nr = self.engine.add_capsule_shape(pose, radius, height, mat_nr, collision_groups, attachment_groups)

                            elif type =="plane":
                                n = wp.vec3(0.0, 1.0, 0.0)   # todo, support general planes
                                d = trans_t[1]
                                plane_size = 10.0
                                plane_depth = 0.1
                                shape_nr = self.engine.add_plane_shape(n, d, plane_size, plane_depth, mat_nr, collision_groups , attachment_groups)

                            elif type == "dynamicMesh" or type == "staticMesh":
                                shape_nr = self.engine.add_mesh_shape(trans_points, mesh_tri_ids, mat_nr, collision_groups, attachment_groups)

                            if shape_nr >= 0:
                                body_nr = self.engine.add_body(prim.GetName(), shape_nr, density, is_static)
                                obj_created = True

                                self.update_obj_nr.append(body_nr)
                                if is_static:
                                    inv_mat = np.linalg.inv(trans_mat)
                                    self._add_object("staticBody", body_nr, prim, None, inv_mat, -inv_mat @ trans_t)
                                else:
                                    self.update_obj_type.append("dynamicBody")
                                    if visible:
                                        out_prim = self._clone_prim(prim)
                                        
                                        self._add_object("dynamicBody", body_nr, None, out_prim)
                                        prim.SetActive(False)

                        elif type == "softBody":
                            
                            thickness = 0.05
                            mesh_nr = self.engine.add_soft_body(trans_points, mesh_tri_ids, thickness, spacing, density, compliance, compression_compliance, 0)

                            if visible:
                                out_prim = self._clone_prim(prim)
                                self._add_object("softBody", mesh_nr, None, out_prim)
                                prim.SetActive(False)

                        elif type == "animMesh":

                            anim_nr = self.engine.add_anim_mesh(trans_points, mesh_tri_ids)

                            if not visible:
                                vis = prim.GetAttribute("visibility")
                                if vis:
                                    vis.Set("invisible")
                                self._add_object("animMesh", mesh_nr, prim, None)
                            
                        elif type == "visMesh":
                            mesh_nr = self.engine.add_vis_mesh(trans_points, mesh_tri_ids)

                            out_prim = self._clone_prim(prim)
                            self._add_object("visMesh", mesh_nr, prim, out_prim)

                            vis = prim.GetAttribute("visibility")
                            if vis:
                                vis.Set("invisible")

        print("finalizing...")
        with wp.ScopedDevice("cuda:0"):
            self.engine.finalize()    
        print("finalizing done")


    def update_pre_solve(self):

        if self.engine is None:
            return

        if self.anim_running:
            self.usd_time += self.engine.solver.time_step * self.stage.GetTimeCodesPerSecond()

        # update vertices of animated and visual meshes

        for i in range(len(self.update_obj_nr)):

            # find points of anim mesh

            in_prim = self.update_obj_in_prim[i]
            if not in_prim:
                continue

            type = self.update_obj_type[i]

            if type == "animMesh" or type == "visMesh":

                mesh = UsdGeom.Mesh(in_prim)
                points = mesh.GetPointsAttr()
                num = points.GetNumTimeSamples()
                if num == 0:
                    continue

                # loop animation

                duration = points.GetTimeSamples()[-1]
                periods = wp.floor(self.usd_time / duration)
                mesh_time = self.usd_time - periods * duration

                if mesh_time < self.start_anim_time:
                    mesh_time = self.start_anim_time

                # transform mesh from usd to sim space

                trans_mat, trans_t = self.get_global_transform(in_prim, mesh_time, True)
                trans_points = points.Get(mesh_time) @ trans_mat + trans_t

                # send update to solver

                if self.update_obj_type[i] == "anim":
                    self.engine.update_anim_mesh(self.update_obj_nr[i], trans_points)

                elif self.update_obj_type[i] == "vis":
                    self.engine.update_vis_mesh(self.update_obj_nr[i], trans_points)

            elif type == "staticBody":

                trans_mat, trans_t = self.get_global_transform(in_prim, mesh_time, True)
                rel_mat = trans_mat @ self.update_inv_trans_mat[i]
                rel_t = trans_mat @ self.update_inv_trans_t[i] + trans_t

                wp_mat = wp.mat33(
                    rel_mat[0,0], rel_mat[0,1], rel_mat[0,2],
                    rel_mat[1,0], rel_mat[1,1], rel_mat[1,2],
                    rel_mat[2,0], rel_mat[2,1], rel_mat[2,2])
                wp_t = wp.vec3(rel_t[0], rel_t[1], rel_t[2])

                self.engine.set_relative_body_pose(self.update_obj_nr, wp.transform(wp_t, wp.quat_from_matrix(wp_mat)))


    def update_post_solve(self):

        if self.engine is None:
            return

        self.prim_cache.SetTime(self.usd_time)

        # read back visual objets

        for i in range(len(self.update_obj_nr)):

            out_prim = self.update_obj_out_prim[i]
            if not out_prim:
                continue

            mesh = UsdGeom.Mesh(out_prim)
            type = self.update_obj_type[i]

            if type == "visMesh":
                
                verts = self.engine.get_vis_mesh_verts(self.update_obj_nrs[i])
                trans_verts = verts @ np.linalg.inv(self.stage_trans_mat)
                mesh.SetPointsAttr().Set(trans_verts)

            elif type == "softBody":

                verts = self.engine.get_soft_body_verts(self.update_obj_nrs[i])
                trans_verts = verts @ np.linalg.inv(self.stage_trans_mat)
                mesh.SetPointsAttr().Set(trans_verts)

            elif type == "dynamicBody":

                body_nr = self.update_obj_nr[i]

                if self.first_update:
                    verts = self.engine.get_body_verts(body_nr)
                    trans_verts = verts @ np.linalg.inv(self.stage_trans_mat)
                    mesh.SetPointsAttr().Set(trans_verts)
                    
                pose = self.engine.get_body_pose(body_nr)
                t = wp.mul(wp.transform_get_translation(pose), 1.0/self.stage_scale)
                q = wp.transform_get_rotation(pose)
                c0 = wp.quat_rotate(wp.vec3(1.0, 0.0, 0.0))
                c1 = wp.quat_rotate(wp.vec3(0.0, 1.0, 0.0))
                c2 = wp.quat_rotate(wp.vec3(0.0, 0.0, 1.0))

                usd_mat = None
                if UsdGeom.GetStageUpAxis(self.stage) == 'Z':
                    usd_mat = Gf.Matrix4d(
                        c0[0], -c2[0], c1[0], 0.0,
                        c0[1], -c2[1], c1[1], 0.0,
                        c0[2], -c2[2], c1[2], 0.0,
                        t[0], -t[2], t[1], 1.0)
                else:
                    usd_mat = Gf.Matrix4d(
                        c0[0], c0[1], c0[2], 0.0,
                        c1[0], c1[1], c1[2], 0.0,
                        c2[0], c2[1], c2[2], 0.0,
                        t[0], t[1], t[2], 1.0)

                xform = UsdGeom.Xform(mesh)
                xform.GetOrderedXformOps()[0].Set(usd_mat)


        self.first_update = False


