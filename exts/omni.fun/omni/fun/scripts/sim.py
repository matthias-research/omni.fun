# Copyright 2022 Matthias Müller - Ten Minute Physics, 
# https://www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import math
import warp as wp

from pxr import Usd, UsdGeom, Gf, UsdShade


class Sphere():

    def __init__(self, prim, radius, transform):

        self.prim = prim
        self.radius = radius
        self.transform = transform


    def update_stage(self):

        p = wp.transform_get_translation(self.transform)
        q = wp.transform_get_rotation(self.transform)

        xform_ops = self.prim.GetOrderedXformOps()
        xform_ops[1].Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])), 0.0)
        xform_ops[0].Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])), 0.0)

    
def _clone_primvar(self, prim, prim_clone, name, time=0.0):

    try:
        attr = UsdGeom.Primvar(prim.GetAttribute(name))
        prim_clone.CreatePrimvar(name, attr.GetTypeName(), attr.GetInterpolation()).Set(attr.Get(time))
    
    except:
        pass


class Sim():

    def __init__(self, stage):

        self.stage = stage
        self.up = wp.vec3(0.0, 1.0, 0.0)
        self.gravity = wp.vec3(0.0, -10.0, 0.0)

        if UsdGeom.GetStageUpAxis(stage) == 'Z':
            self.up = wp.vec3(0.0, 0.0, 1.0)
            self.gravity = wp.vec3(0.0, 0.0, -10.0)

        self.gravity = wp.mul(self.gravity, 1.0 / UsdGeom.GetStageMetersPerUnit(stage))

        self.spheres = []

        prim_cache = UsdGeom.XformCache()

        for prim in self.stage.Traverse():

            if prim.GetTypeName() == "Mesh":

                mesh = UsdGeom.Mesh(prim)
                name = prim.GetName()

                if name.find("sphere") >= 0:

                    # read existing prim

                    m = prim_cache.GetLocalToWorldTransform(prim)
                    trans = np.array([m[3,0], m[3,1], m[3,2]])

                    points = np.array(mesh.GetPointsAttr().Get(0.0))
                    center = np.average(points, axis = 0)
                    trans = trans + center
                    points = points - center
                    radius = np.max(np.linalg.norm(points, axis=1))

                    prim.SetActive(False)

                    # create new prim

                    new_prim_path = '/' + name + '_fun'
                    UsdGeom.Mesh.Define(self.stage, new_prim_path)
                    new_prim = UsdGeom.Mesh(self.stage.GetPrimAtPath(new_prim_path))
                    new_mesh = UsdGeom.Mesh(new_prim)

                    xform = UsdGeom.Xform(new_mesh)
                    xform.ClearXformOpOrder()
                    xform.AddXformOp(UsdGeom.XformOp.TypeOrient)
                    xform.AddXformOp(UsdGeom.XformOp.TypeTranslate)

                    new_mesh.GetPointsAttr().Set(points)
                    new_mesh.GetFaceVertexIndicesAttr().Set(mesh.GetFaceVertexIndicesAttr().Get(0.0))
                    new_mesh.GetFaceVertexCountsAttr().Set(mesh.GetFaceVertexCountsAttr().Get(0.0))

                    _clone_primvar(prim, new_prim, "primvars:UVMap")    
                    _clone_primvar(prim, new_prim, "primvars:Texture")

                    try:
                        mat = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
                        UsdShade.MaterialBindingAPI(new_prim).Bind(mat)
                    except:
                        pass

                    p = wp.vec3(trans[0], trans[1], trans[2])
                    q = wp.quat(0.0, 0.0, 0.0, 1.0)

                    sphere = Sphere(new_prim, radius, wp.transform(p, q))
                    sphere.update_stage()
                    self.spheres.append(sphere)


    def on_update(self):

        stage = self.stage
        if not stage:
            return

        for sphere in self.spheres:
            pass

