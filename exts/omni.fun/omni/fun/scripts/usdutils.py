from pxr import Usd, UsdGeom, Gf, UsdShade
import numpy as np
import warp as wp

prim_cache = None


def get_global_transform(prim, time, return_mat44):

    if prim_cache is None:
        prim_cache = UsdGeom.XformCache()
    prim_cache.SetTime(time)

    m = prim_cache.GetLocalToWorldTransform(prim)
    if return_mat44:
        return wp.mat44(
            m[0][0], m[1][0], m[2][0], m[3][0],
            m[0][1], m[1][1], m[2][1], m[3][1],
            m[0][2], m[1][2], m[2][2], m[3][2],
            m[0][3], m[1][3], m[2][3], m[3][3])
    else:
        A = np.array([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]) 
        b = np.array([m[3][0], m[3][1], m[3][2]])
        return A, b


def set_transform(mesh, trans, quat):

    usd_mat = Gf.Matrix4d()
    usd_mat.SetRotateOnly(Gf.Quatd(*quat))
    usd_mat.SetTranslateOnly(Gf.Vec3d(*trans))

    xform = UsdGeom.Xform(mesh)
    xform.GetOrderedXformOps()[0].Set(usd_mat)


def clone_primvar(self, prim, prim_clone, name, time=0.0):

    try:
        attr = UsdGeom.Primvar(prim.GetAttribute(name))
        prim_clone.CreatePrimvar(name, attr.GetTypeName(), attr.GetInterpolation()).Set(attr.Get(time))
    except:
        pass


def clone_prim(stage, prim):

    vis = prim.GetAttribute("visibility")
    if vis:
        vis.Set("invisible")

    mesh = UsdGeom.Mesh(prim)
    clone_prim_path = '/' + str(prim.GetPath()).replace("/", "_") + '_clone'

    UsdGeom.Mesh.Define(stage, clone_prim_path)
    
    prim_clone = UsdGeom.Mesh(stage.GetPrimAtPath(clone_prim_path))
    mesh_clone = UsdGeom.Mesh(prim_clone)
    stage.GetPrimAtPath(clone_prim_path).SetActive(True) 

    xform = UsdGeom.Xform(mesh_clone)
    xform.ClearXformOpOrder()
    xform.AddXformOp(UsdGeom.XformOp.TypeTransform)

    trans_mat, trans_t = get_global_transform(prim, 0.0, True)
    trans_points = mesh.GetPointsAttr().Get(0.0) @ trans_mat + trans_t
    
    normal_mat = np.array([\
        trans_mat[0,:] / np.linalg.norm(trans_mat[0,:]), \
        trans_mat[1,:] / np.linalg.norm(trans_mat[1,:]), \
        trans_mat[2,:] / np.linalg.norm(trans_mat[2,:])])
    trans_normals = mesh.GetNormalsAttr().Get(0.0) @ normal_mat

    mesh_clone.GetPointsAttr().Set(trans_points)
    mesh_clone.GetNormalsAttr().Set(trans_normals)
    mesh_clone.SetNormalsInterpolation(mesh.GetNormalsInterpolation())
    mesh_clone.GetFaceVertexIndicesAttr().Set(mesh.GetFaceVertexIndicesAttr().Get(0.0))
    mesh_clone.GetFaceVertexCountsAttr().Set(mesh.GetFaceVertexCountsAttr().Get(0.0))

    mesh_clone.GetCornerIndicesAttr().Set(mesh.GetCornerIndicesAttr().Get(0.0))
    mesh_clone.GetCornerSharpnessesAttr().Set(mesh.GetCornerSharpnessesAttr().Get(0.0))
    mesh_clone.GetCreaseIndicesAttr().Set(mesh.GetCreaseIndicesAttr().Get(0.0))
    mesh_clone.GetCreaseLengthsAttr().Set(mesh.GetCreaseLengthsAttr().Get(0.0))
    mesh_clone.GetCreaseSharpnessesAttr().Set(mesh.GetCreaseSharpnessesAttr().Get(0.0))
    mesh_clone.GetSubdivisionSchemeAttr().Set(mesh.GetSubdivisionSchemeAttr().Get(0.0))
    mesh_clone.GetInterpolateBoundaryAttr().Set(mesh.GetInterpolateBoundaryAttr().Get(0.0))
    mesh_clone.GetFaceVaryingLinearInterpolationAttr().Set(mesh.GetFaceVaryingLinearInterpolationAttr().Get(0.0))
    mesh_clone.GetTriangleSubdivisionRuleAttr().Set(mesh.GetTriangleSubdivisionRuleAttr().Get(0.0))
    mesh_clone.GetHoleIndicesAttr().Set(mesh.GetHoleIndicesAttr().Get(0.0))

    for attr in prim.GetAttributes():
        type = str(attr.GetTypeName())
        if type.find("texCoord") >= 0:
            clone_primvar(prim, prim_clone, attr.GetName())

    try:
        mat = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
        UsdShade.MaterialBindingAPI(prim_clone).Bind(mat)
    except:
        pass

    return prim_clone
    

def hide_clones(stage):
    if stage is None:
        return
    
    for prim in stage.Traverse():
        if str(prim.GetName()).find("_clone") >= 0:
            prim.SetActive(False)
        else:
            vis = prim.GetAttribute("visibility")
            if vis:
                vis.Set("inherited")
