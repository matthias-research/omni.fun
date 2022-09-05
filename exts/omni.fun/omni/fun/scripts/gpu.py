# Copyright 2022 Matthias Müller - Ten Minute Physics, 
# https://www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import warp as wp


@wp.struct
class GpuData:
    sphere_pos: wp.array(dtype=wp.vec3)
    sphere_vel: wp.array(dtype=wp.vec3)
    sphere_radius: wp.array(dtype=float)
    sphere_mass: wp.array(dtype=float)
    mesh_id: wp.uint64
    mesh_verts: wp.array(dtype=wp.vec3)
    mesh_tri_ids: wp.array(dtype=int)


@wp.func
def closest_point_on_triangle(
        p: wp.vec3, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3):
    
    e0  = p1 - p0
    e1  = p2 - p0
    tmp = p0 - p

    a = wp.dot(e0, e0)
    b = wp.dot(e0, e1)
    c = wp.dot(e1, e1)
    d = wp.dot(e0, tmp)
    e = wp.dot(e1, tmp)
    coords = wp.vec3(b*e - c*d, b*d - a*e, a*c - b*b)
    
    x = 0.0
    y = 0.0
    z = 0.0

    if coords[0] <= 0.0:
        if c != 0.0:
            y = -e / c
    elif coords[1] <= 0.0:
        if a != 0.0:
            x = -d / a
    elif coords[0] + coords[1] > coords[2]:
        den = a + c - b - b
        num = c + e - b - d
        if den != 0.0:
            x = num / den
            y = 1.0 - x
    else:
        if coords[2] != 0.0:
            x = coords[0] / coords[2]
            y = coords[1] / coords[2]
        
    x = wp.clamp(x, 0.0, 1.0)
    y = wp.clamp(y, 0.0, 1.0)
    
    bary = wp.vec3(1.0 - x - y, x, y)

    return bary


@wp.kernel
def solve_collisions(
        dt: float,
        num_spheres: int,
        data: GpuData):
    
    sphere_nr = wp.tid()
    pos = data.sphere_pos[sphere_nr]
    r = data.sphere_radius[sphere_nr]

    # bounds for overlap test with bounding volume hierarchy

    bounds_lower = pos - wp.vec3(r, r, r)
    bounds_upper = pos + wp.vec3(r, r, r)

    # query

    query = wp.mesh_query_aabb(data.mesh_id, bounds_lower, bounds_upper)
    tri_nr = int(0)

    while (wp.mesh_query_aabb_next(query, tri_nr)):

        if tri_nr < num_spheres:
            # sphere - sphere collision
            pass

        else:
            # sphere - triangle collision
            tri_nr = tri_nr - num_spheres
                
            p0 = data.mesh_verts[data.mesh_tri_ids[3*tri_nr]]
            p1 = data.mesh_verts[data.mesh_tri_ids[3*tri_nr + 1]]
            p2 = data.mesh_verts[data.mesh_tri_ids[3*tri_nr + 2]]

            hit = closest_point_on_triangle(pos, p0, p1, p2)

            

        


