# Copyright 2022 Matthias Müller - Ten Minute Physics, 
# https://www.youtube.com/c/TenMinutePhysics
# www.matthiasMueller.info/tenMinutePhysics
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import warp as wp


@wp.struct
class SimData:
    spheres_pos: wp.array(dtype=wp.vec3)
    spheres_prev_pos: wp.array(dtype=wp.vec3)
    spheres_pos_corr: wp.array(dtype=wp.vec3)
    spheres_vel: wp.array(dtype=wp.vec3)
    spheres_radius: wp.array(dtype=float)
    spheres_inv_mass: wp.array(dtype=float)
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
def dev_integrate_spheres(
    dt: float,
    gravity: wp.vec3,
    data: SimData):

    sphere_nr = wp.tid()
    w = data.spheres_inv_mass[sphere_nr]
    if w > 0.0:
        data.spheres_vel[sphere_nr] += gravity * dt
        data.spheres_prev_pos[sphere_nr] = data.spheres_pos[sphere_nr]
        data.spheres_pos[sphere_nr] += data.spheres_vel[sphere_nr] * dt


def integrate_spheres(num_spheres: int, dt: float, gravity: wp.vec3, data: SimData, device):
    wp.launch(kernel = dev_integrate_spheres, 
                inputs = [dt, gravity, data], dim=num_spheres, device=device)


@wp.kernel
def dev_update_spheres(
    dt: float,
    jacobi_scale: float, 
    data: SimData):

    sphere_nr = wp.tid()
    w = data.spheres_inv_mass[sphere_nr]
    if w > 0.0:
        data.spheres_pos[sphere_nr] = data.spheres_pos[sphere_nr] + jacobi_scale * data.spheres_pos_corr
        data.spheres_vel[sphere_nr] = (data.spheres_pos[sphere_nr] - data.spheres_prev_pos[sphere_nr]) / dt


def update_spheres(num_spheres: int, dt: float, jacobi_scale: float, data: SimData, device):
    wp.launch(kernel = dev_update_spheres, 
                inputs = [dt, jacobi_scale, data], dim=num_spheres, device=device)


@wp.kernel
def dev_solve_mesh_collisions(
        data: SimData):
    
    sphere_nr = wp.tid()
    w = data.spheres_inv_mass[sphere_nr]
    if w > 0.0:
        pos = data.spheres_pos[sphere_nr]
        r = data.spheres_radius[sphere_nr]

        # query bounding volume hierarchy

        bounds_lower = pos - wp.vec3(r, r, r)
        bounds_upper = pos + wp.vec3(r, r, r)

        query = wp.mesh_query_aabb(data.mesh_id, bounds_lower, bounds_upper)
        tri_nr = int(0)

        while (wp.mesh_query_aabb_next(query, tri_nr)):
                
            p0 = data.mesh_verts[data.mesh_tri_ids[3*tri_nr]]
            p1 = data.mesh_verts[data.mesh_tri_ids[3*tri_nr + 1]]
            p2 = data.mesh_verts[data.mesh_tri_ids[3*tri_nr + 2]]

            hit = closest_point_on_triangle(pos, p0, p1, p2)

            n = pos - hit                
            d = wp.length(n)
            if d < r:
                n = wp.normalize(n)
                data.spheres_pos[sphere_nr] = data.spheres_pos[sphere_nr] + n * (r - d)

            
def solve_mesh_collisions(num_spheres: int, data: SimData, device):
    wp.launch(kernel = dev_solve_mesh_collisions, 
                inputs = [data], dim=num_spheres, device=device)

            
@wp.kernel
def dev_solve_sphere_collisions(
        num_spheres: int,
        data: SimData):
    
    i0 = wp.tid()
    p0 = data.spheres_pos[i0]
    r0 = data.spheres_radius[i0]
    w0 = data.spheres_inv_mass[i0]

    # simpe O(n^2) collision detection

    for i1 in range(num_spheres):
        if i1 > i0:

            p1 = data.spheres_pos[i1]
            r1 = data.spheres_radius[i1]
            w1 = data.spheres_inv_mass[i1]
            w = w0 + w1

            if w > 0.0:
                n = p1 - p0
                d = wp.length(n)
                n = wp.noramlize(n)

                if d < r0 + r1:
                    corr = n * (r0 + r1 - d) / w
                    data.spheres_corr[i0] = data.spheres_corr[i0] - corr * w0
                    data.spheres_corr[i1] = data.spheres_corr[i1] - corr * w0

            
def solve_sphere_collisions(num_spheres: int, data: SimData, device):
    wp.launch(kernel = dev_solve_sphere_collisions, 
                inputs = [num_spheres, data], dim=num_spheres, device=device)

            

        