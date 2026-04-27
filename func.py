### Version 1.1

import numpy as np
from numpy import sin, cos, sqrt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifftshift, fftfreq
from mpmath import mp, mpf, matrix
import json
mp.dps = 20
calc_mpmath = False
def ellipse_ab_inc(a, b, inc):
    q_plus_p = 2*a
    q_times_p = b**2 / (np.sin(inc)**2)
    p = (q_plus_p + np.sqrt(q_plus_p**2 - 4*q_times_p)) / 2
    q = q_plus_p - p
    # print("p =", p)
    # print("q =", q)
    return p, q
def hyperbola_ab_inc(a, b, inc):
    p_minus_q = 2*a
    p_times_q = b**2 / (np.sin(inc)**2)
    p_plus_q = np.sqrt(p_minus_q**2 + 4 * p_times_q)
    p = (p_plus_q + p_minus_q) / 2
    q = (p_plus_q - p_minus_q) / 2
    # print("p =", p)
    # print("q =", q)
    return p, q
def ellipse_pq_inc(p,q,inc):
    a = (p + q) / 2
    b = np.sqrt(p * q) * np.sin(inc)
    return a, b
def hyperbola_pq_inc(p,q,inc):
    if p < q:
        a = (q - p) / 2
    else:
        a = (p - q) / 2
    b = np.sqrt(p * q) * np.sin(inc)
    return a, b

def calc_theta1_ell(p, q, inc):
    theta_1 = np.arctan(q*np.sin(inc*2)/(p + q*np.cos(inc*2)))
    return theta_1

def calc_theta3_hyp(p, q, inc):
    theta_3 = np.arctan(p*np.sin(inc*2)/(p - q*np.cos(inc*2)))
    return theta_3

def Yvalue_hyperbola(a, b, x0, x):
    return b * np.sqrt(((x - x0) / a) ** 2 - 1)
def Yvalue_ellipse(a, b, x0, x):
    return b * np.sqrt(1 - ((x - x0) / a) ** 2)

def Yvalue_hyperbola_3D(a, b, x0, x, z):
    r2 = b**2 * (((x - x0) / a) ** 2 - 1)
    return np.sqrt(r2 - z**2)

def Yvalue_ellipse_3D(a, b, x0, x, z):
    r2 = b**2 * (1 - ((x - x0) / a) ** 2)
    return np.sqrt(r2 - z**2)

def calc_ell_theta(a,b,theta3,x0):
    A = (cos(theta3)**2)/a**2 + (sin(theta3)**2)/b**2
    B = -2*x0*cos(theta3)/a**2
    C = (x0**2)/a**2 - 1
    D = B**2 - 4*A*C
    if D.any() < 0:
        print("Error in calc_ell_theta: D<0")
        return None
    x1 = (-B + sqrt(D)) / (2*A)
    x2 = (-B - sqrt(D)) / (2*A)
    return x1, x2

def calc_hyp_theta(a,b,theta3,x0):
    A = (cos(theta3)**2)/a**2 - (sin(theta3)**2)/b**2
    B = -2*x0*cos(theta3)/a**2
    C = (x0**2)/a**2 - 1
    D = B**2 - 4*A*C
    if D.any() < 0:
        print("Error in calc_hyp_theta: D<0")
        return None
    x1 = (-B + sqrt(D)) / (2*A)
    x2 = (-B - sqrt(D)) / (2*A)
    return x1, x2

def rotation_2D(x, y, angle_rad):
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot

# def calc_intersections_coeffs(points, rays, c, positive=True):
#     if points.ndim == 1:
#         points = points.reshape(1, -1)
#     if points.shape[0] == 3:
#         # print("calc 3D normal vector coefficients")
#         x0 = points[0, :]
#         y0 = points[1, :]
#         z0 = points[2, :]
#         vx = rays[0, :]
#         vy = rays[1, :]
#         vz = rays[2, :]

#         A = c[0]*vx**2 + c[1]*vy**2 + c[2]*vz**2 + c[3]*vz*vy + c[4]*vz*vx + c[5]*vx*vy
#         B = 2*c[0]*x0*vx + 2*c[1]*y0*vy + 2*c[2]*z0*vz + c[3]*(z0*vy + y0*vz) + c[4]*(z0*vx + x0*vz) + c[5]*(x0*vy + y0*vx) + c[6]*vx + c[7]*vy + c[8]*vz
#         C = c[0]*x0**2 + c[1]*y0**2 + c[2]*z0**2 + c[3]*z0*y0 + c[4]*z0*x0 + c[5]*x0*y0 + c[6]*x0 + c[7]*y0 + c[8]*z0 + c[9]
#         D = B**2 - 4*A*C
#         if D.any() < 0:
#             print("No real intersection points.")
#             return None

#         if positive:
#             k = (-B + np.sqrt(D)) / (2*A)
#         else:
#             k = (-B - np.sqrt(D)) / (2*A)
#         intersections = points + k * rays 
#         return intersections
#     elif points.shape[0] == 2:
#         # print("calc 2D normal vector coefficients")
#         x0 = points[0, :]
#         y0 = points[1, :]
#         vx = rays[0, :]
#         vy = rays[1, :]
#         A = c[0]*vx**2 + c[1]*vy**2 + c[5]*vx*vy
#         B = 2*c[0]*x0*vx + 2*c[1]*y0*vy + c[5]*(x0*vy + y0*vx) + c[6]*vx + c[7]*vy
#         C = c[0]*x0**2 + c[1]*y0**2 + c[5]*x0*y0 + c[6]*x0 + c[7]*y0 + c[9]
#         D = B**2 - 4*A*C
#         if D.any() < 0:
#             print("No real intersection points.")
#             return None
#         if positive:
#             k = (-B + np.sqrt(D)) / (2*A) 
#         else:
#             k = (-B - np.sqrt(D)) / (2*A)
#         intersections = points + k * rays
#         return intersections
#     else:
#         raise ValueError("Input points must be 2D or 3D.")

def calc_intersections_coeffs(points, rays, c, positive=True, calc_mpmath=calc_mpmath):
    if calc_mpmath:
        print("Calculating intersections using mpmath for high precision...")
        if points.ndim == 1:
            points = points.reshape(-1, 1)
        if rays.ndim == 1:
            rays = rays.reshape(-1, 1)
        
        dim, num_pts = points.shape
        # 係数 c を mpmath オブジェクトに変換
        c_mp = [mp.mpf(val) for val in c]
        
        # 出力用の配列 (mpmathオブジェクトを一時保持するため dtype=object)
        intersections = np.empty((dim, num_pts), dtype=object)

        for i in range(num_pts):
            # 各成分を mpmath.mpf に変換
            p0 = [mp.mpf(points[j, i]) for j in range(dim)]
            v = [mp.mpf(rays[j, i]) for j in range(dim)]
            
            if dim == 3:
                x0, y0, z0 = p0
                vx, vy, vz = v
                # 3D 係数計算
                A = c_mp[0]*vx**2 + c_mp[1]*vy**2 + c_mp[2]*vz**2 + \
                    c_mp[3]*vz*vy + c_mp[4]*vz*vx + c_mp[5]*vx*vy
                B = 2*c_mp[0]*x0*vx + 2*c_mp[1]*y0*vy + 2*c_mp[2]*z0*vz + \
                    c_mp[3]*(z0*vy + y0*vz) + c_mp[4]*(z0*vx + x0*vz) + \
                    c_mp[5]*(x0*vy + y0*vx) + c_mp[6]*vx + c_mp[7]*vy + c_mp[8]*vz
                C = c_mp[0]*x0**2 + c_mp[1]*y0**2 + c_mp[2]*z0**2 + \
                    c_mp[3]*z0*y0 + c_mp[4]*z0*x0 + c_mp[5]*x0*y0 + \
                    c_mp[6]*x0 + c_mp[7]*y0 + c_mp[8]*z0 + c_mp[9]
            else: # dim == 2
                x0, y0 = p0
                vx, vy = v
                # 2D 係数計算
                A = c_mp[0]*vx**2 + c_mp[1]*vy**2 + c_mp[5]*vx*vy
                B = 2*c_mp[0]*x0*vx + 2*c_mp[1]*y0*vy + \
                    c_mp[5]*(x0*vy + y0*vx) + c_mp[6]*vx + c_mp[7]*vy
                C = c_mp[0]*x0**2 + c_mp[1]*y0**2 + c_mp[5]*x0*y0 + \
                    c_mp[6]*x0 + c_mp[7]*y0 + c_mp[9]

            # 判別式
            D = B**2 - 4*A*C
            if D < 0:
                print("No real intersection points.")
                return None

            # 解の公式
            if positive:
                k = (-B + mp.sqrt(D)) / (2*A)
            else:
                k = (-B - mp.sqrt(D)) / (2*A)
            
            # 交点座標の算出
            for j in range(dim):
                intersections[j, i] = p0[j] + k * v[j]
        
        # 最終的に NumPy の float64 配列として返す
        return intersections.astype(np.float64)
 
    else:
        # (既存の else ブロックの処理)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[0] == 3:
            x0 = points[0, :]
            y0 = points[1, :]
            z0 = points[2, :]
            vx = rays[0, :]
            vy = rays[1, :]
            vz = rays[2, :]

            A = c[0]*vx**2 + c[1]*vy**2 + c[2]*vz**2 + c[3]*vz*vy + c[4]*vz*vx + c[5]*vx*vy
            B = 2*c[0]*x0*vx + 2*c[1]*y0*vy + 2*c[2]*z0*vz + c[3]*(z0*vy + y0*vz) + c[4]*(z0*vx + x0*vz) + c[5]*(x0*vy + y0*vx) + c[6]*vx + c[7]*vy + c[8]*vz
            C = c[0]*x0**2 + c[1]*y0**2 + c[2]*z0**2 + c[3]*z0*y0 + c[4]*z0*x0 + c[5]*x0*y0 + c[6]*x0 + c[7]*y0 + c[8]*z0 + c[9]
            D = B**2 - 4*A*C
            if np.any(D < 0):
                print("No real intersection points.")
                return None

            if positive:
                k = (-B + np.sqrt(D)) / (2*A)
            else:
                k = (-B - np.sqrt(D)) / (2*A)
            intersections = points + k * rays 
            return intersections
        elif points.shape[0] == 2:
            x0 = points[0, :]
            y0 = points[1, :]
            vx = rays[0, :]
            vy = rays[1, :]
            A = c[0]*vx**2 + c[1]*vy**2 + c[5]*vx*vy
            B = 2*c[0]*x0*vx + 2*c[1]*y0*vy + c[5]*(x0*vy + y0*vx) + c[6]*vx + c[7]*vy
            C = c[0]*x0**2 + c[1]*y0**2 + c[5]*x0*y0 + c[6]*x0 + c[7]*y0 + c[9]
            D = B**2 - 4*A*C
            if np.any(D < 0):
                print("No real intersection points.")
                return None
            if positive:
                k = (-B + np.sqrt(D)) / (2*A) 
            else:
                k = (-B - np.sqrt(D)) / (2*A)
            intersections = points + k * rays
            return intersections
        else:
            raise ValueError("Input points must be 2D or 3D.")

# def calc_norm_vector_coeffs(points,c, calc_mpmath=True):
#     ### points: (N, 3), coeffs: (10)
#     if points.ndim == 1:
#         points = points.reshape(1, -1)
#     if points.shape[0] == 3:
#         # print("calc 3D normal vector coefficients")
#         x = points[0, :]
#         y = points[1, :]
#         z = points[2, :]
#         nx = 2*c[0]*x + c[4]*z + c[5]*y + c[6]
#         ny = 2*c[1]*y + c[3]*z + c[5]*x + c[7]
#         nz = 2*c[2]*z + c[3]*y + c[4]*x + c[8]
#         n_vec = np.vstack((nx, ny, nz))
#         n_vec = n_vec / np.linalg.norm(n_vec, axis=0)  # 正規化
#         return n_vec
#     elif points.shape[0] == 2:
#         # print("calc 2D normal vector coefficients")
#         x = points[0, :]
#         y = points[1, :]
#         nx = 2*c[0]*x + c[5]*y + c[6]
#         ny = 2*c[1]*y + c[5]*x + c[7]
#         n_vec = np.vstack((nx, ny))
#         n_vec = n_vec / np.linalg.norm(n_vec, axis=0)  # 正規化
#         return n_vec
#     else:
#         raise ValueError("Input points must be 2D or 3D.")

def calc_norm_vector_coeffs(points, c, calc_mpmath=calc_mpmath):
    if calc_mpmath:
        # mpmathの精度設定（必要に応じて設定）
        # mp.mp.dps = 50 

        if points.ndim == 1:
            points = points.reshape(-1, 1)
        
        dim, num_pts = points.shape
        c_mp = [mp.mpf(val) for val in c]
        
        # 出力用の配列 (mpmathオブジェクトを保持するため dtype=object)
        n_vec_mp = np.empty((dim, num_pts), dtype=object)

        for i in range(num_pts):
            if dim == 3:
                x = mp.mpf(points[0, i])
                y = mp.mpf(points[1, i])
                z = mp.mpf(points[2, i])
                
                nx = 2*c_mp[0]*x + c_mp[4]*z + c_mp[5]*y + c_mp[6]
                ny = 2*c_mp[1]*y + c_mp[3]*z + c_mp[5]*x + c_mp[7]
                nz = 2*c_mp[2]*z + c_mp[3]*y + c_mp[4]*x + c_mp[8]
                
                # 正規化
                norm = mp.sqrt(nx**2 + ny**2 + nz**2)
                n_vec_mp[0, i] = nx / norm
                n_vec_mp[1, i] = ny / norm
                n_vec_mp[2, i] = nz / norm

            elif dim == 2:
                x = mp.mpf(points[0, i])
                y = mp.mpf(points[1, i])
                
                nx = 2*c_mp[0]*x + c_mp[5]*y + c_mp[6]
                ny = 2*c_mp[1]*y + c_mp[5]*x + c_mp[7]
                
                # 正規化
                norm = mp.sqrt(nx**2 + ny**2)
                n_vec_mp[0, i] = nx / norm
                n_vec_mp[1, i] = ny / norm

        # NumPyのfloat64行列に戻して返す
        return n_vec_mp.astype(np.float64)

    else:
        ### points: (N, 3), coeffs: (10)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[0] == 3:
            # print("calc 3D normal vector coefficients")
            x = points[0, :]
            y = points[1, :]
            z = points[2, :]
            nx = 2*c[0]*x + c[4]*z + c[5]*y + c[6]
            ny = 2*c[1]*y + c[3]*z + c[5]*x + c[7]
            nz = 2*c[2]*z + c[3]*y + c[4]*x + c[8]
            n_vec = np.vstack((nx, ny, nz))
            n_vec = n_vec / np.linalg.norm(n_vec, axis=0)  # 正規化
            return n_vec
        elif points.shape[0] == 2:
            # print("calc 2D normal vector coefficients")
            x = points[0, :]
            y = points[1, :]
            nx = 2*c[0]*x + c[5]*y + c[6]
            ny = 2*c[1]*y + c[5]*x + c[7]
            n_vec = np.vstack((nx, ny))
            n_vec = n_vec / np.linalg.norm(n_vec, axis=0)  # 正規化
            return n_vec
        else:
            raise ValueError("Input points must be 2D or 3D.")

def calc_reflection(rays, n_vec, calc_mpmath = calc_mpmath):
    if calc_mpmath:
        if rays.ndim == 1:
            rays = rays.reshape(-1, 1)
        if n_vec.ndim == 1:
            n_vec = n_vec.reshape(-1, 1)
            
        dim, num_rays = rays.shape
        # 出力用の空のNumPy配列を作成
        reflected_rays = np.empty((dim, num_rays), dtype=object)

        for i in range(num_rays):
            # 1. mpmath.mpf に変換
            v = [mp.mpf(rays[j, i]) for j in range(dim)]
            n = [mp.mpf(n_vec[j, i]) for j in range(dim)]
            
            # 2. 内積の計算 (dot product)
            dot_product = sum(vk * nk for vk, nk in zip(v, n))
            
            # 3. 反射ベクトルの計算: r = v - 2 * (v・n) * n
            r = [vk - 2 * dot_product * nk for vk, nk in zip(v, n)]
            
            # 4. 2Dの場合の正規化処理 (else側のロジックに合わせる)
            if dim == 2:
                norm = mp.sqrt(sum(rk**2 for rk in r))
                r = [rk / norm for rk in r]
            
            # NumPy配列に格納
            for j in range(dim):
                reflected_rays[j, i] = r[j]
        
        # 計算結果を float64 の NumPy 配列として返す（高精度を維持したままなら dtype=object のままでも可）
        return reflected_rays.astype(np.float64)
    else:
        if rays.ndim == 1:
            rays = rays.reshape(1, -1)
        if rays.shape[0] == 3:
            # print("calc 3D reflection")
            vx = rays[0, :]
            vy = rays[1, :]
            vz = rays[2, :]
            nx = n_vec[0, :]
            ny = n_vec[1, :]
            nz = n_vec[2, :]
            dot_product = vx*nx + vy*ny + vz*nz
            reflected_rays = rays - 2 * dot_product * n_vec
            return reflected_rays
        elif rays.shape[0] == 2:
            # print("calc 2D reflection")
            # print("rays shape:", rays.shape)
            # print("n_vec :", n_vec)
            vx = rays[0, :]
            vy = rays[1, :]
            nx = n_vec[0, :]
            ny = n_vec[1, :]
            dot_product = vx*nx + vy*ny
            reflected_rays = rays - 2 * dot_product * n_vec
            reflected_rays = reflected_rays / np.linalg.norm(reflected_rays, axis=0)  # 正規化
            return reflected_rays
        else:
            raise ValueError("Input points must be 2D or 3D.")

def calc_intersections_plane_coeffs(points, rays, c, positive=True):
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[0] == 3:
        # print("calc 3D normal vector coefficients")
        x0 = points[0, :]
        y0 = points[1, :]
        z0 = points[2, :]
        vx = rays[0, :]
        vy = rays[1, :]
        vz = rays[2, :]
        k = -(c[6]*x0 + c[7]*y0 + c[8]*z0 + c[9]) / (c[6]*vx + c[7]*vy + c[8]*vz)
        intersections = points + k * rays 
        return intersections
    elif points.shape[0] == 2:
        # print("calc 2D normal vector coefficients")
        x0 = points[0, :]
        y0 = points[1, :]
        vx = rays[0, :]
        vy = rays[1, :]
        k = -(c[6]*x0 + c[7]*y0 + c[9]) / (c[6]*vx + c[7]*vy)
        intersections = points + k * rays
        return intersections
    else:
        raise ValueError("Input points must be 2D or 3D.")
def shift_x_coeffs(coeffs, shift):
    coeffs_shifted = coeffs.copy()
    coeffs_shifted[6] += 2*coeffs[0]*-shift
    coeffs_shifted[7] += coeffs[5]*-shift
    coeffs_shifted[8] += coeffs[4]*-shift
    coeffs_shifted[9] += coeffs[0]*shift**2 + coeffs[6]*-shift
    return coeffs_shifted
def shift_y_coeffs(coeffs, shift):
    coeffs_shifted = coeffs.copy()
    coeffs_shifted[7] += 2*coeffs[1]*-shift
    coeffs_shifted[6] += coeffs[5]*-shift
    coeffs_shifted[8] += coeffs[3]*-shift
    coeffs_shifted[9] += coeffs[1]*shift**2 + coeffs[7]*-shift
    return coeffs_shifted
def shift_z_coeffs(coeffs, shift):
    coeffs_shifted = coeffs.copy()
    coeffs_shifted[8] += 2*coeffs[2]*-shift
    coeffs_shifted[6] += coeffs[4]*-shift
    coeffs_shifted[7] += coeffs[3]*-shift
    coeffs_shifted[9] += coeffs[2]*shift**2 + coeffs[8]*-shift
    return coeffs_shifted
def rotation_matrix(axis, theta):
    axis = axis / np.linalg.norm(axis)  # 単位ベクトルに正規化
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    ux, uy, uz = axis

    # クロス積行列
    cross_product_matrix = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])

    # ロドリゲスの回転行列
    R = np.eye(3) * cos_theta + (1 - cos_theta) * np.outer(axis, axis) + cross_product_matrix * sin_theta
    return R

def rotate_coeffs_general_axis(coeffs, axis, theta, center):
    # 移動と回転を行う
    coeffs = shift_x_coeffs(coeffs, -center[0])
    coeffs = shift_y_coeffs(coeffs, -center[1])
    coeffs = shift_z_coeffs(coeffs, -center[2])

    # 係数の分解
    cxx, cyy, czz, cyz, cxz, cxy, cx, cy, cz, cc = coeffs

    # 回転行列を取得
    R = rotation_matrix(axis, theta).T

    cxx_dash = cxx * R[0,0]**2    + cyy * R[1,0]**2    + czz * R[2,0]**2    + cxy * R[0,0] * R[1,0] + cxz * R[2,0] * R[0,0] + cyz * R[1,0] * R[2,0]
    cyy_dash = cxx * R[0,1]**2    + cyy * R[1,1]**2    + czz * R[2,1]**2    + cxy * R[0,1] * R[1,1] + cxz * R[2,1] * R[0,1] + cyz * R[1,1] * R[2,1]
    czz_dash = cxx * R[0,2]**2    + cyy * R[1,2]**2    + czz * R[2,2]**2    + cxy * R[0,2] * R[1,2] + cxz * R[2,2] * R[0,2] + cyz * R[1,2] * R[2,2]
    cxy_dash = 2*cxx*R[0,0]*R[0,1]+ 2*cyy*R[1,0]*R[1,1]+ 2*czz*R[2,0]*R[2,1]+ cxy*(R[0,1]*R[1,0]+R[0,0]*R[1,1]) + cxz*(R[2,1]*R[0,0]+R[2,0]*R[0,1]) + cyz*(R[1,1]*R[2,0]+R[1,0]*R[2,1])
    cxz_dash = 2*cxx*R[0,0]*R[0,2]+ 2*cyy*R[1,0]*R[1,2]+ 2*czz*R[2,0]*R[2,2]+ cxy*(R[0,2]*R[1,0]+R[0,0]*R[1,2]) + cxz*(R[2,2]*R[0,0]+R[2,0]*R[0,2]) + cyz*(R[1,2]*R[2,0]+R[1,0]*R[2,2])
    cyz_dash = 2*cxx*R[0,1]*R[0,2]+ 2*cyy*R[1,1]*R[1,2]+ 2*czz*R[2,1]*R[2,2]+ cxy*(R[0,1]*R[1,2]+R[0,2]*R[1,1]) + cxz*(R[2,1]*R[0,2]+R[2,2]*R[0,1]) + cyz*(R[1,1]*R[2,2]+R[1,2]*R[2,1])
    cx_dash = cx * R[0, 0] + cy * R[1, 0] + cz * R[2, 0]
    cy_dash = cx * R[0, 1] + cy * R[1, 1] + cz * R[2, 1]
    cz_dash = cx * R[0, 2] + cy * R[1, 2] + cz * R[2, 2]
    cc_dash = cc
    coeffs_new = [cxx_dash, cyy_dash, czz_dash, cyz_dash, cxz_dash, cxy_dash, cx_dash, cy_dash, cz_dash, cc_dash]

    # 元の位置に戻す
    coeffs_new = shift_x_coeffs(coeffs_new, center[0])
    coeffs_new = shift_y_coeffs(coeffs_new, center[1])
    coeffs_new = shift_z_coeffs(coeffs_new, center[2])

    return coeffs_new, rotation_matrix(axis, theta)

def psf_fft(wavelength, theta_array, phase_array, pad_factor=4):
    """
    1次元の集光計算（FFT法）
    
    Parameters:
        wavelength (float): 波長 [mなど]
        theta_array (ndarray): 等間隔の角度配列 [rad]
        phase_array (ndarray): 位相配列 [rad]
        pad_factor (int): パディング倍率（解像度の向上）
    """
    # 1. パラメータ
    N_orig = len(theta_array)
    N_pad = N_orig * pad_factor
    dtheta = theta_array[1] - theta_array[0]

    # 2. 角度領域の光場 (複素振幅)
    # パディング領域は強度が0（窓関数外）として扱われる
    E_theta = np.exp(1j * phase_array)

    # 3. FFT で焦点面の光場を計算
    # fft(x, n) は n が len(x) より大きい場合、自動的に末尾に0を付け足す
    # ゼロが中心に来るように ifftshift を適用
    E_theta_shifted = ifftshift(E_theta)
    Ex_raw = fft(E_theta_shifted, n=N_pad) * dtheta
    Ex = fftshift(Ex_raw)

    # 4. 焦点面位置 x の軸
    # fftfreq は [0, 1, ..., N/2, -N/2, ..., -1] / (N * dtheta) の順で周波数を返す
    # これを fftshift することで、空間周波数 u [rad⁻¹] を計算
    u = fftshift(fftfreq(N_pad, dtheta))
    x = wavelength * u  # 空間周波数から実空間位置へ変換

    # 5. 強度
    I_x = np.abs(Ex)**2

    # 6. 正規化（ピークを1に）
    I_x /= np.max(I_x)
    
    return x, I_x

def curvature_radius(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 曲率半径の計算
    curvature_radius = ((dx**2 + dy**2)**1.5) / np.abs(dx * ddy - dy * ddx)
    
    return curvature_radius

def curvature_radius_poly(x, y, deg=4):
    """
    多項式フィッティングを用いて曲率半径を計算する
    
    Parameters:
        x (ndarray): 座標データ（独立変数）
        y (ndarray): プロファイルデータ（従属変数）
        deg (int): 多項式の次数（局所的な曲率なら2以上、複雑な形状なら高めに設定）
        
    Returns:
        R (ndarray): 各x点における曲率半径
    """
    # 1. 多項式フィッティング (y = p[0]*x^deg + ... + p[deg])
    coeffs = np.polyfit(x, y, deg)
    p = np.poly1d(coeffs)
    
    # 2. 解析的な導関数を取得
    p1 = np.polyder(p, 1) # 1次導関数 y'
    p2 = np.polyder(p, 2) # 2次導関数 y''
    
    # 3. 各点 x での値を計算
    dy = p1(x)
    ddy = p2(x)
    
    # 4. 曲率半径の公式 R = (1 + y'^2)^1.5 / |y''|
    # ※ xが等間隔の独立変数の場合、dx=1, ddx=0 となりこの簡略式が使える
    R = (1 + dy**2)**1.5 / np.abs(ddy)
    
    return R

def calc_intersections_plane_f_df(points, rays, initial_focus,defocus):
    coeffs_det_ = np.zeros(10)
    coeffs_det_[6] = 1
    coeffs_det_[9] = -(initial_focus + defocus)
    points_det_ = calc_intersections_plane_coeffs(points, rays, coeffs_det_)
    return points_det_

def size_on_detector(points, num, all=True):    
    points_z_u = (points[2, 0] + points[2,num-1]) / 2
    points_z_l = (points[2,-1] + points[2,num*(num-1)]) / 2
    points_y_u = (points[1, 0] + points[1,num*(num-1)]) / 2
    points_y_l = (points[1,-1] + points[1,num-1]) / 2
    if all:
        size_z = np.max(points[2, :]) - np.min(points[2, :])
        size_y = np.max(points[1, :]) - np.min(points[1, :])
        size_z *= np.sign(points_z_u - points_z_l)
        size_y *= np.sign(points_y_u - points_y_l)
    else:
        size_z = points_z_u - points_z_l
        size_y = points_y_u - points_y_l
    return size_z, size_y
### Ast Defocus 調整
def Auto_focus(points,rays, focus_initial, defocus, num_ray=None):
    defocus_array = np.linspace(-defocus, defocus, 3)
    size_det_d_z = []
    size_det_d_y = []
    for i, d in enumerate(defocus_array):
        points_det_d = calc_intersections_plane_f_df(points, rays, focus_initial, d)
        size_z, size_y = size_on_detector(points_det_d, num_ray)
        size_det_d_z.append(size_z)
        size_det_d_y.append(size_y)
    size_det_d_z = np.array(size_det_d_z)
    size_det_d_y = np.array(size_det_d_y)
    pz = np.polyfit(defocus_array, size_det_d_z, 1)
    py = np.polyfit(defocus_array, size_det_d_y, 1)
    ### R2乗値を計算
    # 予測値を計算
    y_pred_z = np.polyval(pz, defocus_array)
    y_pred_y = np.polyval(py, defocus_array)

    # 残差平方和（SSE）
    ss_res_z = np.sum((size_det_d_z - y_pred_z)**2)
    ss_res_y = np.sum((size_det_d_y - y_pred_y)**2)

    # 全変動（SST）
    ss_tot_z = np.sum((size_det_d_z - np.mean(size_det_d_z))**2)
    ss_tot_y = np.sum((size_det_d_y - np.mean(size_det_d_y))**2)

    # 決定係数 R²
    r2_z = 1 - ss_res_z/ss_tot_z
    r2_y = 1 - ss_res_y/ss_tot_y

    print(f"R² (z): {r2_z}")
    print(f"R² (y): {r2_y}")

    defocus_fine = -pz[1] / pz[0]
    ast = -py[1] / py[0] - defocus_fine
    # print(f"Focal length calculated from size on detector vs defocus = {defocus_fine} mm")
    # print(f"Astigmatism calculated from size on detector vs defocus = {ast} mm")
    plt.figure(figsize=(8, 6))
    plt.plot(defocus_array, size_det_d_z, 'o-', label='Size on Detector (z)', color='red')
    plt.plot(defocus_array, size_det_d_y, 'o-', label='Size on Detector (y)', color='blue')
    plt.xlabel('Defocus (mm)')
    plt.ylabel('Size on Detector (mm)')
    plt.title("Size on Detector vs Defocus")
    plt.legend()
    plt.grid()
    return defocus_fine, ast