### Version 1.1

import numpy as np
import func as f
from numpy import sin, cos, tan, sqrt, arcsin, arccos, arctan
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.interpolate import interp1d
from scipy.fft import fft, fftshift, fftfreq
from scipy.ndimage import gaussian_filter
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import traceback

global wolter1_V, wolter1_H

def calc_grazing_angle_with_xy(x, y, L_s2vrtlf):
    theta_a = np.arctan(y/x)
    theta_b = np.arctan(y/(L_s2vrtlf - x))
    theta = (theta_a + theta_b) / 2
    return theta_a, theta_b, theta
def get_params_from_entries_opt_vars(entry_map, opt_vars):
    params = {}
    options = {}
    for key, entry in entry_map.items():
        try:
            params[key] = float(entry.get())
            print(f"Parameter {key} set to {params[key]} (float)")
        except Exception:
            params[key] = entry.get()
            print(f"Parameter {key} set to {params[key]} (string)")
    for key, var in opt_vars.items():
        options[key] = var.get()
        print(f"Option {key} set to {options[key]}")
    return params, options
class Wolter1:
    def __init__(self, p_e, q_e, theta_e, l_e, p_h, q_h, datapitch=0.1,sagittal=1, margin_inside=0, margin_outside=0):
        self.p_e = p_e
        self.q_e = q_e
        self.theta_e = theta_e * 1e-3
        self.l_e = l_e
        self.p_h = p_h
        self.q_h = q_h
        self.datapitch = datapitch
        self.sagittal = sagittal
        self.margin_inside = margin_inside
        self.margin_outside = margin_outside
        print(f"Initialized Wolter1 with p_e={p_e} mm, q_e={q_e} mm, theta_e={theta_e} mrad, l_e={l_e} mm, p_h={p_h} mm, q_h={q_h} mm, datapitch={datapitch} mm, sagittal={sagittal} mm, margin_inside={margin_inside} mm, margin_outside={margin_outside} mm")
    def calc_Ellipse(self, axs=None, canvas=None):
        self.a_e, self.b_e = f.ellipse_pq_inc(self.p_e, self.q_e, self.theta_e)
        self.c_e = np.sqrt(self.a_e**2 - self.b_e**2)
        theta1 = f.calc_theta1_ell(self.p_e, self.q_e, self.theta_e)
        theta2 = self.theta_e
        theta3 = 2*self.theta_e - theta1
        print("theta1 =", theta1)
        self.L_s2f = 2*self.c_e
        self.x_e = self.p_e*cos(theta1)
        self.y_e = self.p_e*sin(theta1)
        x_e_u = self.x_e - self.l_e/2 - self.margin_outside
        x_e_l = self.x_e + self.l_e/2 + self.margin_inside
        y_e_u = f.Yvalue_ellipse(self.a_e, self.b_e, self.c_e, x_e_u)
        y_e_l = f.Yvalue_ellipse(self.a_e, self.b_e, self.c_e, x_e_l)
        print("x_e_u =", x_e_u)
        print("y_e_u =", y_e_u)
        self.theta1_u, self.theta3_u, theta_u = calc_grazing_angle_with_xy(x_e_u, y_e_u, self.L_s2f)
        self.theta1_l, self.theta3_l, theta_l = calc_grazing_angle_with_xy(x_e_l, y_e_l, self.L_s2f)
        self.theta2_u = (self.theta1_u + self.theta3_u)/2
        self.theta2_l = (self.theta1_l + self.theta3_l)/2
        self.m_u = tan(self.theta3_u) / tan(self.theta1_u)
        self.m_l = tan(self.theta3_l) / tan(self.theta1_l)
        print("theta1_u =", self.theta1_u)
        print("theta1_l =", self.theta1_l)
        self.NA_s = sin((self.theta1_u - self.theta1_l)/2)
        self.NA_f = sin((self.theta3_l - self.theta3_u)/2)
        number_of_points_e = int((x_e_l - x_e_u) / self.datapitch) + 1
        self.x_e_array = np.linspace(x_e_u, x_e_l, number_of_points_e)
        self.y_e_array = f.Yvalue_ellipse(self.a_e, self.b_e, self.c_e, self.x_e_array)

        _, _, self.theta2_array = calc_grazing_angle_with_xy(self.x_e_array, self.y_e_array, self.L_s2f)

        print(f"x_e_array = {self.x_e_array[0:2]} ... {self.x_e_array[-2:]}")
        print(f"y_e_array = {self.y_e_array[0:2]} ... {self.y_e_array[-2:]}")
        print("theta1 =", theta1)
        print("theta2 =", theta2)
        print("theta3 =", theta3)
        print("NA_s =", self.NA_s)
        print("NA_f =", self.NA_f)
        print("L_s2f",self.L_s2f)
        print("demagnification_u =", self.m_u)
        print("demagnification_l =", self.m_l)
        self.x_e_array_rotated, self.y_e_array_rotated = f.rotation_2D(self.x_e_array, self.y_e_array, 0)
        x_offset = np.min(self.x_e_array_rotated)
        y_offset = np.min(self.y_e_array_rotated)
        self.x_e_array_rotated -= x_offset
        self.y_e_array_rotated -= y_offset
        self.y_e_array_rotated = -self.y_e_array_rotated


        self.rot_angle_ell = -np.arctan((self.y_e_array_rotated[-1] - self.y_e_array_rotated[0]) / (self.x_e_array_rotated[-1] - self.x_e_array_rotated[0]))
        self.x_e_array_final, self.y_e_array_final = f.rotation_2D(self.x_e_array_rotated, self.y_e_array_rotated, self.rot_angle_ell)
        self.x_e_array_final -= np.min(self.x_e_array_final)
        self.y_e_array_final -= np.min(self.y_e_array_final)
        ### 曲率半径を計算
        self.R_e_array = f.curvature_radius_poly(self.x_e_array_final, self.y_e_array_final,deg=10)

        ### グラフ描画

        # fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1 = axs[0]
        color1 = 'tab:blue'
        ax1.set_xlabel('Position [mm]')
        ax1.set_ylabel('Height [μm]', color=color1)
        ax1.plot(self.x_e_array_final, self.y_e_array_final * 1e3, label='Ellipse', color='blue')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x')
        # 右軸: R vs x
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Radius of Curvature [m]', color=color2)
        ax2.plot(self.x_e_array_final, self.R_e_array*1e-3, label='R', color='red')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax1.set_title("Ellipse")

        ax = axs[1]
        ax.plot(self.x_e_array, self.theta2_array*1e3, color='blue')
        ax.set_xlabel('Position [mm]')
        ax.set_ylabel('Grazing Angle (mrad)')
        ax.set_title("Grazing Angle on Ellipse")
        
        
    def calc_1D_axis(self, axs=None, canvas=None):
        self.a_e, self.b_e = f.ellipse_pq_inc(self.p_e, self.q_e, self.theta_e)
        self.c_e = np.sqrt(self.a_e**2 - self.b_e**2)
        theta1 = f.calc_theta1_ell(self.p_e, self.q_e, self.theta_e)
        theta2 = self.theta_e
        theta3 = 2*self.theta_e - theta1
        print("theta1 =", theta1)
        self.L_s2vrtlf = 2*self.c_e
        self.x_e = self.p_e*cos(theta1)
        self.y_e = self.p_e*sin(theta1)
        x_e_u = self.x_e - self.l_e/2 - self.margin_outside
        x_e_l = self.x_e + self.l_e/2 + self.margin_inside
        y_e_u = f.Yvalue_ellipse(self.a_e, self.b_e, self.c_e, x_e_u)
        y_e_l = f.Yvalue_ellipse(self.a_e, self.b_e, self.c_e, x_e_l)
        print("x_e_u =", x_e_u)
        print("y_e_u =", y_e_u)
        self.theta1_u, self.theta3_u, theta_u = calc_grazing_angle_with_xy(x_e_u, y_e_u, self.L_s2vrtlf)
        self.theta1_l, self.theta3_l, theta_l = calc_grazing_angle_with_xy(x_e_l, y_e_l, self.L_s2vrtlf)

        self.theta1_c, self.theta3_c, _ = calc_grazing_angle_with_xy(self.x_e, self.y_e, self.L_s2vrtlf)
        
        print("theta1_u =", self.theta1_u)
        print("theta1_l =", self.theta1_l)
        self.NA_s = sin((self.theta1_u - self.theta1_l)/2)
        print("NA_s =", self.NA_s)
        number_of_points_e = int((x_e_l - x_e_u) / self.datapitch) + 1
        self.x_e_array = np.linspace(x_e_u, x_e_l, number_of_points_e)
        self.y_e_array = f.Yvalue_ellipse(self.a_e, self.b_e, self.c_e, self.x_e_array)

        _, _, self.theta2_array = calc_grazing_angle_with_xy(self.x_e_array, self.y_e_array, self.L_s2vrtlf)

        theta5 = arcsin(self.p_h*sin(theta3)/self.q_h)
        theta4 = (theta5 - theta3) / 2
        theta_h = theta4 ### 楕円中心の反射点基準
        self.a_h, self.b_h = f.hyperbola_pq_inc(self.p_h, self.q_h, theta_h)
        self.c_h = np.sqrt(self.a_h**2 + self.b_h**2)
        x_h_focus = self.p_h*cos(theta3)
        self.y_h = self.p_h*sin(theta3)
        self.x_h = self.L_s2vrtlf - x_h_focus
        x_h_u_focus, _ = f.calc_hyp_theta(self.a_h, self.b_h, self.theta3_u, self.c_h)
        x_h_l_focus, _ = f.calc_hyp_theta(self.a_h, self.b_h, self.theta3_l, self.c_h)
        x_h_c_focus, _ = f.calc_hyp_theta(self.a_h, self.b_h, self.theta3_c, self.c_h)
        print("x_h_u_focus =", x_h_u_focus)
        print("x_h_l_focus =", x_h_l_focus)
        print("x_h_c_focus =", x_h_c_focus)
        print("x_h_u =", self.L_s2vrtlf - x_h_u_focus)
        print("x_h_l =", self.L_s2vrtlf - x_h_l_focus)
        print("x_h_c =", self.L_s2vrtlf - x_h_c_focus)
        y_h_u = f.Yvalue_hyperbola(self.a_h, self.b_h, self.c_h, x_h_u_focus)
        y_h_l = f.Yvalue_hyperbola(self.a_h, self.b_h, self.c_h, x_h_l_focus)
        number_of_points_h = int((x_h_l_focus - x_h_u_focus) / self.datapitch) + 1
        self.x_h_focus_array = np.linspace(x_h_l_focus, x_h_u_focus, number_of_points_h)
        self.y_h_array = f.Yvalue_hyperbola(self.a_h, self.b_h, self.c_h, self.x_h_focus_array)

        self.theta3_h_array = arctan(self.y_h_array/self.x_h_focus_array)
        self.theta5_h_array = arcsin(self.y_h_array/(self.x_h_focus_array - self.c_h*2))
        self.theta4_array = (self.theta5_h_array - self.theta3_h_array) / 2


        self.x_h_array = self.L_s2vrtlf - self.x_h_focus_array

        coeff2_e_origin = np.polyfit(self.x_e_array, self.y_e_array, 2)
        coeff2_h_origin = np.polyfit(self.x_h_array, self.y_h_array, 2)
        ### 交点の計算
        A = coeff2_e_origin[0] - coeff2_h_origin[0]
        B = coeff2_e_origin[1] - coeff2_h_origin[1]
        C = coeff2_e_origin[2] - coeff2_h_origin[2]
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            print("Error: No real intersection points.")
            return
        sqrt_disc = np.sqrt(discriminant)
        self.x_intersect2_origin = (-B + sqrt_disc) / (2*A)
        print("x_intersect2_origin =", self.x_intersect2_origin)
        self.y_intersect2_origin = coeff2_e_origin[0] * self.x_intersect2_origin**2 + coeff2_e_origin[1] * self.x_intersect2_origin + coeff2_e_origin[2]

        self.L_s2f = self.L_s2vrtlf - 2*self.c_h
        tan_u = self.y_h_array[-1]/(self.L_s2f - self.x_h_array[-1])
        tan_l = self.y_h_array[0]/(self.L_s2f - self.x_h_array[0])

        self.NA_f = sin((arctan(tan_u) - arctan(tan_l)) / 2)
        print("tan_u", tan_u)
        print("tan_l", tan_l)
        
        print("y_h_u =", y_h_u)
        print("y_h_l =", y_h_l)
        print(f"x_e_array = {self.x_e_array[0:2]} ... {self.x_e_array[-2:]}")
        print(f"y_e_array = {self.y_e_array[0:2]} ... {self.y_e_array[-2:]}")
        print(f"x_h_array = {self.x_h_array[0:2]} ... {self.x_h_array[-2:]}")
        print(f"y_h_array = {self.y_h_array[0:2]} ... {self.y_h_array[-2:]}")
        print(f"L_s2vrtlf = {self.L_s2vrtlf}")
        print(f"L_s2f = {self.L_s2f}")
        print("theta1 =", theta1)
        print("theta2 =", theta2)
        print("theta3 =", theta3)
        print("theta4 =", theta4)
        print("theta5 =", theta5)
        print("a_e =", self.a_e)
        print("b_e =", self.b_e)
        print("a_h =", self.a_h)
        print("b_h =", self.b_h)
        print("NA_s =", self.NA_s)
        print("NA_f =", self.NA_f)
        self.m_u = tan_u / tan(self.theta1_u)
        self.m_l = tan_l / tan(self.theta1_l)
        print("demagnification_u =", self.m_u)
        print("demagnification_l =", self.m_l)
        self.wd = self.L_s2f - self.x_h_array[-1]
        print("WD =", self.wd)
        # plt.show()
        self.rot_angle_wolter = np.arctan((np.max(self.y_e_array) - np.min(self.y_h_array)) / (np.max(self.x_h_array) - np.min(self.x_e_array)))
        print("Rotation angle (rad) =", self.rot_angle_wolter)
        self.x_e_array_rotated, self.y_e_array_rotated = f.rotation_2D(self.x_e_array, self.y_e_array, self.rot_angle_wolter)
        self.x_h_array_rotated, self.y_h_array_rotated = f.rotation_2D(self.x_h_array, self.y_h_array, self.rot_angle_wolter)
        x_offset = np.min(self.x_e_array_rotated)
        y_offset = np.min(self.y_e_array_rotated)
        self.x_e_array_rotated -= x_offset
        self.y_e_array_rotated -= y_offset
        self.x_h_array_rotated -= x_offset
        self.y_h_array_rotated -= y_offset
        self.y_e_array_rotated = -self.y_e_array_rotated
        self.y_h_array_rotated = -self.y_h_array_rotated

        coeff2_e = np.polyfit(self.x_e_array_rotated, self.y_e_array_rotated, 2)
        coeff2_h = np.polyfit(self.x_h_array_rotated, self.y_h_array_rotated, 2)
        ### 交点の計算
        A = coeff2_e[0] - coeff2_h[0]
        B = coeff2_e[1] - coeff2_h[1]
        C = coeff2_e[2] - coeff2_h[2]
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            print("Error: No real intersection points.")
            return
        sqrt_disc = np.sqrt(discriminant)
        x_intersect2 = (-B - sqrt_disc) / (2*A)
        print("x_intersect2 =", x_intersect2)
        y_intersect2 = coeff2_e[0] * x_intersect2**2 + coeff2_e[1] * x_intersect2 + coeff2_e[2]

        ell2int = x_intersect2 - np.max(self.x_e_array_rotated)
        hyp2int = np.min(self.x_h_array_rotated) - x_intersect2
        print("ell to int =", ell2int)
        print("hyp to int =", hyp2int)
        slope_e = np.arctan(coeff2_e[1])
        slope_h = np.arctan(coeff2_h[1])
        print("slope_e (rad) =", slope_e)
        print("slope_h (rad) =", slope_h) 

        self.rot_angle_ell = -np.arctan((self.y_e_array_rotated[-1] - self.y_e_array_rotated[0]) / (self.x_e_array_rotated[-1] - self.x_e_array_rotated[0]))
        self.rot_angle_hyp = -np.arctan((self.y_h_array_rotated[-1] - self.y_h_array_rotated[0]) / (self.x_h_array_rotated[-1] - self.x_h_array_rotated[0]))
        self.x_e_array_final, self.y_e_array_final = f.rotation_2D(self.x_e_array_rotated, self.y_e_array_rotated, self.rot_angle_ell)
        self.x_h_array_final, self.y_h_array_final = f.rotation_2D(self.x_h_array_rotated, self.y_h_array_rotated, self.rot_angle_hyp)
        self.x_e_array_final -= np.min(self.x_e_array_final)
        self.y_e_array_final -= np.min(self.y_e_array_final)
        self.x_h_array_final -= np.min(self.x_h_array_final)
        self.y_h_array_final -= np.min(self.y_h_array_final)

        
        ### 曲率半径を計算        
        self.R_e_array = f.curvature_radius_poly(self.x_e_array_final, self.y_e_array_final,deg=10)
        self.R_h_array = f.curvature_radius_poly(self.x_h_array_final, self.y_h_array_final,deg=10)
        ### graph drawing
        ax1 = axs[0]
        color1 = 'tab:blue'
        ax1.set_xlabel('Position [mm]')
        ax1.set_ylabel('Height [μm]', color=color1)
        ax1.plot(self.x_e_array_final, self.y_e_array_final * 1e3, label='Ellipse', color='blue')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x')
        # 右軸: R vs x
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Radius of Curvature [m]', color=color2)
        ax2.plot(self.x_e_array_final, self.R_e_array*1e-3, label='R', color='red')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax1.set_title("Wolter1 Ellipse")
        # plt.savefig("wolter1_1D_axis_ellipse.png", dpi=300)
        # fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1 = axs[1]
        color1 = 'tab:green'
        ax1.set_xlabel('Position [mm]')
        ax1.set_ylabel('Height [μm]', color=color1)
        ax1.plot(self.x_h_array_final, self.y_h_array_final * 1e3, label='Hyperbola', color='green')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x')
        # 右軸: R vs x
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Radius of Curvature [m]', color=color2)
        ax2.plot(self.x_h_array_final, self.R_h_array*1e-3, label='R', color='red')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax1.set_title("Wolter1 Hyperbola")
        # plt.savefig("wolter1_1D_axis_hyperbola.png", dpi=300)
        
        # plt.figure(figsize=(8, 6))
        ax = axs[2]
        ax.plot(self.x_e_array_rotated, self.y_e_array_rotated, label='Ellipse Rotated', color='blue')
        ax.plot(self.x_h_array_rotated, self.y_h_array_rotated, label='Hyperbola Rotated', color='green')
        ax.scatter(x_intersect2, y_intersect2, color='red', label='Intersection Point')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend()
        ax.grid()
        ax.set_title(f"Wolter1 ell to intersection={ell2int:.1f} mm hyp to intersection={hyp2int:.1f} mm\n slope_e={1e3*slope_e:.3f} mrad slope_h={1e3*slope_h:.3f} mrad\n slope_e={np.rad2deg(slope_e):.3f} deg slope_h={np.rad2deg(slope_h):.3f} deg")
        # plt.savefig("wolter1_1D_axis_rotated.png", dpi=300)
        ax = axs[3]
        ax.plot(self.x_e_array, self.theta2_array*1e3, label='Ellipse', color='blue')
        ax.plot(self.x_h_array, self.theta4_array*1e3, label='Hyperbola', color='green')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('Grazing Angle (mrad)')
        ax.legend()
        ax.set_title("Grazing Angles on Wolter1")

    def calc_1D_axis_raytrace(self):
        num = 3000
        points_0 = np.zeros((2, num))
        rays_0 = np.zeros((2, num))
        rays0_angle = np.linspace(self.theta1_u, self.theta1_l, num)
        rays_0[0, :] = np.cos(rays0_angle)
        rays_0[1, :] = np.sin(rays0_angle)
        
        coeffs_e = np.zeros(10)
        coeffs_e[0] = 1/self.a_e**2
        coeffs_e[1] = 1/self.b_e**2
        coeffs_e[6] = -2*self.c_e/self.a_e**2
        coeffs_e[9] = self.c_e**2/self.a_e**2 - 1

        points1 = f.calc_intersections_coeffs(points_0, rays_0, coeffs_e, positive=True)
        n_vec_e = f.calc_norm_vector_coeffs(points1, coeffs_e)
        n_vec_e = n_vec_e / np.linalg.norm(n_vec_e, axis=0)
        rays_1 = f.calc_reflection(rays_0, n_vec_e)
        coeffs_h = np.zeros(10)
        coeffs_h[0] = 1/self.a_h**2
        coeffs_h[1] = -1/self.b_h**2
        coeffs_h[6] = -2*(self.L_s2vrtlf - self.c_h)/self.a_h**2
        coeffs_h[9] = (self.L_s2vrtlf - self.c_h)**2/self.a_h**2 - 1

        points2 = f.calc_intersections_coeffs(points1, rays_1, coeffs_h, positive=False)
        n_vec_h = f.calc_norm_vector_coeffs(points2, coeffs_h)
        n_vec_h = n_vec_h / np.linalg.norm(n_vec_h, axis=0)
        rays_2 = f.calc_reflection(rays_1, n_vec_h)

        ### rays_2が等角度ピッチになるようにrays_0を再定義
        rays_2_angle = np.arctan2(rays_2[1, :], rays_2[0, :])
        rays_2_angle_uniform = np.linspace(rays_2_angle[0], rays_2_angle[-1], num)
        print(f"rays_2_angle = {rays_2_angle[0]} ... {rays_2_angle[-1]} rad")
        print(f"rays_2_angle_uniform = {rays_2_angle_uniform[0]} ... {rays_2_angle_uniform[-1]} rad pitch = {rays_2_angle_uniform[1] - rays_2_angle_uniform[0]} rad")
        
        interp_func = interp1d(rays_2_angle, rays_0, axis=1, kind='linear')
        rays_0_uniform = interp_func(rays_2_angle_uniform)
        points1 = f.calc_intersections_coeffs(points_0, rays_0_uniform, coeffs_e, positive=True)
        n_vec_e = f.calc_norm_vector_coeffs(points1, coeffs_e)
        n_vec_e = n_vec_e / np.linalg.norm(n_vec_e, axis=0)
        rays_1 = f.calc_reflection(rays_0_uniform, n_vec_e)
        points2 = f.calc_intersections_coeffs(points1, rays_1, coeffs_h, positive=False)
        n_vec_h = f.calc_norm_vector_coeffs(points2, coeffs_h)
        n_vec_h = n_vec_h / np.linalg.norm(n_vec_h, axis=0)
        rays_2 = f.calc_reflection(rays_1, n_vec_h)
        rays_2_angle = np.arctan2(rays_2[1, :], rays_2[0, :])

        print("rays_0_uniform.shape =", rays_0_uniform.shape)
        print("n_vec_e.shape =", n_vec_e.shape)
        inc_angle_e = np.pi/2 - np.arccos(np.sum(rays_0_uniform * n_vec_e, axis=0))
        inc_angle_h = -np.pi/2 + np.arccos(np.sum(rays_1 * n_vec_h, axis=0))
        plt.figure(figsize=(8, 6))
        plt.plot(points1[0, :] - np.min(points1[0, :]), inc_angle_e, label='Incidence Angle on Ellipse', color='blue')
        plt.plot(points2[0, :] - np.min(points1[0, :]), inc_angle_h, label='Incidence Angle on Hyperbola', color='green')
        plt.xlabel('x (mm)')
        plt.ylabel('Grazing Incidence Angle (rad)')
        plt.legend()
        plt.savefig("wolter1_incidence_angles.png", dpi=300)
        print(f"rays_2_angle = {rays_2_angle[0]} ... {rays_2_angle[-1]} rad pitch = {rays_2_angle[1] - rays_2_angle[0]} rad")

        coeffs_det = np.zeros(10)
        coeffs_det[6] = 1
        coeffs_det[9] = -(self.L_s2vrtlf - 2*self.c_h)
        points3 = f.calc_intersections_plane_coeffs(points2, rays_2, coeffs_det)

        plt.figure(figsize=(8, 6))
        plt.plot(points1[0, :], points1[1, :], 'o', label='Reflection Points on Ellipse', color='blue')
        plt.plot(points2[0, :], points2[1, :], 'o', label='Reflection Points on Hyperbola', color='green')
        plt.plot(points3[0, :], points3[1, :], 'x', label='Focus Points', color='red')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.legend()
        plt.grid()
        plt.title("Ray Tracing on Wolter1 1D Axisymmetric")
        # plt.axis('equal')
        
        plt.figure(figsize=(8, 6))
        plt.plot(points3[0, :], points3[1, :], 'x', label='Focus Points', color='red')

        slope_error_rms = 2 # 1 μrad rms
        rng = np.random.default_rng()
        slopes = rng.standard_normal(num)
        ### 2次をfit
        p = np.polyfit(points1[0, :], slopes, 2)
        slopes -= np.polyval(p, points1[0, :])
        # スケーリング: RMSを指定の値に合わせる
        slopes = slopes * (slope_error_rms / np.std(slopes))
        print(f"data pitch points1 = {points1[0, 1] - points1[0, 0]} mm to points1 = {points1[0, -1] - points1[0, -2]} mm")
        datapitch_points1 = np.mean(np.diff(points1[0, :]))

        thereshold = 100e-3 # μm
        sigma = thereshold / datapitch_points1
        print(f"Gaussian filter sigma = {sigma} points")
        slopes_lp = gaussian_filter(slopes, sigma=sigma)
        slopes_hp = slopes - slopes_lp
        slopes_lp = slopes_lp * (slope_error_rms / np.std(slopes_lp))
        slopes_hp = slopes_hp * (slope_error_rms / np.std(slopes_hp))

        shape_error_lp = np.cumsum(slopes_lp) * datapitch_points1
        shape_error_hp = np.cumsum(slopes_hp) * datapitch_points1

        energy = 100 # eV
        wavelength = 1240 / energy # nm
        wave_error_lp = 2 * shape_error_lp * (np.sin(np.max(inc_angle_e)) + np.sin(np.max(inc_angle_h))) * 2 * np.pi / wavelength
        wave_error_hp = 2 * shape_error_hp * (np.sin(np.max(inc_angle_e)) + np.sin(np.max(inc_angle_h))) * 2 * np.pi / wavelength
        pad_factor = 32
        theta_array = rays_2_angle-np.mean(rays_2_angle)
        print(f"theta_array = {theta_array[0]} ... {theta_array[-1]} rad pitch = {theta_array[1] - theta_array[0]} rad")
        x_focus, psf_lp = f.psf_fft(wavelength * 1e-9, theta_array, wave_error_lp, pad_factor=pad_factor)
        x_focus, psf_hp = f.psf_fft(wavelength * 1e-9, theta_array, wave_error_hp, pad_factor=pad_factor)
        x_focus_nm = x_focus * 1e9
        num_over_half_lp = np.sum(psf_lp >= 0.5)
        fwhm_nm_lp = num_over_half_lp * (x_focus_nm[1] - x_focus_nm[0])
        num_over_half_hp = np.sum(psf_hp >= 0.5)
        fwhm_nm_hp = num_over_half_hp * (x_focus_nm[1] - x_focus_nm[0])
        print(f"FWHM with slope and shape errors = {fwhm_nm_lp:.3f} nm")
        print(f"FWHM with only slope errors = {fwhm_nm_hp:.3f} nm")
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_focus_nm, psf_lp, label='PSF with Slope Errors (Low-Pass)', color='magenta')
        plt.xlim(-10e3, 10e3)
        plt.xlabel('Focus Position (nm)')
        plt.ylabel('Normalized Intensity')
        plt.title(f"PSF at Focus with Slope Errors (Low-Pass)\nslope_error_rms = {slope_error_rms:.2f} μrad\nFWHM = {fwhm_nm_lp:.6f} nm")
        plt.legend()

        plt.figure(figsize=(8, 6))
        plt.plot(x_focus_nm, psf_hp, label='PSF with Slope Errors (High-Pass)', color='cyan')
        plt.xlim(-10e3, 10e3)
        plt.xlabel('Focus Position (nm)')
        plt.ylabel('Normalized Intensity')
        plt.title(f"PSF at Focus with Slope Errors (High-Pass)\nslope_error_rms = {slope_error_rms:.2f} μrad\nFWHM = {fwhm_nm_hp:.6f} nm")
        plt.legend()

        # plt.figure(figsize=(8, 6))
        # plt.plot(points1[0, :], slopes)
        # plt.xlabel('x (mm)')
        # plt.ylabel('Slope Error (μrad)')

        plt.figure(figsize=(8, 6))
        plt.plot(points1[0, :], shape_error_lp)
        plt.xlabel('x (mm)')
        plt.ylabel('Shape Error (nm)')
        plt.title(f"Shape Error for Each Mirror (Low-Pass) slope_error_rms = {slope_error_rms:.2f} μrad")

        plt.figure(figsize=(8, 6))
        plt.plot(points1[0, :], shape_error_hp)
        plt.xlabel('x (mm)')
        plt.ylabel('Shape Error (nm)')
        plt.title(f"Shape Error for Each Mirror (High-Pass) slope_error_rms = {slope_error_rms:.2f} μrad")

        plt.show()


    def calc_2D_axis(self, axs=None, canvas=None):
        self.calc_1D_axis(axs=axs, canvas=canvas)
        numpoints_z = int(self.sagittal / self.datapitch) + 1
        self.z_array = np.linspace(-self.sagittal/2, self.sagittal/2, numpoints_z)
        self.X_e_2D, self.Z_e_2D = np.meshgrid(self.x_e_array, self.z_array)
        self.Y_e_2D = f.Yvalue_ellipse_3D(self.a_e, self.b_e, self.c_e, self.X_e_2D, self.Z_e_2D)
        self.X_h_focus_2D, self.Z_h_2D = np.meshgrid(self.x_h_focus_array, self.z_array)
        self.Y_h_2D = f.Yvalue_hyperbola_3D(self.a_h, self.b_h, self.c_h, self.X_h_focus_2D, self.Z_h_2D)
        self.X_h_2D = self.L_s2vrtlf - self.X_h_focus_2D

        ###
        numpoints_x = int((np.max(self.x_h_array) - np.min(self.x_e_array)) / self.datapitch) + 1
        self.x_wolter = np.linspace(np.min(self.x_e_array), np.max(self.x_h_array), numpoints_x)
        self.X_wolter_2D, self.Z_wolter_2D = np.meshgrid(self.x_wolter, self.z_array)
        self.Y_wolter_2D = np.zeros_like(self.X_wolter_2D)
        for i in range(self.Z_wolter_2D.shape[0]):
            for j in range(self.Z_wolter_2D.shape[1]):
                x = self.X_wolter_2D[i, j]
                z = self.Z_wolter_2D[i, j]
                if x < self.x_intersect2_origin:
                    self.Y_wolter_2D[i, j] = f.Yvalue_ellipse_3D(self.a_e, self.b_e, self.c_e, x, z)
                else:
                    self.Y_wolter_2D[i, j] = f.Yvalue_hyperbola_3D(self.a_h, self.b_h, self.c_h, self.L_s2vrtlf - x, z)


        # plt.figure(figsize=(8, 6))
        # plt.plot(self.X_e_2D[self.X_e_2D.shape[0]//2, :], self.Y_e_2D[self.X_e_2D.shape[0]//2, :], label='Ellipse', color='blue')
        # plt.plot(self.X_h_2D[self.X_h_2D.shape[0]//2, :], self.Y_h_2D[self.X_h_2D.shape[0]//2, :], label='Hyperbola', color='green')
        # plt.xlabel('x (mm)')
        # plt.ylabel('y (mm)')
        # plt.legend()
        # plt.grid()
        # plt.show()

        # plt.figure(figsize=(8, 6))
        # plt.contourf(self.X_e_2D, self.Z_e_2D, self.Y_e_2D, levels=50, cmap='jet')
        # plt.colorbar(label='y (mm)')
        # plt.title('Ellipse Surface')
        # plt.xlabel('x (mm)')
        # plt.ylabel('z (mm)')
        # plt.grid()
        # plt.figure(figsize=(8, 6))
        # plt.contourf(self.X_h_2D, self.Z_h_2D, self.Y_h_2D, levels=50, cmap='jet')
        # plt.colorbar(label='y (mm)')
        # plt.title('Hyperbola Surface')
        # plt.xlabel('x (mm)')
        # plt.ylabel('z (mm)')
        # plt.grid()

        self.X_e_2D_rotated, self.Y_e_2D_rotated = f.rotation_2D(self.X_e_2D, self.Y_e_2D, self.rot_angle_wolter)
        self.X_h_2D_rotated, self.Y_h_2D_rotated = f.rotation_2D(self.X_h_2D, self.Y_h_2D, self.rot_angle_wolter)
        x_offset = np.min(self.X_e_2D_rotated)
        y_offset = np.min(self.Y_e_2D_rotated)
        self.X_e_2D_rotated -= x_offset
        self.Y_e_2D_rotated -= y_offset
        self.X_h_2D_rotated -= x_offset
        self.Y_h_2D_rotated -= y_offset
        self.Y_e_2D_rotated = -self.Y_e_2D_rotated
        self.Y_h_2D_rotated = -self.Y_h_2D_rotated

        self.X_wolter_2D_rotated, self.Y_wolter_2D_rotated = f.rotation_2D(self.X_wolter_2D, self.Y_wolter_2D, self.rot_angle_wolter)
        x_offset = np.min(self.X_wolter_2D_rotated)
        y_offset = np.min(self.Y_wolter_2D_rotated)
        self.X_wolter_2D_rotated -= x_offset
        self.Y_wolter_2D_rotated -= y_offset
        self.Y_wolter_2D_rotated = -self.Y_wolter_2D_rotated

        # plt.figure(figsize=(8, 6))
        # plt.contourf(self.X_e_2D_rotated, self.Z_e_2D, self.Y_e_2D_rotated, levels=50, cmap='jet')
        # plt.colorbar(label='y (mm)')
        # plt.title('Ellipse Surface Rotated')
        # plt.xlabel('x (mm)')
        # plt.ylabel('z (mm)')
        # plt.grid()
        # plt.figure(figsize=(8, 6))
        # plt.contourf(self.X_h_2D_rotated, self.Z_h_2D, self.Y_h_2D_rotated, levels=50, cmap='jet')
        # plt.colorbar(label='y (mm)')
        # plt.title('Hyperbola Surface Rotated')
        # plt.xlabel('x (mm)')
        # plt.ylabel('z (mm)')
        # plt.grid()

        # plt.figure(figsize=(8, 6))
        # plt.contourf(self.X_e_2D_rotated, self.Z_e_2D, self.Y_e_2D_rotated, levels=50, cmap='jet')
        # plt.contourf(self.X_h_2D_rotated, self.Z_h_2D, self.Y_h_2D_rotated, levels=50, cmap='jet')
        # plt.colorbar(label='y (mm)')
        # plt.title('Ellipse Surface Rotated')
        # plt.xlabel('x (mm)')
        # plt.ylabel('z (mm)')
        # plt.grid()
        

class Wolter3:
    def __init__(self, p_e, q_e, theta_e, l_e, p_h, q_h, datapitch=0.1,sagittal=1, margin_inside=0, margin_outside=0):
        self.p_e = p_e
        self.q_e = q_e
        self.theta_e = theta_e * 1e-3
        self.l_e = l_e
        self.p_h = p_h
        self.q_h = q_h
        self.datapitch = datapitch
        self.sagittal = sagittal
        self.margin_inside = margin_inside
        self.margin_outside = margin_outside
        print(f"Initialized Wolter3 with p_e={p_e} mm, q_e={q_e} mm, theta_e={theta_e} mrad, l_e={l_e} mm, p_h={p_h} mm, q_h={q_h} mm, datapitch={datapitch} mm, sagittal={sagittal} mm, margin_inside={margin_inside} mm, margin_outside={margin_outside} mm")
    def calc_1D_axis(self, axs=None, canvas=None):
        self.a_e, self.b_e = f.ellipse_pq_inc(self.p_e, self.q_e, self.theta_e)
        self.c_e = np.sqrt(self.a_e**2 - self.b_e**2)
        theta5 = f.calc_theta1_ell(self.p_e, self.q_e, self.theta_e)
        theta4 = self.theta_e
        theta3 = 2*self.theta_e - theta5
        print("theta5 =", theta5)
        self.L_s2vrtlf = 2*self.c_e
        self.x_e = self.p_e*cos(theta5)
        self.y_e = self.p_e*sin(theta5)
        x_e_u = self.x_e - self.l_e/2 - self.margin_outside
        x_e_l = self.x_e + self.l_e/2 + self.margin_inside
        y_e_u = f.Yvalue_ellipse(self.a_e, self.b_e, self.c_e, x_e_u)
        y_e_l = f.Yvalue_ellipse(self.a_e, self.b_e, self.c_e, x_e_l)
        print("x_e_u =", x_e_u)
        print("y_e_u =", y_e_u)
        self.theta5_u, self.theta3_u, theta_u = calc_grazing_angle_with_xy(x_e_u, y_e_u, self.L_s2vrtlf)
        self.theta5_l, self.theta3_l, theta_l = calc_grazing_angle_with_xy(x_e_l, y_e_l, self.L_s2vrtlf)

        self.theta5_c, self.theta3_c, _ = calc_grazing_angle_with_xy(self.x_e, self.y_e, self.L_s2vrtlf)
        
        print("theta5_u =", self.theta5_u)
        print("theta5_l =", self.theta5_l)
        self.NA_s = sin((self.theta5_u - self.theta5_l)/2)
        print("NA_s =", self.NA_s)
        number_of_points_e = int((x_e_l - x_e_u) / self.datapitch) + 1
        self.x_e_array = np.linspace(x_e_u, x_e_l, number_of_points_e)
        self.y_e_array = f.Yvalue_ellipse(self.a_e, self.b_e, self.c_e, self.x_e_array)

        _, _, self.theta4_array = calc_grazing_angle_with_xy(self.x_e_array, self.y_e_array, self.L_s2vrtlf)

        theta1 = arcsin(self.p_h*sin(theta3)/self.q_h)
        theta2 = (theta3 - theta1) / 2
        theta_h = theta2 ### 楕円中心の反射点基準
        self.a_h, self.b_h = f.hyperbola_pq_inc(self.p_h, self.q_h, theta_h)
        self.c_h = np.sqrt(self.a_h**2 + self.b_h**2)
        x_h_focus = self.p_h*cos(theta3)
        self.y_h = self.p_h*sin(theta3)
        self.x_h = self.L_s2vrtlf - x_h_focus
        x_h_u_focus, _ = f.calc_hyp_theta(self.a_h, self.b_h, self.theta3_u, self.c_h)
        x_h_l_focus, _ = f.calc_hyp_theta(self.a_h, self.b_h, self.theta3_l, self.c_h)
        x_h_c_focus, _ = f.calc_hyp_theta(self.a_h, self.b_h, self.theta3_c, self.c_h)
        print("x_h_u_focus =", x_h_u_focus)
        print("x_h_l_focus =", x_h_l_focus)
        print("x_h_c_focus =", x_h_c_focus)
        print("x_h_u =", self.L_s2vrtlf + x_h_u_focus)
        print("x_h_l =", self.L_s2vrtlf + x_h_l_focus)
        print("x_h_c =", self.L_s2vrtlf + x_h_c_focus)
        y_h_u = f.Yvalue_hyperbola(self.a_h, self.b_h, self.c_h, x_h_u_focus)
        y_h_l = f.Yvalue_hyperbola(self.a_h, self.b_h, self.c_h, x_h_l_focus)
        number_of_points_h = int((x_h_l_focus - x_h_u_focus) / self.datapitch) + 1
        self.x_h_focus_array = np.linspace(x_h_l_focus, x_h_u_focus, number_of_points_h)
        self.y_h_array = f.Yvalue_hyperbola(self.a_h, self.b_h, self.c_h, self.x_h_focus_array)

        self.theta3_h_array = arctan(self.y_h_array/-self.x_h_focus_array)
        self.theta1_h_array = arcsin(self.y_h_array/(-self.x_h_focus_array + self.c_h*2))
        self.theta2_array = (self.theta3_h_array - self.theta1_h_array) / 2


        self.x_h_array = self.L_s2vrtlf + self.x_h_focus_array

        self.L_s2f = self.L_s2vrtlf + 2*self.c_h
        tan_u = self.y_h_array[-1]/(self.L_s2f - self.x_h_array[-1])
        tan_l = self.y_h_array[0]/(self.L_s2f - self.x_h_array[0])

        self.NA_f = sin((arctan(tan_u) - arctan(tan_l)) / 2)
        print("tan_u", tan_u)
        print("tan_l", tan_l)
        
        print("y_h_u =", y_h_u)
        print("y_h_l =", y_h_l)
        print(f"x_e_array = {self.x_e_array[0:2]} ... {self.x_e_array[-2:]}")
        print(f"y_e_array = {self.y_e_array[0:2]} ... {self.y_e_array[-2:]}")
        print(f"x_h_array = {self.x_h_array[0:2]} ... {self.x_h_array[-2:]}")
        print(f"y_h_array = {self.y_h_array[0:2]} ... {self.y_h_array[-2:]}")
        print(f"L_s2vrtlf = {self.L_s2vrtlf}")
        print(f"L_s2f = {self.L_s2f}")
        print("theta1 =", theta1)
        print("theta2 =", theta2)
        print("theta3 =", theta3)
        print("theta4 =", theta4)
        print("theta5 =", theta5)
        print("a_e =", self.a_e)
        print("b_e =", self.b_e)
        print("a_h =", self.a_h)
        print("b_h =", self.b_h)
        print("NA_s =", self.NA_s)
        print("NA_f =", self.NA_f)
        self.m_u = tan(self.theta5_u) / tan_u
        self.m_l = tan(self.theta5_l) / tan_l
        print("magnification_u =", self.m_u)
        print("magnification_l =", self.m_l)
        self.wd = self.L_s2f - self.x_h_array[-1]
        print("WD =", self.wd)
        # plt.show()
        self.p_e_array = np.sqrt(self.x_e_array**2 + self.y_e_array**2)
        self.q_e_array = np.sqrt((self.L_s2vrtlf - self.x_e_array)**2 + self.y_e_array**2)
        self.p_h_array = np.sqrt((self.L_s2vrtlf - self.x_h_array)**2 + self.y_h_array**2)
        self.q_h_array = np.sqrt((self.L_s2f - self.x_h_array)**2 + self.y_h_array**2)

        self.rot_angle_ell = -np.arctan((self.y_e_array[-1] - self.y_e_array[0]) / (self.x_e_array[-1] - self.x_e_array[0]))
        self.rot_angle_hyp = -np.arctan((self.y_h_array[-1] - self.y_h_array[0]) / (self.x_h_array[-1] - self.x_h_array[0]))
        self.x_e_array_final, self.y_e_array_final = f.rotation_2D(self.x_e_array, self.y_e_array, self.rot_angle_ell)
        self.x_h_array_final, self.y_h_array_final = f.rotation_2D(self.x_h_array, self.y_h_array, self.rot_angle_hyp)
        self.x_e_array_final -= np.min(self.x_e_array_final)
        self.y_e_array_final -= np.min(self.y_e_array_final)
        self.y_e_array_final = -self.y_e_array_final
        self.x_h_array_final -= np.min(self.x_h_array_final)
        self.y_h_array_final -= np.min(self.y_h_array_final)

        
        ### 曲率半径を計算        
        self.R_e_array = f.curvature_radius_poly(self.x_e_array_final, self.y_e_array_final,deg=10)
        self.R_h_array = f.curvature_radius_poly(self.x_h_array_final, self.y_h_array_final,deg=10)
        ### graph drawing
        ax1 = axs[0]
        color1 = 'tab:blue'
        ax1.set_xlabel('Position [mm]')
        ax1.set_ylabel('Height [μm]', color=color1)
        ax1.plot(self.x_e_array_final, self.y_e_array_final * 1e3, label='Ellipse', color='blue')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x')
        # 右軸: R vs x
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Radius of Curvature [m]', color=color2)
        ax2.plot(self.x_e_array_final, self.R_e_array*1e-3, label='R', color='red')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax1.set_title("Wolter3 Ellipse")
        # plt.savefig("wolter1_1D_axis_ellipse.png", dpi=300)
        # fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1 = axs[1]
        color1 = 'tab:green'
        ax1.set_xlabel('Position [mm]')
        ax1.set_ylabel('Height [μm]', color=color1)
        ax1.plot(self.x_h_array_final, self.y_h_array_final * 1e3, label='Hyperbola', color='green')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x')
        # 右軸: R vs x
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Radius of Curvature [m]', color=color2)
        ax2.plot(self.x_h_array_final, self.R_h_array*1e-3, label='R', color='red')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax1.set_title("Wolter3 Hyperbola")
        # plt.savefig("wolter1_1D_axis_hyperbola.png", dpi=300)
        
        # plt.figure(figsize=(8, 6))
        # ax = axs[2]
        # ax.plot(self.x_e_array_final, self.y_e_array_final, label='Ellipse Rotated', color='blue')
        # ax.plot(self.x_h_array_final, self.y_h_array_final, label='Hyperbola Rotated', color='green')
        # ax.scatter(x_intersect2, y_intersect2, color='red', label='Intersection Point')
        # ax.set_xlabel('x (mm)')
        # ax.set_ylabel('y (mm)')
        # ax.legend()
        # ax.grid()
        # ax.set_title(f"Wolter1 ell to intersection={ell2int:.1f} mm hyp to intersection={hyp2int:.1f} mm\n slope_e={1e3*slope_e:.3f} mrad slope_h={1e3*slope_h:.3f} mrad\n slope_e={np.rad2deg(slope_e):.3f} deg slope_h={np.rad2deg(slope_h):.3f} deg")
        # plt.savefig("wolter1_1D_axis_rotated.png", dpi=300)
        ax = axs[3]
        ax.plot(self.x_e_array, self.theta4_array*1e3, label='Ellipse', color='blue')
        ax.plot(self.x_h_array, self.theta2_array*1e3, label='Hyperbola', color='green')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('Grazing Angle (mrad)')
        ax.legend()
        ax.set_title("Grazing Angles on Wolter3")




def get_rotation_matrix_to_x(v):
    """
    ベクトルvが新しい座標系のX軸を向くようにするための回転行列を計算する
    """
    v = np.array(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.eye(3)
    
    # 1. 新しいX軸 (正規化されたベクトルv)
    ex = v / norm
    
    # 2. 新しいY軸を求めるための補助ベクトルを選択
    # exが(0, 0, 1)に近い場合は(0, 1, 0)を使用、そうでなければ(0, 0, 1)を使用
    if abs(ex[2]) < 0.9:
        a = np.array([0, 0, 1])
    else:
        a = np.array([0, 1, 0])
    
    # 3. 新しいY軸 (aとexの外積を正規化)
    ey = np.cross(a, ex)
    ey /= np.linalg.norm(ey)
    
    # 4. 新しいZ軸 (exとeyの外積)
    ez = np.cross(ex, ey)
    
    # 5. 回転行列 R の構築
    # 行ベクトルとして並べることで、xyz系からXYZ系への変換行列になる
    R = np.array([ex, ey, ez])
    
    return R

def calc_AKB_raytrace_1_1(wolter1_V, wolter1_H):
    num_ray = 31 ### 奇数に設定
    num_ray_center = num_ray // 2
    num_ray_squre = num_ray**2
    points_0 = np.zeros((3, num_ray_squre))
    source_angle_center_V = (wolter1_V.theta1_u + wolter1_V.theta1_l) / 2
    rays0_angle_v = np.linspace(wolter1_V.theta1_u, wolter1_V.theta1_l, num_ray) - source_angle_center_V
    source_angle_center_H = (wolter1_H.theta1_u + wolter1_H.theta1_l) / 2
    rays0_angle_h = np.linspace(wolter1_H.theta1_u, wolter1_H.theta1_l, num_ray) - source_angle_center_H

    yaw_H_initial = ((wolter1_V.theta1_u + wolter1_V.theta3_u) + (wolter1_H.theta1_u + wolter1_H.theta3_u)) / 2
    print(f"Initial yaw angle for H mirror (rad) = {yaw_H_initial}")
    
    rays_0 = np.zeros((3, num_ray_squre))
    for i in range(num_ray):
        rays0_angle_v_here = rays0_angle_v[i]
        rays_0[1, num_ray * i:num_ray * (i + 1)] = np.tan(rays0_angle_h)
        rays_0[2, num_ray * i:num_ray * (i + 1)] = np.tan(rays0_angle_v_here)
        rays_0[0, num_ray * i:num_ray * (i + 1)] = 1.
    rays_0 = rays_0 / np.linalg.norm(rays_0, axis=0)
    ### 初期曲面係数定義    
    axis_V_x = np.array([1, 0, 0])
    axis_V_y = np.array([0, 1, 0])
    axis_V_z = np.array([0, 0, 1]) 

    coeffs_e_V = np.zeros(10)
    coeffs_e_V[0] = 1/wolter1_V.a_e**2
    coeffs_e_V[2] = 1/wolter1_V.b_e**2
    coeffs_e_V[9] = -1
    coeffs_e_V = f.shift_x_coeffs(coeffs_e_V, wolter1_V.c_e)
    coeffs_e_V, rotate_matrix_e_V = f.rotate_coeffs_general_axis(coeffs_e_V, axis_V_y, source_angle_center_V,[0,0,0])

    coeffs_h_V = np.zeros(10)
    coeffs_h_V[0] = 1/wolter1_V.a_h**2
    coeffs_h_V[2] = -1/wolter1_V.b_h**2
    coeffs_h_V[9] =  -1
    coeffs_h_V = f.shift_x_coeffs(coeffs_h_V, wolter1_V.L_s2vrtlf - wolter1_V.c_h)
    coeffs_h_V, rotate_matrix_h_V = f.rotate_coeffs_general_axis(coeffs_h_V, axis_V_y, source_angle_center_V,[0,0,0])

    axis_H_x = np.array([1, 0, 0])
    axis_H_y = np.array([0, 1, 0])
    axis_H_z = np.array([0, 0, 1])
    coeffs_e_H = np.zeros(10)
    coeffs_e_H[0] = 1/wolter1_H.a_e**2
    coeffs_e_H[1] = 1/wolter1_H.b_e**2
    coeffs_e_H[9] = -1
    coeffs_e_H = f.shift_x_coeffs(coeffs_e_H, wolter1_H.c_e)
    coeffs_e_H, rotate_matrix_e_H = f.rotate_coeffs_general_axis(coeffs_e_H, axis_H_z, -source_angle_center_H,[0,0,0])

    coeffs_h_H = np.zeros(10)
    coeffs_h_H[0] = 1/wolter1_H.a_h**2
    coeffs_h_H[1] = -1/wolter1_H.b_h**2
    coeffs_h_H[9] = -1
    coeffs_h_H = f.shift_x_coeffs(coeffs_h_H, wolter1_H.L_s2vrtlf - wolter1_H.c_h)
    coeffs_h_H, rotate_matrix_h_H = f.rotate_coeffs_general_axis(coeffs_h_H, axis_H_z, -source_angle_center_H,[0,0,0])
    ### 初期光線追跡
    points1_V = f.calc_intersections_coeffs(points_0, rays_0, coeffs_e_V, positive=True)
    n_vec_e_V = f.calc_norm_vector_coeffs(points1_V, coeffs_e_V)
    rays_1_V = f.calc_reflection(rays_0, n_vec_e_V)

    points2_V = f.calc_intersections_coeffs(points1_V, rays_1_V, coeffs_h_V, positive=False)
    n_vec_h_V = f.calc_norm_vector_coeffs(points2_V, coeffs_h_V)
    rays_2_V = f.calc_reflection(rays_1_V, n_vec_h_V)

    if True:
        rays_2_V_mean = (rays_2_V[:,0] + rays_2_V[:, -1] + rays_2_V[:, num_ray-1] + rays_2_V[:, num_ray*(num_ray-1)]) / 4
        rays_2_V_mean = rays_2_V_mean / np.linalg.norm(rays_2_V_mean)
        yaw_H_initial = -np.arctan2(rays_2_V_mean[2], rays_2_V_mean[0])
        print(f"Initial yaw angle for H mirror after first raytrace (rad) = {yaw_H_initial}")
    
    points3_H = f.calc_intersections_coeffs(points2_V, rays_2_V, coeffs_e_H, positive=True)
    ### Yaw adjustment for H mirror
    points3_H_mean = (points3_H[:, num_ray_center] + points3_H[:, -num_ray_center-1]) / 2
    coeffs_e_H, rotate_matrix_e_H = f.rotate_coeffs_general_axis(coeffs_e_H, axis_H_y, yaw_H_initial,points3_H_mean)
    coeffs_h_H, rotate_matrix_h_H = f.rotate_coeffs_general_axis(coeffs_h_H, axis_H_y, yaw_H_initial,points3_H_mean)
    axis_H_x = rotate_matrix_e_H @ axis_H_x
    axis_H_y = rotate_matrix_e_H @ axis_H_y
    axis_H_z = rotate_matrix_e_H @ axis_H_z
    print(f"Yaw rotate_matrix_H:\n{rotate_matrix_e_H}")
    print(f"Axis H x after yaw rotation: {axis_H_x}")
    print(f"Axis H y after yaw rotation: {axis_H_y}")
    print(f"Axis H z after yaw rotation: {axis_H_z}")
    alignment = False
    if alignment:
        # coeffs_h_H, rotate_matrix_h_H = f.rotate_coeffs_general_axis(coeffs_h_H, axis_H_x, 50e-6,points3_H_mean)
        coeffs_h_H, rotate_matrix_h_H = f.rotate_coeffs_general_axis(coeffs_h_H, axis_H_z, 2e-6,points3_H_mean)

    
    ### re raytrace after yaw adjustment
    points3_H = f.calc_intersections_coeffs(points2_V, rays_2_V, coeffs_e_H, positive=True)
    n_vec_e_H = f.calc_norm_vector_coeffs(points3_H, coeffs_e_H)
    rays_3_H = f.calc_reflection(rays_2_V, n_vec_e_H)

    points4_H = f.calc_intersections_coeffs(points3_H, rays_3_H, coeffs_h_H, positive=False)
    n_vec_h_H = f.calc_norm_vector_coeffs(points4_H, coeffs_h_H)
    rays_4_H = f.calc_reflection(rays_3_H, n_vec_h_H)

    rays_4_H_mean = (rays_4_H[:,0] + rays_4_H[:, -1] + rays_4_H[:, num_ray-1] + rays_4_H[:, num_ray*(num_ray-1)]) / 4
    rays_4_H_mean = rays_4_H_mean / np.linalg.norm(rays_4_H_mean)
    points_det = f.calc_intersections_plane_f_df(points4_H, rays_4_H, wolter1_V.L_s2f, 0)


    R = get_rotation_matrix_to_x(rays_4_H_mean)
    # 変換後のベクトルを確認 ( [norm(v), 0, 0] になるはず )
    print("Rotation matrix R:\n", R)
    print("Original rays_4_H mean direction:", rays_4_H_mean)
    print("Transformed rays_4_H mean direction (should be along x-axis):", R @ rays_4_H_mean)
    rays_4_H_transformed = np.zeros_like(rays_4_H)
    points_4_H_transformed = np.zeros_like(points4_H)
    for i in range(rays_4_H.shape[1]):
        rays_4_H_transformed[:, i] = R @ rays_4_H[:, i]
        points_4_H_transformed[:, i] = R @ points4_H[:, i]
        
    print("Transformed rays_4_H (should be along x-axis):")
    # defocus_array = np.array([-1, 0, 1])
    # size_det_d_z = []
    # size_det_d_y = []
    # for i, d in enumerate(defocus_array):
    #     points_det_d = f.calc_intersections_plane_f_df(points_4_H_transformed, rays_4_H_transformed, wolter1_V.L_s2f, d)
    #     size_z, size_y = f.size_on_detector(points_det_d, num)
    #     size_det_d_z.append(size_z)
    #     size_det_d_y.append(size_y)
    # size_det_d_z = np.array(size_det_d_z)
    # size_det_d_y = np.array(size_det_d_y)
    # pz = np.polyfit(defocus_array, size_det_d_z, 1)
    # py = np.polyfit(defocus_array, size_det_d_y, 1)
    # ### zサイズの傾きから焦点距離を計算
    # defocus_fine = -pz[1] / pz[0]
    # ast = -py[1] / py[0] - defocus_fine
    # print(f"Focal length calculated from size on detector vs defocus = {defocus_fine:.2f} mm")
    # print(f"Astigmatism calculated from size on detector vs defocus = {ast:.2f} mm")
    
    # defocus_fine, ast = f.Auto_focus(points_4_H_transformed, rays_4_H_transformed, wolter1_V.L_s2f, 1)
    
    ### Ast Defocus 調整後の焦点位置でのサイズを計算し、再度Ast Defocus 調整をするiteration
    ### TODO write fine tuning 
    for j in range(3):
        defocus_fine, ast = f.Auto_focus(points_4_H_transformed, rays_4_H_transformed, wolter1_V.L_s2f, 10**-j, num_ray=num_ray)
        coeffs_e_H = f.shift_x_coeffs(coeffs_e_H, -ast)
        coeffs_h_H = f.shift_x_coeffs(coeffs_h_H, -ast)

        points3_H = f.calc_intersections_coeffs(points2_V, rays_2_V, coeffs_e_H, positive=True)
        n_vec_e_H = f.calc_norm_vector_coeffs(points3_H, coeffs_e_H)
        rays_3_H = f.calc_reflection(rays_2_V, n_vec_e_H)

        points4_H = f.calc_intersections_coeffs(points3_H, rays_3_H, coeffs_h_H, positive=False)
        n_vec_h_H = f.calc_norm_vector_coeffs(points4_H, coeffs_h_H)
        rays_4_H = f.calc_reflection(rays_3_H, n_vec_h_H)

        rays_4_H_mean = (rays_4_H[:,0] + rays_4_H[:, -1] + rays_4_H[:, num_ray-1] + rays_4_H[:, num_ray*(num_ray-1)]) / 4
        rays_4_H_mean = rays_4_H_mean / np.linalg.norm(rays_4_H_mean)
        points_det = f.calc_intersections_plane_f_df(points4_H, rays_4_H, wolter1_V.L_s2f + defocus_fine, 0)

        R = get_rotation_matrix_to_x(rays_4_H_mean)
        # 変換後のベクトルを確認 ( [norm(v), 0, 0] になるはず )
        # print("Rotation matrix R:\n", R)
        # print("Original rays_4_H mean direction:", rays_4_H_mean)
        # print("Transformed rays_4_H mean direction (should be along x-axis):", R @ rays_4_H_mean)
        rays_4_H_transformed = np.zeros_like(rays_4_H)
        points_4_H_transformed = np.zeros_like(points4_H)
        for i in range(rays_4_H.shape[1]):
            rays_4_H_transformed[:, i] = R @ rays_4_H[:, i]
            points_4_H_transformed[:, i] = R @ points4_H[:, i]
        points_det_t = f.calc_intersections_plane_f_df(points_4_H_transformed, rays_4_H_transformed, wolter1_V.L_s2f + defocus_fine, 0)
        size_z, size_y = f.size_on_detector(points_det_t, num_ray)
        print(f"Iteration {j}: defocus_fine = {defocus_fine} mm, ast = {ast} mm, size_z = {size_z} mm, size_y = {size_y} mm")
        if np.abs(ast) < 1e-9:
            print("Astigmatism is sufficiently small, stopping iteration.")
            break
    print("Transformed rays_4_H (should be along x-axis):")
    defocus_m = -10**(-j-1)
    defocus_p = 10**(-j-1)
    points_det_t_m = f.calc_intersections_plane_f_df(points_4_H_transformed, rays_4_H_transformed, wolter1_V.L_s2f + defocus_fine, defocus_m)
    points_det_t_p = f.calc_intersections_plane_f_df(points_4_H_transformed, rays_4_H_transformed, wolter1_V.L_s2f + defocus_fine, defocus_p)
    opl_s_to_1 = np.linalg.norm(points1_V - points_0, axis=0)
    opl_1_to_2 = np.linalg.norm(points2_V - points1_V, axis=0)
    opl_2_to_3 = np.linalg.norm(points3_H - points2_V, axis=0)
    opl_3_to_4 = np.linalg.norm(points4_H - points3_H, axis=0)
    opl_4_to_det = np.linalg.norm(points_det_t - points_4_H_transformed, axis=0)
    opl_total = opl_s_to_1 + opl_1_to_2 + opl_2_to_3 + opl_3_to_4 + opl_4_to_det
    points_det_t_matrix =np.zeros((3, num_ray, num_ray))
    points_det_t_m_matrix =np.zeros((3, num_ray, num_ray))
    points_det_t_p_matrix =np.zeros((3, num_ray, num_ray))
    opl_total_matrix = np.zeros((num_ray, num_ray))
    for i in range(num_ray):
        for j in range(num_ray):
            points_det_t_matrix[:, i, j] = points_det_t[:, num_ray * i + j]
            points_det_t_m_matrix[:, i, j] = points_det_t_m[:, num_ray * i + j]
            points_det_t_p_matrix[:, i, j] = points_det_t_p[:, num_ray * i + j]
            opl_total_matrix[i, j] = opl_total[num_ray * i + j]
    opl_total_matrix = opl_total_matrix - (np.max(opl_total_matrix) + np.min(opl_total_matrix)) / 2
    plt.figure(figsize=(8, 6))
    plt.imshow(opl_total_matrix, extent=[-rays0_angle_h[-1], rays0_angle_h[-1], rays0_angle_v[0], rays0_angle_v[-1]], aspect='auto', origin='lower',interpolation=None, cmap='jet')
    plt.colorbar(label='Optical Path Length (mm)')
    plt.xlabel('Initial Ray Angle in H direction (rad)')
    plt.ylabel('Initial Ray Angle in V direction (rad)')
    plt.title('Optical Path Length from Source to Detector')

    plt.figure(figsize=(8, 6))
    # plt.plot(points_4_H_transformed[0, :], points_4_H_transformed[2, :], 'o', label='Reflection Points on Hyperbola H', color='magenta')
    plt.plot(points_det_t_matrix[0, :, :].flatten(), points_det_t_matrix[2, :, :].flatten(), 'x', label='Focus Points', color='red')
    plt.plot(points_det_t_m_matrix[0, :, :].flatten(), points_det_t_m_matrix[2, :, :].flatten(), 'x', label=f'Focus Points defocus {defocus_m}mm', color='blue')
    plt.plot(points_det_t_p_matrix[0, :, :].flatten(), points_det_t_p_matrix[2, :, :].flatten(), 'x', label=f'Focus Points defocus {defocus_p}mm', color='cyan')
    plt.plot([points_det_t_m_matrix[0, :, 0].flatten(), points_det_t_p_matrix[0, :, 0].flatten()], [points_det_t_m_matrix[2, :, 0].flatten(), points_det_t_p_matrix[2, :, 0].flatten()], label='Focus Points edge1', color='black')
    plt.plot([points_det_t_m_matrix[0, :, -1].flatten(), points_det_t_p_matrix[0, :, -1].flatten()], [points_det_t_m_matrix[2, :, -1].flatten(), points_det_t_p_matrix[2, :, -1].flatten()], label='Focus Points edge4', color='purple')
    # plt.axis('equal')
    plt.xlabel('x (mm)')
    plt.ylabel('z (mm)')
    # plt.legend()
    plt.grid()
    plt.title("Ray Tracing on AKB1 1D - Side View")
    plt.figure(figsize=(8, 6))
    # plt.plot(points_4_H_transformed[0, :], points_4_H_transformed[1, :], 'o', label='Reflection Points on Hyperbola H', color='magenta')
    plt.plot(points_det_t_matrix[0, :, :].flatten(), points_det_t_matrix[1, :, :].flatten(), 'x', label='Focus Points', color='red')
    plt.plot(points_det_t_m_matrix[0, :, :].flatten(), points_det_t_m_matrix[1, :, :].flatten(), 'x', label=f'Focus Points defocus {defocus_m}mm', color='blue')
    plt.plot(points_det_t_p_matrix[0, :, :].flatten(), points_det_t_p_matrix[1, :, :].flatten(), 'x', label=f'Focus Points defocus {defocus_p}mm', color='cyan')
    plt.plot([points_det_t_m_matrix[0, 0, :].flatten(), points_det_t_p_matrix[0, 0, :].flatten()], [points_det_t_m_matrix[1, 0, :].flatten(), points_det_t_p_matrix[1, 0, :].flatten()], label='Focus Points edge2', color='gray')
    plt.plot([points_det_t_m_matrix[0, -1, :].flatten(), points_det_t_p_matrix[0, -1, :].flatten()], [points_det_t_m_matrix[1, -1, :].flatten(), points_det_t_p_matrix[1, -1, :].flatten()], label='Focus Points edge3', color='orange')
    # plt.axis('equal')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    # plt.legend()
    plt.grid()
    plt.title("Ray Tracing on AKB1 1D - Top View")
    

    plt.figure(figsize=(8, 6))
    plt.plot(points_det_t_matrix[1, :, :].flatten(), points_det_t_matrix[2, :, :].flatten(), 'x', label='Focus Points', color='red')
    plt.plot(points_det_t_matrix[1, 0, 0], points_det_t_matrix[2, 0, 0], 'x', label='Focus Points edge', color='blue')
    plt.plot(points_det_t_matrix[1, -1, -1], points_det_t_matrix[2, -1, -1], 'x', label='Focus Points edge', color='green')
    plt.plot(points_det_t_matrix[1, 0, -1], points_det_t_matrix[2, 0, -1], 'x', label='Focus Points edge', color='cyan')
    plt.plot(points_det_t_matrix[1, -1, 0], points_det_t_matrix[2, -1, 0], 'x', label='Focus Points edge', color='magenta')
    plt.axis('equal')
    plt.xlabel('y (mm)')
    plt.ylabel('z (mm)')
    plt.title("Spot diagram on Detector")

    fig, ax = plt.subplots(3,figsize=(8, 12),sharex=True,sharey=True)
    ax[0].plot(points_det_t_matrix[1, :, :].flatten(), points_det_t_matrix[2, :, :].flatten(), 'x', label='Focus Points', color='red')
    ax[0].set_title('Focus Points')
    ax[1].plot(points_det_t_m_matrix[1, :, :].flatten(), points_det_t_m_matrix[2, :, :].flatten(), 'x', label=f'Focus Points defocus {defocus_m}mm', color='blue')
    ax[1].set_title(f'Focus Points defocus {defocus_m}mm')
    ax[2].plot(points_det_t_p_matrix[1, :, :].flatten(), points_det_t_p_matrix[2, :, :].flatten(), 'x', label=f'Focus Points defocus {defocus_p}mm', color='cyan')
    ax[2].set_title(f'Focus Points defocus {defocus_p}mm')
    for a in ax:
        a.set_aspect('equal', adjustable='box')
        a.set_xlabel('y (mm)')
        a.set_ylabel('z (mm)')
        a.legend()
        a.grid()
    plt.show()

    

    plt.figure(figsize=(8, 6))
    plt.plot(points1_V[0, :], points1_V[2, :], 'o', label='Reflection Points on Ellipse V', color='blue')
    plt.plot(points2_V[0, :], points2_V[2, :], 'o', label='Reflection Points on Hyperbola V', color='green')
    plt.plot(points3_H[0, :], points3_H[2, :], 'o', label='Reflection Points on Ellipse H', color='cyan')
    plt.plot(points3_H[0, num_ray_center], points3_H[2, num_ray_center], 'o', label='Reflection Points on Ellipse H', color='k')
    plt.plot(points3_H[0, -num_ray_center-1], points3_H[2, -num_ray_center-1], 'o', label='Reflection Points on Ellipse H', color='k')
    plt.plot(points4_H[0, :], points4_H[2, :], 'o', label='Reflection Points on Hyperbola H', color='magenta')
    plt.plot(points_det[0, :], points_det[2, :], 'x', label='Focus Points', color='red')
    plt.xlabel('x (mm)')
    plt.ylabel('z (mm)')
    plt.legend()
    plt.grid()
    plt.title("Ray Tracing on AKB1 - Side View")

    plt.figure(figsize=(8, 6))
    plt.plot(points1_V[0, :], points1_V[1, :], 'o', label='Reflection Points on Ellipse V', color='blue')
    plt.plot(points2_V[0, :], points2_V[1, :], 'o', label='Reflection Points on Hyperbola V', color='green')
    plt.plot(points3_H[0, :], points3_H[1, :], 'o', label='Reflection Points on Ellipse H', color='cyan')
    plt.plot(points3_H[0, num_ray_center], points3_H[1, num_ray_center], 'o', label='Reflection Points on Ellipse H', color='k')
    plt.plot(points3_H[0, -num_ray_center-1], points3_H[1, -num_ray_center-1], 'o', label='Reflection Points on Ellipse H', color='k')
    plt.plot(points4_H[0, :], points4_H[1, :], 'o', label='Reflection Points on Hyperbola H', color='magenta')
    plt.plot(points_det[0, :], points_det[1, :], 'x', label='Focus Points', color='red')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.legend()
    plt.grid()
    plt.title("Ray Tracing on AKB1 - Top View")
    # plt.axis('equal')
    
    # plt.figure(figsize=(8, 6))
    # plt.plot(points_det[1, :], points_det[2, :], 'x', label='Focus Points', color='red')
    # plt.plot(points_det[1, 0], points_det[2, 0], 'x', label='Focus Points edge', color='blue')
    # plt.plot(points_det[1, -1], points_det[2, -1], 'x', label='Focus Points edge', color='green')
    # plt.plot(points_det[1, num-1], points_det[2, num-1], 'x', label='Focus Points edge', color='cyan')
    # plt.plot(points_det[1, num*(num-1)], points_det[2, num*(num-1)], 'x', label='Focus Points edge', color='green')

def run_Ellipse(axs=None, canvas=None):
    try:
        params, options = get_params_from_entries_opt_vars(entries, opt_vars)
        # Wolter1のインスタンス化と実行
        clear_axs(axs, canvas)
        Ellipse1 = Wolter1(
            params["p_e"], 
            params["q_e"], 
            params["theta_e"], 
            params["l_e"], 
            0, 
            0, 
            datapitch=params["datapitch"],
            sagittal=0,
            margin_inside=params["margin_inside"],
            margin_outside=params["margin_outside"]
        )
        Ellipse1.calc_Ellipse(axs, canvas)
        # plt.show()
        if canvas is not None:
            canvas.draw() # GUIを更新
        print("完了", "Wolter1の実行が完了しました。")
        
    except ValueError as e:
        messagebox.showerror("入力エラー", f"すべての項目に数値を入力してください。")
        print("ValueError:", e)
    except Exception as e:
        # 2. 詳細なエラー情報を文字列として取得
        error_detail = traceback.format_exc()
        # コンソールにすべて出力
        print("--- Detailed Error Report ---")
        print(error_detail)
        messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{e}")
def run_KB(axs_a=None, canvas_a=None, axs_b=None, canvas_b=None):
    try:
        params, options = get_params_from_entries_opt_vars(entries, opt_vars)
        clear_axs(axs_a, canvas_a)
        clear_axs(axs_b, canvas_b)
        # Wolter1のインスタンス化と実行
        Ellipse1 = Wolter1(
            params["p_e"], 
            params["q_e"], 
            params["theta_e"], 
            params["l_e"], 
            0, 
            0, 
            datapitch=params["datapitch"],
            sagittal=0,
            margin_inside=params["margin_inside"],
            margin_outside=params["margin_outside"]
        )
        Ellipse1.calc_Ellipse(axs=axs_a, canvas=canvas_a)
        plt.savefig("shape_ellipse1.png", dpi=300)
        Ellipse2 = Wolter1(
            params["p_e_H"], 
            params["q_e_H"], 
            params["theta_e_H"], 
            params["l_e_H"], 
            0, 
            0, 
            datapitch=params["datapitch"],
            sagittal=0,
            margin_inside=params["margin_inside"],
            margin_outside=params["margin_outside"]
        )
        Ellipse2.calc_Ellipse(axs=axs_b, canvas=canvas_b)
        # plt.savefig("shape_ellipse2.png", dpi=300)
        ax = axs_b[4]
        ax.plot(Ellipse1.x_e_array, Ellipse1.y_e_array, label='Ellipse V', color='blue')
        ax.scatter(Ellipse1.x_e, Ellipse1.y_e, color='red', label='Ellipse Center Point V')
        ax.scatter(Ellipse1.L_s2f, 0, color='orange', label='Focus Point V')
        ax.plot(Ellipse2.x_e_array, Ellipse2.y_e_array, label='Ellipse H', color='cyan')
        ax.scatter(Ellipse2.x_e, Ellipse2.y_e, color='magenta', label='Ellipse Center Point H')
        ax.scatter(Ellipse2.L_s2f, 0, color='orange', label='Focus Point H')
        ax.set_xlabel('major axis (mm)')
        ax.set_ylabel('minor axis (mm)')
        ax.legend()
        ax.grid()
        print("===== Ellipse V =====")
        print(f"Length of mirror: {Ellipse1.x_e_array[-1] - Ellipse1.x_e_array[0]} mm")
        print(f"self.L_s2f: {Ellipse1.L_s2f} mm")
        print("===== Ellipse H =====")
        print(f"Length of mirror: {Ellipse2.x_e_array[-1] - Ellipse2.x_e_array[0]} mm")
        print(f"self.L_s2f: {Ellipse2.L_s2f} mm")
        print("===== KB =====")
        print(f"Distance between two mirrors: {Ellipse2.x_e_array[0] - Ellipse1.x_e_array[-1]} mm")
        print(f"Overall length: {Ellipse2.x_e_array[-1] - Ellipse1.x_e_array[0]} mm")
        print(f"Distance from source to focus: {Ellipse1.L_s2f} mm")
        print(f"Working distance: {Ellipse1.L_s2f - Ellipse2.x_e_array[-1]} mm")
        canvas_a.draw() # GUIを更新
        canvas_b.draw() # GUIを更新
        # plt.show()
        if options["data_output"]:
            canvas_a.figure.savefig(f"Results_KB_1D_axis_V.png", dpi=300)
            canvas_b.figure.savefig(f"Results_KB_1D_axis_H.png", dpi=300)
        # if options["raytrace"]:
        #     calc_KB_raytrace(wolter1_V, wolter1_H)
        #     plt.show()
        
    except ValueError as e:
        messagebox.showerror("入力エラー", f"すべての項目に数値を入力してください。")
        print("ValueError:", e)
    except Exception as e:
        # 2. 詳細なエラー情報を文字列として取得
        error_detail = traceback.format_exc()
        # コンソールにすべて出力
        print("--- Detailed Error Report ---")
        print(error_detail)
        messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{e}")
def run_autoKB(axs=None, canvas=None):
    try:
        params, options = get_params_from_entries_opt_vars(entries, opt_vars)
        clear_axs(axs, canvas)
        p1 = params["p1"]
        m1 = params["m1"]
        Length1 = params["Length1"]
        theta1 = params["theta1"]
        Length2 = params["Length2"]
        theta2 = params["theta2"]
        spacing = params["spacing"]
        
        q1 = p1 / m1
        p2 = p1 + Length1/2 + spacing + Length2/2

        # Wolter1のインスタンス化と実行
        Ellipse1 = Wolter1(
            p1, 
            q1, 
            theta1, 
            Length1, 
            0, 
            0, 
            datapitch=params["datapitch"],
            sagittal=0,
            margin_inside=0,
            margin_outside=0
        )
        Ellipse1.calc_Ellipse(axs=axs, canvas=canvas)
        plt.savefig("shape_ellipse1.png", dpi=300)
        A = 1
        B = 2*p2-4*p2*sin(theta2*1e-3)**2
        C = p2**2 - 4*Ellipse1.c_e**2
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            raise ValueError("No real solution for q2. Please check the input parameters.")
        q2_prime = (-B - np.sqrt(discriminant)) / (2*A)
        q2 = (-B + np.sqrt(discriminant)) / (2*A)
        print(f"q2_prime = {q2_prime}, q2 = {q2}")
        Ellipse2 = Wolter1(
            p2, 
            q2, 
            theta2, 
            Length2, 
            0, 
            0, 
            datapitch=params["datapitch"],
            sagittal=0,
            margin_inside=0,
            margin_outside=0
        )
        Ellipse2.calc_Ellipse(axs=axs[3:], canvas=canvas)
        
        spacing_after_calc = np.min(Ellipse2.x_e_array) - np.max(Ellipse1.x_e_array)
        wd = Ellipse1.L_s2f - np.max(Ellipse2.x_e_array)
        overall_length = np.max(Ellipse2.x_e_array) - np.min(Ellipse1.x_e_array)



            
            
        ax = axs[5]
        ax.plot(Ellipse1.x_e_array, Ellipse1.y_e_array, label='Ellipse V', color='blue')
        ax.scatter(Ellipse1.x_e, Ellipse1.y_e, color='red', label='Ellipse Center Point V')
        ax.scatter(Ellipse1.L_s2f, 0, color='orange', label='Focus Point V')
        ax.plot(Ellipse2.x_e_array, Ellipse2.y_e_array, label='Ellipse H', color='cyan')
        ax.scatter(Ellipse2.x_e, Ellipse2.y_e, color='magenta', label='Ellipse Center Point H')
        ax.scatter(Ellipse2.L_s2f, 0, color='orange', label='Focus Point H')
        ax.set_xlabel('major axis (mm)')
        ax.set_ylabel('minor axis (mm)')
        ax.legend()
        ax.grid()
        if options["data_output"]:
            canvas.figure.savefig(f"Results_AKB_1D_axis.png", dpi=300)
            file_txt = f"KB_parameters.txt"
            with open(file_txt, 'w') as f:
                f.write(f"===== Ellipse V =====\n")
                f.write(f"p_v: {p1} mm\n")
                f.write(f"q_v: {q1} mm\n")
                f.write(f"theta_v: {theta1} mrad\n")
                f.write(f"Length_v: {Length1} mm\n")
                f.write(f"demagnification_u: {Ellipse1.m_u}\n")
                f.write(f"demagnification_l: {Ellipse1.m_l}\n")
                f.write(f"NA_v: {Ellipse1.NA_f}\n")
                f.write(f"theta_v: {Ellipse1.theta2_u*1e3:.2f} - {Ellipse1.theta2_l*1e3:.2f} mrad\n")
                f.write(f"aperture: {Length1 * np.sin(theta1*1e-3):.2f} mm\n")
                f.write(f"===== Ellipse H =====\n")
                f.write(f"p_h: {p2} mm\n")
                f.write(f"q_h: {q2} mm\n")
                f.write(f"theta_h: {theta2} mrad\n")
                f.write(f"Length_h: {Length2} mm\n")
                f.write(f"demagnification_u: {Ellipse2.m_u}\n")
                f.write(f"demagnification_l: {Ellipse2.m_l}\n")
                f.write(f"NA_h: {Ellipse2.NA_f}\n")
                f.write(f"theta_h: {Ellipse2.theta2_u*1e3:.2f} - {Ellipse2.theta2_l*1e3:.2f} mrad\n")
                f.write(f"aperture: {Length2 * np.sin(theta2*1e-3):.2f} mm\n")
                f.write(f"===== KB system =====\n")
                f.write(f"spacing: {spacing_after_calc}\n")
                f.write(f"working distance: {wd}\n")
                f.write(f"overall length: {overall_length}\n")
        canvas.draw() # GUIを更新
        print("完了", "Wolter1の実行が完了しました。")
        
    except ValueError as e:
        messagebox.showerror("入力エラー", f"すべての項目に数値を入力してください。")
        print("ValueError:", e)
    except Exception as e:
        # 2. 詳細なエラー情報を文字列として取得
        error_detail = traceback.format_exc()
        # コンソールにすべて出力
        print("--- Detailed Error Report ---")
        print(error_detail)
        messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{e}")
def run_wolter1_1D(axs=None, canvas=None):
    try:
        params, options = get_params_from_entries_opt_vars(entries, opt_vars)
        # Wolter1のインスタンス化と実行
        clear_axs(axs, canvas)
        wolter1 = Wolter1(
            params["p_e"], 
            params["q_e"], 
            params["theta_e"], 
            params["l_e"], 
            params["p_h"], 
            params["q_h"], 
            datapitch=params["datapitch"],
            sagittal=0,
            margin_inside=params["margin_inside"],
            margin_outside=params["margin_outside"]
        )
        wolter1.calc_1D_axis(axs, canvas)
        wolter_x_combined, wolter_y_combined = polyfit_wolter1_from_ell_hyp(
            wolter1.x_e_array_rotated, wolter1.y_e_array_rotated, wolter1.x_h_array_rotated, wolter1.y_h_array_rotated, degree=20,datapitch=wolter1.datapitch)
        print(f"Check datapitch: {wolter_x_combined[1] - wolter_x_combined[0]} mm, expected: {wolter1.datapitch} mm")
        print(f"Length of mirror: {wolter_x_combined[-1] - wolter_x_combined[0]} mm")
        print(f"Length of Ellipse: {wolter1.x_e_array[-1] - wolter1.x_e_array[0]} mm")
        print(f"Length of Hyperbola: {wolter1.x_h_array[-1] - wolter1.x_h_array[0]} mm")
        axs[5].plot(wolter_x_combined, wolter_y_combined, label='Combined Wolter1', color='black')
        axs[5].scatter(wolter1.x_e_array_rotated, wolter1.y_e_array_rotated, label='Ellipse Rotated', color='blue')
        axs[5].scatter(wolter1.x_h_array_rotated, wolter1.y_h_array_rotated, label='Hyperbola Rotated', color='green')
        axs[5].set_xlabel('x (mm)')
        axs[5].set_ylabel('y (mm)')
        axs[5].legend()
        axs[5].grid()
        axs[5].set_title(f"Combined Wolter1 Surface")

        # plt.figure(figsize=(8, 6))
        ax = axs[4]
        ax.plot(wolter1.x_e_array, wolter1.y_e_array, label='Ellipse', color='blue')
        ax.scatter(wolter1.x_e, wolter1.y_e, color='red', label='Ellipse Center Point')
        ax.plot(wolter1.x_h_array, wolter1.y_h_array, label='Hyperbola', color='green')
        ax.scatter(wolter1.x_h, wolter1.y_h, color='purple', label='Hyperbola Center Point')
        ax.scatter(wolter1.L_s2vrtlf - 2*wolter1.c_h, 0, color='orange', label='Focus Point')
        ax.scatter(wolter1.x_intersect2_origin, wolter1.y_intersect2_origin, color='red', label='Intersection Point')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend()
        ax.grid()
        ax.set_title(f"WD = {wolter1.wd:.2f} mm, NA = {wolter1.NA_f:.3e}, Demag = {wolter1.m_u:.2f} - {wolter1.m_l:.2f} x")
        
        # plt.savefig("wolter1_1D_axis.png", dpi=300)
        if options["data_output"]:
            ### wolter_x_combinedの保存
            np.savetxt(f"wolter1_surface_1D_datapitch{wolter1.datapitch}.csv", np.column_stack((wolter_x_combined, wolter_y_combined)), delimiter=",", header="x_mm,y_mm", comments="")
            ### グラフの保存
            # fig = axs[0].get_figure()
            canvas.figure.savefig(f"Results_wolter1_1D_axis.png", dpi=300)
        if canvas is not None:
            canvas.draw() # GUIを更新
        if options["raytrace"]:
            wolter1.calc_1D_axis_raytrace()
            plt.show()
        print("完了", "Wolter1の実行が完了しました。")
        
    except ValueError as e:
        messagebox.showerror("入力エラー", f"すべての項目に数値を入力してください。")
        print("ValueError:", e)
    except Exception as e:
        # 2. 詳細なエラー情報を文字列として取得
        error_detail = traceback.format_exc()
        # コンソールにすべて出力
        print("--- Detailed Error Report ---")
        print(error_detail)
        messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{e}")
def run_wolter1_2D(axs=None, canvas=None):
    try:
        params, options = get_params_from_entries_opt_vars(entries, opt_vars)
        clear_axs(axs, canvas)
        # Wolter1のインスタンス化と実行
        wolter1 = Wolter1(
            params["p_e"], 
            params["q_e"], 
            params["theta_e"], 
            params["l_e"], 
            params["p_h"], 
            params["q_h"], 
            datapitch=params["datapitch"],
            sagittal=params["sagittal"],
            margin_inside=params["margin_inside"],
            margin_outside=params["margin_outside"]
        )
        wolter1.calc_2D_axis(axs, canvas)
        ax = axs[4]
        ax.plot(wolter1.x_e_array, wolter1.y_e_array, label='Ellipse', color='blue')
        ax.scatter(wolter1.x_e, wolter1.y_e, color='red', label='Ellipse Center Point')
        ax.plot(wolter1.x_h_array, wolter1.y_h_array, label='Hyperbola', color='green')
        ax.scatter(wolter1.x_h, wolter1.y_h, color='purple', label='Hyperbola Center Point')
        ax.scatter(wolter1.L_s2vrtlf - 2*wolter1.c_h, 0, color='orange', label='Focus Point')
        ax.scatter(wolter1.x_intersect2_origin, wolter1.y_intersect2_origin, color='red', label='Intersection Point')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y RoC (mm)')
        ax.legend()
        ax.grid()
        ax.set_title(f"WD = {wolter1.wd:.2f} mm, NA = {wolter1.NA_f:.3e}, Demag = {wolter1.m_u:.2f} - {wolter1.m_l:.2f} x")
        ax = axs[5]
        contour = ax.contourf(wolter1.X_wolter_2D_rotated, wolter1.Z_wolter_2D, wolter1.Y_wolter_2D_rotated, levels=50, cmap='jet')
        plt.colorbar(contour, ax=ax, label='y (mm)')
        ax.set_title('Wolter1 Surface Rotated')
        # ax.set_title(f"WD = {wolter1.wd:.2f} mm, NA = {wolter1.NA_f:.3e}, Demag = {wolter1.m_u:.2f} - {wolter1.m_l:.2f} x")
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('z (mm)')
        ax.grid()

        # plt.savefig("wolter1_2D_axis_rotated.png", dpi=300)
        if options["data_output"]:
            canvas.figure.savefig(f"Results_wolter1_2D_axis.png", dpi=300)
            np.savetxt(f"wolter1_surface_datapitch{wolter1.datapitch}.csv", wolter1.Y_wolter_2D_rotated, delimiter=",")
            np.savetxt(f"wolter1_surface_datapitch{wolter1.datapitch}.txt", wolter1.Y_wolter_2D_rotated)
        canvas.draw() # GUIを更新
        # plt.show()
        
        print("完了", "Wolter1の実行が完了しました。")
        
    except ValueError as e:
        messagebox.showerror("入力エラー", f"すべての項目に数値を入力してください。")
        print("ValueError:", e)
    except Exception as e:
        # 2. 詳細なエラー情報を文字列として取得
        error_detail = traceback.format_exc()
        # コンソールにすべて出力
        print("--- Detailed Error Report ---")
        print(error_detail)
        messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{e}")
def run_AKB(axs_a=None, canvas_a=None, axs_b=None, canvas_b=None):
    try:
        params, options = get_params_from_entries_opt_vars(entries, opt_vars)
        clear_axs(axs_a, canvas_a)
        clear_axs(axs_b, canvas_b)
        # Wolter1のインスタンス化と実行
        wolter1_V = Wolter1(
            params["p_e"], 
            params["q_e"], 
            params["theta_e"], 
            params["l_e"], 
            params["p_h"], 
            params["q_h"], 
            datapitch=params["datapitch"],
            sagittal=0,
            margin_inside=0,
            margin_outside=0
        )
        wolter1_V.calc_1D_axis(axs=axs_a, canvas=canvas_a)
        wolter1_H = Wolter1(
            params["p_e_H"],
            params["q_e_H"],
            params["theta_e_H"],
            params["l_e_H"],
            params["p_h_H"],
            params["q_h_H"],
            datapitch=params["datapitch"],
            sagittal=0,
            margin_inside=0,
            margin_outside=0
        )
        wolter1_H.calc_1D_axis(axs=axs_b, canvas=canvas_b)
        # plt.figure(figsize=(8, 6))
        ax = axs_b[4]
        ax.plot(wolter1_V.x_e_array, wolter1_V.y_e_array, label='Ellipse V', color='blue')
        ax.scatter(wolter1_V.x_e, wolter1_V.y_e, color='red', label='Ellipse Center Point V')
        ax.plot(wolter1_V.x_h_array, wolter1_V.y_h_array, label='Hyperbola V', color='green')
        ax.scatter(wolter1_V.x_h, wolter1_V.y_h, color='purple', label='Hyperbola Center Point V')
        ax.scatter(wolter1_V.L_s2vrtlf - 2*wolter1_V.c_h, 0, color='orange', label='Focus Point V')
        ax.scatter(wolter1_V.x_intersect2_origin, wolter1_V.y_intersect2_origin, color='red', label='Intersection Point V')
        ax.plot(wolter1_H.x_e_array, wolter1_H.y_e_array, label='Ellipse H', color='cyan')
        ax.scatter(wolter1_H.x_e, wolter1_H.y_e, color='magenta', label='Ellipse Center Point H')
        ax.plot(wolter1_H.x_h_array, wolter1_H.y_h_array, label='Hyperbola H', color='lime')
        ax.scatter(wolter1_H.x_h, wolter1_H.y_h, color='yellow', label='Hyperbola Center Point H')
        ax.scatter(wolter1_H.L_s2vrtlf - 2*wolter1_H.c_h, 0, color='orange', label='Focus Point H')
        ax.scatter(wolter1_H.x_intersect2_origin, wolter1_H.y_intersect2_origin, color='red', label='Intersection Point H')
        ax.set_xlabel('major axis (mm)')
        ax.set_ylabel('minor axis (mm)')
        ax.legend()
        ax.grid()
        # plt.savefig("AKB_1D_axis.png", dpi=300)

        wolter_x_combined_V, wolter_y_combined_V = polyfit_wolter1_from_ell_hyp(
            wolter1_V.x_e_array_rotated, wolter1_V.y_e_array_rotated, wolter1_V.x_h_array_rotated, wolter1_V.y_h_array_rotated, degree=20,datapitch=wolter1_V.datapitch)
        print(f"Check datapitch: {wolter_x_combined_V[1] - wolter_x_combined_V[0]} mm, expected: {wolter1_V.datapitch} mm")
        axs_a[5].plot(wolter_x_combined_V, wolter_y_combined_V, label='Combined Wolter1 V', color='black')
        axs_a[5].scatter(wolter1_V.x_e_array_rotated, wolter1_V.y_e_array_rotated, label='Ellipse Rotated V', color='blue')
        axs_a[5].scatter(wolter1_V.x_h_array_rotated, wolter1_V.y_h_array_rotated, label='Hyperbola Rotated V', color='green')
        axs_a[5].set_xlabel('x (mm)')
        axs_a[5].set_ylabel('y (mm)')
        axs_a[5].legend()
        axs_a[5].grid()
        axs_a[5].set_title(f"Combined Wolter1 V Surface")

        wolter_x_combined_H, wolter_y_combined_H = polyfit_wolter1_from_ell_hyp(
            wolter1_H.x_e_array_rotated, wolter1_H.y_e_array_rotated, wolter1_H.x_h_array_rotated, wolter1_H.y_h_array_rotated, degree=20,datapitch=wolter1_H.datapitch)
        print(f"Check datapitch: {wolter_x_combined_H[1] - wolter_x_combined_H[0]} mm, expected: {wolter1_H.datapitch} mm")
        axs_b[5].plot(wolter_x_combined_H, wolter_y_combined_H, label='Combined Wolter1 H', color='black')
        axs_b[5].scatter(wolter1_H.x_e_array_rotated, wolter1_H.y_e_array_rotated, label='Ellipse Rotated H', color='blue')
        axs_b[5].scatter(wolter1_H.x_h_array_rotated, wolter1_H.y_h_array_rotated, label='Hyperbola Rotated H', color='green')
        axs_b[5].set_xlabel('x (mm)')
        axs_b[5].set_ylabel('y (mm)')
        axs_b[5].legend()
        axs_b[5].grid()
        axs_b[5].set_title(f"Combined Wolter1 H Surface")
        print("===== Wolter1 V =====")
        print(f"Check datapitch: {wolter_x_combined_V[1] - wolter_x_combined_V[0]} mm, expected: {wolter1_V.datapitch} mm")
        print(f"Length of mirror: {wolter_x_combined_V[-1] - wolter_x_combined_V[0]} mm")
        print(f"Length of Ellipse: {wolter1_V.x_e_array[-1] - wolter1_V.x_e_array[0]} mm")
        print(f"Length of Hyperbola: {wolter1_V.x_h_array[-1] - wolter1_V.x_h_array[0]} mm")
        print(f"self.L_s2f: {wolter1_V.L_s2f} mm")
        print("===== Wolter1 H =====")
        print(f"Check datapitch: {wolter_x_combined_H[1] - wolter_x_combined_H[0]} mm, expected: {wolter1_H.datapitch} mm")
        print(f"Length of mirror: {wolter_x_combined_H[-1] - wolter_x_combined_H[0]} mm")
        print(f"Length of Ellipse: {wolter1_H.x_e_array[-1] - wolter1_H.x_e_array[0]} mm")
        print(f"Length of Hyperbola: {wolter1_H.x_h_array[-1] - wolter1_H.x_h_array[0]} mm")
        print(f"self.L_s2f: {wolter1_H.L_s2f} mm")
        canvas_a.draw() # GUIを更新
        canvas_b.draw() # GUIを更新
        # plt.show()
        if options["data_output"]:
            canvas_a.figure.savefig(f"Results_AKB_1D_axis_V.png", dpi=300)
            canvas_b.figure.savefig(f"Results_AKB_1D_axis_H.png", dpi=300)
        if options["raytrace"]:
            calc_AKB_raytrace_1_1(wolter1_V, wolter1_H)
            plt.show()
        print("完了", "Wolter1の実行が完了しました。")
        
    except ValueError as e:
        error_detail = traceback.format_exc()
        # コンソールにすべて出力
        print("--- Detailed Error Report ---")
        print(error_detail)
        messagebox.showerror("入力エラー", f"すべての項目に数値を入力してください。")
        print("ValueError:", e)
    except Exception as e:
        # 2. 詳細なエラー情報を文字列として取得
        error_detail = traceback.format_exc()
        # コンソールにすべて出力
        print("--- Detailed Error Report ---")
        print(error_detail)
        messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{e}")

def run_Wolter3(axs=None, canvas=None):
    try:
        params, options = get_params_from_entries_opt_vars(entries, opt_vars)
        # Wolter3のインスタンス化と実行
        clear_axs(axs, canvas)
        wolter3 = Wolter3(
            params["p_e_3"], 
            params["q_e_3"], 
            params["theta_e_3"], 
            params["l_e_3"], 
            params["p_h_3"], 
            params["q_h_3"], 
            datapitch=params["datapitch_3"],
            sagittal=0,
            margin_inside=0,
            margin_outside=0
        )
        wolter3.calc_1D_axis(axs, canvas)

        ax = axs[4]
        ax.plot(wolter3.x_e_array, wolter3.y_e_array, label='Ellipse', color='blue')
        ax.scatter(wolter3.x_e, wolter3.y_e, color='red', label='Ellipse Center Point')
        ax.plot(wolter3.x_h_array, wolter3.y_h_array, label='Hyperbola', color='green')
        ax.scatter(wolter3.x_h, wolter3.y_h, color='purple', label='Hyperbola Center Point')
        ax.scatter([0], [0], color='black', label='Object Point')
        ax.scatter(wolter3.L_s2f, 0, color='orange', label='Image Point')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.legend()
        ax.grid()
        ax.set_title(f"WD = {wolter3.wd:.2f} mm, Mag = {wolter3.m_u:.2f} - {wolter3.m_l:.2f} x")
        print("===== Wolter3 =====")
        print(f"Length of Ellipse: {wolter3.x_e_array[-1] - wolter3.x_e_array[0]} mm")
        print(f"Length of Hyperbola: {wolter3.x_h_array[-1] - wolter3.x_h_array[0]} mm")
        
        coeffs_theta = np.polyfit(wolter3.x_h_array, wolter3.theta2_array, deg=20)
        theta2_at_center = np.polyval(coeffs_theta, (wolter3.x_h_array[-1] + wolter3.x_h_array[0])/2)
        print(f"Grazing angle of Hyperbola center: {theta2_at_center} rad")

        coeffs_y_h = np.polyfit(wolter3.x_h_array, wolter3.y_h_array, deg=20)
        x_h_at_center = (wolter3.x_h_array[-1] + wolter3.x_h_array[0])/2
        y_h_at_center = np.polyval(coeffs_y_h, x_h_at_center)
        p_h_at_center = np.sqrt((x_h_at_center - wolter3.L_s2vrtlf)**2 + y_h_at_center**2)
        q_h_at_center = np.sqrt((x_h_at_center - wolter3.L_s2f)**2 + y_h_at_center**2)
        print(f"At Hyperbola center: x = {x_h_at_center} mm, y = {y_h_at_center} mm, p_h = {p_h_at_center} mm, q_h = {q_h_at_center} mm")
        index = np.argmin(np.abs(wolter3.x_h_array - x_h_at_center))
        print(f"At Hyperbola center (from array): x = {wolter3.x_h_array[index]} mm, y = {wolter3.y_h_array[index]} mm, p_h = {wolter3.p_h_array[index]} mm, q_h = {wolter3.q_h_array[index]} mm")
        print(f"At Ellipse reflection point: x = {wolter3.x_e_array[index]} mm, y = {wolter3.y_e_array[index]} mm, p_e = {wolter3.p_e_array[index]} mm, q_e = {wolter3.q_e_array[index]} mm")
        # plt.savefig("wolter1_1D_axis.png", dpi=300)
        if options["data_output"]:
            ### wolter_x_combinedの保存
            np.savetxt(f"wolter3_ell_surface_1D_datapitch{wolter3.datapitch}.csv", np.column_stack((wolter3.x_e_array_final, wolter3.y_e_array_final)), delimiter=",", header="x_mm,y_mm", comments="")
            np.savetxt(f"wolter3_hyp_surface_1D_datapitch{wolter3.datapitch}.csv", np.column_stack((wolter3.x_h_array_final, wolter3.y_h_array_final)), delimiter=",", header="x_mm,y_mm", comments="")
            ### グラフの保存
            # fig = axs[0].get_figure()
            canvas.figure.savefig(f"Results_wolter3_1D_axis.png", dpi=300)
        if canvas is not None:
            canvas.draw() # GUIを更新
        if options["raytrace"]:
            wolter3.calc_1D_axis_raytrace()
            plt.show()
        print("完了", "Wolter3の実行が完了しました。")
        
    except ValueError as e:
        messagebox.showerror("入力エラー", f"すべての項目に数値を入力してください。")
        print("ValueError:", e)
    except Exception as e:
        # 2. 詳細なエラー情報を文字列として取得
        error_detail = traceback.format_exc()
        # コンソールにすべて出力
        print("--- Detailed Error Report ---")
        print(error_detail)
        messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{e}")

# def run_wolter_intersection(axs=None, canvas=None):
#     try:
#         params, options = get_params_from_entries_opt_vars(entries, opt_vars)
#         a_e = params["a_e"]
#         b_e = params["b_e"]
#         l_e_int = params["l_e_int"]
#         a_h = params["a_h"]
#         b_h = params["b_h"]
#         l_h_int = params["l_h_int"]
#         datapitch = params["datapitch"]
#         c_e = np.sqrt(a_e**2 - b_e**2)
#         c_h = np.sqrt(a_h**2 + b_h**2)
#         l_s2vrtlf = 2*c_e
#         l_s2f = 2*c_e - 2*c_h
#         def ellipse(x, a, b, x0):
#             return b * np.sqrt(1 - ((x - x0) / a)**2)
#         def hyperbola(x, a, b, x0):
#             return b * np.sqrt(((x - x0) / a)**2 - 1)
#         x_common = np.linspace(0,l_s2f, int(l_s2f/datapitch))

#         y_ell = ellipse(x_common, a_e, b_e, c_e)
#         y_hyp = hyperbola(x_common, a_h, b_h, 2*c_e -c_h)

#         x_intersection = x_common[np.argmin(np.abs(y_ell - y_hyp))]
#         x_start = x_intersection - l_e_int
#         x_end = x_intersection + l_h_int
#         x_mirror = np.linspace(x_start, x_end, int((x_end - x_start)/datapitch))
#         y_mirror_ell = ellipse(x_mirror, a_e, b_e, c_e)
#         y_mirror_hyp = hyperbola(x_mirror, a_h, b_h, 2*c_e -c_h)
#         y_mirror_combined = np.where(x_mirror <= x_intersection, y_mirror_ell, y_mirror_hyp)

#         rot_angle = -np.arctan((y_mirror_combined[-1] - y_mirror_combined[0]) / (x_mirror[-1] - x_mirror[0]))
#         x_mirror_rotated, y_mirror_rotated = f.rotation_2D(x_mirror, y_mirror_combined, rot_angle)
#         x_offset = x_mirror_rotated[0]
#         y_offset = y_mirror_rotated[0]
#         x_mirror_rotated -= x_offset
#         y_mirror_rotated -= y_offset
#         y_mirror_rotated = -y_mirror_rotated
#         plt.figure(figsize=(8, 6))
#         plt.plot(x_mirror_rotated, y_mirror_rotated, label='Combined Mirror', color='black')
#         plt.xlabel('x (mm)')
#         plt.ylabel('y (mm)')

#         np.savetxt("wolter1_mirror_profile.csv", np.column_stack((x_mirror_rotated, y_mirror_rotated)), delimiter=",", header="x_mm,y_mm", comments="")



#         print(f"Intersection Point: x = {x_intersection:.2f} mm, y = {ellipse(x_intersection, a_e, b_e, c_e):.2f} mm")
        
#         plt.figure(figsize=(8, 6))
#         plt.plot(x_common, y_ell, label='Ellipse', color='blue', linestyle='--')
#         plt.plot(x_common, y_hyp, label='Hyperbola', color='green', linestyle='--')
#         plt.plot(x_mirror, y_mirror_combined, label='Combined Mirror', color='black')
#         plt.scatter(x_intersection, ellipse(x_intersection, a_e, b_e, c_e), color='red', label='Intersection Point')
#         plt.xlabel('x (mm)')
#         plt.ylabel('y (mm)')
#         plt.legend()
#         plt.grid()
#         plt.show()
#         print("完了", "run_wolter_intersectionの実行が完了しました。")
        
#     except ValueError as e:
#         messagebox.showerror("入力エラー", f"すべての項目に数値を入力してください。")
#         print("ValueError:", e)
#     except Exception as e:
#         # 2. 詳細なエラー情報を文字列として取得
#         error_detail = traceback.format_exc()
#         # コンソールにすべて出力
#         print("--- Detailed Error Report ---")
#         print(error_detail)
#         messagebox.showerror("エラー", f"予期せぬエラーが発生しました:\n{e}")

# グラフを閉じる関数
def close_plots():
    plt.close('all')
def save_settings():
    # 各Entryから現在の値を取得して辞書を作成
    settings = {key: entry.get() for key, entry in entries.items()}
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="設定を保存"
    )
    
    if file_path:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4, ensure_ascii=False)
            messagebox.showinfo("保存", "設定を保存しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました: {e}")

def load_settings():
    file_path = filedialog.askopenfilename(
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        title="設定を読み込む"
    )
    
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # 読み込んだ値を各Entryに反映
            for key, value in settings.items():
                if key in entries:
                    entries[key].delete(0, tk.END)
                    entries[key].insert(0, value)
            
            messagebox.showinfo("読み込み", "設定を読み込みました。")
        except Exception as e:
            messagebox.showerror("エラー", f"読み込みに失敗しました: {e}")
def clear_axs(axs, canvas):
    reset_6plots_layout(axs, canvas)
    # """
    # axs: 初期化時に作成したメインの4つの軸のリスト [ax1, ax2, ax3, ax4]
    # canvas: 描画先のキャンバス
    # """
    # for ax in axs:
    #     # 1. フィギュアがまだ存在するか確認（重要！）
    #     if ax is not None and ax.figure is not None:
    #         # 2. メイン軸をクリア（タイトルやプロットを消す）
    #         ax.clear()
            
    #         # 3. twinx() などで後から追加された「外側の軸」だけを削除する
    #         # ax.figure.axes には、そのフィギュアに乗っている全ての軸が入っている
    #         for extra_ax in list(ax.figure.axes):
    #             # 元々のメインの4つの軸に含まれていない軸（twinxで増えた軸）なら削除
    #             if extra_ax not in axs:
    #                 extra_ax.remove()
                    
    # # 4. 最後に一度だけ再描画
    # canvas.draw()

def reset_6plots_layout(axs, canvas):
    """
    既存のCanvas内のFigureをクリアし、2x3のレイアウトを再作成する
    axs: 既存の軸リスト（中身を入れ替えます）
    canvas: 既存のキャンバス
    """
    fig = canvas.figure
    fig.clear()  # Figureの中身（以前の軸やカラーバー）を完全に消去
    
    # リストの中身をクリア（参照を維持するため list.clear() を使用）
    axs.clear()
    
    # 2x3のグラフを新しく作成して追加
    for i in range(1, 7):
        ax = fig.add_subplot(2, 3, i)
        ax.set_title(f"Plot {i}")
        axs.append(ax)
    
    # レイアウトの微調整
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # 描画の反映
    canvas.draw()
# 共通オプション定義
calc_options = [("raytrace", False), ("data_output", False)]

def polyfit_wolter1_from_ell_hyp(x_ell, y_ell, x_hyp, y_hyp, degree=20,datapitch=0.1):
    # 1. 楕円と双曲線の中心点を計算
    p_ell = np.polyfit(x_ell, y_ell, degree)
    p_hyp = np.polyfit(x_hyp, y_hyp, degree)
    # 2. 2つのデータをくっつけ datapitchに基づいてx軸を再生成
    x_combined = np.arange(min(min(x_ell), min(x_hyp)), max(max(x_ell), max(x_hyp)), datapitch)
    y_ell_fit = np.polyval(p_ell, x_combined)
    y_hyp_fit = np.polyval(p_hyp, x_combined)
    # 3. 2つのフィット曲線の高い方を取る（これがWolter1の形状になる）
    y_wolter1 = np.maximum(y_ell_fit, y_hyp_fit)
    return x_combined, y_wolter1
    
    

# --- グラフエリア（6つ）作成用ヘルパー ---
def create_6plots_area(parent_frame):
    # 2x3のグラフを持つFigureを作成
    fig = Figure(figsize=(8, 6), dpi=80)
    fig.subplots_adjust(hspace=0.4, wspace=0.4) # グラフ間の隙間調整
    
    axs = []
    axs.append(fig.add_subplot(2, 3, 1))
    axs.append(fig.add_subplot(2, 3, 2))
    axs.append(fig.add_subplot(2, 3, 3))
    axs.append(fig.add_subplot(2, 3, 4))
    axs.append(fig.add_subplot(2, 3, 5))
    axs.append(fig.add_subplot(2, 3, 6))
    
    for i, ax in enumerate(axs):
        ax.set_title(f"Plot {i+1}")
        
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side="top", fill="both", expand=True)
    
    toolbar = NavigationToolbar2Tk(canvas, parent_frame)
    toolbar.update()
    
    return axs, canvas

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Wolter1 パラメータ入力 & 4連グラフ表示")
    root.geometry("1800x900")

    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill="both", padx=10, pady=10)

    # タブ用のフレーム作成
    tab_basic = ttk.Frame(notebook)
    tab_advanced = ttk.Frame(notebook)
    tab_autodesign = ttk.Frame(notebook)
    tab_wolter3 = ttk.Frame(notebook)

    notebook.add(tab_basic, text=" 基本パラメータ ")
    notebook.add(tab_advanced, text=" AKBパラメータ ")
    notebook.add(tab_autodesign, text=" 自動設計 ")
    notebook.add(tab_wolter3, text="Wolter3 楕円基準")
    # notebook.add(tab_wolter_intersection, text="楕円双曲の交点からの設計")

    entries = {}

    def create_fields(frame, field_list):
        for i, (label_text, default_val) in enumerate(field_list):
            tk.Label(frame, text=label_text).grid(row=i, column=0, padx=10, pady=5, sticky="e")
            entry = tk.Entry(frame)
            entry.insert(0, default_val)
            entry.grid(row=i, column=1, padx=10, pady=5)
            field_key = label_text.split(" ")[0].strip().replace(":", "")
            entries[field_key] = entry

    def setup_tab_layout(tab_frame, fields):
        # 左右分割：左側にパラメータ、右側にグラフエリア
        left_f = tk.Frame(tab_frame)
        left_f.pack(side="left", fill="y", padx=10, pady=10)
        right_f = tk.Frame(tab_frame, bg="white")
        right_f.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        
        create_fields(left_f, fields)
        axs, canvas = create_6plots_area(right_f)
        return left_f, axs, canvas

    # --- 各タブのセットアップ ---
    fields_basic = [
        ("p_e or L1 (mm):", "72170"), ("q_e or L2 (mm):", "1680"), ("theta_e (mrad):", "3.35"),
        ("l_e (長さ mm):", "135"), ("p_h or L2 - L3 (mm):", "1524.7"), ("q_h or L4 (mm):", "780"),
        ("datapitch (mm):", "0.1"), ("sagittal (mm):", "3"),
        ("margin_inside (mm):", "0"), ("margin_outside (mm):", "0")
    ]
    left_basic, axs_basic, canvas_basic = setup_tab_layout(tab_basic, fields_basic)

    fields_advanced = [
        ("p_e_H or L1 (mm):", "72478"), ("q_e_H or L2 (mm):", "1058"), ("theta_e_H (mrad):", "3.35"),
        ("l_e_H (長さ mm):", "135"), ("p_h_H or L2 - L3 (mm):", "902"), ("q_h_H or L4 (mm):", "465"),
    ]
    left_adv, axs_adv, canvas_adv = setup_tab_layout(tab_advanced, fields_advanced)

    fields_autodesign = [
        ("p1 (mm):", "72000"), ("m1 (倍率):", "87"), ("Length1 (長さ mm):", "275"),
        ("theta1 (mrad):", "1.77"), ("Length2 (長さ mm):", "274"), ("theta2 (mrad):", "1.73"),
        ("spacing (mm):", "30")
    ]
    left_auto, axs_auto, canvas_auto = setup_tab_layout(tab_autodesign, fields_autodesign)
    fields_wolter3 = [
        ("p_e_3 (mm):", "95"), ("q_e_3 (mm):", "170.3422156"), ("theta_e_3 (mrad):", "9"),
        ("l_e_3 (長さ mm):", "30"), ("p_h_3 (mm):", "61.8275"), ("q_h_3 (mm):", "896.494"),
        ("datapitch_3 (mm):", "0.1")
    ]
    left_wolter3, axs_wolter3, canvas_wolter3 = setup_tab_layout(tab_wolter3, fields_wolter3)
    # fields_wolter_intersection = [
    #     ("a_e (mm):", "60523"), ("b_e (mm):", "61.60"), ("l_e_int (交点からの長さ mm):", "100"),
    #     ("a_h (mm):", "241.32"), ("b_h (mm):", "3.77"), ("l_h_int (交点からの長さ mm):", "88")
    # ]
    # left_wolter_intersection, axs_wolter_intersection, canvas_wolter_intersection = setup_tab_layout(tab_wolter_intersection, fields_wolter_intersection)
    # 各タブの実行ボタン（例として配置）
    # tk.Button(left_basic, text="Wolter1 実行", command=run_wolter1_1D, bg="lightblue", width=25).grid(row=15, column=0, columnspan=2, pady=10)
    tk.Button(
        left_basic, 
        text="Wolter1 1D 実行",
        command=lambda: run_wolter1_1D(axs_basic, canvas_basic),
        bg="lightblue",
        width=25
    ).grid(row=15, column=0, columnspan=2, pady=10)
    # tk.Button(left_basic, text="Wolter1 2D 実行", command=run_wolter1_2D, bg="lightcyan", width=25).grid(row=16, column=0, columnspan=2, pady=10)
    tk.Button(
        left_basic, 
        text="Wolter1 2D 実行", 
        command=lambda: run_wolter1_2D(axs_basic, canvas_basic), 
        bg="lightcyan", 
        width=25
    ).grid(row=16, column=0, columnspan=2, pady=10)
    # ボタンの command で target_axs と target_canvas を渡す
    tk.Button(
        left_basic, 
        text="Ellipse 実行", 
        command=lambda: run_Ellipse(axs_basic, canvas_basic), 
        bg="lightyellow", 
        width=25
    ).grid(row=17, column=0, columnspan=2, pady=10)

    # tk.Button(left_adv, text="AKB 実行", command=run_AKB, bg="lightgreen", width=25).grid(row=10, column=0, columnspan=2, pady=10)
    tk.Button(
        left_adv, 
        text="AKB 実行", 
        command=lambda: run_AKB(axs_basic, canvas_basic, axs_adv, canvas_adv), 
        bg="lightgreen", 
        width=25
    ).grid(row=10, column=0, columnspan=2, pady=10)

    tk.Button(
        left_adv, 
        text="KB 実行", 
        command=lambda: run_KB(axs_basic, canvas_basic, axs_adv, canvas_adv), 
        bg="lightpink", 
        width=25
    ).grid(row=11, column=0, columnspan=2, pady=10)

    tk.Button(
        left_auto, 
        text="自動設計 開始", 
        command=lambda: run_autoKB(axs_auto, canvas_auto), 
        bg="khaki", 
        width=25
    ).grid(row=10, column=0, columnspan=2, pady=10)
    
    tk.Button(
        left_wolter3, 
        text="Wolter3 実行", 
        command=lambda: run_Wolter3(axs_wolter3, canvas_wolter3), 
        bg="lightblue", 
        width=25
    ).grid(row=17, column=0, columnspan=2, pady=10)
    # tk.Button(left_wolter_intersection, text="Wolter 交点設計 実行", command=lambda: run_wolter_intersection(axs_wolter_intersection, canvas_wolter_intersection), bg="lightblue", width=25).grid(row=10, column=0, columnspan=2, pady=10)

    # --- 下部：共通操作エリア ---
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(fill="x", side="bottom", pady=10)

    # グラフ操作とオプション
    graph_ops_frame = tk.Frame(bottom_frame)
    graph_ops_frame.pack()
    tk.Button(graph_ops_frame, text="グラフをすべて閉じる", command=close_plots, bg="salmon", width=20).pack(side="left", padx=10)

    opt_vars = {}
    for label, default in calc_options:
        var = tk.BooleanVar(value=default)
        tk.Checkbutton(graph_ops_frame, text=label, variable=var).pack(side="left", padx=5)
        opt_vars[label] = var

    # ファイル操作
    file_ops_frame = tk.Frame(bottom_frame)
    file_ops_frame.pack(pady=5)
    tk.Button(file_ops_frame, text="設定を保存", command=save_settings, width=15).pack(side="left", padx=5)
    tk.Button(file_ops_frame, text="設定を読込", command=load_settings, width=15).pack(side="left", padx=5)

    root.mainloop()