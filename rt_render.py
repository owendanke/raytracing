from abc import ABC

import numpy as np
from random import random

import vedo.utils as vu
from vedo import Mesh, pointcloud

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def clamp(x, min, max):
    '''
    Clamps an input between min and max values
    :param x: value to clamp
    :param min: lower bound
    :param max: upper bound
    :return: min <= x <= max
    '''
    if x.all() < min:
        return min
    if x.all() > max:
        return max
    return x


class Matter(object):
    def go(self):
        raise NotImplementedError("Please Implement this method")


class Trimesh(Matter, ABC):
    def __init__(self, color, ka, kd, ks, shininess):
        self.color = np.array(color)    # RGB color
        self.ka = ka  # Ambient coefficient
        self.kd = kd  # Diffuse coefficient
        self.ks = ks  # Specular coefficient
        self.shininess = shininess  # he shininess factor for specular highlights

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def pos(self, t):
        return np.array(self.origin + (t * self.direction))


class Camera:
    def __init__(self, camera_position, image_width, image_height):
        # Camera Definitions
        self.camera_position = camera_position
        self.image_width = image_width
        self.image_height = image_height
        self.direction = None
        self.focus_distance = None
        self.background_color = [0.3, 0.3, 0.3]

        # Image plane definitions
        self.image_plane_height = 2.0
        aspect_ratio = image_width / image_height
        self.image_plane_width = aspect_ratio * self.image_plane_height
        self.image_plane_dist = 1.0

        self.image_sample = 1
        self.image_height_scaled = int(self.image_height // self.image_sample)
        self.image_width_scaled = int(self.image_width // self.image_sample)

        # Light Definitions
        self.light_position = None
        self.light_intensity = None
        self.light_color = None

    def blinn_phong(self, cell_id, hit_point, matter, light_source: np.ndarray, light_color: np.ndarray, light_intensity):
        # compute vectors
        normal = np.array(normalize(hit_point - self.mesh_data.cell_normals[cell_id]))
        light_direction = np.array(normalize(light_source - hit_point))
        half_vector = np.array(normalize(self.camera_position + light_source))

        # ambient component
        ambient = matter.ka * matter.color * light_color * light_intensity
        # diffuse component
        diffuse = matter.kd * light_intensity * max(0, np.dot(normal, light_direction)) * light_color

        # specular component
        specular = matter.ks * light_intensity * light_color * pow(max(0, np.dot(normal, half_vector)), matter.shininess)

        return ambient + diffuse + specular

    def closest(self, point):
        '''
        Finds cell_id nearest the point
        :param point:
        :return: ndarray
        '''
        nearest = self.mesh_data.closest_point(point, return_point_id=True)
        return nearest

    def get_ray_perspective(self, x, y):
        '''
        Constructs a Ray originating from camera and pointing towards pixel (x, y)
        :param x: pixel x value
        :param y: pixel y value
        :return: Ray object
        '''
        # standard basis vectors of image plane
        u_b = np.array([self.image_plane_width, 0, 0])
        v_b = np.array([0, self.image_plane_height, 0])
        w_b = np.array([0, 0, self.image_plane_dist])

        # pixel positions on image plane
        u = u_b / self.image_width_scaled
        v = v_b / self.image_height_scaled

        index_pixel = self.camera_position - w_b - (u_b / 2) - (v_b / 2) + (0.5 * (u + v))
        offset = self.rand_sample()
        sampled_pixels = index_pixel + ((x + offset[0]) * u) + ((y + offset[1]) * v)

        # define ray's origin and direction
        ray_origin = self.camera_position
        # ray_direction = index_pixel + (x * u) + (y * v) # normal ray
        ray_direction = sampled_pixels - ray_origin

        return Ray(ray_origin, ray_direction)

    def gradient(self, ray_direction: np.array, color_1: np.array, color_2: np.array):
        '''
        Linear gradient from color_1 to color_2 based on ray_direction
        :param ray_direction: nparray
        :param color_1: nparray of rgb values
        :param color_2: nparray of rgb values
        :return:
        '''
        # find unit vector direction
        unit_direction = normalize(ray_direction)
        # create a blending coefficient for color gradient
        blend = 0.5 * (unit_direction[1] + 1)
        # return a
        return (1 - blend) * color_1 + blend * color_2

    def lambert(self, hit_point, sphere, light_source: np.array, light_intensity):
        normal = normalize(hit_point - sphere.center)
        light_direction = normalize(light_source - hit_point)
        l = sphere.kd * light_intensity * max(0, np.dot(normal, light_direction))
        return sphere.color * l

    def load_mesh(self, filename):
        '''
        Load mesh using Vedo
        :param filename:
        :return:
        '''
        self.mesh_data = Mesh(filename).scale(0.1, reset=True, origin=False)
        self.mesh_data.rotate_x(180)
        self.mesh_data.compute_normals(cells=True, points=False)

    def ray_intersect(self, direction):
        '''
        Finds any intersections with the loaded mesh to
        :param direction: ndarray
        :return: ndarray | tuple[ndarray, ndarray]
        '''
        return self.mesh_data.intersect_with_line(self.camera_position, direction, return_ids=True)

    def rand_sample(self):
        '''
        vector to a random point in [-.5,-.5]-[+.5,+.5] unit square.
        :return: ndarray of size [3,]
        '''
        return np.array([random() - 0.5, random() - 0.5, 0])

    def render_scene(self, obj_mat, aa_samples):
        '''
        Render the scene
        :param obj_mat:
        :param aa_samples:
        :return: ndarray of size img_height x img_width x rgb
        '''
        # Initial pixel colors of the scene (final output image)
        pixel_colors = np.zeros((self.image_height_scaled, self.image_width_scaled, 3))

        # check aa_samples for correct loop
        if aa_samples <= 0:
            aa_samples = 1

        # loop through every pixel
        for y in range(self.image_height_scaled):
            for x in range(self.image_width_scaled):

                for sample in range(aa_samples):
                    # create a new Ray
                    ray = self.get_ray_perspective(x, y)

                    # get ray-mesh intersection with vedo
                    cell_pos, cell_id = self.ray_intersect(ray.direction)

                    # ray intersects mesh
                    if 0 < len(cell_id):
                        # find the closest point to the camera on the ray
                        distance, closest_pt = vu.closest(self.camera_position, cell_pos)

                        # given a closest point, find the closest cell_id
                        cell_id = self.mesh_data.closest_point(closest_pt, return_cell_id=True)

                        # compute shading for mesh
                        shading = self.blinn_phong(
                            cell_id, closest_pt, obj_mat, self.light_position, self.light_color, self.light_intensity)

                        # update pixel rgb value to shaded color
                        pixel_colors[y][x] = clamp(shading, 0, 1)
                    else:
                        # update pixel rgb value to background color
                        pixel_colors[y][x] = self.background_color

        # return the image
        return pixel_colors

    def set_light(self, light_position, light_intensity, light_color):
        '''
        Change the lighting of the scene
        :param light_position:
        :param light_intensity:
        :param light_color:
        :return: None
        '''
        self.light_position = np.array(light_position)
        self.light_intensity = light_intensity
        self.light_color = np.array(light_color)