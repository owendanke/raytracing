import numpy as np
from vedo import Plotter, Mesh, Arrow, Point
import rt_render
import matplotlib.pyplot as plt
from trimesh import transformations

def v_plot(mesh):
    vplt = Plotter(bg='blackboard', size=['1920', '1080']).add_shadows()
    vplt.show(Mesh(mesh), camera_model, light_model)

def render_iter(mesh, camera_pos):
    material = rt_render.Mesh_mat(color=[0.1, 0.3, 1], ka=0.5, kd=0.3, ks=0.8, shininess=100)
    cam = rt_render.Camera(camera_pos, 640, 480)
    cam.load_mesh_vedo(mesh)
    cam.set_light([5, 3, 0], 1.0, [1, 1, 1])
    image = cam.render_scene_iterative(material, 1)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def render_vector(mesh, camera_pos):
    rotations = list()
    material = rt_render.Mesh_mat(color=[200, 30, 30], ka=0.9, kd=0.12, ks=0.5, shininess=40)
    cam = rt_render.Camera(camera_pos, 3840, 2160)
    cam.load_mesh_trimesh(mesh)
    cam.set_light([0, -2, 15], 0.4, [1, 1, 1])

    if mesh == "models/zigul_v1.stl":
        rotations.append(transformations.rotation_matrix(-1, [0, 0, 1], cam.mesh_data.center_mass))
        rotations.append(transformations.rotation_matrix(-1.4, [0, 1, 0], cam.mesh_data.center_mass))
    else:
        rotations.clear()

    cam.render_scene_vector(material, rotations)


if __name__ == "__main__":
    mesh = "models/vaz2106.obj"
    camera_pos = [4, 0, 4]
    camera_model = Arrow(camera_pos, [0,0,0], c='red5').scale(0.2)
    light_model = Point([5, 3, 0], c='b').scale(5)

    while(True):
        print("Choices:\n (0) Choose model\n (1) View scene\n (2) Iterative Render\n (3) Vector Render\n (4) Exit\n")
        choice = input()
        if choice =="0":
            print("Path to mesh: ")
            mesh = input()
        elif choice == "1":
            v_plot(mesh)
        elif choice == "2":
            render_iter(mesh, camera_pos)
        elif choice == "3":
            render_vector(mesh, camera_pos)
        elif choice == "4":
            break
        else:
            print("Please choose a valid option")