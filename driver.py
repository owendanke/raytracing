import numpy as np
from vedo import Plotter, Mesh, Arrow, Point
from rt_render import Camera, Trimesh
import matplotlib.pyplot as plt



#

#
#plt.show(model)

def v_plot(mesh):
    vplt = Plotter(bg='blackboard', size=['1920', '1080']).add_shadows()
    vplt.show(Mesh(mesh), camera_model, light_model)

def render(mesh, camera_pos, light_pos):
    material = Trimesh(color=[0.1, 0.3, 1], ka=0.5, kd=0.3, ks=0.8, shininess=100)
    cam = Camera(camera_pos, 1280, 720)
    cam.load_mesh(mesh)
    cam.set_light(light_pos, 1.0, [1, 1, 1])
    image = cam.render_scene(material, 1)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    mesh = "models/vaz2106.obj"
    camera_pos = [4, 0, 4]
    camera_model = Arrow(camera_pos, [0,0,0], c='red5').scale(0.2)
    light_pos = [5, 3, 0]
    light_model = Point([5, 3, 0], c='b').scale(5)

    while(True):
        print("Choices:\n (0) Choose model\n (1) View scene\n (2) Render scene\n (3) Exit\n")
        choice = input()
        if choice =="0":
            print("Path to mesh: ")
            mesh = input()
        elif choice == "1":
            v_plot(mesh)
        elif choice == "2":
            render(mesh, camera_pos, light_pos)
        elif choice == "3":
            break
        else:
            print("Please choose a valid option")