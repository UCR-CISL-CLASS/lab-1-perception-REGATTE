import glob
import os
import sys
import time
import pygame
import numpy as np
import carla
import argparse
import logging
import random
import cv2
import queue

# Importing Detector class from detector.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from detector import Detector  # Ensure detector.py is in the same directory or adjust the path

# Initialize pygame
pygame.init()
display = None

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()

# Edge pairs for bounding box drawing
edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def process_image(image, world, ego_vehicle, K, world_2_camera):
    """Callback function to process and display camera images in pygame and OpenCV with bounding boxes."""
    global display

    # Convert CARLA image to pygame format
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # BGRA format
    img = array[:, :, :3]  # Remove alpha channel

    # Draw bounding boxes
    for npc in world.get_actors().filter('*vehicle*'):
        if npc.id != ego_vehicle.id:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(ego_vehicle.get_transform().location)
            if dist < 50:
                forward_vec = ego_vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - ego_vehicle.get_transform().location
                if forward_vec.dot(ray) > 1:
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    for edge in edges:
                        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                        p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                        cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 1)

    # Create a pygame surface and display it
    surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    if display is None:
        display = pygame.display.set_mode((image.width, image.height))
    display.blit(surface, (0, 0))
    pygame.display.flip()  # Update the pygame display

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        ego_vehicle = None
        ego_cam = None

        # Enable synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True  # Enable synchronous mode
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)

        # --------------
        # Spawn ego vehicle
        # --------------
        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'ego')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color', ego_color)
        spawn_points = world.get_map().get_spawn_points()
        if spawn_points:
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
            print('Ego vehicle spawned')

        # Enable autopilot
        ego_vehicle.set_autopilot(True)

        # --------------
        # Use camera from detector
        # --------------
        detector = Detector()  # Assuming detector defines sensor setup
        sensors = detector.sensors()

        for sensor in sensors:
            if sensor['id'] == 'Center':
                camera_bp = world.get_blueprint_library().find(sensor['type'])
                camera_bp.set_attribute('image_size_x', str(sensor['width']))
                camera_bp.set_attribute('image_size_y', str(sensor['height']))
                camera_bp.set_attribute('fov', str(sensor['fov']))
                camera_transform = carla.Transform(
                    carla.Location(x=sensor['x'], y=sensor['y'], z=sensor['z']),
                    carla.Rotation(pitch=sensor['pitch'], yaw=sensor['yaw'], roll=sensor['roll'])
                )
                ego_cam = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
                ego_cam.listen(image_queue.put)

        # Camera intrinsic matrix setup
        image_w = camera_bp.get_attribute('image_size_x').as_int()  # Use as_int for integers
        image_h = camera_bp.get_attribute('image_size_y').as_int()  # Use as_int for integers
        fov = camera_bp.get_attribute('fov').as_float()  # Use as_float for floating point numbers
        K = build_projection_matrix(image_w, image_h, fov)

        # --------------
        # Main loop to visualize the camera feed with bounding boxes
        # --------------
        while True:
            world_snapshot = world.wait_for_tick()
            image = image_queue.get()

            # Get the world to camera matrix
            world_2_camera = np.array(ego_cam.get_transform().get_inverse_matrix())

            process_image(image, world, ego_vehicle, K, world_2_camera)

            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

    finally:
        # Restore settings and cleanup
        settings.synchronous_mode = False
        world.apply_settings(settings)
        client.stop_recorder()
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            ego_vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done with simulation')
