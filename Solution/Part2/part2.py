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

# Importing Detector class from detector.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from detector import Detector  # Ensure detector.py is in the same directory or adjust the path

# Initialize pygame
pygame.init()
display = None

def process_image(image):
    """Callback function to process and display camera images in pygame."""
    global display

    # Convert CARLA image to pygame format
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # BGRA format
    array = array[:, :, :3]  # Remove alpha channel

    # Create a pygame surface and display it
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    if display is None:
        display = pygame.display.set_mode((image.width, image.height))

    display.blit(surface, (0, 0))
    pygame.display.flip()  # Update the pygame display

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        ego_vehicle = None
        ego_cam = None

        # --------------
        # Spawn ego vehicle
        # --------------
        ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'ego')
        print('\nEgo role_name is set')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color', ego_color)
        print('\nEgo color is set')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_spawn_points > 0:
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
            print('\nEgo is spawned')
        else:
            logging.warning('Could not find any spawn points')

        # --------------
        # Use camera placement from detector.py
        # --------------
        detector = Detector()  # Create an instance of Detector
        sensors = detector.sensors()  # Get the sensor configuration

        # Extract the sensor configuration for the left camera
        for sensor in sensors:
            if sensor['id'] == 'Center':  # Visualize only the left camera
                camera_bp = world.get_blueprint_library().find(sensor['type'])
                camera_bp.set_attribute('image_size_x', str(sensor['width']))
                camera_bp.set_attribute('image_size_y', str(sensor['height']))
                camera_bp.set_attribute('fov', str(sensor['fov']))
                
                # Set camera transformation based on detector.py values
                camera_transform = carla.Transform(
                    carla.Location(x=sensor['x'], y=sensor['y'], z=sensor['z']),
                    carla.Rotation(pitch=sensor['pitch'], yaw=sensor['yaw'], roll=sensor['roll'])
                )
                
                # Spawn the left camera attached to the ego vehicle
                ego_cam = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
                print(f"\nCamera {sensor['id']} attached to the vehicle")

                # Start listening to the camera feed
                ego_cam.listen(lambda image: process_image(image))

        # Main loop to keep the pygame window open and handle events
        while True:
            world_snapshot = world.wait_for_tick()
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

    finally:
        client.stop_recorder()
        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            ego_vehicle.destroy()
        pygame.quit()  # Make sure to quit pygame when the script ends

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')
