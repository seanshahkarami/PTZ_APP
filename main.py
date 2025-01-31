import os
import sys
import argparse
from PIL import Image
from source.bring_data import center_and_maximize_object, get_image_from_ptz_position, publish_images
from source.object_detector import DetectorFactory

def get_argparser():
    parser = argparse.ArgumentParser("PTZ APP")
    parser.add_argument(
        "-ki",
        "--keepimages",
        action="store_true",
        help="Keep collected images in persistent folder for later use",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        help="An integer with the number of iterations (PTZ rounds) to be run (default=5).",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-obj",
        "--objects",
        help="Objects to capture with the camera (comma-separated, e.g., 'person,car,dog')",
        type=str,
        default="person",
    )
    parser.add_argument(
        "-un",
        "--username",
        help="The username of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-pw",
        "--password",
        help="The password of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-ip", 
        "--cameraip", 
        help="The ip of the PTZ camera.", 
        type=str, 
        default=""
    )
    parser.add_argument(
        "-ps", 
        "--panstep", 
        help="The step of pan in degrees.", 
        type=int, 
        default=15
    )
    parser.add_argument(
        "-tv", 
        "--tilt", 
        help="The tilt value in degrees.", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "-zm", 
        "--zoom", 
        help="The zoom value.", 
        type=int, 
        default=1
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model to use (e.g., 'yolo11n', 'Florence-base')",
        type=str,
        default="yolo11n"
    )

    return parser

def look_for_object(args):
    # Parse comma-separated objects into list
    objects = [obj.strip().lower() for obj in args.objects.split(',')]
    pans = [angle for angle in range(0, 360, args.panstep)]
    tilts = [args.tilt for _ in range(len(pans))]
    zooms = [args.zoom for _ in range(len(pans))]

    # Create detector using factory
    try:
        detector = DetectorFactory.create_detector(args.model)
    except ValueError as e:
        print(f"Error creating detector: {str(e)}")
        sys.exit(1)
    
    for iteration in range(args.iterations):
        for pan, tilt, zoom in zip(pans, tilts, zooms):
            image_path, detection = get_image_from_ptz_position(
                args, objects, pan, tilt, zoom, detector, None
            )
            print(f"img path: {image_path}")
            if detection is None:  # If no objects found
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print(f'             no objects found               ')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                continue

            reward = detection['reward']
            if reward > 0.99:  # High reward means low confidence
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print(f'             low confidence detection      ')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                continue
            
            label = detection['label']
            bbox = detection['bbox']
            print('reward: ', reward)
            print('type(reward): ', type(reward))
            image = Image.open(image_path)
            print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
            print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
            print(f'         following {label} object           ')
            print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
            print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
            center_and_maximize_object(args, bbox, image, reward)

            if not args.keepimages:
                os.remove(image_path)

        publish_images()

def main():
    args = get_argparser().parse_args()
    look_for_object(args)

if __name__ == "__main__":
    main()