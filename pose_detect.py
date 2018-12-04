import cv2
import argparse
from openpose import Openpose, draw_person_pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--img', '-i', help='image file path')
    parser.add_argument('--precise', '-p', action='store_true', help='do precise inference')
    args = parser.parse_args()

    # load model
    openpose = Openpose(weights_file = args.weights, training = False)

    # read image
    img = cv2.imread(args.img)

    # inference
    poses, _ = openpose.detect(img, precise=args.precise)

    # draw and save image
    img = draw_person_pose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), poses)

    print('Saving result into result.png...')
    cv2.imwrite('result.png', img)