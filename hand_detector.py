import cv2
import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn.functional as F
from entity import params
from models.HandNet import HandNet

class HandDetector(object):
    def __init__(self, weights_file):
        print('Loading HandNet...')
        self.model = HandNet()
        self.model.load_state_dict(torch.load(weights_file))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def detect(self, hand_img, fast_mode=False, hand_type="right"):
        if hand_type == "left":
            hand_img = cv2.flip(hand_img, 1)

        hand_img_h, hand_img_w, _ = hand_img.shape

        resized_image = cv2.resize(hand_img, (params["hand_inference_img_size"], params["hand_inference_img_size"]))
        x_data = np.array(resized_image[np.newaxis], dtype=np.float32).transpose(0, 3, 1, 2) / 256 - 0.5
        x_data = torch.tensor(x_data).to(self.device)
        x_data.requires_grad = False
        with torch.no_grad():
            hs = self.model(x_data)
            
            heatmaps = F.interpolate(hs[-1], (hand_img_h, hand_img_w), mode='bilinear', align_corners=True).cpu().numpy()[0]

        if hand_type == "left":
            heatmaps = cv2.flip(heatmaps.transpose(1, 2, 0), 1).transpose(2, 0, 1)

        keypoints = self.compute_peaks_from_heatmaps(heatmaps)

        return keypoints

    def compute_peaks_from_heatmaps(self, heatmaps):
        keypoints = []

        for i in range(heatmaps.shape[0] - 1):
            heatmap = gaussian_filter(heatmaps[i], sigma=params['gaussian_sigma'])
            max_value = heatmap.max()
            if max_value > params['hand_heatmap_peak_thresh']:
                coords = np.array(np.where(heatmap==max_value)).flatten().tolist()
                keypoints.append([coords[1], coords[0], max_value]) # x, y, conf
            else:
                keypoints.append(None)

        return keypoints

def draw_hand_keypoints(orig_img, hand_keypoints, left_top):
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img = orig_img.copy()
    left, top = left_top

    finger_colors = [
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
    ]

    for i, finger_indices in enumerate(params["fingers_indices"]):
        for finger_line_index in finger_indices:
            keypoint_from = hand_keypoints[finger_line_index[0]]
            keypoint_to = hand_keypoints[finger_line_index[1]]

            if keypoint_from:
                keypoint_from_x, keypoint_from_y, _ = keypoint_from
                cv2.circle(img, (keypoint_from_x + left, keypoint_from_y + top), 3, finger_colors[i], -1)

            if keypoint_to:
                keypoint_to_x, keypoint_to_y, _ = keypoint_to
                cv2.circle(img, (keypoint_to_x + left, keypoint_to_y + top), 3, finger_colors[i], -1)

            if keypoint_from and keypoint_to:
                cv2.line(img, (keypoint_from_x + left, keypoint_from_y + top), (keypoint_to_x + left, keypoint_to_y + top), finger_colors[i], 1)

    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face detector')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--img', '-i', help='image file path')
    args = parser.parse_args()

    # load model
    hand_detector = HandDetector(args.weights)

    # read image
    img = cv2.imread(args.img)

    # inference
    hand_keypoints = hand_detector.detect(img, hand_type="right")

    # draw and save image
    img = draw_hand_keypoints(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), hand_keypoints, (0, 0))
    print('Saving result into result.png...')
    cv2.imwrite('result.png', img)
