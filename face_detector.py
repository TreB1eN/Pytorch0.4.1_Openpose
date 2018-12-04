import cv2
import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn.functional as F
from entity import params
from models.FaceNet import FaceNet

class FaceDetector(object):
    def __init__(self, weights_file):
        print('Loading FaceNet...')
        self.model = FaceNet()
        self.model.load_state_dict(torch.load(weights_file))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def detect(self, face_img, fast_mode=False):
        face_img_h, face_img_w, _ = face_img.shape

        resized_image = cv2.resize(face_img, (params["face_inference_img_size"], params["face_inference_img_size"]))
        x_data = np.array(resized_image[np.newaxis], dtype=np.float32).transpose(0, 3, 1, 2) / 256 - 0.5
        x_data = torch.tensor(x_data).to(self.device)
        x_data.requires_grad = False

        with torch.no_grad():
            hs = self.model(x_data)            
            heatmaps = F.interpolate(hs[-1], (face_img_h, face_img_w), mode='bilinear', align_corners=True).cpu().numpy()[0]

        keypoints = self.compute_peaks_from_heatmaps(heatmaps)
        return keypoints

    def compute_peaks_from_heatmaps(self, heatmaps):
        keypoints = []

        for i in range(heatmaps.shape[0] - 1):
            heatmap = gaussian_filter(heatmaps[i], sigma=params['gaussian_sigma'])
            max_value = heatmap.max()
            if max_value > params['face_heatmap_peak_thresh']:
                coords = np.array(np.where(heatmap==max_value)).flatten().tolist()
                keypoints.append([coords[1], coords[0], max_value]) # x, y, conf
            else:
                keypoints.append(None)

        return keypoints

def draw_face_keypoints(orig_img, face_keypoints, left_top):
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img = orig_img.copy()
    left, top = left_top

    for keypoint in face_keypoints:
        if keypoint:
            x, y, conf = keypoint
            cv2.circle(img, (x + left, y + top), 2, (255, 255, 0), -1)

    for face_line_index in params["face_line_indices"]:
        keypoint_from = face_keypoints[face_line_index[0]]
        keypoint_to = face_keypoints[face_line_index[1]]

        if keypoint_from and keypoint_to:
            keypoint_from_x, keypoint_from_y, _ = keypoint_from
            keypoint_to_x, keypoint_to_y, _ = keypoint_to
            cv2.line(img, (keypoint_from_x + left, keypoint_from_y + top), (keypoint_to_x + left, keypoint_to_y + top), (255, 255, 0), 1)

    return img

def crop_face(img, rect):
    orig_img_h, orig_img_w, _ = img.shape
    crop_center_x = rect[0] + rect[2] / 2
    crop_center_y = rect[1] + rect[3] / 2
    crop_width = rect[2] * params['face_crop_scale']
    crop_height = rect[3] * params['face_crop_scale']
    crop_left = max(0, int(crop_center_x - crop_width / 2))
    crop_top = max(0, int(crop_center_y - crop_height / 2))
    crop_right = min(orig_img_w-1, int(crop_center_x + crop_width / 2))
    crop_bottom = min(orig_img_h-1, int(crop_center_y + crop_height / 2))
    cropped_face = img[crop_top:crop_bottom, crop_left:crop_right]
    max_edge_len = np.max(cropped_face.shape[:-1])
    padded_face = np.zeros((max_edge_len, max_edge_len, cropped_face.shape[-1]), dtype=np.uint8)
    padded_face[0:cropped_face.shape[0], 0:cropped_face.shape[1]] = cropped_face

    return padded_face, (crop_left, crop_top)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face detector')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--img', '-i', help='image file path')
    args = parser.parse_args()

    # load model
    face_detector = FaceDetector(args.weights)

    # read image
    img = cv2.imread(args.img)

    # inference
    face_keypoints = face_detector.detect(img)

    # draw and save image
    img = draw_face_keypoints(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face_keypoints, (0, 0))
    print('Saving result into result.png...')
    cv2.imwrite('result.png', img)
