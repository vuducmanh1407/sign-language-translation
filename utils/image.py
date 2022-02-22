import cv2
import glob
import numpy as np
import torch


def read_images_from_folder(path):
    img_list = sorted(glob.glob(path))
    img_list = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list]

    return img_list

def resize_images(img_list, resolution=None):
    if resolution is not None:
        if img_list[0].shape[0] != resolution[0] and img_list[0].shape[1] != resolution[1]:
            img_list = [cv2.resize(img, resolution, interpolation=cv2.INTER_AREA) for img in img_list]
    return img_list

def normalize_and_to_tensor(img_list):
    video = np.array(img_list)
    video = video / 127.5 - 1
    video = torch.from_numpy(video.transpose((0, 3, 1, 2))).float()
    return video

def load_input(path, resolution):
    img_list = read_images_from_folder(path)
    resized_img_list = resize_images(img_list, resolution)
    norm_video = normalize_and_to_tensor(resized_img_list)

    return {
        "original_image_list": img_list,
        "resized_image": norm_video
    }


if __name__ == "__main__":
    video = read_images_from_folder("./dataset/05July_2010_Monday_tagesschau-1205/*")
    video = resize_images(video, (256, 256))
    video = normalize_and_to_tensor(video)
    print(video.shape)


