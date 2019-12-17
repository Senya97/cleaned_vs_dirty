import argparse
from os.path import join
from os import listdir
import cv2
from tqdm import tqdm
import albumentations as A


def create_argument_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='../../data/train/', help='path to train dataset')
    return parser


def show(name, image):
    print(name)
    cv2.imshow(name, image)
    cv2.waitKey(0)


def augmentation(image):
    height, width = image.shape[:2]
    crop_height = height * 3 // 4
    crop_width = width * 3 // 4

    aug = A.Compose([

        A.OneOf([
            A.RandomBrightnessContrast(p=0.33),
            A.RandomGamma(p=0.33),
            A.CLAHE(p=0.33),
        ], p=0),

        A.OneOf([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
        ], p=0.75),

        A.OneOf([
            A.MultiplicativeNoise(multiplier=0.5, p=0.2),
            A.MultiplicativeNoise(multiplier=1.5, p=0.2),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=0.3),
            A.MultiplicativeNoise(multiplier=[0.2, 0.3], elementwise=True, p=0.1),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
        ], p=0.25),

        A.OneOf([
            A.JpegCompression(quality_lower=99, quality_upper=100, p=0.25),
            A.JpegCompression(quality_lower=59, quality_upper=60, p=0.25),
            A.JpegCompression(quality_lower=39, quality_upper=40, p=0.25),
            A.JpegCompression(quality_lower=19, quality_upper=20, p=0.25),
        ], p=0.25),

        A.OneOf([
            A.Blur(blur_limit=8, p=0.5),
            A.Blur(blur_limit=15, p=0.5),
        ], p=0.25),

        A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.2),

        A.CenterCrop(height=crop_height, width=crop_width, p=0.2),
    ], p=1)

    image = aug(image=image)['image']
    return image


def main(args):
    data_path = args.data_path
    folders = ['cleaned', 'dirty']
    for folder in folders:
        folder_path = join(data_path, folder)
        images = listdir(folder_path)
        for i, name in tqdm(enumerate(images)):
            image_path = join(folder_path, name)
            image = cv2.imread(image_path)

            image = augmentation(image)
            name = '{}_{}_{}'.format(i, folder, name)
            save_path = join(data_path, 'tmp', name)
            cv2.imwrite(save_path, image)

    pass


if __name__ == '__main__':
    argument_parser = create_argument_parser()
    arguments = argument_parser.parse_args()
    main(arguments)