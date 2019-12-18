import argparse
from os.path import join
from os import listdir, makedirs
from shutil import rmtree, copy
from tqdm import tqdm
import cv2
import albumentations as A


def create_argument_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_path', type=str, default='../data/default/', help='')
    parser.add_argument('--output_path', type=str, default='../data/dataset/', help='')
    return parser

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
    input_path = args.input_path
    output_path = args.output_path

    # create folders
    rmtree(output_path, ignore_errors=True)
    for folder in ['train', 'val']:
        for sub_folder in ['cleaned', 'dirty']:
            makedirs(join(output_path, folder, sub_folder))

    # split data
    for folder in ['cleaned', 'dirty']:
        source_folder = join(input_path, 'train', folder)
        images = listdir(source_folder)
        for i, image in enumerate(images):
            if i % 6 != 0:
                dest_folder = join(output_path, 'train', folder)
            else:
                dest_folder = join(output_path, 'val', folder)
            copy(join(source_folder, image), (join(dest_folder, image)))

    # add augmentations
    for folder in ['cleaned', 'dirty']:
        folder_path = join(output_path, 'train', folder)
        images = listdir(folder_path)
        for i, name in tqdm(enumerate(images)):
            image_path = join(folder_path, name)
            image = cv2.imread(image_path)

            image = augmentation(image)

            name = '{}_{}_{}'.format(i, folder, name)
            save_path = join(folder_path, name)
            cv2.imwrite(save_path, image)


    pass


if __name__ == '__main__':
    argument_parser = create_argument_parser()
    arguments = argument_parser.parse_args()
    main(arguments)