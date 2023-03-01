import os
import shutil
import argparse


def create_imagenet_format(data_path, root):
    os.makedirs(root, exist_ok=True)
    
    class_names = os.listdir(data_path)
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(root, 'train', class_name), exist_ok=True)
            image_path = os.path.join(class_path, 'train', 'good')
            images = os.listdir(image_path)
            # copy all normal train images to train folder
            for image in images:
                shutil.copy(os.path.join(image_path, image), os.path.join(root, 'train', class_name, image))
                print("Copy {} -> {}".format(os.path.join(image_path, image), os.path.join(root, 'train', class_name, image)))
            
            os.makedirs(os.path.join(root, 'val', class_name), exist_ok=True)
            image_path = os.path.join(class_path, 'test', 'good')
            images = os.listdir(image_path)
            # copy all normal test images to val folder
            for image in images:
                shutil.copy(os.path.join(image_path, image), os.path.join(root, 'val', class_name, image))
                print("Copy {} -> {}".format(os.path.join(image_path, image), os.path.join(root, 'val', class_name, image)))


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description='One-for-All: Proposal Masked Cross-Class Anomaly Detection')
        parser.add_argument("--data_path", type=str, default="")
        parser.add_argument("--root", type=str, default="data/Mvtec-ImageNet")
        args = parser.parse_args()

        return args

    args = parse_args()
    
    create_imagenet_format(args.data_path, args.root)
    print("Setup train dataset done...")
                
    