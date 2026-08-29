import os
import glob
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str)
    args = parser.parse_args()

    image_folder = args.image_folder
    image_path = glob.glob(os.path.join(image_folder, "*.png"))
    print(len(image_path))
    for p in image_path:
        base = os.path.basename(p)
        name = base.split(".")[0]
        while len(name) < 3:
            name = "0" + name
        new_path = os.path.join(image_folder, name + ".png")
        os.rename(p, new_path) 