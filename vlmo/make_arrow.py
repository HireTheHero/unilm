import argparse

# from vlmo.utils.write_coco_karpathy import make_arrow
from vlmo.utils.write_hm import make_arrow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="coco")
    parser.add_argument("--arrows_root", type=str, default="coco/arrows")
    args = parser.parse_args()
    make_arrow(args.root, args.arrows_root)
