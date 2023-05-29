import argparse
import os
from collections import OrderedDict

import torch


def main(input_path, output_path, map_from, map_to):
    ckpt = torch.load(input_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict(
        {k.replace(map_from, map_to): v for k, v in ckpt["state_dict"].items()}
    )
    ckpt["state_dict"] = new_state_dict
    if output_path is None:
        root, ext = os.path.splitext(input_path)
        output_path = root + "-keys=remapped" + ext
    torch.save(ckpt, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        help="Path to lightning checkpoint",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Output path for new state dict",
        default=None,
        required=False,
    )
    parser.add_argument("--map_from", required=True)
    parser.add_argument("--map_to", required=True)
    args = parser.parse_args()
    main(**vars(args))
