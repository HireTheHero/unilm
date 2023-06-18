import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict

from typing import Dict, List


def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def jsonls2dict(
    jsonls_path: str,
    splits: List[str] = ["train", "dev_seen", "dev_unseen", "test_seen", "test_unseen"],
    split_map: Dict[str, str] = {
        "train": "train",
        "dev_seen": "val",
        "dev_unseen": "val",
        "test_seen": "test",
        "test_unseen": "test",
    },
    column_map: Dict[str, str] = {"img": "image", "id": "image_id", "text": "caption"},
):
    list_captions = []
    for split in splits:
        jsonlpath = f"{jsonls_path}/{split}.jsonl"
        caption = pd.read_json(jsonlpath, lines=True).rename(columns=column_map)
        caption["split"] = split_map[split]
        list_captions.append(caption)
    df_captions = pd.concat(list_captions, ignore_index=True).dropna()
    captions = df_captions.to_dict()
    return captions


def make_arrow(root: str, dataset_root: str):
    captions = jsonls2dict(root)

    iid2captions = defaultdict(list)
    iid2split = dict()

    images = list(captions["image"].values())
    splits = list(captions["split"].values())
    texts = list(captions["caption"].values())

    for image, split, text in tqdm(zip(images, splits, texts)):
        filename = image.split("/")[-1]
        iid2split[filename] = split
        iid2captions[filename].append(text)

    paths = list(glob(f"{root}/img/*.png"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths),
        len(caption_paths),
        len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "restval", "test"]:
        batches = [b for b in bs if b[-1] == split]

        dataframe = pd.DataFrame(
            batches,
            columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/coco_caption_karpathy_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
