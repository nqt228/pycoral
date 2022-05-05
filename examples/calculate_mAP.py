import argparse
import collections

import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import time
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import tflite_runtime.interpreter as tflite 
import pandas as pd

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])
def gen_images(image_ids):
    print("Generating image data...")
    results = []
    for idx, image_id in enumerate(image_ids):
        results.append({
            "id": idx,
        })
    return results

def gen_categories(df):
    result = []
    result.append({
        "id": 1,
        "name": "lice",
        "supercategory": "none"
    })
    return result

def gen_annotations(df, image_ids):
    results = []
    k = 0
    for idx, image_id in enumerate(image_ids):
        for i, row in df[df["filename"] == image_id].iterrows():
            results.append({
                "id": k,
                "image_id": idx,
                "category_id": 1,
                "bbox": np.array([
                    row["xmin"],
                    row["ymin"],
                    row["xmax"],
                    row["ymax"]
                ]),
                "segmentation": [],
                "ignore":0,
                "area": (row["xmax"] - row["xmin"]) * (row["ymax"] - row["ymin"]),
                "iscrowd": 0,
            }) 
            k += 1
    
    return results

def decode_prediction_string(pred_str):
        data = list(map(float, pred_str.split(" ")))
        data = np.array(data)

        return data.reshape(-1, 6)

def gen_predictions(df, image_ids):
    print("Generating prediction data...")
    k = 0
    results = []
    
    for i, row in df.iterrows():
        
        image_id = row["image_id"]
        preds = decode_prediction_string(row["pred_str"])

        for j, pred in enumerate(preds):

            results.append({
                "id": k,
                "image_id": int(np.where(image_ids == image_id)[0]),
                "category_id": int(pred[0]) + 1,
                "bbox": np.array([
                    pred[2], pred[3], pred[4], pred[5]
                ]),
                "segmentation": [],
                "ignore": 0,
                "area": (pred[4] - pred[2]) * (pred[5] - pred[3]),
                "iscrowd": 0,
                "score": pred[1]
            })

            k += 1
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt',
        required=True,
        help='ground truth csv file')
    parser.add_argument('--pred', help='prediction csv file', required=True)
    args = parser.parse_args()

    gt_csv = args.gt
    pred_csv = args.pred

    true_df = pd.read_csv(gt_csv)   
    image_ids = true_df['filename'].unique()
    annotations = {
        "type": "instances",
        "images": gen_images(image_ids),
        "categories":gen_categories(true_df),
        "annotations": gen_annotations(true_df, image_ids)
    }

    predictions = {
        "images": annotations["images"].copy(),
        "categories": annotations["categories"].copy(),
        "annotations": None
    }

    pred_df = pd.read_csv(pred_csv)
    if pred_df is not None:
        predictions["annotations"] = gen_predictions(pred_df, image_ids)
        n_imgs = -1

    coco_ds = COCO()
    coco_ds.dataset = annotations
    coco_ds.createIndex()

    coco_dt = COCO()
    coco_dt.dataset = predictions
    coco_dt.createIndex()

    imgIds = sorted(coco_ds.getImgIds())

    if n_imgs > 0:
        imgIds = np.random.choice(imgIds, n_imgs)

    # print([x /100 for x in range(50, 100, 5)])
    thrs_range = [x /100 for x in range(50, 100, 5)]

    cocoEval = COCOeval(coco_ds, coco_dt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.params.useCats = True
    cocoEval.params.iouType = "bbox"
    cocoEval.params.iouThrs = np.array(thrs_range)

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
  main()
