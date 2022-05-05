import argparse
import collections
import os
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import time
import pandas as pd
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import tflite_runtime.interpreter as tflite 
import detectlices

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])
def non_max_suppression(objects, threshold):
  """Returns a list of indexes of objects passing the NMS.

  Args:
    objects: result candidates.
    threshold: the threshold of overlapping IoU to merge the boxes.

  Returns:
    A list of indexes containings the objects that pass the NMS.
  """
  if len(objects) == 1:
    return [0]

  boxes = np.array([o.bbox for o in objects])
  xmins = boxes[:, 0]
  ymins = boxes[:, 1]
  xmaxs = boxes[:, 2]
  ymaxs = boxes[:, 3]

  areas = (xmaxs - xmins) * (ymaxs - ymins)
  scores = [o.score for o in objects]
  idxs = np.argsort(scores)

  selected_idxs = []
  while idxs.size != 0:

    selected_idx = idxs[-1]
    selected_idxs.append(selected_idx)

    overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
    overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
    overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
    overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

    w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
    h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

    intersections = w * h
    unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
    ious = intersections / unions

    idxs = np.delete(
        idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

  return selected_idxs


def draw_objects(draw, objs, scale_factor, labels):
  """Draws the bounding box and label for each object."""
  COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype=np.uint8)
  for obj in objs:
    bbox = obj.bbox
    color = tuple(int(c) for c in COLORS[obj.id])
    draw.rectangle([(bbox.xmin * scale_factor, bbox.ymin * scale_factor),
                    (bbox.xmax * scale_factor, bbox.ymax * scale_factor)],
                   outline=color, width=3)
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=15)
    draw.text((bbox.xmin * scale_factor + 4, bbox.ymin * scale_factor + 4),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill=color, font=font)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        required=True,
        help='Model path')
    parser.add_argument('--label', help='Labels file path.')
    parser.add_argument('--input', help='input file path')
    parser.add_argument('--yolo', help='if model is yolov5', default=0)
    args = parser.parse_args()
    image_folder = args.input
    model_path = args.model
    yolov5 = args.yolo
    print("Evaluate yolo model: ", yolov5)

    pred_dict = {
        'image_id': [],
        'pred_str': []
    }
    from os.path import isfile, join
    image_list = [f for f in os.listdir(image_folder) if isfile(join(image_folder, f))]
    count = 0

    for path in image_list:
        # Open image.
        if path[-4:] != '.jpg':
            print(path)
            continue
        image = Image.open(join(image_folder, path)).convert('RGB')

        objects_by_label = dict()
        img_size = image.size

        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()

        labels = read_label_file(args.label) if args.label else {}
        # Resize the image for input
        _, scale = common.set_resized_input(
            interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

        # Run inference
        interpreter.invoke()
        if yolov5 != '0':
            objs = detectlices.get_objects(interpreter, score_threshold=0.3, image_scale=scale)
        else:
            objs = detect.get_objects(interpreter, score_threshold=0.3, image_scale=scale)

        objects_by_label = dict()
        for obj in objs:
            bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
            label = labels.get(obj.id, '')
            objects_by_label.setdefault(label,[]).append(Object(label, obj.score, bbox))
        for label, objects in objects_by_label.items():
            idxs = non_max_suppression(objects, 0.5)
            for idx in idxs:
                obj = objects[idx]
                print(obj)
                # bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
                bbox = obj.bbox
                pred_dict['image_id'].append(os.path.basename(path))
                pred_str = "0 " + "{:.4f}".format(obj.score) + " " + " ".join([str(x) for x in bbox])
                pred_dict['pred_str'].append(pred_str)

        # Resize again to a reasonable size for display
        # display_width = 500
        # scale_factor = display_width / image.width
        # height_ratio = image.height / image.width
        # image = image.resize((display_width, int(display_width * height_ratio)))
        # draw_objects(ImageDraw.Draw(image), objs, scale_factor, labels)
        # image.save('binh_' +str(count)+'.jpg')
        # count += 1

    print(pred_dict)
    pred_df = pd.DataFrame.from_dict(data=pred_dict)
    pred_df.to_csv('prediction.csv')

if __name__ == '__main__':
  main()
