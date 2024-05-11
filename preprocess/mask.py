import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm.auto import tqdm

"""
src: https://pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
"""


def transform(img_fp, net, width, height, min_confidence, layerNames):
    image = cv2.imread(img_fp)
    masked = image.copy()
    (H, W) = image.shape[:2]
    (newW, newH) = (width, height)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return masked


def save(img_fn, source_dir, target_dir_masked, net, width, height, min_confidence, layerNames):
    img_fp = os.path.join(source_dir, img_fn)
    img_fp_masked = os.path.join(target_dir_masked, img_fn)
    img_masked, img_inpainted = transform(img_fp)
    cv2.imwrite(img_fp_masked, img_masked)


import argparse


def main():
    parser = argparse.ArgumentParser(description="Text detection and image inpainting")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing the images to process")
    parser.add_argument("--target_dir_masked", type=str, required=True, help="Directory to save masked images")
    parser.add_argument("--east_path", type=str, required=True, help="Path to EAST text detector model")
    parser.add_argument("--width", type=int, default=320, help="Width for resizing images")
    parser.add_argument("--height", type=int, default=320, help="Height for resizing images")
    parser.add_argument("--min_confidence", type=float, default=0.5, help="Minimum confidence for detecting text")
    args = parser.parse_args()

    net = cv2.dnn.readNet(args.east_path)
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    img_fns = [x for x in os.listdir(args.source_dir) if x.endswith('.png')]
    tasks = [(fn, args.source_dir, args.target_dir_masked, args.target_dir_inpainted, net, args.width, args.height,
              args.min_confidence, layerNames) for fn in img_fns]

    with Pool(64) as pool:
        list(tqdm(pool.imap_unordered(lambda p: save(*p), tasks), total=len(tasks)))


if __name__ == "__main__":
    main()
