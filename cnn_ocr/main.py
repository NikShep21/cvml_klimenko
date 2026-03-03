import cv2
import numpy as np
from pathlib import Path
from skimage.measure import label, regionprops
from skimage.io import imread

def class_to_char(name):
    if len(name) == 2 and name[0] == "s":
        return name[1]
    return name

def extractor(image):
    if image.ndim == 3:
        gray = np.mean(image, 2).astype("u1")
    else:
        gray = image.astype("u1")

    binary = gray > 0
    props = regionprops(label(binary))

    if not props:
        return np.zeros(8, dtype="f4")

    p = props[0]
    return np.array([*p.moments_hu, p.eccentricity], dtype="f4")


def merge_next_by_x_overlap(bboxes, min_overlap=11):

    bboxes.sort(key=lambda b: b[1])  
    merged = []
    i = 0

    while i < len(bboxes):
        b1 = bboxes[i]

        if i + 1 < len(bboxes):
            b2 = bboxes[i + 1]

            inter = min(b1[3], b2[3]) - max(b1[1], b2[1])

            if inter >= min_overlap:
                merged.append((
                    min(b1[0], b2[0]), min(b1[1], b2[1]),
                    max(b1[2], b2[2]), max(b1[3], b2[3])
                ))
                i += 2
                continue

        merged.append(b1)
        i += 1

    return merged


def extract_symbol_patches(image):
    gray = np.mean(image, 2).astype("u1")
    binary = gray > 0

    props = regionprops(label(binary))
    if not props:
        return []

    bboxes = [p.bbox for p in props]
    bboxes = merge_next_by_x_overlap(bboxes)

    res = []
    for b in bboxes:
        minr, minc, maxr, maxc = b
        patch = binary[minr:maxr, minc:maxc]
        res.append((b, patch))

    res.sort(key=lambda t: t[0][1])
    return res


def make_train(train_dir: Path):
    X, y = [], []
    classes = []
    for cls in sorted([d for d in train_dir.iterdir() if d.is_dir()]):
        classes.append(cls.name)

    letters = [class_to_char(c) for c in classes]

    for idx, cls_name in enumerate(classes, start=1):
        cls_dir = train_dir / cls_name
        for p in cls_dir.glob("*.png"):
            img = imread(p)
            X.append(extractor(img))
            y.append(idx)

    X = np.array(X, dtype="f4").reshape(-1, 8)
    y = np.array(y, dtype="f4").reshape(-1, 1)
    return X, y, letters


def detect_spaces(sorted_bboxes):
    xs = [b[1] for b in sorted_bboxes]
    xe = [b[3] for b in sorted_bboxes]
    gaps = [xs[i] - xe[i - 1] for i in range(1, len(xs))]
    if not gaps:
        return set()

    gmed = float(np.median(gaps))
    space_after = set()
    for i in range(1, len(xs)):
        gap = xs[i] - xe[i - 1]
        if gap > 2.5 * gmed:
            space_after.add(i - 1)
    return space_after


def main():
    root = Path(__file__).parent / "task"
    train_dir = root / "train"

    train, responses, letters = make_train(train_dir)

    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, responses)

    k = 5

    for p in sorted(root.glob("*.png")):
        img = imread(p)

        symbols = extract_symbol_patches(img)
        if not symbols:
            print(p.stem, "")
            continue

        bboxes = []
        for bbox, _ in symbols:
            bboxes.append(bbox)
        space_after = detect_spaces(bboxes)

        feat_list = []
        for _, patch in symbols:
            feat_list.append(extractor(patch))

        feats = np.array(feat_list, dtype="f4")   

        _, results, _, _ = knn.findNearest(feats, k)

        res = []
        class_ids = results.flatten().astype(int)

        for i, cls_id in enumerate(class_ids):
            res.append(letters[cls_id - 1])
            if i in space_after:
                res.append(" ")

        print(p.stem, "".join(res))


if __name__ == "__main__":
    main()