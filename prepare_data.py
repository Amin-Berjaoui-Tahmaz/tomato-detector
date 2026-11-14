"""AgRobTomato (Pascal VOC, CC BY 4.0) -> COCO splits that RF-DETR reads.

Download first (137 MB, https://zenodo.org/records/5596799, md5 890666716924415720f073b06a9a02a3)
and unzip into data/raw/agrob/.

Split by recording pass, not at random: frames are 3 s apart with ~60% overlap, so a random
split would put near-duplicate frames in train and test and inflate mAP.
  test  = the whole "side" pass (152 frames), never seen in training
  valid = last 15% of the other pass (contiguous block), used only to pick the best epoch
  train = the rest
"""
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SRC = Path("data/raw/agrob/Dataset-Greenhouse_Tomato_AgRob")
OUT = Path("data/coco")
# 4 source classes -> 3. "riped" has only 30 boxes in the whole set, too few to learn alone,
# and reddish vs ripe is a fuzzy line anyway; both mean "pick soon".
CLASSES = ["unripe", "turning", "ripe"]
MAP = {"unriped": 0, "breaking": 1, "reddish": 2, "riped": 2}


def split_of(stems: list[str]) -> dict[str, str]:
    main = sorted(s for s in stems if "_side_" not in s)
    cut = int(len(main) * 0.85)
    out = {s: "train" for s in main[:cut]} | {s: "valid" for s in main[cut:]}
    return out | {s: "test" for s in stems if "_side_" in s}


def main() -> None:
    stems = [p.stem for p in sorted((SRC / "Annotations").glob("*.xml"))]
    splits = split_of(stems)
    cocos = {
        sp: {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)]}
        for sp in ("train", "valid", "test")
    }
    shutil.rmtree(OUT, ignore_errors=True)
    for sp in cocos:
        (OUT / sp).mkdir(parents=True)
    for img_id, stem in enumerate(stems):
        sp, coco = splits[stem], cocos[splits[stem]]
        root = ET.parse(SRC / "Annotations" / f"{stem}.xml").getroot()
        w, h = int(root.find("size/width").text), int(root.find("size/height").text)
        shutil.copy(SRC / "JPEGImages" / f"{stem}.jpg", OUT / sp / f"{stem}.jpg")
        coco["images"].append({"id": img_id, "file_name": f"{stem}.jpg", "width": w, "height": h})
        for obj in root.iter("object"):
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            if x2 <= x1 or y2 <= y1:
                continue
            coco["annotations"].append({
                "id": len(coco["annotations"]), "image_id": img_id, "category_id": MAP[obj.find("name").text],
                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
            })
    for sp, coco in cocos.items():
        (OUT / sp / "_annotations.coco.json").write_text(json.dumps(coco))
        counts = [sum(a["category_id"] == i for a in coco["annotations"]) for i in range(len(CLASSES))]
        print(f"{sp:5} {len(coco['images']):3} images  " + "  ".join(f"{n}={c}" for n, c in zip(CLASSES, counts)))


if __name__ == "__main__":
    main()
