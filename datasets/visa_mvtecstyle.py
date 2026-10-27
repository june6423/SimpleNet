"""VisA loader with the MVTec input geometry (Option A).

The images and masks get the transforms of ``MVTecDataset``: Resize(resize),
then CenterCrop(imagesize).  The crop discards the border of each non-square
VisA image.  The split comes from the official ``split_csv/1cls.csv``; the
data folder is not scanned.
"""

import os

from torchvision import transforms

from .highres_base import read_manifest
from .mvtec import DatasetSplit, MVTecDataset
from .visa import _CLASSNAMES


def _binarize_mask(mask):
    """Map every non-zero VisA mask value to 255, as in MVTec masks."""
    return mask.convert("L").point(lambda value: 255 if value > 0 else 0)


class VisAMVTecStyleDataset(MVTecDataset):
    def __init__(self, source, classname, **kwargs):
        if classname not in _CLASSNAMES:
            raise ValueError(f"Unknown VisA category {classname!r}.")
        super().__init__(source, classname, **kwargs)
        self.transform_mask = transforms.Compose(
            [transforms.Lambda(_binarize_mask), *self.transform_mask.transforms]
        )

    def get_image_data(self):
        manifest = os.path.join(self.source, "split_csv", "1cls.csv")
        imgpaths_per_class = {}
        data_to_iterate = []
        for classname in self.classnames_to_use:
            records = read_manifest(manifest, self.source, classname, self.split.value)
            if not records:
                raise ValueError(f"No VisA records for {classname} split={self.split.value}.")
            imgpaths_per_class[classname] = {}
            for record in records:
                anomaly = "anomaly" if record.is_anomaly else "good"
                imgpaths_per_class[classname].setdefault(anomaly, []).append(record.image_path)
                mask_path = record.mask_path if record.is_anomaly else None
                data_to_iterate.append([classname, anomaly, record.image_path, mask_path])
        return imgpaths_per_class, data_to_iterate


__all__ = ["DatasetSplit", "VisAMVTecStyleDataset"]
