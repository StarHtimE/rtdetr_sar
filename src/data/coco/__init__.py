from .coco_dataset import (
    CocoDetection, 
    mscoco_category2label,
    mscoco_label2category,
    mscoco_category2name,
)

from .hrsid_dataset import (
    HRSIDDetection,
    hrsid_category2name,
)

from .aircraft_dataset import (
    AircraftDetection,
    aircraft_category2name,
    aircraft_category2label,
    aircraft_label2category,
)

from .coco_eval import *

from .coco_utils import get_coco_api_from_dataset