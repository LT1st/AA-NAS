# AA-NAS
中文版本请看：[中文readme](README.md)
## File Structure
| Folder/File Name | Description |
|------------------|-------------|
| scripts/         | shell startup scripts |
| main.py          | Python startup script |
| testing/         | AA data |
| data/            | NAS training datasets |
| weights/         | weights |
| NAS/             | code for NAS |
| AA/              | code for AA |

# TODO List
- [x] 7.4 Fix bugs caused by changes in file structure
- [x] 7.7 Resolve FNA environment compatibility issues
- [ ] Replace deformable approach with ...

## Data Processing
- [ ] Ensure that the coco dataset used for NAS is not contaminated by using a cleaning script before use
- [ ] The data redundancy is high, should it be dynamically or manually cleared during training?
- [ ] How to operate the dataset when creating a soft link for coco?

## AA using patches
- [ ] Multiple model deformable
- [ ] Compare cyclic training with deformable training

## NAS for OD
- [ ] Muti-shot search is too slow
- [ ] How to add data for zero-shot search? https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/installation.md
- [x] One-shot search is most suitable DetNAS https://github.com/LT1st/DetNAS

### Open-source Reproduction Goals
| Paper     | Characteristics | Time | Evaluation | Link |
|-----------|----------------|------|------------|------|
| tiny NAS  | Training-Free  | 6 hours | Zero-shot, AE needs optimization objective adjustment | https://github.com/alibaba/lightweight-neural-architecture-search/tree/main |
| NAS-FCOS  |                |      |            |      |
| NAS-FPN   |                | 32 days | Early work, too slow | https://github.com/LT1st/NAS_FPN_Tensorflow |
| NEAS      |                |      |            | https://github.com/LT1st/NEAS |
| FNA       |                |      | Network evolution approach, good | https://github.com/LT1st/FNA |
| ZenNAS    | Training-Free |      |            | https://github.com/LT1st/ZenNAS |
| HITDET    |                |      |            | https://github.com/LT1st/HitDet.pytorch |
| OPANAS    |                |      |            | https://github.com/LT1st/OPANAS |

## API