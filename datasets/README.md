---
license: apache-2.0
task_categories:
- object-detection
- image-classification
tags:
- SARFish
- Illegal Fishing
- Computer Vision
- Complex-Valued
- Synthetic Aperture Radar
pretty_name: SARFish Dataset
size_categories:
- n<1K
---

SARFish is a [Synthetic Aperture Radar (SAR)](https://sentinel.esa.int/web/sentinel/missions/sentinel-1/instrument-payload) imagery dataset for the purpose of training, validating and testing supervised machine learning models on the tasks of ship detection, classification, and length regression. The SARFish dataset builds on the excellent work of the [xView3-SAR dataset](https://iuu.xview.us/dataset) (2021) and consists of two parts:

1. Data -  Extends the xView3-SAR dataset to include [Single Look Complex (SLC)](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/single-look-complex) as well as [Ground Range Detected (GRD)](https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-1-sar/products-algorithms/level-1-algorithms/ground-range-detected) imagery data taken directly from the European Space Agency (ESA) Copernicus Programme [Open Access Hub Website](https://scihub.copernicus.eu/).
2. Labels - Derives labels from the xView3-SAR dataset providing maritime object location, vessel classification and vessel length information.

### Quick Links

The following are links to the Kaggle competitions for each of the tracks of the SARFish challenge along with the SARFish dataset and GitHub repo:

- Data:
    - [SARFish](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish)
    - [SARFishSample](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFishSample)
- [Labels](https://iuu.xview.us/download-links)
- Challenge:
    - [Maritime Object Detection Track](https://www.kaggle.com/competitions/sarfish-maritime-object-detection)
    - [Maritime Object Classification Track](https://www.kaggle.com/competitions/sarfish-maritime-object-classification)
    - [Vessel Length Regression Track](https://www.kaggle.com/competitions/sarfish-vessel-length-regression)
- [GitHub repo](https://github.com/RitwikGupta/SARFish)
- [Mailbox](SARFish.Dataset@defence.gov.au)
- [DAIRNet](https://www.dairnet.com.au/events/workshop-on-complex-valued-deep-learning-and-sarfish-challenge/)

The [GitHub repo](https://github.com/RitwikGupta/SARFish) describes how to:
- Download the dataset.
- Run the SARFish_demo jupyter notebook.
- Load imagery products and groundtruth labels,
- Train and evaluate a reference/baseline model using the dataset.

### Dataset summary - What does the SARFish dataset consist of?

The following table summarises the sizes of the full size and sample SARFish dataset.

| dataset | coincident GRD, SLC products | compressed (GB) | uncompressed (GB) |
| --- | --- | --- | --- |
| SARFishSample | 1 | 4.3 | 8.2 |
| SARFish | 753 | 3293 | 6468 |

The following table summarises the partitions of the dataset:

| Partition | Coincident products | Labels Provided | Unique maritime object  labels | |
| --- | --- | --- | --- | --- |
| | | | SLC | GRD |
| train | 553 | True | 63071 | 64054 |
| validation | 50 | True | 18906 | 19222 |
| public | 150 | False | 58744 | 60008 |
| | | Total | 140721 | 143284 |

### How to access the SARFish dataset

The SARFish dataset is available for download at:
- [full SARFish dataset](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish)
- [sample SARFish dataset](https://huggingface.co/datasets/ConnorLuckettDSTG/SARFishSample)

#### Full SARFish dataset

Make sure you have at least enough storage space for the uncompressed dataset.

```bash
cd /path/to/large/storage/location
```

[Create|login] to a [huggingface](https://huggingface.co) account.

Login to the huggingface command line interface.

```bash
huggingface-cli login
```

Copy the access token in settings/Access Tokens from your huggingface account. Clone the dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/ConnorLuckettDSTG/SARFish
```

#### SARFish sample dataset

Substitute the final command for the full dataset with the following:

```bash
git clone https://huggingface.co/datasets/ConnorLuckettDSTG/SARFishSample
```

Follow the instructions of the github repo README to check the md5sums of the data and unzip them.

#### Labels

The SARFish dataset labels are derived from the labels supplied with the [xView-3 SAR dataset](https://iuu.xview.us/dataset). The SARFish dataset labels are available for download from the [DIU website](https://iuu.xview.us/download-links). Be sure to take into account country restrictions.

### Data

SARFish extends the xView3-SAR dataset by providing products from the [Sentinel-1 C-band SAR satellite constellation](https://sentinel.esa.int/web/sentinel/missions/sentinel-1) operated by the European Space Agency’s (ESA) Copernicus Programme available on their [Open Access Hub Website](https://scihub.copernicus.eu/) in both real-valued GRD and complex-valued SLC product types.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16538278%2F01f81bd0507e1a2512937dafe3d6e4cd%2FSentinel_1_processing_levels_summary.png?generation=1704576734729193&alt=media)

The above image shows a condensed summary of the image formation pipeline of the Sentinel-1 products provided by the Sentinel-1 Mission Performance Center. Note that the SLC and GRD products both share a common ancestor.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16538278%2F7751ca1ec0fa5cdea8f13465c0ecbda0%2Frelationship_between_the_xView3_and_SARFish_datasets.png?generation=1704576840602914&alt=media)

The above image shows the relationship between the xView3-SAR and SARFish datasets.

#### Summary table

The following table compares the GRD and SLC products of the SARFish dataset [3][4]

| | | |
| --- | --- | --- |
| Platform | Sentinel-1 (A, B) |  |
| Operator | European Space Agency (ESA) Sentinel-1 Mission Performance Center |  |
| Sensor | CBand SAR |  |
| Mode | Interferometric Wide Swath (IW) |  |
| Polarisations | VV, VH |  |
| Ground range coverage (km) | 251.8 |  |
| Product type | SLC | GRD |
| Pixel value | Complex | Magnitude Detected |
| Data type | Complex Int16 | Unsigned Int16 |
| Azimuth pixel spacing (m) | 2.3 | 10 |
| Range pixel spacing (m) | 14.1 | 10 |

#### Ground Range Detected (GRD) Products

GRD products consist of two 'detected' imagery products in VH, VV polarisations. The imagery data is stored in GeoTiff format. Also included in the dataset are no_data masks and shoreline files which are used to evaluate 'close-to-shore' maritime object detection tasks.

#### Single Look Complex (SLC) Products

![SARFish Single Look Complex (SLC) example swath 1](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16538278%2F96c7236965f001af2121a1bfb512490e%2Fscaled_detected_full_SLC_vh_swath_1_annotated.png?generation=1704092495548996&alt=media)

![SARFish Single Look Complex (SLC) example swath 2](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16538278%2Fda4797f41133145d534a6bb892aca843%2Fscaled_detected_full_SLC_vh_swath_2_annotated.png?generation=1704092615622983&alt=media)

![SARFish Single Look Complex (SLC) example swath 3](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16538278%2F71ce540d31a14a293c7ccd1d8c0cf710%2Fscaled_detected_full_SLC_vh_swath_3_annotated.png?generation=1704092648447034&alt=media)

The figures above show the 'swaths' comprising a SARFish SLC product in VH polarisation with groundtruth maritime object. labels The complex data has been 'detected' [3] by projecting the complex-valued data onto the real numbers for visualisation and displayed on decibel scale where the dynamic range is between 15 and 60 dB.  Note that the SLC products have non-square (x, y): 2.3 × 14.1 m pixel spacing. The native format of the data is Complex Int16.

![SARFish SLC footprint](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16538278%2Fa982a5ee2a3027b9e147c6e98823ddac%2FSARFish_SLC_footprint_polygon_optimised.png?generation=1704092702046069&alt=media)

The figure above shows the footprint of the first swath of the example SLC product in context. The footprint was plotted using Clyde D'Cruz' ["openstreetmap WKT playground"](https://clydedacruz.github.io/openstreetmap-wkt-playground/).

![SARFish SLC VH polarisation ship example](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16538278%2Fd4cd496ddc5a699c9629f20a1f8c767f%2FSLC_region_vh_annotation_optimised_cropped.png?generation=1704092802631790&alt=media)

![SARFish SLC VV polarisation ship example](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F16538278%2Fc75c73f734028887484f699a892448e1%2FSLC_region_vv_optimised_cropped.png?generation=1704092864387849&alt=media)

The above images show detail of a labelled vessel in a SLC product in both VH (above) and VV (below) polarisations. Note the differences in the speckle and side-lobing artefacts on the vessel between polarisations and the non-square pixel spacing.

### Labels

#### Location labels

The labels denote the image pixel and geographic coordinate location of the maritime object.

| field     | data_type | description |
| --------- | ----------- | --------- |
| detect\_lat | float | latitude of detection in World Geodetic System (WGS) 84 coordinates |
| detect\_lon | float | longitude of detection in WGS84 coordinates |
| detect\_scene\_row | int | pixel row of scene containing detection |
| detect\_scene\_column | int | pixel column of scene containing detection |

#### Classification Labels

The labels for the maritime object classification are organised in the same hierarchical structure as the xView3-SAR challenge labels:

```bash
label_heirarchy:
└── maritime_objects
    └── vessels
        └── fishing_vessels
```

They are denoted by the following columns in the labels:

| field     | data_type | description |
| --------- | ----------- | --------- |
| is\_vessel | bool | True if detection is a vessel, False otherwise |
| is\_fishing | bool | True if detection is a fishing vessel, False otherwise |

The maritime object categories are labelled using boolean values to the following questions:

- is the maritime object a vessel?
- is the vessel a fishing vessel?

The following table shows the combinations of hierarchical classification labels present in the SARFish dataset:

| is\_vessel | is\_fishing |
|------------:|-------------:|
| False | nan |
| True | nan |
| | False |
| | True |
| nan | nan |

#### Vessel Length Labels

The vessel lengths are denoted in the following column in the labels:

| field     | data_type | description |
| --------- | ----------- | --------- |
| vessel\_length\_m | float | length of vessel in meters; only provided where available from AIS |

#### Detailed labels summary

| field     | data_type | description |
| --------- | ----------- | --------- |
| partition | str: \{"train", "validation"\} | split of the dataset |
| product\_type | str: \{"GRD", "SLC"\} | product type of the data |
| scene\_id | str | unique xView3 scene ID for challenge purposes |
| detect\_id | str | unique detection ID in the format: {scene\_id}\_{detect\_lat}\_{detect\_lon} |
| \{product\_type\}\_product\_identifier | str | The Copernicus Sentinel-1 product identifier for the designated product type |
| detect\_lat | float | latitude of detection in World Geodetic System (WGS) 84 coordinates |
| detect\_lon | float | longitude of detection in WGS84 coordinates |
| detect\_scene\_row | int | pixel row of scene containing detection |
| detect\_scene\_column | int | pixel column of scene containing detection |
| top | float | pixel row of the top left corner of the bounding box, where available |
| left | float | pixel column of the top left corner of the bounding box, where available |
| bottom | float | pixel row of the bottom right corner of the bounding box, where available |
| right | float | pixel column of the bottom right corner of the bounding box, where available |
| vessel\_length\_m | float | length of vessel in meters; only provided where available from AIS |
| source | str: \{AIS, AIS/Manual, Manual\} | source of detection (AIS, manual label, or both) |
| is\_vessel | bool | True if detection is a vessel, False otherwise |
| is\_fishing | bool | True if detection is a fishing vessel, False otherwise |
| global\_shoreline\_vector\_distance\_from\_shore\_km | float | distance from shore of detection in kilometers as determined using the global shoreline vectors projected into the pixel space of the SARFish products  |
| xView3\_shoreline\_vector\_distance\_from\_shore\_km | float | distance from shore of detection in kilometers as determined using the  xView3-SAR shoreline vectors projected into the pixel space of the SARFish products  |
| confidence | str: \{HIGH, MEDIUM, LOW\} | level of confidence for is\_vessel and is\_fishing labels |

### Source

The Sentinel-1 GRD and SLC products were downloaded the University of Alaska's Alaska Satellite Facillity (ASF) which operates NASA's Distributed Active Archive Center (DAAC).

- [website](https://asf.alaska.edu/)
- [registration](https://urs.earthdata.nasa.gov/users/new)
- [download](https://datapool.asf.alaska.edu/)
- API docs
	- [basics](https://docs.asf.alaska.edu/api/basics/)
	- [keywords](https://docs.asf.alaska.edu/api/keywords/)
	- [tools](https://docs.asf.alaska.edu/api/tools/)


[1]. Tri-Tan Cao, Connor Luckett, Jerome Williams, Tristrom Cooke, Ben Yip, Arvind Rajagopalan, and Sebastien Wong. Sarfish: Space-based maritime surveillance using complex synthetic aperture radar imagery. In 2022 International Conference on Digital Image Computing: Techniques and Applications (DICTA), pages 1–8. IEEE, 2022.

[2] xview3-sar: Detecting dark fishing activity using synthetic aperture radar imagery. arXiv:2206.00897v4 [cs.CV], Nov 2022.

[3] M. Bourbigot, H. Johnsen, R. Piantanida, and G. Hajduch, Sentinel-1 Product Definition. Sentinel-1 Mission Performance Centre, 2016. [Online]. Available: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/document-library/-/asset_publisher/1dO7RF5fJMbd/content/sentinel-1-product-definition

[4] S. N. R. Chandra, J. Christopherson, and K. A. Casey, 2020 Joint Agency Commercial Imagery Evaluation—Remote sensing satellite compendium. US Geological Survey, 2020.
