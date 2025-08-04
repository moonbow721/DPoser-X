# Dataset Preparation
## Body-only Data
To train DPoser, we use the [AMASS](https://amass.is.tue.mpg.de/) dataset. You have two options for dataset preparation:

- **Option 1: Process the Dataset Yourself**  
  Download the AMASS dataset and process it using the following script:
  ```shell
  python -m lib/data/script.py
  ```
  Ensure you follow this directory structure:
  ```
  ${ROOT}  
  |-- data  
  |  |-- body_data
  |        |-- body_normalizer
  |           |-- axis_normalize1.pt
  |           |-- axis_normalize2.pt
  |           |-- rot6d_normalize1.pt
  |           |-- rot6d_normalize2.pt
  |        |-- version1  
  |           |-- test  
  |              |-- betas.pt  
  |              |-- pose_body.pt  
  |              |-- root_orient.pt  
  |           |-- train  
  |           |-- valid  
  ```

- **Option 2: Use Preprocessed Data**  
  Alternatively, download the processed data directly from [Google Drive](https://drive.google.com/file/d/1TQi_wKxJU3TTcVko-oPlWvp8L12lNR7F/view?usp=sharing).

## Hand-only Data
For hand data, we use several publicly available datasets as detailed in our paper. While we recommend using [HaMeR](https://github.com/geopavlakos/hamer) to access the original datasets, you have two options:

- **Option 1: Process the Datasets Yourself**
  Process the raw datasets using scripts under `./hand_process`. The processed data should contain MANO parameters.

- **Option 2: Use Our Preprocessed Data**
  Download our preprocessed data from [Google Drive](https://drive.google.com/file/d/1VlJLJ_FHeU8q9-jeG9HrBuPlqEvKVpMU/view?usp=drive_link). The data should be organized as follows:
  ```
  ${ROOT}  
  |-- data  
  |  |-- hand_data              
  |        |-- dataset_params   
  |           |-- *.npz        
  |        |-- hand_normalizer
  |           |-- axis_normalize1.pt
  |           |-- axis_normalize2.pt
  |           |-- rot6d_normalize1.pt
  |           |-- rot6d_normalize2.pt 
  |        |-- reinterhand_mocap.pt  
  |        |-- reference_batch.pt 
  |        |-- statistics.npz  
  ```


## Face-only Data
For face data, due to copyright restrictions, we cannot directly share the used datasets. Please obtain them from:

- [MICA](https://github.com/Zielon/MICA)
- [WCPA](https://tianchi.aliyun.com/competition/entrance/531961/information)

Utilize the scripts under `./face_process` to process the raw datasets. The processed data should be organized as follows:
```
${ROOT}  
|-- data  
|  |-- face_data
|        |-- betas300_normalizer
|           |-- axis_normalize1.pt
|           |-- axis_normalize2.pt
|        |-- betas_normalizer
|           |-- axis_normalize1.pt
|           |-- axis_normalize2.pt
|        |-- expression_normalizer
|           |-- axis_normalize1.pt
|           |-- axis_normalize2.pt
|        |-- FACEWAREHOUSE
|           |-- merged_flame_data.npz
|        |-- FLORENCE
|           |-- merged_flame_data.npz
|        |-- FRGC
|           |-- merged_flame_data.npz
|        |-- FT
|           |-- merged_flame_data.npz
|        |-- jaw_normalizer
|           |-- axis_normalize1.pt
|           |-- axis_normalize2.pt
|        |-- LYHM
|           |-- merged_flame_data.npz
|        |-- LYHM_TRAIN
|           |-- merged_flame_data.npz
|        |-- LYHM_VALID
|           |-- merged_flame_data.npz
|        |-- STIRLING
|           |-- merged_flame_data.npz
|        |-- WCPA
|           |-- merged_flame_data.npz
|        |-- WCPAPRE_TRAIN
|           |-- merged_flame_data.npz
|        |-- WCPAPRE_VALID
|           |-- merged_flame_data.npz
|        |-- WCPA_TRAIN
|           |-- merged_flame_data.npz
|        |-- WCPA_VALID
|           |-- merged_flame_data.npz
|        |-- reference_batch_expression.pt
|        |-- reference_batch_shape.pt
|        |-- statistics_expression.npz
|        |-- statistics_shape.npz
```


## Whole-body Data
We utilize the [ARCTIC](https://github.com/zc-alexfan/arctic), [EgoBody](https://sanweiliti.github.io/egobody/egobody.html), GRAB subset in [AMASS](https://amass.is.tue.mpg.de/), and [BEAT2](https://pantomatrix.github.io/EMAGE/) datasets for whole-body model training.
The whole-body data is prepared using the scripts under `./whole_body_process`. We also provide the preprocessed data in [Google Drive](https://drive.google.com/file/d/1e_sF8b1aSi8BhXaQonOzL4ZaQ8BDCZu6/view?usp=sharing). The processed data should be organized as follows:

```
${ROOT}  
|-- data  
|  |-- wholebody_data
|     |-- Arctic
|       |-- merged_smplx
|         |-- train
|            |-- body_pose.pt
|            |-- global_orient.pt
|            |-- left_hand_pose.pt
|            |-- right_hand_pose.pt
|         |-- val
|     |-- BEAT
|       |-- train
|         |-- body_pose.pt
|         |-- expression.pt
|         |-- global_orient.pt
|         |-- jaw_pose.pt
|         |-- left_hand_pose.pt
|         |-- right_hand_pose.pt
|       |-- val
|       |-- test
|     |-- EgoBody
|       |-- smplx_parameters
|          |-- smplx_camera_wearer_test.pt
|          |-- smplx_camera_wearer_train.pt
|          |-- smplx_camera_wearer_val.pt
|          |-- smplx_interactee_test.pt
|          |-- smplx_interactee_train.pt
|          |-- smplx_interactee_val.pt
|     |-- GRAB
|       |-- train
|         |-- body_pose.pt
|         |-- expression.pt
|         |-- global_orient.pt
|         |-- jaw_pose.pt
|         |-- left_hand_pose.pt
|         |-- right_hand_pose.pt
|     |-- reference_batch.pt
|     |-- statistics.npz
```


## Building the FAISS Index

To evaluate generation quality using the `dNN` metrics described in our paper (with `--task eval_generation`), you need to build a FAISS index from the prepared datasets. 

After you have finished preparing the datasets as described above, run the `build_faiss.py` script. This script will process the data and generate two files for each dataset type:
- `all_data.pt`: a merged file containing all relevant data samples
- `faiss_index.bin`: the FAISS index used for efficient nearest neighbor search

These files are required for computing the `dNN` metrics during evaluation.
