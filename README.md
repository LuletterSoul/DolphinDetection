# DolphinDetection
 A project for dolphin detection based online video stream.


# Project Structure

```
├── data            
│   └── candidates                      // video workspace
│       └──03171503                     // daily data
│             └─── 0                    // video index
│               ├── original-streams    // save origianl video clips
│               ├── render-streams      // save render video clips with bboxs
│               ├── crops               // save bboxs patch
│               ├── ...
│             └─── 2                    // video index
│               ├── ...
│               ├── ...
│       └──03171530
│             └─── 0                    // video index
│       └── ...
│   └── offline                         // offilne video file
│       └── 0                           // save some videos from video indexed by 0
│       └── 1                           // save some videos from video indexed by 0
│       └── 2                           // save some videos from video indexed by 0
├── class                               // classifier training datasets
├── labels                              // Related label work
├── detection                           // Module of building object detection
│       └── ssd                         // ssd detection
│       └── capture.py                  // video capture
│       └── monitor.py                  // video services management
│       └── controller.py               // video controller
│       └── render.py                   // video generation
│       └── ...
├── model                               // checkpoints such as ssd, classfier,object tracker
├── pysot                               // object tracker service
├── interface                           // module of service interfaces
├── log                                 // log file
├── vcfg                                // video configutations
│       └── server-prod.py              // server configuration for production env
│       └── server-test.py              // server configuration for test env
│       └── ...
│       └── video-prod.py               // video configuraiton for production env
├──  config.py                          // video configuraiton object
├──  app.py                             // system entry
├──  requirements.txt                   // project dependency

```

# Download model checkponts

Dowload the model checkponts from [Google Drive](https://drive.google.com/open?id=1f_VvqUfYJ7xv5b2cDP6pKiU8xlO9WL-e), put all checkpoints in `model/`.

# Build Environment

cd`     DolphinDetection/`
## Create a conda env 
```
conda create -n dol python=3.8

```
## Activate python env
```
conda activate dol
```
## Install packages

```
pip install -r requirements.txt         # install packages
```

## Build pysot

```
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
```

```
python setup.py build_ext --inplace
```

Change pip mirror to `aliyun` if download slowly.


# Run on prod env

Run detection for video monitor indexed by 5 in PROD mode, configurations are loaded from vcfg/*-prod.json.
```
python app.py --env prod --http_ip 192.168.0.116 --http_port 8080 --cd_id 0 --dt_id 0 --enable 5 --use_sm 5 --send_msg -log_level INFO
```

# Run on test env

Run detection for video monitor indexed by 8 in TEST mode, configurations are loaded from vcfg/*-test.json. 
If the configuration set the item `online=offine`,it should be placed a video file at least at `data/offline/8`.
## Run single video detection

```
python app.py --env test --http_ip 127.0.0.1 --http_port 8080 --cd_id 0 --dt_id 0 --run_direct --enable 8 --use_sm 8 --log_level INFO
```

## Run multiples videos detection

Run detection for video monitor indexed by 8,9 in TEST mode.
```
python app.py --env test --http_ip 127.0.0.1 --http_port 8080 --cd_id 0 --dt_id 0 --run_direct --enable 8,9 --use_sm 8,9 --log_level INFO
```
Set `log_level=DEBUG`  to see more debug information. Remove `--run_direct` to activate `cron` timing run service.

# Shutdown system

 Input double `Enter`  to shutdown system.

# Analysis offline video

See video-*json ,set item `online` to `offline` to 
load corresponding video file.










