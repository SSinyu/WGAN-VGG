# WGAN_VGG [DEPRECATED]
Implementation of Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss
https://arxiv.org/abs/1708.00961    

<img src="https://github.com/SSinyu/WGAN_VGG/blob/master/img/wgan_vgg.PNG" width="550"/> 

-----

### DATASET

The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge by Mayo Clinic (I can't share this data, you should ask at the URL below if you want)  
https://www.aapm.org/GrandChallenge/LowDoseCT/

The data_path should look like:


    data_path
    ├── L067
    │   ├── quarter_3mm
    │   │       ├── L067_QD_3_1.CT.0004.0001 ~ .IMA
    │   │       ├── L067_QD_3_1.CT.0004.0002 ~ .IMA
    │   │       └── ...
    │   └── full_3mm
    │           ├── L067_FD_3_1.CT.0004.0001 ~ .IMA
    │           ├── L067_FD_3_1.CT.0004.0002 ~ .IMA
    │           └── ...
    ├── L096
    │   ├── quarter_3mm
    │   │       └── ...
    │   └── full_3mm
    │           └── ...      
    ...
    │
    └── L506
        ├── quarter_3mm
        │       └── ...
        └── full_3mm
                └── ...     

-------

## Use
Check the arguments.

1. run `python prep.py` to convert 'dicom file' to 'numpy array'
2. run `python main.py --load_mode=0` to training. If the available memory(RAM) is more than 10GB, it is faster to run `--load_mode=1`.
3. run `python main.py --mode='test' --test_iters=***` to test.

