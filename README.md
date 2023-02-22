# DeepPilot4Pose

## Available for the following versions
- Ubuntu 20.04


# DeepPilot4Pose: 
## Recommended system
- Ubuntu 16.04
- ROS kinetic Kame
- Python 2.7.15
- Cuda 9.0
- Cudnn 7.3.0
- Tensorflow 1.12.0
- Keras 2.2.4

## DeepPilot

```bash
git clone https://github.com/QuetzalCpp/DeepPilot.git
cd DeepPilot
```

### Additional Resources
- [DeepPilot Models pretrained](https://inaoepedu-my.sharepoint.com/:f:/g/personal/carranza_inaoe_edu_mx/EslxVDqc9zBMmiV4mDH48KUBAcAHu0Ypt1rZLL6ifOjyoA?e=VYtMyT)
- [Datasets to train DeepPilot](https://inaoepedu-my.sharepoint.com/:f:/g/personal/carranza_inaoe_edu_mx/EslxVDqc9zBMmiV4mDH48KUBAcAHu0Ypt1rZLL6ifOjyoA?e=VYtMyT)

### Train DeepPilot

```bash
cd /bebop_ws/src/DeepPilot/DeepPilot_network
python train_deeppilot.py
```

### Start DeepPilot

```bash
cd /bebop_ws/src/DeepPilot/DeepPilot_network
python evaluation_mosaic-6img.py
```

## Reference
If you use any of data, model or code, please cite the following reference:

Rojas-Perez, L.O., & Martinez-Carranza, J. (2020). DeepPilot: A CNN for Autonomous Drone Racing. Sensors, 20(16), 4524.
https://doi.org/10.3390/s20164524

```
@article{rojas2020deeppilot,
  title={DeepPilot: A CNN for Autonomous Drone Racing},
  author={Rojas-Perez, Leticia Oyuki and Martinez-Carranza, Jose},
  journal={Sensors},
  volume={20},
  number={16},
  pages={4524},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
## Related References


 ## Acknowledgements
We are thankful for the processing time granted by the National Laboratory of Supercomputing (LNS) under the project 201902063C. The first author is thankful to Consejo Nacional de Ciencia y Tecnología (CONACYT) for the scholarship No. 924254. We are also thankful for the partial financial support granted via the FORDECYT project no. 296737 “Consorcio en Inteligencia Artificial” for the development of this work.


