# Trilobite-inspired neural nanophotonic light-field camera with extreme depth-of-field

Qingbin Fan, Weizhu Xu, Xuemei Hu, Wenqi Zhu, Tao Yue, Cheng Zhang, Feng Yan, Lu Chen, Henri J. Lezec, Yanqing Lu, Amit Agrawal and Ting Xu

This code implements a reconstruction algorithm based on multi-scale convolutional neural network.

## Testing

To perform inference on real-world captures, just run the 'test.py' file in your terminal. The code will load a checkpoint of  the neural network from 'checkpoint/checkpoint_50.pth' and process images in 'test_images'. The reconstructed images will be saved in 'restored_images' with the same name in 'test_images'.

## Requirements

The code has been tested with Python 3.7.4 using PyTorch1.4.0 running on Linux and the following library packages are installed to run this code:

```
PyTorch >= 1.4.0
PIL == 7.0.0
opencv-python == 4.3.0
Numpy
Scipy
matplotlib
```

## Citation


Fan, Q., Xu, W., Hu, X. et al. Trilobite-inspired neural nanophotonic light-field camera with extreme depth-of-field. Nat Commun 13, 2130 (2022). 

https://doi.org/10.1038/s41467-022-29568-y


## License

