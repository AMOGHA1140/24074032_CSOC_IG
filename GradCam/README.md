## What are the files all about?

`a_gradcam_visualisation.ipynb` - File containing functions and the code to get the grad-cam visualization presented in the pdf (the prefix a is just to get the file on top). Note that a part of the code also contains visualization for fine-tuned model too. 

`ensemble_models.ipynb` - Combines Resnet26 and Xception(5.9M) and tests it on the test set, ensuring appropriate transformations for the difference models.

`mapping.json` - Contains the mapping from "class number" to "class name" for caltech-256 dataset (1-indexed numbering)

`mnist_cnn.ipynb` - my implementation of CNNs on MNIST database with almost 99% test accuracy.

`main.py` - implementation of custom functions used throughout the project

`resnet18_scratch_caltech.ipynb` - using the default resnet18 from torchvision.models, but training from scratch on just caltech256. 

`resnet28_self_implementation.ipynb` - implementing the architecture similar to resnet18, but using bottleneck layers instead of simple ones, same number of times as in resnet18. 

`self_implementation_xception.ipynb` - implementing 5.9M parameter xception myself, and training on just caltech

`xception_fine_tuned.ipynb` - FIne tuning the default xception model from timm by just unfreezing the final layers. 



