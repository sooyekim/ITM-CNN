# ITM-CNN
This is the official project repository for our paper ITM-CNN, presented in ACCV 2018.

We provide the test code along with the trained weights. The test code generates an HDR video file (YUV) of 10 bits/pixel, 
after the PQ-OETF, in the BT.2020 color container, from an SDR video (YUV) of 8 bits/pixel, after the Gamma curve, in the BT.709 color container.
The resulting HDR YUV file can be viewed on HDR TVs after encoding with the above specifications.

More details can be found in the paper.

**Reference**: Soo Ye Kim, Dae Eun Kim, Munchurl Kim. ITM-CNN: Learning the Inverse Tone Mapping from Low Dynamic Range Video to High Dynamic Range Displays using Convolutional Neural Networks. 
*Asian Conference on Computer Vision*, 2018.

### Requirements
Our code is implemented using MatConvNet. (MATLAB required)

Appropriate installations of MatConvNet must be made through the official website: <http://www.vlfeat.org/matconvnet/>
Detailed instructions on installing MatConvNet can be found in: <http://www.vlfeat.org/matconvnet/install/>

## Test code

### Quick Start
1. Download all files.
2. Place the files in **/+dagnn/** to **\<MatConvNet\>/matlab/+dagnn**
3. Set the input and output file specifications in lines 2-8 of **test.m** (or **test_lowmem.m**)
4. Run **test.m** (or **test_lowmem.m**)

### Description
- The provided test code will generate an HDR YUV video file of 10 bits/pixel, after the PQ-OETF, in the BT.2020 color container.
- The generated file can be viewed on HDR TVs after encoding using an appropriate codec for HDR videos (e.g. HEVC) with the above specifications.
- We provide options for CPU & GPU computations.

In case of an 'out of memory' error depending on the memory capacity of the GPU, we also provide a low memory version of the code (**test_lowmem.m**).
This code divides the input frame into (factor)x(factor) blocks and processes each block serially.
Thus, it may produce unpleasant artifacts such as vertical and horizontal lines in the block boundaries, and may consume 
additional time compared to the original code (**testm.m**). The low memory version is not the official implementation.

Note that **test.m** has been tested on an **NVIDIA TITAN Xp GPU**.

## Train code

Updated on 16.12.2019.

### Quick Start
1. Download all files.
2. Place the files in **/+dagnn/** to **\<MatConvNet\>/matlab/+dagnn**
3. Run in order: **train_1.m** -> **train_2.m** -> **train_3.m**

### Description
- The three training scripts correspond to the three training phases mentioned in the paper.
- The trained weights will be stored in the *net* folder.

## Contact
Please contact me via email (sooyekim@kaist.ac.kr) for any problems regarding the released code.
