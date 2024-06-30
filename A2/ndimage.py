import numpy as np
import cv2
import matplotlib.pyplot as plt
import array
import math
import random
from scipy import ndimage,spatial
from collections.abc import Iterable

print("imported")

o = ndimage.correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])

print(o)

def sobel(input,axis=-1,output=None,mode='reflect',cval=0.0):
    '''
    This function is used to calculate the derivative of the image using sobel filter
    Args:
    input: input image
    axis: axis along which the derivative is calculated
    output: output image
    mode: mode of the filter
    cval: constant value
    Returns:
    The derivative of the image
    '''
    input =np.asarray(input)
    axis = normailze_axis_index(axis,input.ndim)
    output = get_output(output,input)
    modes = normalize_sequence(mode,input.ndim)
    ndimage.correlate1d(input, [-1, 0, 1], axis, output, modes[axis], cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        ndimage.correlate1d(output, [1, 2, 1], ii, output, modes[ii], cval, 0)

    return output

def correlate1d(input, weights, axis=-1, output=None, mode="reflect",cval=0.0, origin=0):
    input = np.asarray(input)        
    weights = np.asarray(weights)
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input or complex_weights:
        if complex_weights:
            weights = weights.conj()
            weights = weights.astype(np.complex128, copy=False)
        kwargs = dict(axis=axis, mode=mode, origin=origin)
        output = get_output(output, input, complex_output=True)
        return _complex_via_real_components(correlate1d, input, weights,output, cval, **kwargs)

    output = get_output(output, input)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = normailze_axis_index(axis, input.ndim)
    if (origin < -(len(weights) // 2)) or (origin > (len(weights) - 1) // 2):
        raise ValueError('Invalid origin; origin must satisfy '
                         '-(len(weights) // 2) <= origin <= '
                         '(len(weights)-1) // 2')
    mode = _extend_mode_to_code(mode)
    print("mode: ", mode)
    
    ndimage.correlate1d(input, weights, axis, output, mode, cval,origin)
    print("output: \n", output)
    return output

def _extend_mode_to_code(mode):
    """Convert an extension mode to the corresponding integer code.
    """
    print("mode :",mode)
    if mode == 'nearest':
        return 0
    elif mode == 'wrap':
        return 1
    elif mode in ['reflect', 'grid-mirror']:
        return 2
    elif mode == 'mirror':
        return 3
    elif mode == 'constant':
        return 4
    elif mode == 'grid-wrap':
        return 5
    elif mode == 'grid-constant':
        return 6
    else:
        raise RuntimeError('boundary mode not supported')
    
def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    """Complex convolution via a linear combination of real convolutions."""
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input and complex_weights:
        # real component of the output
        func(input.real, weights.real, output=output.real, cval=np.real(cval), **kwargs)
        output.real -= func(input.imag, weights.imag, output=None,cval=np.imag(cval), **kwargs)
        # imaginary component of the output
        func(input.real, weights.imag, output=output.imag,cval=np.real(cval), **kwargs)
        output.imag += func(input.imag, weights.real, output=None,cval=np.imag(cval), **kwargs)
    elif complex_input:
        func(input.real, weights, output=output.real, cval=np.real(cval),**kwargs)
        func(input.imag, weights, output=output.imag, cval=np.imag(cval),**kwargs)
    else:
        if np.iscomplexobj(cval):
            raise ValueError("Cannot provide a complex-valued cval when the input is real.")
        func(input, weights.real, output=output.real, cval=cval, **kwargs)
        func(input, weights.imag, output=output.imag, cval=cval, **kwargs)
    return output


def normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(input, str)
    if not is_str and isinstance(input, Iterable):
        normalized = list(input)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized

def get_output(output, input, shape=None, complex_output=False):
    if shape is None:
        shape = input.shape
    if output is None:
        if not complex_output:
            output = np.zeros(shape, dtype=input.dtype.name)
        else:
            complex_type = np.promote_types(input.dtype, np.complex64)
            output = np.zeros(shape, dtype=complex_type)
    elif isinstance(output, (type, np.dtype)):
        # Classes (like `np.float32`) and dtypes are interpreted as dtype
        if complex_output and np.dtype(output).kind != 'c':
            print("promoting specified output dtype to complex")
            output = np.promote_types(output, np.complex64)

        output = np.zeros(shape, dtype=output)
    elif isinstance(output, str):
        # testsuite only appears to cover
        # f->np.float32 here
        f_dict = {"f": np.float32,
                  "d": np.float64,
                  "F": np.complex64,
                  "D": np.complex128}
        output = f_dict[output]
        if complex_output and np.dtype(output).kind != 'c':
            raise RuntimeError("output must have complex dtype")
        output = np.zeros(shape, dtype=output)
    elif output.shape != shape:
        raise RuntimeError("output shape not correct")
    elif complex_output and output.dtype.kind != 'c':
        raise RuntimeError("output must have complex dtype")
    return output

def normailze_axis_index(axis,input):
    print("axis",axis)
    print("input",input)
    if axis < -input or axis >= input:
        msg = f"axis{axis} is out of range for array of dimension{input}"
        raise ValueError(msg)
    
    if axis<0:
        axis = axis +input

    return axis

def harriscornerdetector(Image):
  '''
  Extra Credit Part
  Goal:Compute Harris Features
  Steps Followed
  1.Apply two filters on the entire image to get the derivative image of x axis and y axis
  2.Computed the harris matrix of each pixel in its neighborhood using a gaussian mask and the derivative images.
  Sum over a 5X5 window. Apply 5X5 Guassian mask with .5 standard deviation 
  3.Then compute the harris score using the matrix
  The response is given by the formula 
  R_score=det(M)-alpha*trace(M)^2 for each pixel window
  4.Finally we take the eigenvector corresponding to the first eigenvalue as the orientation of the feature transformed to radian by atan() and atan()+pi  
  
  '''
  
  
  harris = np.zeros(Image.shape[:2])
#   orientations = np.zeros(Image.shape[:2])
  #Step 1
    i_x = ndimage.sobel(Image, axis=-1)
    plt.imshow(result, cmap='gray')
    plt.title("thresholded")
    plt.show()
    i_y = ndimage.sobel(Image, axis=0) 
      print(i_y)
  i_x_sqr = i_x**2
  print(i_x_sqr)
  i_y_sqr = i_y**2
  print(i_y_sqr)
  i_x_times_i_y = i_x*i_y
  print(i_x_times_i_y)

  #Step 2
  s = 0.9 #Sigma
  G = 31  #Gauss Mask   
  truncate_SD = G/(s*2)
  sumix2 = ndimage.gaussian_filter(i_x_sqr, s, truncate=truncate_SD)
  print(sumix2)
  sumiy2 = ndimage.gaussian_filter(i_y_sqr, s, truncate=truncate_SD)
  print(sumiy2)
  sumixiy2 = ndimage.gaussian_filter(i_x_times_i_y, s, truncate=truncate_SD)
  print(sumixiy2)
  
  #Step 3
  alpha = 0.01
  det = sumix2*sumiy2 - sumixiy2 **2
  print("det/n",det)
  trace = sumix2+sumiy2
  print(trace)
  harris = det- alpha*(trace**2)
  print(harris)
#   orientations = np.degrees(np.arctan2(i_y.flatten(),i_x.flatten()).reshape(orientations.shape)) 
#   return harris, orientations


def LocalMaxima(Image):
    '''
    This function takes a np array containing the Harris score at
    each pixel and returns an np array containing True/False at
    each pixel, depending on whether the pixel is a local maxima 
    Steps adopted
    1.Calculate the local maxima image
    2.And find the maximum pixels in the 7X7 window
    3.Then return true when pixel is the maximum, otherwise false
    '''
    destImage = np.zeros_like(Image, bool)
    harrisImage_max = ndimage.maximum_filter(Image, size=(81,81))
    destImage = (Image == harrisImage_max)
    return destImage


def detectKeypoints(image):
    '''
    This function takes in the  image and returns detected keypoints
    Steps:
    1.Grayscale image used for Harris detection
    2.Call harriscornerdetector() which gives the harris score at each pixel
    position
    3.Compute local maxima in the Harris image
    4.Update the cv2.KeyPoint() class objects with the coordinate, size, angle and response
    '''
    image = image.astype(np.float32)
    image /= 255.
    h, w = image.shape[:2]
    keypoints = []
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris, orientation = harriscornerdetector(grayImage)
    maxi = LocalMaxima(harris)

    for y in range(h):
        for x in range(w):
            if not maxi[y, x]:
                continue

            f = cv2.KeyPoint()
            f.pt = x, y
            f.size = 1
            #f.angle = orientation[y, x]
            #f.response = harris[y, x]
            keypoints.append(f)

    return keypoints

k=correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
print("k\n",k)
image_path = "data/Images/Field/1.jpg"
image = cv2.imread(image_path)

# Detect keypoints
keypoints = detectKeypoints(image)

print(len(keypoints))
#Visualize keypoints (without plotting on the image)
# for keypoint in keypoints:
#     print("Keypoint coordinates:", keypoint.pt)
# Visualize keypoints
for keypoint in keypoints:
    center = (int(keypoint.pt[0]), int(keypoint.pt[1]))
    cv2.circle(image, center, 3, (0, 0, 255), -1)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()        