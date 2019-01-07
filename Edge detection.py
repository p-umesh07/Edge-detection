import cv2
import numpy as np
import math

def show(title, image):
    cv2.imwrite(title,image)

def get_gradient(image):
    grad_x = image.copy() #[[0]*900 for i in range(image.shape[0])] #np.zeros((600,900))
    grad_y = image.copy() #[[0]*900 for i in range(image.shape[1])] #np.zeros((600,900))
    for grad_i in range(1, image.shape[0]-1):
        for grad_j in range(1,image.shape[1]-1):
            grad_x[grad_i-1][grad_j-1] = image[grad_i+1][grad_j]-image[grad_i-1][grad_j]
    for grad_i in range(1, image.shape[0]-1):
        for grad_j in range(1,image.shape[1]-1):
            grad_y[grad_i-1][grad_j-1] = image[grad_i][grad_j+1]-image[grad_i][grad_j-1]
    grad_x = np.asarray(grad_x)
    grad_y = np.asarray(grad_y)
    show("X-Gradient.png",grad_x)
    show("Y-Gradient.png",grad_y)
    return grad_x, grad_y

def getGaussianKernel(kSize, sigma=0.1):
    if (kSize%2==0):
        kSize=kSize+1
    kSizeHalf = (int)(kSize/2)
    l=-1*kSizeHalf
    h=kSizeHalf
    kernel = np.zeros((kSize, kSize))
    sum=0
    for i in np.arange( l, h+1 ):
        for j in np.arange( l, h+1 ):
            kernel[i+kSizeHalf][j+kSizeHalf] = (1/( 2*(math.pi)*(sigma**2)*( (math.e)**( ( ((i/(2*kSize))**2)+((j/(2*kSize))**2) )/( 2*(sigma**2) ) ) ) ))
            sum = sum + kernel[i+kSizeHalf][j+kSizeHalf]
    for i in np.arange( l, h+1 ):
        for j in np.arange( l, h+1 ):
            kernel[i+kSizeHalf][j+kSizeHalf] = (kernel[i+kSizeHalf][j+kSizeHalf]) / sum
    return kernel

def sobel_operator():
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    print("Sobel X-axis",sobel_x)
    #sobel_x = np.asarray(sobel_x)
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    print("Sobel Y-axis",sobel_y)
    #sobel_y = np.asarray(sobel_y)
    return (1/8)*sobel_x, (1/8)*sobel_y

def padding(image):
    image_pad = np.full((600,900),0)
    for i in range(1,image_pad.shape[0]-1):
        for j in range(1,image_pad.shape[1]-1):
            image_pad[i][j] = image[i-1][j-1]
    return image_pad

def convolution(image, sobel, edge, axis):
    max1 = image[1][1]
    for i in range(1, image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            edge[i][j] = sobel[0][0]*image[i-1][j-1]+sobel[0][1]*image[i-1][j]+sobel[0][2]*image[i-1][j+1]+sobel[1][0]*image[i][j-1]+sobel[1][1]*image[i][j]+sobel[1][2]*image[i][j+1]+sobel[2][0]*image[i+1][j-1]+sobel[2][1]*image[i+1][j]+sobel[2][2]*image[i+1][j+1]
            if(math.fabs(edge[i-1][j-1])>max1):
                max1 = math.fabs(edge[i][j])
    edge = np.asarray(edge)
    pos_edge = edge.copy()
    #pos_edge = (np.abs(edge) / np.max(np.abs(edge)))*255
    for i in range(0,edge.shape[0]):
        for j in range(0,edge.shape[1]):
            pos_edge[i][j] = ((edge[i][j]) / max1)*255
    cv2.imwrite(axis,pos_edge)
    return pos_edge

def merge(m,x,y):
    for i in range(0,m.shape[0]):
        for j in range(0,m.shape[1]):
            m[i][j] = math.sqrt( ((x[i][j])**2) + ((y[i][j])**2) )
    cv2.imwrite("Merged edge detection.png",m)

def normalization(x, name):
    x = (np.absolute(np.asarray(x)) / (np.asarray(x).max()))*255
    cv2.namedWindow('Normalized', cv2.WINDOW_NORMAL)
    cv2.imwrite(name+'.png', x)

def main():
    image = cv2.imread("/Users/poojaumesh/Downloads/proj1_cse573-2/task1.png",0)
    edge_x = [[0]*900 for j in range(image.shape[0])]
    edge_y = [[0]*900 for j in range(image.shape[1])]
    edge = image.copy()
    print(image.shape)
    show("Original image.png",image)
    #grad_x, grad_y = get_gradient(image)
    
    gaussianKernel = getGaussianKernel(5, 0.13301)
    print(gaussianKernel)
    convoluted_img_gaussianKernel = convolution(image, gaussianKernel,edge,"Gaussian image.png")
    #displayImage(convoluted_img_gaussianKernel, "convoluted_img_gaussianKernel")
    cv2.imwrite("convoluted_img_gaussianKernel.png", convoluted_img_gaussianKernel)#writeImage(convoluted_img_gaussianKernel, 'convoluted_img_gaussianKernel')
    img = convoluted_img_gaussianKernel
    
    sobel_x, sobel_y = sobel_operator()
    print(sobel_x)
    print(sobel_y)
    #padded_image = padding(image)
    x_edge = convolution(img, sobel_x, edge_x, "Horizontal edge detection.png")
    y_edge = convolution(img, sobel_y, edge_y, "Vertical edge detection.png")
#normalization(x_edge, 'NormX')
#normalization(y_edge, 'NormY')
#merged = image.copy()
#merge(merged, x_edge,y_edge)

main()
