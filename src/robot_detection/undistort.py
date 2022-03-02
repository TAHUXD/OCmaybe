import cv2
import sys
import numpy as np
# You should replace these 3 lines with the output in calibration step
DIM=(960, 600)
K=np.array([[806.2172727144507, 0.0, 435.0782605288524], [0.0, 865.8147968858042, 349.3906287372319], [0.0, 0.0, 1.0]])
D=np.array([[-0.002437521948018968], [-0.939151071175389], [1.7530142401098896], [-1.2065884991627005]])
# DIM=(480, 300)
# K=np.array([[666.9415965508599, 0.0, 431.95115471194333], [0.0, 717.6422902502163, 278.5371490457156], [0.0, 0.0, 1.0]])
# D=np.array([[0.037063993055043515], [-0.2506574643279681], [0.09389006223318914], [0.08024155824229262]])

class undistort(object):

    def __init__(self):
        print ('undistort running')

    def undistort(self, img, balance=0.0, dim2=(480, 300), dim3=(480, 300)):    
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort   
        assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"    
        if not dim2:
            dim2 = dim1    
        if not dim3:
            dim3 = dim1    
        
        scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img
        # cv2.imshow("undistorted", undistorted_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()