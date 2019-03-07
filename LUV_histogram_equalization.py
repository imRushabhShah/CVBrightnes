import cv2
import numpy as np
import sys

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)


color = inputImage

x,y,_=color.shape
color1 = color[int(w1*x):int(w2*x),int(h1*y):int(h2*y),:]

def Img_to_non_linear_rgb(color):
    rgb=color/255
    return rgb

def non_linear_rgb_to_Image(rgb):
    (x,y,_)=rgb.shape
    image=np.zeros((x,y,3),dtype='double')
    for i in range(x):
        for j in range(y):
            for k in range(3):
                image[i][j][k] = int(rgb[i][j][k]*255)
    return image

def linear_to_non_linear(lRGB):
    (x,y,_)=lRGB.shape
    nlRGB=np.zeros((x,y,3),dtype='double')
    for i in range(x):
        for j in range(y):
            for k in range(3):
                if lRGB[i][j][k] < 0.00304:
                    val = lRGB[i][j][k]*12.92
                else:
                    val = ((lRGB[i][j][k]**(1/2.4))*1.055)-0.055
                if val > 1:
                    val=1
                elif val < 0:
                    val=0
                nlRGB[i][j][k] = val
    return nlRGB

def non_linear_to_linear(nlRGB):
    (x,y,_)=nlRGB.shape
    lRGB=np.zeros((x,y,3),dtype='double')
    for i in range(x):
        for j in range(y):
            for k in range(3):
                if nlRGB[i][j][k] < 0.03928:
                    lRGB[i][j][k] = nlRGB[i][j][k]/12.92
                else:
                    lRGB[i][j][k] = ((nlRGB[i][j][k]+0.055)/1.055)**(2.4)
    return lRGB

InvMAt = np.matrix("0.412453 0.35758 0.180423 ;0.212671 0.71516 0.072169; 0.019334 0.119193 0.950227")

def rgb_to_XYZ(rgb):
    (x,y,_)=rgb.shape
    XYZ=np.zeros((x,y,3),dtype='double')
    for i in range(x):
        for j in range(y):
            XYZ[i][j]=np.matmul(InvMAt,np.transpose(rgb[i][j]))  
    return XYZ

MAt = np.matrix("3.240479 -1.53715 -0.498535 ; -0.969256 1.875991 0.041556; 0.055648 -0.204043 1.057311")

def XYZ_to_rgb(XYZ):
    (x,y,_)=XYZ.shape
    rgb=np.zeros((x,y,3),dtype='double')
    for i in range(x):
        for j in range(y):
            rgb[i][j]=np.matmul(MAt,np.transpose(XYZ[i][j]))
    return rgb

uw = 3.8/19.22
vw = 9/19.22

def XYZ_to_LUV(XYZ):
    (x,y,_)=XYZ.shape
    LUV=np.zeros((x,y,3),dtype='double')
    for i in range(x):
        for j in range(y):
            X = XYZ[i][j][0]
            Y = XYZ[i][j][1]
            Z = XYZ[i][j][2]
            
            t=Y
            d = X + (15*Y) + (3*Z)
            if t > 0.008856:
                L = (116*(t**(1/3))-16)
            else:
                L = 903.3*t
            if L >100:
                L = 100
            elif L < 0:
                L = 0
                
            if d == 0:
                LUV[i][j][1] = 0
                LUV[i][j][2] = 0
            else:
                u_ = (4*X)/d
                v_ = (9*Y)/d
                LUV[i][j][1] = 13*L*(u_ - uw)
                LUV[i][j][2] = 13*L*(v_ - vw)
            LUV[i][j][0] = L
    return LUV


def LUV_to_XYZ(LUV):
    (x,y,_)=LUV.shape
    XYZ=np.zeros((x,y,3),dtype='double')
    for i in range(x):
        for j in range(y):
            L = LUV[i][j][0]
            u = LUV[i][j][1]
            v = LUV[i][j][2]
            if L > 7.9996:
                Y = ((L+16)/(116))**(3)
            else:
                Y = L/903.3

            if L ==0:
                XYZ[i][j][0] = 0
                XYZ[i][j][1] = Y
                XYZ[i][j][2] = 0
            
            else:    
                u_ = (u + (13 * uw * L))/(13 * L)
                v_ = (v + (13 * vw * L))/(13 * L)

                
                if v_ == 0:
                    XYZ[i][j][0] = 0
                    XYZ[i][j][1] = Y
                    XYZ[i][j][2] = 0
                else:
                    XYZ[i][j][0] = Y * 2.25 * (u_/v_)
                    XYZ[i][j][1] = Y
                    XYZ[i][j][2] = (Y * (3 - (0.75*u_)-(5 * v_)))/v_
                
    return XYZ

def LinearStreaching(LUV):
    (x,y,_)=LUV.shape
    L = LUV[:,:,0]
    A=0
    B=100
    a = L[0][0]
    b = L[0][0]
    for i in L:
        for j in i:
            if a>j:
                a=j
            if b<j:
                b=j
    for i in range(x):
        for j in range(y):
            LUV[i][j][0] = (LUV[i][j][0]-a)*(B-A)/(b-a)
    return LUV
def HistEqualization(LUV):
    (x,y,_)=LUV.shape
    L = LUV[:,:,0]
    H=np.zeros((102),dtype='uint')
    F=np.zeros((102),dtype='uint')
    G=np.zeros((102),dtype='uint')
    for i in L:
        for j in i:
            H[int(j)+1] += 1
    for i in range(1,102):
        F[i] = H[i] + F [i-1]
    
    for i in range(1,102):
        tp = np.floor((F[i] + F [i-1])*101*0.5/F[101])
        if tp >100:
            tp = 100
        G[i] = tp
    
    
    for i in range(x):
        for j in range(y):
            LUV[i][j][0] = G[int(LUV[i][j][0])+1]  
    return LUV

nlrgb=Img_to_non_linear_rgb(color1)
lrgb=non_linear_to_linear(nlrgb)
XYZ=rgb_to_XYZ(lrgb)
LUV=XYZ_to_LUV(XYZ)


LUV_new = HistEqualization(LUV)


XYZ = LUV_to_XYZ(LUV_new)
lrgb = XYZ_to_rgb(XYZ)
nlrgb = linear_to_non_linear(lrgb)
Image = non_linear_rgb_to_Image(nlrgb)


color[int(w1*x):int(w2*x),int(h1*y):int(h2*y),:]=Image

cv2.imwrite(name_output, color)


cv2.namedWindow('write_window', cv2.WINDOW_AUTOSIZE)
cv2.imshow('write_window', color)

cv2.waitKey(0)
cv2.destroyAllWindows()