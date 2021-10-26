import sys , os
import cv2
import numpy as np
from pytesseract import image_to_string
class PanCardExtractor():
    def __init__(self) -> None:
        self.kernal = np.ones((2,2),np.uint8)
    
    def Identifier():
        """
        If distance of y between income tax deparment vs pan number is higher than 250 than old
        variant else New variant

        for x1[1][1] - x[1][1] >=250:
            old
        else:
            new 
        
        """

    def ExtractName(self,data):
        pass

    def basicTransform(self,img):
        _, mask = cv2.threshold(img,80,255,cv2.THRESH_BINARY_INV)
        img = cv2.bitwise_not(mask)
        return img

    def panExtract(self,image):
        panColor = cv2.imread(image)
        panColor = cv2.resize(panColor,(1200,743))
        adjusted = cv2.convertScaleAbs(panColor, alpha=1.5, beta=0)
        panImage = cv2.imread(image,0)
        meanImg = panImage.mean()
        #panImage = panImage / meanImg
        print("panImage",panImage.shape)
        panImage = cv2.resize(panImage,(1200,743))
        _, mask = cv2.threshold(panImage,90,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        dst = cv2.dilate(mask,self.kernal,iterations = 1)
        dst = cv2.bitwise_not(dst) 
        kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT,(31,5))
        clossing = cv2.morphologyEx((255-dst),cv2.MORPH_CLOSE,kernel_)
        contours , hierarchy = cv2.findContours(clossing,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        allBoxes = []
        for cnt , high in zip(contours,hierarchy[0]):
            x,y,w,h = cv2.boundingRect(cnt)
            if h > 20 and w >30 and x <450:
                cv2.rectangle(panColor,(x,y),(x+w,y+h),(255,127,0),1)
                cells = adjusted[y:y+h,x:x+w]
                #x,y,c = cells.shape
                # Cells2 = cv2.resize(cells,(y,x))
                gray = cv2.cvtColor(cells,cv2.COLOR_BGR2GRAY)
                #_, gray = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
                cv2.imshow("gray",gray)
                #cv2.imshow("mask",clossing)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                data = image_to_string(cells,config='--psm 7')
                print("pan :" , data)
                allBoxes.append([data,[x,y,x+w,y+h]])
        cv2.imshow("Binary",cv2.resize(panColor,(600,375)))
        cv2.imshow("mask",meanImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #data = image_to_string(dst)
        print("pan :" , data)
        print("original ",allBoxes)
        allBoxes.reverse()
        return allBoxes
        
class AadharExtraction():
    def __init__(self) -> None:
        pass

    def panExtract(self,):
        pass

class PassportExtractor():
    def __init__(self) -> None:
        pass

    def panExtract(self,):
        pass


class IDextract():
    def __init__(self) -> None:
        pass
    def Application(self,Image):
        image = cv2.imread(Image)
        data = image_to_string(panImage)
if __name__ == "__main__":
    pan = PanCardExtractor()

    outPuts = pan.panExtract(sys.argv[1])
    print("outPuts :" , outPuts)