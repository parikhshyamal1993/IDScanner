import sys , os
import cv2
import numpy as np
from pytesseract import image_to_string
import re

class PanCardExtractor():
    def __init__(self) -> None:
        self.kernal = np.ones((2,2),np.uint8)
    


    def Identifier(self,data):
        """
        If distance of y between income tax deparment vs pan number is higher than 250 than old
        variant else New variant

        for x1[1][1] - x[1][1] >=250:
            old
        else:
            new 
        
        """
        IncomeTaxIdentityList = ["INCOME TAX" ,"TAX","INCOME"]
        PanCardIdentityList = ["Permanent", "Account" ,"Number"]
        IncomeLine = 0
        PanCard = 0
        for i in range(len(data)):
            for items in IncomeTaxIdentityList:
                if re.findall(items , data[i][0]):
                    IncomeLine = data[i]
                    break
            for items in PanCardIdentityList:
                if re.findall(items , data[i][0]):
                    PanCard = data[i]
                    break

        if PanCard[1][1] - IncomeLine[1][1] > 250:
            return 2
        else:
            return 1

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
        typeIDList = []
        for cnt , high in zip(contours,hierarchy[0]):
            x,y,w,h = cv2.boundingRect(cnt)
            if h > 20 and w >30 and x <550:
                cv2.rectangle(panColor,(x,y),(x+w,y+h),(0,255,100),3)
                cells = adjusted[y-5:y+h,x:x+w]
                gray = cv2.cvtColor(cells,cv2.COLOR_BGR2GRAY)
                data = image_to_string(cells,config='--psm 7')
                allBoxes.append([data,[x,y,x+w,y+h]])
        cv2.imshow("Binary",cv2.resize(panColor,(600,375)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        allBoxes.reverse()
        return allBoxes

    def run(self,Image):
        HOCR = self.panExtract(Image)
        #print("Output:",HOCR)
        typeId = self.Identifier(HOCR)
        print("pan type" , typeId)
        if len(HOCR) >2:

            if typeId == 2:
                output = self.ExtractionType2(HOCR)
            elif typeId == 1:
                output = self.ExtractionType1(HOCR)
            #print("Pan EXtract",output)
            return output
        else:
            return " "

    def ExtractionType2(self,data):
        output = {}
        IncomeTaxIdentityList = ["INCOME TAX" ,"TAX","INCOME"]
        PanCardIdentityList = ["Permanent", "Account" ,"Number"]
        IncomeLine = 0
        PanCard = 0
        for i in range(len(data)):
            #print("items :",i)
            for items in PanCardIdentityList:
                if re.findall(items , data[i][0]):
                    PanCard = data[i]
                    output["PAN"] = re.sub(r'[^\w\s]','',re.sub('\n\x0c', '', data[i+1][0]))
                    break
            
            for items in IncomeTaxIdentityList:
                if re.findall(items , data[i][0]):
                    #print("ID name",data[i])
                    IncomeLine = data[i]
                    #print("Name:",data[i+1])
                    output["Name"] = re.sub(r'[^\w\s]','',re.sub('\n\x0c', '', data[i+1][0]))
                    #print("Fathers Name",data[i+2])
                    output["Fathers Name"] = re.sub(r'[^\w\s]','',re.sub('\n\x0c', '', data[i+2][0]))
                    #print("Date ",data[i+3])
                    output["Date"] = re.sub('\n\x0c', '', data[i+3][0])

                    break
        return output

    def ExtractionType1(self,data):
        output = {}
        IncomeTaxIdentityList = ["INCOME TAX" ,"TAX","INCOME"]
        PanCardIdentityList = ["Permanent", "Account" ,"Number"]
        DateList = ["Date of Birth","Date","Birth"]
        IncomeLine = 0
        PanCard = 0
        for i in range(len(data)):
            #print("items :",i)
            for items in PanCardIdentityList:
                if re.findall(items , data[i][0]):
                    PanCard = re.sub('\n\x0c', '', data[i][0])
                    output["PAN"] = re.sub(r'[^\w\s]','',re.sub('\n\x0c', '', data[i+1][0]))
                    #print("PAN",data[i][0],data[i+1][0])
                    #print("Name:",data[i+3])
                    output["Name"] = re.sub(r'[^\w\s]','', re.sub('\n\x0c', '', data[i+3][0]))
                    #print("Fathers Name",data[i+5])
                    output["Fathers Name"] = re.sub(r'[^\w\s]','',re.sub('\n\x0c', '', data[i+5][0]))
                    output["Data"] = re.sub('\n\x0c', '', data[i+8][0])
                    break
        
        return output



        

class AadharExtraction():
    def __init__(self) -> None:
        self.kernal = np.ones((2,2),np.uint8)

    def AadharExtract(self,image):
        y , x = 1200 , 749
        panColor = cv2.imread(image)
        panColor = cv2.resize(panColor,(y,x))
        #adjusted = cv2.convertScaleAbs(panColor, alpha=1.0, beta=0)
        panImage = cv2.cvtColor(panColor,cv2.COLOR_BGR2GRAY)
        #panImage = panImage / meanImg
        print("panImage",panImage.shape)
        thresh1 = cv2.adaptiveThreshold(panImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 199, 10)
         
        # cv2.imshow("dst",cv2.resize(thresh1,(600,375)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT,(23,1))
        clossing = cv2.morphologyEx(thresh1,cv2.MORPH_OPEN,kernel_)
        
        #clossing[clossing<140] = 0
        contours , hierarchy = cv2.findContours(clossing,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        allBoxes = []
        typeIDList = []
        for cnt , high in zip(contours,hierarchy[0]):
            x,y,w,h = cv2.boundingRect(cnt)
            if h > 20 and w >30 and h <60:
                cv2.rectangle(panColor,(x,y),(x+w,y+h),(0,255,100),3)
                cells = panColor[y:y+h,x:x+w]
                data = image_to_string(cells,config='--psm 7')
                allBoxes.append([re.sub("\n\x0c","",data),[x,y,x+w,y+h]])
        cv2.imshow("Binary",cv2.resize(panColor,(600,375)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        allBoxes.reverse()
        return allBoxes

    def run(self,image):
        data = self.AadharExtract(image)
        output = {}
        HOCR = {}
        output["Aadhar number"] = ""
        IDIdentityList = ["Government of India" ,"Government","India"]
        GenderIdentityList = ["male", "female" ,"transgender"]
        DateList = ["Birth","Year","YoB"]
        FatherList = ["Father"]
        IncomeLine = 0
        PanCard = 0
        for i in range(len(data)):
            print("items :",data[i][0])
            for items in IDIdentityList:

                if re.findall(items.lower() , data[i][0].lower()):
                    output["Name"] = data[i+2][0]
                    HOCR["Name"] = data[i+2][1]
                    #print("Fathers Name",data[i+5])
                    break
            for items in GenderIdentityList:
                if re.findall(items.lower() , data[i][0].lower()):
                    try:
                        gender = data[i][0].split("/")[-1]
                    except:
                        gender = data[i][0]

                    output["gender"] = gender
                    HOCR["gender"] = items
                    #print("Fathers Name",data[i+5])
                    break
            for items in DateList:
                if re.findall(items.lower() , data[i][0].lower()):
                    print("date",data[i][0])
                    date = "".join([inte for inte in data[i][0].split() if inte.isdigit()])
                    output["date"] = date
                    HOCR["date"] = data[i][1]
                    #print("Fathers Name",data[i+5])
                    break
            if re.sub(" ", "",data[i][0]).isdigit():
                print("numbers",data[i][0])
                output["Aadhar number"] += data[i][0]


        return output


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
    aadhar = AadharExtraction()
    outPuts = pan.run(sys.argv[1])
    print("output :",outPuts)