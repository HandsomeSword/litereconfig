import cv2



def hog_extractor(self, input_image):
    
    winSize = (320, 480)
    input_image = cv2.resize(input_image,winSize)
    blockSize = (80, 80)  # 105
    blockStride = (80, 80)
    cellSize = (16, 16)
    Bin = 9  # 3780
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, Bin)
    return hog.compute(input_image)[:, 0]



