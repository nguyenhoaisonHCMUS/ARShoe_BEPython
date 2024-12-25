import asyncio
from typing import Annotated
from fastapi.responses import JSONResponse
import cv2
# from skimage.io import imread => replace to imageio.v3
import imageio.v3 as iio
import os

from utils import *


ImgPath = 'data/barefeet_4.jpg'


from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins 
origins = [
    "http://localhost",  # allow frontend running on localhost
    "http://localhost:8000",  # allow frontend on FastAPI dev server
]

# CORS to app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.post("/")
async def read_root(file: Annotated[bytes, File()]): # file: UploadFile = File(...)
    feetSize = round(getSize(file))
    await asyncio.sleep(2)
    return JSONResponse(content={"size": feetSize})

@app.get("/hello")
async def get_hello():
    return JSONResponse(content={"message": "hello"})


def getSize(input_image):

    # oimg = imread(ImgPath)
    # oimg = iio.imread(ImgPath)
    oimg = iio.imread(input_image)

    if not os.path.exists('output'):
        os.makedirs('output')



    preprocessedOimg = preprocess(oimg)
    cv2.imwrite('output/preprocessedOimg.jpg', preprocessedOimg)

    clusteredImg = kMeans_cluster(preprocessedOimg)
    cv2.imwrite('output/clusteredImg.jpg', clusteredImg)

    edgedImg = edgeDetection(clusteredImg)
    cv2.imwrite('output/edgedImg.jpg', edgedImg)

    boundRect, contours, contours_poly, img = getBoundingBox(edgedImg)
    pdraw = drawCnt(boundRect[1], contours, contours_poly, img)
    cv2.imwrite('output/pdraw.jpg', pdraw)

    croppedImg, pcropedImg = cropOrig(boundRect[1], clusteredImg)
    cv2.imwrite('output/croppedImg.jpg', croppedImg)
    

    newImg = overlayImage(croppedImg, pcropedImg)
    cv2.imwrite('output/newImg.jpg', newImg)

    fedged = edgeDetection(newImg)
    fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
    fdraw = drawCnt(fboundRect[2], fcnt, fcntpoly, fimg)
    cv2.imwrite('output/fdraw.jpg', fdraw)

    return  calcFeetSize(pcropedImg, fboundRect)/10