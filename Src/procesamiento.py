import numpy as np
import matplotlib.pyplot as plt
import math

def capaRoja(img):#slising
    newImg = np.copy(img)
    newImg[:,:,1] = 0
    newImg[:,:,2] = 0
    return newImg

def capaVerde(img):
    newImg = np.copy(img)
    newImg[:,:,0] = 0
    newImg[:,:,2] = 0
    return newImg

def capaAzul(img):
    newImg = np.copy(img)
    newImg[:,:,0] = 0
    newImg[:,:,1] = 0
    return newImg

def capaAmarillo(img):
    newImg = np.copy(img)
    newImg[:,:,0] = 1
    newImg[:,:,1] = 1
    return newImg

def capaCyan(img):
    newImg = np.copy(img)
    newImg[:,:,1] = 1
    newImg[:,:,2] = 1
    return newImg

def capaMagenta(img):
    newImg = np.copy(img)
    newImg[:,:,0] = 1
    newImg[:,:,2] = 1
    return newImg


def capaNegativa(img):
    newImg = np.copy(img)
    newImg = 1-newImg
    return newImg

def unirCanales(R,G,B):
    return R+G+B

def GrisAverage(img):
    newImg = (img[:,:,0]+img[:,:,1]+img[:,:,2])/3
    return newImg

def midgray(img):
    newImg = (np.maximum(img[:,:,0],img[:,:,1],img[:,:,2])+np.minimum(img[:,:,0],img[:,:,1],img[:,:,2]))/2
    return newImg

def ajusteBrillo(img,brillo):
    newImg = np.copy(img)
    return newImg+brillo

def ajusteCanal(img,brillo,canal):
    newImg = np.copy(img)
    newImg[:,:,canal]=newImg[:,:,canal]+brillo
    return newImg

def ajusteContraste(img, k, tipo):
    newImg = np.copy(img)
    if tipo==1:
        newImg = k*np.log10(img+1)
    else:
        newImg = k*np.log10(img-1)
    return newImg

def binarizar(img,umbral):
    newImg = np.copy(img)
    newImg = (newImg[:,:,0]+newImg[:,:,1]+newImg[:,:,2]/3)
    newImg = (newImg > umbral)
    return newImg

