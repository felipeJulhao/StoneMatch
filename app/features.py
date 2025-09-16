import cv2, numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from .config import LBP_P, LBP_R, RESIZE_TARGET

def gray_world_wb(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    b,g,r = img[...,0].mean(), img[...,1].mean(), img[...,2].mean()
    gray = (b+g+r)/3.0
    gains = np.array([gray/max(b,1e-6), gray/max(g,1e-6), gray/max(r,1e-6)], np.float32)
    img *= gains
    return np.clip(img,0,255).astype(np.uint8)

def segment_main_object(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,41,10)
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h,w = gray.shape
        return img_bgr, np.ones((h,w), np.uint8)*255
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray); cv2.drawContours(mask,[c],-1,255,-1)
    x,y,w,h = cv2.boundingRect(c)
    return img_bgr[y:y+h, x:x+w], mask[y:y+h, x:x+w]

def _resize_align(crop, mask, target=RESIZE_TARGET):
    h,w = crop.shape[:2]; scale = target / max(1,min(h,w))
    crop = cv2.resize(crop,(int(w*scale),int(h*scale)), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask,(crop.shape[1],crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys,xs = np.nonzero(mask); pts = np.column_stack((xs,ys)).astype(np.float32)
    if len(pts)>10:
        mean, eig = cv2.PCACompute(pts, mean=None, maxComponents=2)
        ang = -np.degrees(np.arctan2(eig[0,1], eig[0,0]))
        M = cv2.getRotationMatrix2D(tuple(mean[0]), ang, 1.0)
        crop = cv2.warpAffine(crop, M, (crop.shape[1],crop.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (mask.shape[1],mask.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return crop, mask

def _lab_hist(lab, mask, bins=(16,16,16)):
    hist = cv2.calcHist([lab],[0,1,2],mask,bins,[0,256,0,256,0,256]).flatten().astype(np.float32)
    s = hist.sum(); return hist/s if s>0 else hist

def _lbp_hist(gray, mask, P=LBP_P, R=LBP_R):
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    m = mask.astype(bool); vals = lbp[m].ravel().astype(np.float32)
    bins = P+2
    hist,_ = np.histogram(vals, bins=bins, range=(0,bins))
    hist = hist.astype(np.float32); s=hist.sum()
    return hist/s if s>0 else hist

def _gabor_energy(gray, mask, freqs=(0.1,0.2,0.3,0.4),
                  thetas=(0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6)):
    m = mask.astype(bool); g = gray.astype(np.float32)/255.0; feats=[]
    for f in freqs:
        for t in thetas:
            real, imag = gabor(g, frequency=f, theta=t)
            feats.append(((real**2+imag**2)[m].mean()).astype(np.float32))
    return np.array(feats, np.float32)

def preprocess_and_features(img_bgr: np.ndarray):
    img  = gray_world_wb(img_bgr)
    crop, mask = segment_main_object(img)
    crop, mask = _resize_align(crop, mask)

    lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    m    = mask.astype(bool)
    labp = lab[m] if m.any() else lab.reshape(-1,3)
    mu   = labp.mean(axis=0).astype(np.float32)
    sd   = labp.std(axis=0).astype(np.float32)

    v = np.concatenate([
        mu, sd,              # 6
        _lab_hist(lab, mask),# 4096
        _lbp_hist(gray, mask), # 18
        _gabor_energy(gray, mask), # 24
        np.array([float(lab[...,0][m].mean() if m.any() else lab[...,0].mean()),
                  float((lab[...,0]>240).sum()/max(lab[...,0].size,1))], np.float32) # 2
    ]).astype(np.float32)

    return crop, mask, v
