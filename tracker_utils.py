'''
Created on Oct 22, 2015

@author: krsna
'''
import cv2
import os, sys, commands
import numpy as np
import scipy
from scipy.io import loadmat, savemat
from pylab import *
from numpy.linalg import norm, svd
from scipy.linalg import qr
import time

top = 0
bottom = 1
left = 0
right = 1
def merge_collided_bboxes( bbox_list, perc_overlap ):
    # For every bbox...
    for this_bbox in bbox_list:
        
        # Collision detect every other bbox:
        for other_bbox in bbox_list:
            if this_bbox is other_bbox: continue  # Skip self
            
            # Assume a collision to start out with:
            has_collision = True
            
            # These coords are in screen coords, so > means 
            # "lower than" and "further right than".  And < 
            # means "higher than" and "further left than".
            # We also inflate the box size by 20% to deal with
            # fuzziness in the data.  (Without this, there are many times a bbox
            # is short of overlap by just one or two pixels.)
            
            this_overlap = 1+(perc_overlap/100.0)
            other_overlap = 1-(perc_overlap/100.0)
            if (this_bbox[bottom][0]*this_overlap < other_bbox[top][0]*other_overlap): has_collision = False
            if (this_bbox[top][0]*other_overlap> other_bbox[bottom][0]*this_overlap): has_collision = False
            
            if (this_bbox[right][1]*this_overlap < other_bbox[left][1]*other_overlap): has_collision = False
            if (this_bbox[left][1]*other_overlap > other_bbox[right][1]*this_overlap): has_collision = False
            
            if has_collision:
                # merge these two bboxes into one, then start over:
                top_left_x = min( this_bbox[left][0], other_bbox[left][0] )
                top_left_y = min( this_bbox[left][1], other_bbox[left][1] )
                bottom_right_x = max( this_bbox[right][0], other_bbox[right][0] )
                bottom_right_y = max( this_bbox[right][1], other_bbox[right][1] )
                
                new_bbox = ( (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) )
                
                bbox_list.remove( this_bbox )
                bbox_list.remove( other_bbox )
                bbox_list.append( new_bbox )
                
                # Start over with the new list:
                return merge_collided_bboxes( bbox_list, perc_overlap )
    
    # When there are no collions between boxes, return that list:
    return bbox_list



def trim_boxes_by_area(bounding_box_list, thr):
    box_areas = []
    for box in bounding_box_list:
        box_width = box[right][0] - box[left][0]
        box_height = box[bottom][0] - box[top][0]
        box_areas.append( box_width * box_height )
         
        #cv.Rectangle( display_image, box[0], box[1], cv.CV_RGB(255,0,0), 1)
     
    average_box_area = 0.0
    if len(box_areas): average_box_area = float( sum(box_areas) ) / len(box_areas)
     
#                  pl.plot(frame_count,log(average_box_area),marker='*' )
#                 pl.draw()
     
    trimmed_box_list = []
    for box in bounding_box_list:
        box_width = box[right][0] - box[left][0]
        box_height = box[bottom][0] - box[top][0]
         
        # Only keep the box if it's not a tiny noise box:
        if (box_width * box_height) > average_box_area*float(thr): trimmed_box_list.append( box )

    return trimmed_box_list


def get_bounding_boxes(contours):
    bounding_box_list=[]
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)                    
        rect_points = (( x, y ),( x+w, y+h))
        bounding_box_list.append( rect_points )

    return bounding_box_list

def get_camshift_params(bounding_box_list1,frame1 ):
#     bounding_box_list1 = merge_collided_bboxes(bounding_box_list1, 0)
    track_windows = []
    roi_hists=[]
    for box in bounding_box_list1:
#         print box
        (x,y,w,h) = (box[0][0], box[0][1], (box[1][0] - box[0][0]), (box[1][1] - box[0][1]))
#         print (x,y,w,h) , '-----ini box'
        track_windows.append((x,y,w,h))
        # set up ROI
        roi = frame1[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        roi_hists.append(roi_hist)

    return track_windows, roi_hists

def check_collision(box1, box2):
    has_collision = True
    
    if (box1[bottom][0] < box2[top][0]): has_collision = False
    if (box1[top][0] > box2[bottom][0]): has_collision = False
    
    if (box1[right][1] < box2[left][1]): has_collision = False
    if (box1[left][1] > box2[right][1]): has_collision = False
 
    return has_collision

def get_biggest_box(bounding_box_list,r,c):
    box_area=[]
    for box in bounding_box_list:
        box_area_ = (box[right][0] - box[left][0]) * (box[bottom][1] - box[top][1])
        box_area += [box_area_]
    sum_of_areas = [sum(i) for i in box_area]
#     print sum_of_areas, 'areas of boxes'
    largest_box_idx = [i for i in range(len(sum_of_areas)) if sum_of_areas[i]==max(sum_of_areas)]
    bounding_box_list_LARGEST = [bounding_box_list[i] for i in largest_box_idx]
    norm_areas = [float(i)/(r*c) for i in sum_of_areas]
    return bounding_box_list_LARGEST, norm_areas



def get_flow_img(hsv, frame1_gs, gs_img):
    hsv[...,1] = 255

    flow = cv2.calcOpticalFlowFarneback(frame1_gs, gs_img, 0.5, 3, 9, 3, 5, 1.2, 0)
#     frame1_gs = gs_img

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    flow_gs = cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2GRAY)
    
#     flow_gs_sm = cv2.GaussianBlur(flow_gs, (0,0),5)
    _,flow_gs_thr = cv2.threshold(flow_gs, 40, 255, cv2.THRESH_BINARY)
    flow_gs_blur_ = cv2.medianBlur(flow_gs_thr, 5)
    flow_gs_blur = cv2.cvtColor(flow_gs_blur_, cv2.COLOR_GRAY2RGB)
    
    return flow_gs_blur
#

# def rgb_equalize(frame):
#     rgb_equ = frame.copy()
#     num_channels = rgb_equ.shape[-1]
#     for i in range(num_channels):
#         rgb_equ[:,:,i] = cv2.equalizeHist(frame[:,:,i])
#     
#     return rgb_equ

def rgb_equalize(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2HSV)
    hsv_img_equ = hsv_img.copy()
    v_img = hsv_img[:,:,-1]
    v_img_equ = cv2.equalizeHist(v_img)
    hsv_img_equ[:,:,-1] = v_img_equ
    rgb_img_equ = cv2.cvtColor(hsv_img_equ,cv2.COLOR_HSV2BGR)
    return rgb_img_equ


def inexact_augmented_lagrange_multiplier(X, lmbda=.01, tol=1e-3,
                                          maxiter=100, verbose=True):
    """
    Inexact Augmented Lagrange Multiplier
    """
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate
        E = Eupdate
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
            break
    if verbose:
        print("Finished at iteration %d" % (itr))  
    return A, E



def wthresh(a, thresh):

    #Soft wavelet threshold
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)

#Default threshold of .03 is assumed to be for input in the range 0-1...
#original matlab had 8 out of 255, which is about .03 scaled to 0-1 range

def go_dec(X, thresh=.03, rank=2, power=0, tol=1e-3,
           max_iter=100, random_seed=0, verbose=True):
    m, n = X.shape
    if m < n:
        X = X.T
    m, n = X.shape
    L = X
    S = np.zeros(L.shape)
    itr = 0
    random_state = np.random.RandomState(random_seed)    
    while True:
        Y2 = random_state.randn(n, rank)
        for i in range(power + 1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.T, Y1);
        Q, R = qr(Y2, mode='economic')
        L_new = np.dot(np.dot(L, Q), Q.T)
        T = L - L_new + S
        L = L_new
        S = wthresh(T, thresh)
        T -= S
        err = norm(T.ravel(), 2)
        if (err < tol) or (itr >= max_iter):
            break 
        L += T
        itr += 1
        print 'iteration is', itr
    #Is this even useful in soft GoDec? May be a display issue...
    G = X - L - S
    if m < n:
        L = L.T
        S = S.T
        G = G.T
    if verbose:
        print("Finished at iteration %d" % (itr))
    return L, S, G


