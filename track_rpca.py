'''
Created on Oct 15, 2015

@author: krsna
'''
import cv2
import os, sys, commands
import numpy as np
import scipy
from scipy.io import loadmat, savemat
from pylab import *
from numpy.linalg import norm
from scipy.linalg import qr
import time


'''
http://kastnerkyle.github.io/posts/robust-matrix-decomposition/
https://jeremykarnowski.wordpress.com/2015/08/31/robust-principal-component-analysis-via-admm-in-python/
'''
# Use these to loop later?
movie_dir = 'Antz_scenes'#'../tanaya_shotdetect/scenes/'
#0022.avi
movie_name = '0022.avi'#'0671.avi' #'Antz.avi'
movie_path = os.path.join(movie_dir, movie_name)
print movie_name

from numpy.linalg import norm, svd

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
#         print (x,y,w,h)
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



if False:
    mov = cv2.VideoCapture(movie_path)
    n_frames = mov.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    print n_frames
    
    frame_count = 0
    ret, frame1_ =  mov.read()
    frame1  = cv2.resize(frame1_, (frame1_.shape[1]/2, frame1_.shape[0]/2))
    frame1_gs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#     frame1_gs = cv2.equalizeHist(frame1_gs_)
#     hsv = np.zeros_like(frame1)
#     hsv[...,1] = 255
#     
#     md_avg = frame1.copy().astype('float')
#     X = np.zeros_like(())
    r,c = frame1_gs.shape
    print r,c
    X = frame1_gs.reshape((r*c,1))
    print X.shape
    
    while(mov.isOpened()):
            frame_count += 1
            #print 'retrieving frames'
            ret, frame_orig = mov.read()
            
            if frame_orig is not None:
                print frame_count,
                #0. Resize image
                frame  = cv2.resize(frame_orig, (frame_orig.shape[1]/2, frame_orig.shape[0]/2))
                gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gs_vector = gs_img.reshape((r*c,1))
                
                X = np.hstack((X,gs_vector))
            else: break
    
    print ''   
        
    print X.shape
    n_frames = X.shape[-1]     
    mov.release()
    cv2.destroyAllWindows()
    
    
    RUN_LRMF=True
    
    if RUN_LRMF:
        A, E = inexact_augmented_lagrange_multiplier(X)
        A = A.reshape(r,c,n_frames) * 255.
        E = E.reshape(r, c, n_frames) * 255.
        savemat("./IALM_background_subtraction2.mat", {"1": A, "2": E})
        print 'RPCA complete'
        L, S, G = go_dec(X)
        L = L.reshape(r, c, n_frames) * 255.
        S = S.reshape(r, c, n_frames) * 255.
        G = G.reshape(r, c, n_frames) * 255.
        savemat("./GoDec_background_subtraction2.mat", {"1": L, "2": S, "3": G, })
        print("GoDec complete")
        
    
#     fourcc=cv2.cv.CV_FOURCC('D','I','V','X')
#     writer = cv2.cv.CreateVideoWriter("test.avi", fourcc, 10, (r,c))

DISP = False
if DISP:    
#     ion()
    AE = loadmat("./GoDec_background_subtraction2.mat")
    E = AE['3']
    n_frames = E.shape[-1]
    for i in range(n_frames):
        x = np.asarray(E[:,:,i], 'uint8')
#         y = cv2.GaussianBlur(x,(0,0),5)
        print 'writing  %s_fg.png' % (str(i))
#         cv2.imwrite('%s_fg.png' % (str(i)),x)
        cv2.imshow('x',x)
        k = cv2.cv.WaitKey(30) & 0xff    
#         draw()
#     writer.release()






# from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import slic, felzenszwalb, quickshift, mark_boundaries
from skimage.util import img_as_float
from skimage.measure import structural_similarity as ssim

# Use these to loop later?
movie_dir = '/home/krsna/workspace/animation/tanaya_shotdetect/HTD_scenes/'#'Antz_scenes'#'../tanaya_shotdetect/scenes/'
#0022.avi
# very good example for HTD -0488 
shot_num = '0098'
movie_name = '%s.avi' % (shot_num)#'0022.avi'#'0671.avi' #'Antz.avi'
movie_path = os.path.join(movie_dir, movie_name)
print movie_name
lrmf_path = os.path.join(movie_dir, 'IALM_fgbg_%s.mat' % (shot_num))#"./IALM_background_subtraction2.mat"


ion()



color = np.random.randint(0,255,(100,3))
color = np.vstack(([0,255,0],color))
####  - - incorporating low rank factored matrix back into video

eig_vals=[]

if True:
    AE = loadmat(lrmf_path)
#     AE = loadmat("HTD_outputs/IALM_fgbg_671.mat")
    E = AE['FG']
    A = AE['BG']
    print E.shape
    mov = cv2.VideoCapture(movie_path)
    frame_count = 0
    
    nz_idx1 = np.nonzero(E[:,:,frame_count])
    cov1=cov(nz_idx1)
    max_eig1 = max(eigvals(cov1))
    eig_vals.append(max_eig1)
    
    
    ret, frame1_ =  mov.read()
    frame1  = cv2.resize(frame1_, (frame1_.shape[1]/2, frame1_.shape[0]/2))
    hsv = np.zeros_like(frame1)
    
    frame1_gs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    r,c = frame1_gs.shape
    flow_ref_img = frame1_gs.copy()
    
    E_frame1 = np.asarray(E[:,:,frame_count],'uint8')
    A_frame1 = np.asarray(A[:,:,frame_count],'uint8')
    gs_img_fg1 = frame1_gs *E_frame1
    
    dst1 = cv2.medianBlur(gs_img_fg1, 5)
    
    _, fg_img_thr = cv2.threshold(dst1, 30, 255, cv2.THRESH_BINARY)
    fg_img_rgb1 = cv2.cvtColor(fg_img_thr, cv2.COLOR_GRAY2BGR)
    #cv2.filter2D(E_frame,-1,kernel)
#     _,dst1_ = cv2.threshold(dst1, 0, 255, cv2.THRESH_BINARY)
    contours1, hierarchy1 = cv2.findContours(dst1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_box_list1 = get_bounding_boxes(contours1)
    trimmed_box_list1 = trim_boxes_by_area(bounding_box_list1, 0.9)
    bounding_box_list1 = merge_collided_bboxes( trimmed_box_list1, 20 )
    _,areas = get_biggest_box(bounding_box_list1,r,c)
    print areas, '....areas', bounding_box_list1
    # prepare CAMSHIFT - setup ROIs for tracking
#     frame_camshift = frame1.copy()
    frame_camshift = (fg_img_rgb1) & frame1
    bounding_box_list_to_start, _ = get_biggest_box(bounding_box_list1, r, c)  #bounding_box_list1
    track_windows, roi_hists = get_camshift_params(bounding_box_list_to_start,frame_camshift)
    print 'windows for camshift', track_windows
    
    fg_accumulator = fg_img_rgb1
    fg_adder = fg_img_thr
    
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                    
    INITIALIZE_counter=[0]
    while(mov.isOpened()):
            frame_count += 1
            
            #print 'retrieving frames'
            ret, frame_orig = mov.read()
            if frame_orig is not None:
                nz_idx = np.nonzero(E[:,:,frame_count])
                cov1=cov(nz_idx)
                max_eig = max(eigvals(cov1))
                eig_vals.append(max_eig)
                
                print frame_count,
                #0. Resize image
#                 plot(frame_count, frame_count)
#                 draw()
#                 show()
                frame  = cv2.resize(frame_orig, (frame_orig.shape[1]/2, frame_orig.shape[0]/2))
                rgb_frame = frame.copy()
                frame2 = frame.copy()
#                 cam_frame = frame.copy()
                frame_equ = rgb_equalize(frame)
                
                gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gs_img_cp = gs_img.copy()
                
                gs_img_lapl_ = cv2.Laplacian(gs_img_cp, cv2.CV_32F,3)
                gs_img_lapl = cv2.convertScaleAbs(gs_img_lapl_)
                kernel = np.ones((3,3),np.uint8)
                gs_img_lapl2 = cv2.dilate(gs_img_lapl, kernel)
                gradient = cv2.morphologyEx(gs_img, cv2.MORPH_RECT, kernel)

                
                
                
                gs_img_3ch = cv2.cvtColor(gs_img, cv2.COLOR_GRAY2BGR)
                
                OTSU=False
                if OTSU:
                    otsu_thresh = threshold_otsu(gs_img)
                    otsu_image = closing(gs_img > otsu_thresh,     square(3))
                    otsu_image = np.asarray(otsu_image*255,'uint8' )
                    print otsu_image.shape, '...otsu'
                    otsu_img_rgb = cv2.cvtColor(otsu_image, cv2.COLOR_GRAY2BGR)
                
#                 hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                
                A_frame = np.asarray(A[:,:,frame_count],'uint8')
                bg_img = cv2.cvtColor(A_frame, cv2.COLOR_GRAY2RGB)
#                 _, A_frame_thr = cv2.threshold(A_frame, 240, 255, cv2.THRESH_BINARY)
#                 if frame_count==1: A_frame_blur = cv2.medianBlur(A_frame_thr,3)
#                 A_frame_blur+=cv2.medianBlur(A_frame_thr,3)
                
                E_frame = np.asarray(E[:,:,frame_count]*255,'uint8')
                gs_img_fg = gs_img *E_frame
                
                
                kernel = np.ones((5,5),np.float32)/25
                E_blur = cv2.medianBlur(E_frame, 5)#cv2.filter2D(E_frame,-1,kernel)
                fg_img = E_blur.copy()
                
                _, fg_img_thr = cv2.threshold(fg_img, 30, 255, cv2.THRESH_BINARY)
                fg_rgb_img = cv2.cvtColor(fg_img_thr, cv2.COLOR_GRAY2RGB)
                
                #---- OPTICAL FLOW
                flow_input_img = gs_img.copy()                
                flow_output_img = get_flow_img(hsv, flow_ref_img, flow_input_img)
                flow_ref_img  = flow_input_img
                #--- EO - OPTICAL FLOW
                SLIC=True
                if SLIC:
                    segments_slic = slic(img_as_float(rgb_frame), n_segments = 10, sigma = 5)
#                     slic_img = mark_boundaries(img_as_float(cv2.cvtColor(fg_rgb_img, cv2.COLOR_BGR2RGB)), segments_slic)
               
                # use flow image & fg image masked rgb image for CAM shift?
                cam_frame = ((flow_output_img.copy() | fg_rgb_img)) & rgb_frame
                
                #accumulate all the fg_pixel info and 
                fg_accumulator = (fg_accumulator | (fg_rgb_img))
                fg_adder+=fg_img_thr
                
                # mask the frame with accumulated fg image and then use this for CAMshift to track on
                blanket_frame = frame_equ & fg_accumulator
                
                contours, hierarchy = cv2.findContours(E_blur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                
                bounding_box_list = get_bounding_boxes(contours)
                trimmed_box_list = trim_boxes_by_area(bounding_box_list, 0.9)
                bounding_box_list = merge_collided_bboxes( trimmed_box_list, 20 )
                
                print len(bounding_box_list)
                
                DRAW_ALL_BOXES=True
                if DRAW_ALL_BOXES:
                    for box in bounding_box_list:
    #                     if box != 
                        cv2.rectangle( cam_frame, box[0], box[1], (0,255,0), 1 )
                
                print 'boxes tracked by me', bounding_box_list
                
            #### ------ DO CAMSHIFT NOW
                
                box_after_shift=[]
                for i, track_window in enumerate(track_windows):
                    roi_hist = roi_hists[i]
                    # tracking on cam_frame space - not on the whole frame - call it a blanket frame
                    # other options are cam_frame or just frame
                    hsv = cv2.cvtColor(blanket_frame, cv2.COLOR_BGR2HSV)
                    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                    ret, win_after_shift = cv2.meanShift(dst, track_window, term_crit)
                    x,y,w,h = win_after_shift
                    box_after_shift.append(((x,y), (x+w,y+h)))
                
#                 box_after_shift = merge_collided_bboxes(box_after_shift, 0)
                
                for box in box_after_shift:
                    img2 = cv2.rectangle(frame, box[0], box[1], color[len(INITIALIZE_counter)],2)
#                     cv2.imshow('img2',cam_frame)
                print 'box_after shift',box_after_shift,            
                
                slic_img = mark_boundaries(rgb_frame, segments_slic)
                
                _,areas_after_shift = get_biggest_box(box_after_shift, r, c)
                print areas_after_shift
#                 merge_box_after_shift = merge_collided_bboxes(box_after_shift)
#                 for box in merge_box_after_shift:
#                     img2 = cv2.rectangle(frame, box[0], box[1], 255,2) 
#                     cv2.imshow('img2',cam_frame)
                
                '''-----------HERE -- if boxes tracked by me are overlap with the cam-shifted boxes - keep cam-shift box! DONT change the box-input to camshift''' 
#                 box_after_shift = merge_collided_bboxes(box_after_shift)
                
#                 bounding_box_for_shift = []
#                 overlap_status_all = []
#                 box_not_overlap=[]
#                 for box in bounding_box_list:
#                     for box_i in box_after_shift:
#                         overlap_status = check_collision(box, box_i)
#                         overlap_status_all.append(overlap_status)
#                         if overlap_status == True:
#                             if box_i not in bounding_box_for_shift:
#                                 bounding_box_for_shift.append(box_i)
#                         else:
#                             if box not in bounding_box_for_shift:
# #                                 bounding_box_for_shift.append(box)
#                                 box_not_overlap.append(box)
#                 
#                 _, areas_my_windows = get_biggest_box(bounding_box_list, r, c)
#                 print areas_after_shift, 'after shift'
#                 
                
                ### --- conditions to reinitialize boxes!!
#                 if max(areas_after_shift) > 0.6:
#                     print 'reinitializing'
#                     bounding_box_for_shift,_ = get_biggest_box(bounding_box_list,r,c)
#                  
#                  
#                 if bounding_box_for_shift == []:
#                     print 'REINITIALIZING.........', frame_count
#                     #empty the fg_accumulator - assuming a new fg has come to pic
#                     fg_accumulator=fg_rgb_img
#                     INITIALIZE_counter+=[frame_count]
#                     bounding_box_for_shift, _ = get_biggest_box(bounding_box_list,r,c)
#                 
#                 ### --- END of conditions to reinitialize boxes!!
#                 
#                 print 'new trackers', bounding_box_for_shift
#                 
#                 if len(INITIALIZE_counter)>1: print INITIALIZE_counter, 'frames at which I reinitialized - or where the tracked object fell off!'
                
                # input form cam shift for the next frame
                bounding_box_for_shift = box_after_shift
                track_windows, roi_hists = get_camshift_params(bounding_box_for_shift, cam_frame)
            
                _,test = cv2.threshold(fg_adder, 240, 255, cv2.THRESH_BINARY)
            
                cv2.imshow('x2',fg_accumulator & blanket_frame)#& flow_output_img)
                cv2.imshow('x',np.hstack((slic_img, flow_output_img)))
                cv2.imshow('y',E_frame)
                cv2.imshow('otsu', cam_frame)
                cv2.imshow('z1', frame)
    #             
    #             if frame_count>=200:
    #                 cv2.imwrite('%s_FG_final.png' % (str(frame_count)), cam_frame  )
                time.sleep(0.5)
                k = cv2.cv.WaitKey(30) & 0xff    
            else:
                plot(eig_vals)
                plot(range(len(eig_vals)), [mean(eig_vals) for i in range(len(eig_vals))])
                show()
    