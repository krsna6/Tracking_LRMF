'''
Created on Oct 3, 2015

@author: krsna
'''
import cv2
import os, sys, commands
import numpy as np
import scipy, time

from pylab import *
# Use these to loop later?
movie_dir = 'Antz_scenes'#'/home/krsna/workspace/animation/lrslibrary/output'
movie_name = '0013.avi' #'Antz.avi'
movie_path = os.path.join(movie_dir, movie_name)
print movie_name


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

def get_biggest_box(bounding_box_list):
    box_area=[]
    for box in bounding_box_list:
        box_area_ = (box[right][0] - box[left][0]) * (box[bottom][0] - box[top][0])
        box_area += [box_area_]
    sum_of_areas = [sum(i) for i in box_area]
    largest_box_idx = [i for i in range(len(sum_of_areas)) if sum_of_areas[i]==max(sum_of_areas)]
    bounding_box_list_LARGEST = [bounding_box_list[i] for i in largest_box_idx]
    return bounding_box_list_LARGEST





def detectMotion(image):
    ## gray to binary: threshold = 100 (arbitrary); maxValue = 255; type = cv2.THRESH_BINARY
    flag, binaryImage = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)  # | cv2.THRESH_OTSU) # cv2.THRESH_BINARY = 0

    # Use the Running Average as the static background
    # md_weight = 0.020 leaves artifacts lingering way too long.
    # md_weight = 0.320 works well at 320x240, 15fps.
    # md_weight should be roughly 1/a = num frames.
    cv2.accumulateWeighted(binaryImage, md_average, md_weight)

    # Convert the scale of the moving average.
    runAvg = cv2.convertScaleAbs(md_average)

    # Subtract the current frame from the moving average.
    md_result = cv2.absdiff(binaryImage, runAvg)

    cv2.imshow('Motion detect', md_result)

    #Find the contours
    contours, hierarchy = cv2.findContours(md_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours



if True:
    mov = cv2.VideoCapture(movie_path)
    
    frame_count = 0
    ret, frame1_ =  mov.read()
    frame1  = cv2.resize(frame1_, (frame1_.shape[1]/2, frame1_.shape[0]/2))
    frame1_gs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#     frame1_gs = cv2.equalizeHist(frame1_gs_)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    md_avg = frame1.copy().astype('float')
    
    
    while(1):
            #print 'retrieving frames'
            ret, frame_orig = mov.read()
            frame_count += 1
            print 'frame...', frame_count
            if frame_orig is not None:
                #0. Resize image
                frame  = cv2.resize(frame_orig, (frame_orig.shape[1]/2, frame_orig.shape[0]/2))
                
                # 1. Load color and GS images
                rgb_img = frame.copy()
                gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 
                print gs_img.shape
                
                
#                 flow = cv2.calcOpticalFlowFarneback(frame1_gs, gs_img, 0.5, 3, 9, 3, 5, 1.2, 0)
#                 frame1_gs = gs_img
# 
#                 mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#                 hsv[...,0] = ang*180/np.pi/2
#                 hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#                 flow_rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
#                 flow_gs = cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2GRAY)
#                 flow_gs_sm = cv2.GaussianBlur(flow_gs, (0,0),5)
#                 _,flow_gs_thr = cv2.threshold(flow_gs_sm, 50, 255, cv2.THRESH_BINARY)
#                 
                
                
                
                
                
                
                # 2. equalize images
                rgb_equ = frame.copy()
                num_channels = rgb_equ.shape[-1]
                for i in range(num_channels):
                    rgb_equ[:,:,i] = cv2.equalizeHist(frame[:,:,i])
                
                rgb_sm = cv2.GaussianBlur(rgb_img, (0,0),5)
                
                gs_equ = cv2.equalizeHist(gs_img)
                gs_sm = cv2.GaussianBlur(gs_equ, (0,0),5)
                
#                 md_avg = rgb_sm.copy().astype('float')
                md_res = rgb_sm.copy()
                md_weight = 0.32
#                 flag, binaryImage = cv2.threshold(rgb_sm, 100, 255, cv2.THRESH_BINARY) # | cv2.THRESH_OTSU) # cv2.THRESH_BINARY = 0
                
                
                
                
                # Use the Running Average as the static background
                # md_weight = 0.020 leaves artifacts lingering way too long.
                # md_weight = 0.320 works well at 320x240, 15fps.
                # md_weight should be roughly 1/a = num frames.
                cv2.accumulateWeighted(rgb_sm, md_avg, md_weight)
                # Convert the scale of the moving average.
                run_avg = cv2.convertScaleAbs(md_avg)
                
                # Subtract the current frame from the moving average.
                md_res = cv2.absdiff(rgb_sm, run_avg)
                # change to gray scale - threshold - smooth
                
                md_res_gs = cv2.cvtColor(md_res, cv2.COLOR_BGR2GRAY)
                
#                 avg_md_flow = ( md_res_gs + flow_gs )/2.0
                
                _, md_res_bin = cv2.threshold(md_res_gs, 2, 255, cv2.THRESH_BINARY )
#                 md_res_sm = cv2.GaussianBlur(md_res_bin, (0,0),5)
                _, md_res_sm2 = cv2.threshold(md_res_bin, 240, 255, cv2.THRESH_BINARY )
                fg_img =  md_res_sm2.copy()
                contours, hierarchy = cv2.findContours(md_res_sm2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)#
                
                
                bounding_box_list = get_bounding_boxes(contours)
                trimmed_box_list = trim_boxes_by_area(bounding_box_list, 0.3)
                bounding_box_list = merge_collided_bboxes( trimmed_box_list, 10 )
                
                print len(bounding_box_list)
                for box in bounding_box_list:
#                     if box != 
                    cv2.rectangle( frame, box[0], box[1], (0,255,0), 1 )
                
                print 'boxes tracked by me', bounding_box_list

                
#                 print size(contours)
#                 for cnt in contours:
#                     x,y,w,h = cv2.boundingRect(cnt)
#                     cv2.rectangle(rgb_img,(x,y),(x+w,y+h),(0,255,0),1)
# #                     print rect                
#                 cv_frame_size = cv2.cv.GetSize(cv2.cv.fromarray(frame))
#                 running_avg_image = cv2.cv.CreateImage( cv_frame_size, cv2.cv.IPL_DEPTH_32F, 3 )
#                 cv2.cv.RunningAvg( cv2.cv.fromarray(rgb_sm), running_avg_image, 0.32, None )
#                 running_avg_arr = np.asarray(cv2.cv.GetMat(running_avg_image))
#                 
                #Running average computation
                
                
                
                
                
#                 cv2.imshow('img1',flow_gs)
                cv2.imshow('img2',md_res_bin)    
                cv2.imshow('img3',frame)
                time.sleep(2)
                k = cv2.waitKey(60) & 0xff   


cv2.destroyAllWindows()
mov.release()               