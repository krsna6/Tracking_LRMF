'''
Created on Oct 22, 2015

@author: krsna
'''

from tracker_utils import *
import json

out_dir_name = './HTD_outputs'
os.system('mkdir '+out_dir_name)
movie_path = '/home/krsna/workspace/animation/movies/how_to_train_your_dragon_2.avi'
shot_json = '/home/krsna/workspace/animation/tanaya_shotdetect/scenes_how_to_train_your_dragon_025.json'



shot_dict = json.load(open(shot_json,'rU'))
frame_dict_list = shot_dict['frames']

num_shots = len(frame_dict_list)
print '# of shots = ', num_shots

frame_cut_points = [0]
for shot_i in range(num_shots):
    frame_cut = frame_dict_list[shot_i]['pkt_dts']
    frame_cut_points.append(frame_cut)
   
print frame_cut_points



frame_boundaries = []
for frame_i, frame_num in enumerate(frame_cut_points[1:]) : 
    frame_start = frame_cut_points[frame_i]
    frame_end = frame_num
    frame_bound = [frame_start, frame_end]
    
    frame_boundaries.append(frame_bound)


print frame_boundaries    


def frame_to_vector(frame):
    frame1  = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
    frame1_gs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    r,c = frame1_gs.shape
    X = frame1_gs.reshape((r*c,1))
    return X

mov = cv2.VideoCapture(movie_path)
frame_count = 0
shot_count = 0
frame_within_shot=0

shot_idx=[0]

ret, frame =  mov.read()

#size here is before reshaping - but we use x/2 size - so modify this
r_,c_,ch = frame.shape
r = r_/2
c = c_/2


FIRST_PASS_LRMF=False

if FIRST_PASS_LRMF:
    #-- initialize the first shot vector here

    X = frame_to_vector(frame)
    frame_count+=1

    while(mov.isOpened()):
        ret, frame =  mov.read()
    #     print 'new_shot...'
        which_shot_ = [i for i in range(num_shots) if frame_count in range(frame_boundaries[i][0], frame_boundaries[i][1])]
        which_shot = which_shot_[0]
        
        
        if not which_shot==shot_idx[-1]: 
            print 'shot change', which_shot
            print X.shape
            n_frames = X.shape[-1]
            ##--- Do stuff here----
            ## - at this point X holds all the vectors for the previous shot
            out_mat_name = os.path.join(out_dir_name, 'IALM_fgbg_%s.mat' % (str(which_shot-1)))
            out_npy_name = os.path.join(out_dir_name, 'IALM_fg_%s.npy' % (str(which_shot-1)))
            
            RUN_LRMF = True
            if RUN_LRMF:
                A, E = inexact_augmented_lagrange_multiplier(X)
                A = A.reshape(r,c,n_frames) * 255.
                E = E.reshape(r, c, n_frames) * 255.
                savemat(out_mat_name, {"BG": A, "FG": E})
                np.save(out_npy_name, E)
                print '!!!RPCA complete for shot #', which_shot-1
            
            ##--- once complete re-initialize the X matrix which has columns as vectorized frames and also add a lone vector there so as to update it
            X=[]
            print 'now accumulating shot ', which_shot
            #initialize first vector for the next mtrix here
            X=frame_to_vector(frame)
            
        else: 
            X = np.hstack((X,frame_to_vector(frame)))  
        
    #     print which_shot,
        shot_idx+=which_shot_
        frame_count+=1


SECOND_PASS_TRACK=True
if SECOND_PASS_TRACK:
#     print frame_boundaries[2:]
#     frame_boundaries = frame_boundaries[2:]
# #     num_shots = len(frame_boundaries)
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    while(mov.isOpened()):
        ret, frame =  mov.read()
        frame  = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2))
        frame_equ = rgb_equalize(frame)
        gs_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        which_shot_ = [i for i in range(num_shots) if frame_count in range(frame_boundaries[i][0], frame_boundaries[i][1])]
        
        if which_shot_:
            which_shot = which_shot_[0]
        
            # initialize it for frame 0
            if frame_count==0:
                out_mat_name = os.path.join(out_dir_name, 'IALM_fgbg_%s.mat' % (str(which_shot)))
                AE = loadmat(out_mat_name)
                FG = AE['FG']
                E_frame1 = np.asarray(FG[:,:,frame_count],'uint8')
                gs_img_fg = gs_img*E_frame1
                dst1 = cv2.medianBlur(E_frame1, 5)
                contours1, hierarchy1 = cv2.findContours(dst1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                bounding_box_list1 = get_bounding_boxes(contours1)
                trimmed_box_list1 = trim_boxes_by_area(bounding_box_list1, 0.9)
                bounding_box_list1 = merge_collided_bboxes( trimmed_box_list1, 20 )
                # prepare CAMSHIFT - setup ROIs for tracking
                frame_camshift = frame_equ.copy()
                track_windows, roi_hists = get_camshift_params(bounding_box_list1,frame_camshift)
                print 'windows for camshift', track_windows
                
                
                hsv = np.zeros_like(frame)
                flow_ref_img = gs_img.copy()
    #             BG = AE['BG']
            
            if not which_shot==shot_idx[-1]: 
                print 'shot change', which_shot
                ##--- Do initializing stuff here----
                cv2.destroyAllWindows()
                frame_within_shot=0
                
                ## - at this point X holds all the vectors for the previous shot
                out_mat_name = os.path.join(out_dir_name, 'IALM_fgbg_%s.mat' % (str(which_shot)))
                out_npy_name = os.path.join(out_dir_name, 'IALM_fg_%s.npy' % (str(which_shot)))
                print 'loading...', out_mat_name
                AE = loadmat(out_mat_name)
                FG = AE['FG']
                E_frame1 = np.asarray(FG[:,:,frame_within_shot],'uint8')
                gs_img_fg = gs_img*E_frame1
                dst1 = cv2.medianBlur(E_frame1, 5)
                contours1, hierarchy1 = cv2.findContours(dst1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                bounding_box_list1 = get_bounding_boxes(contours1)
                trimmed_box_list1 = trim_boxes_by_area(bounding_box_list1, 0.9)
                bounding_box_list1 = merge_collided_bboxes( trimmed_box_list1, 20 )
                
                # prepare CAMSHIFT - setup ROIs for tracking
                frame_camshift = frame_equ.copy()
                track_windows, roi_hists = get_camshift_params(bounding_box_list1,frame_camshift)
                print 'windows for camshift', track_windows
               
                
                hsv = np.zeros_like(frame)
                flow_ref_img = gs_img.copy()
                
                
                print 'frame in shot at this point',frame_within_shot
                print 'overall frame', frame_count
                
    #             BG = AE['BG']
                
               
                ##--- once complete re-initialize the X matrix which has columns as vectorized frames and also add a lone vector there so as to update it
                
                print 'now accumulating shot ', which_shot
                #initialize first vector for the next mtrix here
                
            else:
                fg_img_ = np.asarray(FG[:,:,frame_within_shot],'uint8')
                
                if which_shot >= 4:
#                     cv2.imshow('x_%s' % (which_shot),frame)
#                     
#                     cv2.imshow('y_%s' % (which_shot),fg_img_)
                
#                     frame_equ = rgb_equalize(frame)                    
                    E_frame = fg_img_
                    gs_img_fg = gs_img *E_frame
    #                 hsv[...,0]=gs_img_fg                
                    
    #                 gs_img_gray = cv2.threshold(gs_img_fg, 100, 255, cv2.THRESH_BINARY)
                    E_blur = cv2.medianBlur(E_frame, 5)#cv2.filter2D(E_frame,-1,kernel)
                    fg_img = E_blur.copy()
                    fg_rgb_img = cv2.cvtColor(fg_img, cv2.COLOR_GRAY2RGB)
                    
                                    #---- OPTICAL FLOW
                    flow_input_img = gs_img.copy()                
                    flow_output_img = get_flow_img(hsv, flow_ref_img, flow_input_img)
                    # use flow imaage for CAM shift?
                    cam_frame = frame_equ.copy()
                    
                    flow_ref_img  = flow_input_img
    
                    
                    
                    contours, hierarchy = cv2.findContours(E_blur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    
                    bounding_box_list = get_bounding_boxes(contours)
                    trimmed_box_list = trim_boxes_by_area(bounding_box_list, 0.9)
                    bounding_box_list = merge_collided_bboxes( trimmed_box_list, 20 )
                    
                    print len(bounding_box_list)
                    for box in bounding_box_list:
    #                     if box != 
                        cv2.rectangle( frame, box[0], box[1], (0,255,0), 1 )
                    
                    print 'boxes tracked by me', bounding_box_list
                #### ------ DO CAMSHIFT NOW
                    
                    box_after_shift=[]
                    for i, track_window in enumerate(track_windows):
                        roi_hist = roi_hists[i]
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                        ret, win_after_shift = cv2.meanShift(dst, track_window, term_crit)
                        x,y,w,h = win_after_shift
                        box_after_shift.append(((x,y), (x+w,y+h)))
                    
    #                 box_after_shift = merge_collided_bboxes(box_after_shift, 0)
    
                    for box in box_after_shift:
                        img2 = cv2.rectangle(frame, box[0], box[1], 255,2)
    #                     cv2.imshow('img2',cam_frame)
                    print 'box_after shift',box_after_shift           
                    
    #                 merge_box_after_shift = merge_collided_bboxes(box_after_shift)
    #                 for box in merge_box_after_shift:
    #                     img2 = cv2.rectangle(frame, box[0], box[1], 255,2) 
    #                     cv2.imshow('img2',cam_frame)
                    
                    '''-----------HERE -- if boxes tracked by me are overlap with the cam-shifted boxes - keep cam-shift box! DONT change the box-input to camshift''' 
    #                 box_after_shift = merge_collided_bboxes(box_after_shift)
                    
                    bounding_box_for_shift = []
                    overlap_status_all = []
                    for box in bounding_box_list:
                        for box_i in box_after_shift:
                            overlap_status = check_collision(box, box_i)
                            overlap_status_all.append(overlap_status)
                            if overlap_status == True:
                                if box_i not in bounding_box_for_shift:
                                    bounding_box_for_shift.append(box_i)
    #                         else:
    #                             if box not in bounding_box_for_shift:
    #                                 bounding_box_for_shift.append(box)
    
                    
                    if bounding_box_for_shift == []:
                        print 'REINITIALIZING.........', frame_count
                        bounding_box_for_shift = get_biggest_box(bounding_box_list)
    
                    print overlap_status_all
                    print 'new trackers', bounding_box_for_shift
                    
                    
                    
                    # input form cam shift for the next frame
                    track_windows, roi_hists = get_camshift_params(bounding_box_for_shift, cam_frame)
                
                    
                    
                    
                    cv2.imshow('y_%s' % (which_shot),fg_img_)
                    cv2.imshow('x_%s' % (which_shot),frame)
                    cv2.imshow('z_%s' % (which_shot),flow_output_img)
               
                
                
                
                
                
                if not frame_within_shot%100: print frame_within_shot, 
                frame_within_shot+=1
            
        #     print which_shot,
            shot_idx+=which_shot_
            frame_count+=1
            k = cv2.cv.WaitKey(30) & 0xff

        
        
        
        
        
        
        
#         else: print 'waiting for the right shot'