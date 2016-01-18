'''
Created on Oct 27, 2015

@author: krsna
'''
from tracker_utils import *
import sys, commands,os
import numpy as np
# Use these to loop later?
movie_dir = '/home/krsna/workspace/animation/tanaya_shotdetect/HTD_scenes_mkv/'#'Antz_scenes'#'../tanaya_shotdetect/scenes/' str(sys.argv[1])#
#0022.avi
# very good example for HTD -0488 
num_to_try = 10
shot_ids_rand = list(np.random.randint(5,100,num_to_try))#[int(sys.argv[2])] #

def run_CMT(input_path, skip_frames, bbox, SHOW_IMAGES=False, clip_name='main'):
    from numpy import empty, nan
    import CMT
    import util
    CMT = CMT.CMT()
    CMT.estimate_scale = True
    CMT.estimate_rotation = False
    # read video and set delay accordingly
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip_frames)
    
    # something wrong with reading the shot it seems
    if not cap.isOpened():
        print '#Unable to open video input.'
        sys.exit(1)
    
    status, im0 = cap.read()
    im_gray0_ = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    im_gray0 = cv2.equalizeHist(im_gray0_)
    im_draw = np.copy(im0)
    
    # prepare initial bbox values
    bbox_values = [int(v) for v in bbox]
    bbox = array(bbox_values)
    
    # Convert to point representation, adding singleton dimension
    bbox = util.bb2pts(bbox[None, :])

    # Squeeze
    bbox = bbox[0, :]

    tl = bbox[:2]
    br = bbox[2:4]
    
    print '#using', tl, br, 'as init bb'
    
    CMT_TRACKS=[]
    CMT.initialise(im_gray0, tl, br)
    
    frame = 1
    while True:
        # Read image
        status, im = cap.read()
        if not status:
            break
        im_gray_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.equalizeHist(im_gray_)
        im_draw = np.copy(im)

        tic = time.time()
        CMT.process_frame(im_gray)
        toc = time.time()

        # Display results
        # Draw updated estimate
        if CMT.has_result:

            cv2.line(im_draw, CMT.tl, CMT.tr, (255, 0, 0), 4)
            cv2.line(im_draw, CMT.tr, CMT.br, (255, 0, 0), 4)
            cv2.line(im_draw, CMT.br, CMT.bl, (255, 0, 0), 4)
            cv2.line(im_draw, CMT.bl, CMT.tl, (255, 0, 0), 4)
            CMT_rect = [CMT.tl, CMT.tr,CMT.bl,CMT.br]

        util.draw_keypoints(CMT.tracked_keypoints, im_draw, (255, 255, 255))
        # this is from simplescale
        util.draw_keypoints(CMT.votes[:, :2], im_draw)  # blue
        util.draw_keypoints(CMT.outliers[:, :2], im_draw, (0, 0, 255))

        if SHOW_IMAGES:
            cv2.imshow(clip_name, im_draw)
            # Check key input
            k = cv2.cv.WaitKey(10) & 0xff
#         if args.quiet:
#             if CMT.has_result: print '1'#, print_str
#             else: print '0'#, print_str
        # Remember image
        im_prev = im_gray

        # Advance frame number
        frame += 1
        
        print_str =  '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(CMT.center[0], CMT.center[1], CMT.scale_estimate, CMT.active_keypoints.shape[0], 1000 * (toc - tic), frame)
        if CMT.has_result: CMT_TRACKS.append([1,CMT_rect])
        else: CMT_TRACKS.append([0,nan])#, print_str
    return CMT_TRACKS








# shot_num = '0030'
# movie_name = '%s.avi' % (shot_num)#'0022.avi'#'0671.avi' #'Antz.avi'
# movie_path = os.path.join(movie_dir, movie_name)
# print movie_name
# lrmf_path = os.path.join(movie_dir, 'IALM_fgbg_%s.mat' % (shot_num))#"./IALM_background_subtraction2.mat"

vad_init = np.load('voice_init_frame_info.npy')


BOX_TO_TRACK=[]


color = np.random.randint(0,255,(100,3))
color = np.vstack(([0,255,0],color))
####  - - incorporating low rank factored matrix back into video

#for shot_id in shot_ids_rand:
for movie_name in [i for i in os.listdir(movie_dir) if i.endswith('.avi') ][1:3]:
    cv2.destroyAllWindows()
    shot_num = movie_name.split('.')[0]
    #shot_num = str('%04d' % (shot_id,))
    #movie_name = '%s.avi' % (shot_num)#'0022.avi'#'0671.avi' #'Antz.avi'
    shot_num_int = int(shot_num)
    if shot_num_int in [v_i[0][0] for v_i in vad_init]:
        vad_info_ = [v_i for v_i in vad_init if v_i[0][0]==shot_num_int]
        vad_info = vad_info_[0]
        FRAME_CUTOFF = vad_info[1]
        if FRAME_CUTOFF==0 :FRAME_CUTOFF+=5
        else: FRAME_CUTOFF+=5
    else:
        FRAME_CUTOFF = 5
        
        
    movie_path = os.path.join(movie_dir, movie_name)
    print movie_name 
    lrmf_path = os.path.join(movie_dir, 'IALM_fgbg_%s.mat' % (shot_num))#"./IALM_background_subtraction2.mat"
    
    if True:
        AE = loadmat(lrmf_path)
    #     AE = loadmat("HTD_outputs/IALM_fgbg_671.mat")
        E = AE['FG']
        A = AE['BG']
        num_frames = E.shape[-1]
        mid_frame = np.round(E.shape[-1]/2.0)
        mov = cv2.VideoCapture(movie_path)
        
        frame_count = 0
        ret, frame1_ =  mov.read()
        width, height, ch = frame1_.shape
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
        
        frame1_equ = rgb_equalize(frame1)
        cam_frame1 = (fg_img_rgb1) & frame1_equ
        #cv2.filter2D(E_frame,-1,kernel)
    #     _,dst1_ = cv2.threshold(dst1, 0, 255, cv2.THRESH_BINARY)
        contours1, hierarchy1 = cv2.findContours(dst1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        bounding_box_list1 = get_bounding_boxes(contours1)
        trimmed_box_list1 = trim_boxes_by_area(bounding_box_list1, 0.8)
        bounding_box_list1 = merge_collided_bboxes( trimmed_box_list1, 20 )
        _,areas = get_biggest_box(bounding_box_list1,r,c)
        # prepare CAMSHIFT - setup ROIs for tracking
    #     frame_camshift = frame1.copy()
        frame_camshift = (fg_img_rgb1) & frame1
        bounding_box_list_to_start, _ = get_biggest_box(bounding_box_list1, r, c)  #bounding_box_list1
#         bounding_box_list_to_start = bounding_box_list1
        track_windows, roi_hists = get_camshift_params(bounding_box_list_to_start,frame_camshift)
        
        fg_accumulator = fg_img_rgb1
        fg_adder = fg_img_thr
        
        term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        
        
        #FRAME_CUTOFF=33+24
        
        INITIALIZE_counter=[0]
        while (frame_count<=FRAME_CUTOFF):
#                 print frame_count
                frame_count += 1
                #print 'retrieving frames'
                ret, frame_orig = mov.read()
                if frame_orig is not None:
#                     print frame_count,
                    #0. Resize image
                    frame  = cv2.resize(frame_orig, (frame_orig.shape[1]/2, frame_orig.shape[0]/2))
                    my_frame = frame.copy()
                    rgb_frame = frame.copy()
    #                 cam_frame = frame.copy()
                    frame_equ = rgb_equalize(frame)
                    
                    gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gs_img_3ch = cv2.cvtColor(gs_img, cv2.COLOR_GRAY2BGR)
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
                    
                    # use flow image & fg image masked rgb image for CAM shift?
                    cam_frame = (flow_output_img.copy() | fg_rgb_img) & frame_equ
                    
                    #accumulate all the fg_pixel info and 
                    fg_accumulator = (fg_accumulator | (fg_rgb_img))
                    fg_adder+=fg_img_thr
                    
                    # mask the frame with accumulated fg image and then use this for CAMshift to track on
                    blanket_frame = frame_equ & fg_accumulator
                    
                    contours, hierarchy = cv2.findContours(E_blur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                    
                    bounding_box_list = get_bounding_boxes(contours)
                    trimmed_box_list = trim_boxes_by_area(bounding_box_list, 0.9)
                    bounding_box_list = merge_collided_bboxes( trimmed_box_list, 20 )
                                        
                    SHOW_INIT_BOXES=False
                    if frame_count >= mid_frame: 
                        SHOW_INIT_BOXES=True
                    if SHOW_INIT_BOXES:
                        for box in bounding_box_list:
        #                     if box != 
                            cv2.rectangle( frame, box[0], box[1], (0,255,0), 1 )
                        
                    #print 'boxes tracked by me', bounding_box_list
                    
                #### ------ DO CAMSHIFT NOW
                    
                    box_after_shift=[]
                    for i, track_window in enumerate(track_windows):
                        roi_hist = roi_hists[i]
                        # tracking on cam_frame space - not on the whole frame - call it a blanket frame
                        # other options are cam_frame or just frame
                        hsv = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2HSV)
                        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                        ret, win_after_shift = cv2.CamShift(dst, track_window, term_crit)
                        x,y,w,h = win_after_shift
                        if frame_count==FRAME_CUTOFF: 
                            BOX_TO_TRACK=[x*2, y*2, w*2, h*2]
                            #print x*2, y*2, w*2, h*2, '------box after camshift----' 
                        box_after_shift.append(((x,y), (x+w,y+h)))
                        if frame_count<=FRAME_CUTOFF: 
                            for box in box_after_shift:
                                cv2.rectangle( my_frame, box[0], box[1], (0,255,0), 1)
    #                 box_after_shift = merge_collided_bboxes(box_after_shift, 0)
                    
                    for box in box_after_shift:
                        img2 = cv2.rectangle(frame, box[0], box[1], color[len(INITIALIZE_counter)],2)
    #                     cv2.imshow('img2',cam_frame)
                    #print 'box_after shift',box_after_shift,            
                    
                    bounding_box_for_shift = box_after_shift
#                     if not bounding_box_for_shift: track_windows, roi_hists = get_camshift_params(bounding_box_for_shift, cam_frame)
                
                    _,test = cv2.threshold(fg_adder, 240, 255, cv2.THRESH_BINARY)
                

                    #cv2.imshow('y_%s' % (shot_num),np.hstack(((fg_accumulator & rgb_frame), cam_frame, frame, my_frame)))
                    #k = cv2.cv.WaitKey(30) & 0xff
                    
                    if frame_count==FRAME_CUTOFF:
                        print BOX_TO_TRACK
                        if BOX_TO_TRACK:
                            if BOX_TO_TRACK[2] >= 0.9*height and BOX_TO_TRACK[3] >= 0.9*width: 
                                print 'BOX FULL SCREEN??'
                                BOX_TO_TRACK = [ 20,20, 0.8*height, 0.8*width ]
                                print BOX_TO_TRACK
                            frame_to_save = frame_orig.copy()
                            try:
                                cmt_status = run_CMT(movie_path, FRAME_CUTOFF, BOX_TO_TRACK, SHOW_IMAGES=True, clip_name='%s #frames=%s frame_cutoff=%s' % (shot_num, num_frames, FRAME_CUTOFF))
                                print [i[0] for i in cmt_status if i==1]
                                perc_track = len([i[0] for i in cmt_status if i==1])/(float(len(cmt_status)))
                                print perc_track
                                if perc_track >= 0.75: 
                                    cv2.imshow('KEY_FRAME',frame_to_save);k = cv2.cv.WaitKey(30) & 0xff
                            except: 
                                ValueError
                                print 'NOT TRACKED DUE TO SOME ERROR'
                        
                        else: print 'no box detected'
                else: 
                    mov.release()
                    cv2.destroyAllWindows()    
        
