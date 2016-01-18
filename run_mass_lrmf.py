'''
Created on Oct 25, 2015

@author: krsna

'''


from tracker_utils import *
import sys



dst_dir = str(sys.argv[1]) 
#'/home/krsna/workspace/animation/tanaya_shotdetect/HTD_scenes/'
mov_prefix = str(sys.argv[2])
movie_path = os.path.join(dst_dir, '%s.avi' % (mov_prefix))
output_mat_path = os.path.join(dst_dir, 'IALM_fgbg_%s.mat' % (mov_prefix))


if True:
    
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
    
    while(mov.isOpened()):
            frame_count += 1
            #print 'retrieving frames'
            ret, frame_orig = mov.read()
            
            if frame_orig is not None:
                if not frame_count%50: print frame_count,
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
        print 'started LRMF'
        A, E = inexact_augmented_lagrange_multiplier(X)
        A = A.reshape(r,c,n_frames) * 255.
        E = E.reshape(r, c, n_frames) * 255.
        savemat(output_mat_path, {"BG": A, "FG": E})
