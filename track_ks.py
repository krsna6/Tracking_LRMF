#!/usr/bin/env python

# See also: http://sundararajana.blogspot.com/2007/05/motion-detection-using-opencv.html

import cv2.cv as cv
import cv2
import time
import skimage
# from skimage.filters import threshold_otsu
# from skimage.morphology import closing,square
# from skimage.filters import threshold_otsu
# from skimage import measure
# from skimage.color import label2rgb

from scipy import *
from scipy.cluster import vq
import numpy
import sys, os, random, hashlib

from math import *
import pylab as pl
"""
Python Motion Tracker

Reads an incoming video stream and tracks motion in real time.
Detected motion events are logged to a text file.  Also has face detection.
"""

#
# BBoxes must be in the format:
# ( (topleft_x), (topleft_y) ), ( (bottomright_x), (bottomright_y) ) )
top = 0
bottom = 1
left = 0
right = 1

def merge_collided_bboxes( bbox_list ):
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
			
			# We also inflate the box size by 10% to deal with
			# fuzziness in the data.  (Without this, there are many times a bbox
			# is short of overlap by just one or two pixels.)
			
			# 20%
			if (this_bbox[bottom][0]*1.2 < other_bbox[top][0]*0.8): has_collision = False
			if (this_bbox[top][0]*.8 > other_bbox[bottom][0]*1.2): has_collision = False
			
			if (this_bbox[right][1]*1.2 < other_bbox[left][1]*0.8): has_collision = False
			if (this_bbox[left][1]*0.8 > other_bbox[right][1]*1.2): has_collision = False
			
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
				return merge_collided_bboxes( bbox_list )
	
	# When there are no collions between boxes, return that list:
	return bbox_list


def detect_faces( image, haar_cascade, mem_storage ):

	faces = []
	image_size = cv.GetSize( image )

	#faces = cv.HaarDetectObjects(grayscale, haar_cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (20, 20) )
	#faces = cv.HaarDetectObjects(image, haar_cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING )
	#faces = cv.HaarDetectObjects(image, haar_cascade, storage )
	#faces = cv.HaarDetectObjects(image, haar_cascade, mem_storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, ( 16, 16 ) )
	#faces = cv.HaarDetectObjects(image, haar_cascade, mem_storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, ( 4,4 ) )
	faces = cv.HaarDetectObjects(image, haar_cascade, mem_storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, ( image_size[0]/10, image_size[1]/10) )
	
	for face in faces:
		box = face[0]
		cv.Rectangle(image, ( box[0], box[1] ),
			( box[0] + box[2], box[1] + box[3]), cv.RGB(255, 0, 0), 1, 8, 0)


class Target:
	def __init__(self):
		
		READ_FROM_FILE = True
		
# 		if len( sys.argv ) > 1:
		if READ_FROM_FILE:
			self.writer = None
			# 0011, 0012, 0013 good cases
			self.capture = cv.CaptureFromFile( 'Antz_scenes/0022.avi' )
# 			self.capture = cv.CaptureFromFile( 'Antz_clip.avi' )
			frame = cv.QueryFrame(self.capture)
			print frame
			frame_size = cv.GetSize(frame)
	
		frame = cv.QueryFrame(self.capture)
		cv.NamedWindow("Target", 1)
# 		cv.NamedWindow("Scikit", 2)
		#cv.NamedWindow("Target2", 1)
		

	def run(self):
		# Initialize
		#log_file_name = "tracker_output.log"
		#log_file = file( log_file_name, 'a' )
		
# 		pl.ion()
		
		frame = cv.QueryFrame( self.capture )
		frame_size = cv.GetSize( frame )
		
		# Capture the first frame from webcam for image properties
		display_image = cv.QueryFrame( self.capture )
		
		# Greyscale image, thresholded to create the motion mask:
		grey_image = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_8U, 1 )
		gs_image = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_8U, 1 )

		# The RunningAvg() function requires a 32-bit or 64-bit image...
		running_average_image = cv.CreateImage( cv.GetSize(frame), cv.IPL_DEPTH_32F, 3 )
		# ...but the AbsDiff() function requires matching image depths:
		running_average_in_display_color_depth = cv.CloneImage( display_image )
		
		# RAM used by FindContours():
		mem_storage = cv.CreateMemStorage(0)
		
		# The difference between the running average and the current frame:
		difference = cv.CloneImage( display_image )
		
		target_count = 1
		last_target_count = 1
		last_target_change_t = 0.0
		k_or_guess = 1
		codebook=[]
		frame_count=0
		last_frame_entity_list = []
		
		t0 = time.time()
		
		len_of_contours=[]
		len_of_boxes=[]
		area_of_merged_boxes=[]
		all_bounding_boxes=[]
		all_center_points =[]
# 		# For toggling display:
# 		image_list = [ "camera", "difference", "threshold", "display", "faces" ]
# 		image_index = 0   # Index into image_list
# 	
# 	
# 		# Prep for text drawing:
# 		text_font = cv.InitFont(cv.CV_FONT_HERSHEY_COMPLEX, .5, .5, 0.0, 1, cv.CV_AA )
# 		text_coord = ( 5, 15 )
# 		text_color = cv.CV_RGB(255,255,255)

		###############################
		### Face detection stuff
		#haar_cascade = cv.Load( 'haarcascades/haarcascade_frontalface_default.xml' )
# 		haar_cascade = cv.Load( 'haarcascades/haarcascade_frontalface_alt.xml' )
		#haar_cascade = cv.Load( 'haarcascades/haarcascade_frontalface_alt2.xml' )
		#haar_cascade = cv.Load( 'haarcascades/haarcascade_mcs_mouth.xml' )
		#haar_cascade = cv.Load( 'haarcascades/haarcascade_eye.xml' )
		#haar_cascade = cv.Load( 'haarcascades/haarcascade_frontalface_alt_tree.xml' )
		#haar_cascade = cv.Load( 'haarcascades/haarcascade_upperbody.xml' )
		#haar_cascade = cv.Load( 'haarcascades/haarcascade_profileface.xml' )
		
		# Set this to the max number of targets to look for (passed to k-means):
		max_targets = 3
		
		while True:
			
			camera_image = cv.QueryFrame( self.capture )
# 			except: TypeError
			
			frame_count += 1
			
			if camera_image is not None:
				print frame_count, 'frame #...'
				frame_t0 = time.time()
				# Create an image with interactive feedback:
				display_image = cv.CloneImage( camera_image )
				
				cv.CvtColor(display_image, gs_image, cv.CV_RGB2GRAY)
				
				# Create a working "color image" to modify / blur
				color_image = cv.CloneImage( display_image )
	
				# Smooth to get rid of false positives
				cv.Smooth( color_image, color_image, cv.CV_GAUSSIAN, 19, 0 )
				# Use the Running Average as the static background			
				# a = 0.020 leaves artifacts lingering way too long.
				# a = 0.320 works well at 320x240, 15fps.  (1/a is roughly num frames.)
				cv.RunningAvg( color_image, running_average_image, 0.32, None )
				print cv.GetSize(running_average_image), '@@@@@@@@@@@@@@'
				# Convert the scale of the moving average.
				

				cv.ConvertScale( running_average_image, running_average_in_display_color_depth, 1.0, 0.0 )
				
				# Subtract the current frame from the moving average.
				cv.AbsDiff( color_image, running_average_in_display_color_depth, difference )
				
				# Convert the image to greyscale.
				cv.CvtColor( difference, grey_image, cv.CV_RGB2GRAY )
				
				# Threshold the image to a black and white motion mask:
				cv.Threshold( grey_image, grey_image, 2, 255, cv.CV_THRESH_BINARY )
				
				# Smooth and threshold again to eliminate "sparkles"
				cv.Smooth( grey_image, grey_image, cv.CV_GAUSSIAN, 19, 0 )
				cv.Threshold( grey_image, grey_image, 240, 255, cv.CV_THRESH_BINARY )
				
				grey_image_as_array = numpy.asarray( cv.GetMat( grey_image ) )
				
				#### -----SKIMAGE
# 				L = measure.label(grey_image_as_array, connectivity=2)
# 				print L, 'labell'
# 				labeled_image_ = label2rgb(L, grey_image_as_array)
# 				labeled_image = cv.fromarray(labeled_image_)
# 				
				
				
				
				
				
				non_black_coords_array = numpy.where( grey_image_as_array > 3 )
				
# 				for_otsu_image = grey_image_as_array
# 				otsu_thresh = threshold_otsu(for_otsu_image)
# 				print otsu_thresh, 'otsu'
# 				otsu_image = closing(for_otsu_image > otsu_thresh, 	square(3))
				
				
				# Convert from numpy.where()'s two separate lists to one list of (x, y) tuples:
				non_black_coords_array = zip( non_black_coords_array[1], non_black_coords_array[0] )
				
				points = []   # Was using this to hold either pixel coords or polygon coords.
				bounding_box_list = []
	
				# Now calculate movements using the white pixels as "motion" data
				contour = cv.FindContours( grey_image, mem_storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE )
				print len(list(contour)), 'contours....'
				len_of_contours += [len(list(contour))]
# 				img = cv.DrawContours(grey_image, contour, external_color, hole_color, max_level)
# 				pl.plot(frame_count,len(list(contour)),marker='*' )
# 				pl.draw()
# 				cv.ShowImage( "Target", display_image )
				
				while contour:
					
					bounding_rect = cv.BoundingRect( list(contour) )
					point1 = ( bounding_rect[0], bounding_rect[1] )
					point2 = ( bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3] )
					
					bounding_box_list.append( ( point1, point2 ) )
					polygon_points = cv.ApproxPoly( list(contour), mem_storage, cv.CV_POLY_APPROX_DP )
					
# 					i_, (x,y),radius = cv.MinEnclosingCircle(list(contour))
# 					center = (int(x),int(y))
# 					radius = int(radius)
# 					cv.Circle(display_image, center, radius, (0,255,0),2)
					
					# To track polygon points only (instead of every pixel):
# 					points += list(polygon_points)
					
					# Draw the contours:
					cv.DrawContours(color_image, contour, cv.CV_RGB(255,0,0), cv.CV_RGB(0,255,0), cv.CV_FILLED, 3, 0, (0,0) )
					cv.FillPoly( grey_image, [ list(polygon_points), ], cv.CV_RGB(255,255,255), 0, 0 )
					cv.PolyLine( display_image, [ polygon_points, ], 0, cv.CV_RGB(255,255,255), 1, 0, 0 )
					#cv.Rectangle( display_image, point1, point2, cv.CV_RGB(120,120,120), 1)
					
					
					
					contour = contour.h_next()
				
				
				# Find the average size of the bbox (targets), then
				# remove any tiny bboxes (which are prolly just noise).
				# "Tiny" is defined as any box with 1/10th the area of the average box.
				# This reduces false positives on tiny "sparkles" noise.
				box_areas = []
				for box in bounding_box_list:
					box_width = box[right][0] - box[left][0]
					box_height = box[bottom][0] - box[top][0]
					box_areas.append( box_width * box_height )
 					
					#cv.Rectangle( display_image, box[0], box[1], cv.CV_RGB(255,0,0), 1)
 				
				average_box_area = 0.0
				if len(box_areas): average_box_area = float( sum(box_areas) ) / len(box_areas)
 				
#  				pl.plot(frame_count,log(average_box_area),marker='*' )
# 				pl.draw()
 				
				trimmed_box_list = []
				for box in bounding_box_list:
					box_width = box[right][0] - box[left][0]
					box_height = box[bottom][0] - box[top][0]
 					
					# Only keep the box if it's not a tiny noise box:
					if (box_width * box_height) > average_box_area*0.1: trimmed_box_list.append( box )
 				
				# Draw the trimmed box list:
				#for box in trimmed_box_list:
				#	cv.Rectangle( display_image, box[0], box[1], cv.CV_RGB(0,255,0), 2 )
				trimmed_box_list = 	bounding_box_list
				bounding_box_list = merge_collided_bboxes( trimmed_box_list )

				
				# Here are our estimate points to track, based on merged & trimmed boxes:
				estimated_target_count = len( bounding_box_list )
				print estimated_target_count
				len_of_boxes += [estimated_target_count]
# 				pl.plot(frame_count, estimated_target_count, marker='o')
# 				pl.draw()
				print '--------------'
				print len(bounding_box_list), '# boxes' 
				all_bounding_boxes += [bounding_box_list]

				# Draw the merged box list:
				box_area=[]
				for box in bounding_box_list:
					box_area_ = (box[right][0] - box[left][0]) * (box[bottom][0] - box[top][0])
					box_area += [box_area_]
# 					print 'area of merged box...',  box_area_
# 					cv.Rectangle( display_image, box[0], box[1], cv.CV_RGB(0,255,0), 1 )
 				area_of_merged_boxes += [box_area]
				
				sum_of_areas = [sum(i) for i in box_area]
				largest_box_idx = [i for i in range(len(sum_of_areas)) if sum_of_areas[i]==max(sum_of_areas)]
				print largest_box_idx, 'big_box...'
				print '-------------------------'
				# Draw the merged box list:
				bounding_box_list_LARGEST = [bounding_box_list[i] for i in largest_box_idx]

# 				CHANGE THIS TO GET MORE THAN ONE box
				for box in bounding_box_list_LARGEST:
# 					if box != 
					cv.Rectangle( display_image, box[0], box[1], cv.CV_RGB(0,255,0), 1 )
 				area_of_merged_boxes += [box_area]
				
				# ( (topleft_x), (topleft_y) ), ( (bottomright_x), (bottomright_y) ) )
				if False:
					largest_box = bounding_box_list_LARGEST[0]
					#conisder for tracking only if the box cover less than 80% of the screen?????
					full_sc_area = 515524.0
					area_largest_box = (largest_box[right][0] - largest_box[left][0]) * (largest_box[bottom][0] - largest_box[top][0])
					if area_largest_box <= 0.8*full_sc_area:
							disp_array = numpy.asarray( cv.GetMat( display_image ) )
							print largest_box, '!!!!!!!!!!!!!!!!!!!!!!!!!!!'
							top_x, bottom_x, top_y, bottom_y = largest_box[0][0],largest_box[1][0],largest_box[0][1],largest_box[1][1]
							track_window = (top_y, top_x, bottom_y-top_y, bottom_x-top_x)
							roi = disp_array[top_x:bottom_x, top_y: bottom_y]
							print top_x,bottom_x, top_y, bottom_y
							print roi.shape
							hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# 							mask = cv2.inRange(hsv_roi, numpy.array((0., 60.,32.)), numpy.array((180.,255.,255.)))
							mask = cv2.inRange(hsv_roi, numpy.array((0., 60.,32.)), numpy.array((180.,255.,255.)))
							roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
							cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
							
							term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
							
							hsv = cv2.cvtColor(disp_array, cv2.COLOR_BGR2HSV)
							dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
							ret, track_window = cv2.meanShift(dst, track_window, term_crit)
							x,y,w,h = track_window
							img2 = cv2.rectangle(disp_array, (x,y), (x+w,y+h), 255,2)
							cv2.imshow('img2',disp_array)
							k = cv2.waitKey(60) & 0xff
							
				
				
				##CAMSHIFT - on the largest frame
				
# 				roi = display_image[bounding_box_list_LARGEST[0][]];
# 				hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
				
				
				
				
				
				
################################				
				# Don't allow target "jumps" from few to many or many to few.
				# Only change the number of targets up to one target per n seconds.
				# This fixes the "exploding number of targets" when something stops moving
				# and the motion erodes to disparate little puddles all over the place.
 				
				if frame_t0 - last_target_change_t < .350:  # 1 change per 0.35 secs
					estimated_target_count = last_target_count
				else:
					if last_target_count - estimated_target_count > 1: estimated_target_count = last_target_count - 1
					if estimated_target_count - last_target_count > 1: estimated_target_count = last_target_count + 1
					last_target_change_t = frame_t0
 				
				# Clip to the user-supplied maximum:
				estimated_target_count = min( estimated_target_count, max_targets )
 				
				# The estimated_target_count at this point is the maximum number of targets
				# we want to look for.  If kmeans decides that one of our candidate
				# bboxes is not actually a target, we remove it from the target list below.
 				
				# Using the numpy values directly (treating all pixels as points):	
				points = non_black_coords_array
				center_points = []
 				
				if len(points):
 					
					# If we have all the "target_count" targets from last frame,
					# use the previously known targets (for greater accuracy).
					k_or_guess = max( estimated_target_count, 1 )  # Need at least one target to look for.
					if len(codebook) == estimated_target_count: 
						k_or_guess = codebook
 					
					#points = vq.whiten(array( points ))  # Don't do this!  Ruins everything.
					codebook, distortion = vq.kmeans( array( points ), k_or_guess )
 					
					# Convert to tuples (and draw it to screen)
					for center_point in codebook:
						center_point = ( int(center_point[0]), int(center_point[1]) )
						center_points.append( center_point )
						#cv.Circle(display_image, center_point, 10, cv.CV_RGB(255, 0, 0), 2)
						#cv.Circle(display_image, center_point, 5, cv.CV_RGB(255, 0, 0), 3)
 				
				# Now we have targets that are NOT computed from bboxes -- just
				# movement weights (according to kmeans).  If any two targets are
				# within the same "bbox count", average them into a single target.  
				#
				# (Any kmeans targets not within a bbox are also kept.)
				trimmed_center_points = []
				removed_center_points = []
 							
				for box in bounding_box_list:
					# Find the centers within this box:
					center_points_in_box = []
 					
					for center_point in center_points:
						if	center_point[0] < box[right][0] and center_point[0] > box[left][0] and \
							center_point[1] < box[bottom][1] and center_point[1] > box[top][1] :
 							
							# This point is within the box.
							center_points_in_box.append( center_point )
 					
					# Now see if there are more than one.  If so, merge them.
					if len( center_points_in_box ) > 1:
						# Merge them:
						x_list = y_list = []
						for point in center_points_in_box:
							x_list.append(point[0])
							y_list.append(point[1])
 						
						average_x = int( float(sum( x_list )) / len( x_list ) )
						average_y = int( float(sum( y_list )) / len( y_list ) )
 						
						trimmed_center_points.append( (average_x, average_y) )
 						
						# Record that they were removed:
						removed_center_points += center_points_in_box
 						
					if len( center_points_in_box ) == 1:
						trimmed_center_points.append( center_points_in_box[0] ) # Just use it.
 				
				# If there are any center_points not within a bbox, just use them.
				# (It's probably a cluster comprised of a bunch of small bboxes.)
				for center_point in center_points:
					if (not center_point in trimmed_center_points) and (not center_point in removed_center_points):
						trimmed_center_points.append( center_point )
 				
				# Draw what we found:
				#for center_point in trimmed_center_points:
				#	center_point = ( int(center_point[0]), int(center_point[1]) )
				#	cv.Circle(display_image, center_point, 20, cv.CV_RGB(255, 255,255), 1)
				#	cv.Circle(display_image, center_point, 15, cv.CV_RGB(100, 255, 255), 1)
				#	cv.Circle(display_image, center_point, 10, cv.CV_RGB(255, 255, 255), 2)
				#	cv.Circle(display_image, center_point, 5, cv.CV_RGB(100, 255, 255), 3)
 				
				# Determine if there are any new (or lost) targets:
				actual_target_count = len( trimmed_center_points )
				last_target_count = actual_target_count
 				
				# Now build the list of physical entities (objects)
				this_frame_entity_list = []
 				
				# An entity is list: [ name, color, last_time_seen, last_known_coords ]
 				
				for target in trimmed_center_points:
 				
					# Is this a target near a prior entity (same physical entity)?
					entity_found = False
					entity_distance_dict = {}
 					
					for entity in last_frame_entity_list:
 						
						entity_coords= entity[3]
						delta_x = entity_coords[0] - target[0]
						delta_y = entity_coords[1] - target[1]
 				
						distance = sqrt( pow(delta_x,2) + pow( delta_y,2) )
						entity_distance_dict[ distance ] = entity
 					
					# Did we find any non-claimed entities (nearest to furthest):
					distance_list = entity_distance_dict.keys()
					distance_list.sort()
 					
					for distance in distance_list:
 						
						# Yes; see if we can claim the nearest one:
						nearest_possible_entity = entity_distance_dict[ distance ]
 						
						# Don't consider entities that are already claimed:
						if nearest_possible_entity in this_frame_entity_list:
							#print "Target %s: Skipping the one iwth distance: %d at %s, C:%s" % (target, distance, nearest_possible_entity[3], nearest_possible_entity[1] )
							continue
 						
						#print "Target %s: USING the one iwth distance: %d at %s, C:%s" % (target, distance, nearest_possible_entity[3] , nearest_possible_entity[1])
						# Found the nearest entity to claim:
						entity_found = True
						nearest_possible_entity[2] = frame_t0  # Update last_time_seen
						nearest_possible_entity[3] = target  # Update the new location
						this_frame_entity_list.append( nearest_possible_entity )
						#log_file.write( "%.3f MOVED %s %d %d\n" % ( frame_t0, nearest_possible_entity[0], nearest_possible_entity[3][0], nearest_possible_entity[3][1]  ) )
						break
 					
					if entity_found == False:
						# It's a new entity.
						color = ( random.randint(0,255), random.randint(0,255), random.randint(0,255) )
						name = hashlib.md5( str(frame_t0) + str(color) ).hexdigest()[:6]
						last_time_seen = frame_t0
 						
						new_entity = [ name, color, last_time_seen, target ]
						this_frame_entity_list.append( new_entity )
						#log_file.write( "%.3f FOUND %s %d %d\n" % ( frame_t0, new_entity[0], new_entity[3][0], new_entity[3][1]  ) )
 				
				# Now "delete" any not-found entities which have expired:
				entity_ttl = 1.0  # 1 sec.
 				
				for entity in last_frame_entity_list:
					last_time_seen = entity[2]
					if frame_t0 - last_time_seen > entity_ttl:
						# It's gone.
						#log_file.write( "%.3f STOPD %s %d %d\n" % ( frame_t0, entity[0], entity[3][0], entity[3][1]  ) )
						pass
					else:
						# Save it for next time... not expired yet:
						this_frame_entity_list.append( entity )
 				
				# For next frame:
				last_frame_entity_list = this_frame_entity_list
				# Draw the found entities to screen:
				center_points_here =[]
				for entity in this_frame_entity_list:
					center_point = entity[3]
					center_points_here+=[center_point]
					c = entity[1]  # RGB color tuple
					cv.Circle(display_image, center_point, 20, cv.CV_RGB(c[0], c[1], c[2]), 1)
					cv.Circle(display_image, center_point, 15, cv.CV_RGB(c[0], c[1], c[2]), 1)
					cv.Circle(display_image, center_point, 10, cv.CV_RGB(c[0], c[1], c[2]), 2)
					cv.Circle(display_image, center_point,  5, cv.CV_RGB(c[0], c[1], c[2]), 3)
 				
				all_center_points += [center_points_here]
				numpy.save('center_points', all_center_points)
				#print "min_size is: " + str(min_size)
				# Listen for ESC or ENTER key
				c = cv.WaitKey(7) % 0x100
				if c == 27 or c == 10:
					break
				
				# Toggle which image to show
# 				if chr(c) == 'd':
# 					image_index = ( image_index + 1 ) % len( image_list )
# 				
# 				image_name = image_list[ image_index ]
# 				
# 				# Display frame to user
# 				if image_name == "camera":
# 					image = camera_image
# 					cv.PutText( image, "Camera (Normal)", text_coord, text_font, text_color )
# 				elif image_name == "difference":
# 					image = difference
# 					cv.PutText( image, "Difference Image", text_coord, text_font, text_color )
# 				elif image_name == "display":
# 					image = display_image
# 					cv.PutText( image, "Targets (w/AABBs and contours)", text_coord, text_font, text_color )
# 				elif image_name == "threshold":
# 					# Convert the image to color.
# 					cv.CvtColor( grey_image, display_image, cv.CV_GRAY2RGB )
# 					image = display_image  # Re-use display image here
# 					cv.PutText( image, "Motion Mask", text_coord, text_font, text_color )
					
# 				disp_img = numpy.hstack((display_image,difference))
# 				elif image_name == "faces":
# 					# Do face detection
# 					detect_faces( camera_image, haar_cascade, mem_storage )				
# 					image = camera_image  # Re-use camera image here
# 					cv.PutText( image, "Face Detection", text_coord, text_font, text_color )
				
# 				cv.ShowImage( "Target", image )
# 				cv.ShowImage( "Target", camera_image )
# 				cv.ShowImage( "Target", difference )
				#show contours
				print numpy.average(len_of_contours), len(len_of_contours)
				print numpy.average(len_of_boxes), len(len_of_boxes)
				print len(area_of_merged_boxes)#numpy.average(area_of_merged_boxes)
				numpy.save('boxes2.npy', all_bounding_boxes)
				
				
# 				cv.ShowImage( "Target", color_image )
# 				#show boxes
				cv.ShowImage( "Target", grey_image )
				
				time.sleep(2)
# 				cv.ShowImage("Scikit",otsu_image)
# 				cv.CvtColor( grey_image, display_image, cv.CV_GRAY2RGB )
				k = cv.WaitKey(30) & 0xff				
# 				cv.ShowImage( "Target", display_image )
		
# 				if self.writer: 
# 					cv.WriteFrame( self.writer, image );
				
				#log_file.flush()
				
				# If only using a camera, then there is no time.sleep() needed, 
				# because the camera clips us to 15 fps.  But if reading from a file,
				# we need this to keep the time-based target clipping correct:
# 				frame_t1 = time.time()
				
	
# 				# If reading from a file, put in a forced delay:
# 				if not self.writer:
# 					delta_t = frame_t1 - frame_t0
# 					if delta_t < ( 1.0 / 15.0 ): time.sleep( ( 1.0 / 15.0 ) - delta_t )
# 				
# # 			t1 = time.time()
# 			time_delta = t1 - t0
# 			processed_fps = float( frame_count ) / time_delta
# 			print "Got %d frames. %.1f s. %f fps." % ( frame_count, time_delta, processed_fps )
# 		
if __name__=="__main__":
	t = Target()
#	import cProfile
#	cProfile.run( 't.run()' )
	t.run()





