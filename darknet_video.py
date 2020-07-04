# import the necessary packages
import numpy as np
import imutils
import time
import cv2
from scipy.spatial import distance

counter = 0

confidence_val=0.5
threshold_val=0.5

# load the COCO class labels our YOLO model was trained on
#labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
labelsPath = 'model/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
weightsPath = 'model/coco.weights'
configPath = 'model/coco.cfg'

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
input_file = 'LRT.mp4' 
vs = cv2.VideoCapture(input_file)
writer = None
(W, H) = (None, None)
fps =  vs.get(cv2.CAP_PROP_FPS)
print(fps)

frameIndex = 0

cv2.namedWindow('display', cv2.WINDOW_NORMAL)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
#while True:
for z in range(1, 3625):
	# read the next frame from the file
	(grabbed, bgr_frame) = vs.read()
	frame = cv2.resize(bgr_frame, (0, 0), fx=0.25, fy=0.25)

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	if z%2 != 0:
		continue

	if z < 625:
		continue

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	#blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
	#blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	elap = 1/(end - start)
	print("[INFO] current frame rate {:.1f} fps".format(elap))
	print(z)

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > confidence_val and classID == 0 :
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_val, threshold_val)

	coor = []
	border = []
	sd_count = 0

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		#print(len(idxs))

		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
		
			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			#cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			border.append((x, y, x + w, y + h))
			coor.append((x + int(w/2), y + int(h/2)))


		for i in range (len(coor)-1):
			for j in range (i+1,len(coor)):
				D = distance.euclidean(coor[i], coor[j])
				text2 = "{}".format(D)
				cv2.putText(frame, text2, coor[j], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				print(D)
				#if D >= 120 and D < 300:
					#cv2.line(frame, coor[i], coor[j], (255, 0, 0) , 2)
				if D < 220:
					cv2.line(frame, coor[i], coor[j], (0, 0, 255) , 5)
					cv2.rectangle(frame, (border[i][0], border[i][1]), (border[i][2], border[i][3]), (0, 0, 255), 2)
					cv2.rectangle(frame, (border[j][0], border[j][1]), (border[j][2], border[j][3]), (0, 0, 255), 2)
					sd_count = sd_count + 1

		if sd_count != 0:
			sd_count = sd_count + 1

	#display msg on-screen
	cv2.putText(frame, "People count: {}".format(len(idxs)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
	cv2.putText(frame, "People violate: {}".format(sd_count), (100,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)


	#points = np.array(coor)
	#cv2.polylines(frame,[points],0,(255,255,255))

	
	

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter('out.avi', fourcc, fps,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

	#display image
	#cv2.imshow('display',frame)
	#cv2.waitKey(1)

	#if frameIndex >= 4000: # limits the execution to the first 4000 frames
	#	print("[INFO] cleaning up...")
	#	writer.release()
	#	vs.release()
	#	exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
cv2.destroyAllWindows()
