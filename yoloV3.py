import numpy as np
import cv2
import matplotlib.pyplot as plt

probability_minimum = 0.5
threshold = 0.3

pathOut = 'vehicle_detection.mp4'	
frame_array = []

f = open('coco.names')
labels = f.read().strip().split('\n')
f.close()
np.random.seed(42)
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

network = cv2.dnn.readNetFromDarknet('yoloV3.cfg', 'yoloV3.weights')

layers_names_all = network.getLayerNames()
layers_names_output = []
for i in network.getUnconnectedOutLayers() :
	layers_names_output.append(layers_names_all[i[0] - 1])

video_cap = cv2.VideoCapture('video.mp4')
success, image_input = video_cap.read()
image_input_shape = image_input.shape	
h, w = image_input_shape[:2]

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL) 

while success :

	blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (320, 320), [0,0,0], swapRB=True, crop=False)	
	network.setInput(blob)
	output_from_network = network.forward(layers_names_output)
	
	bounding_boxes = []
	confidences = []
	class_numbers = []
	
	for result in output_from_network :
	
		for detection in result :
	
			scores = detection[5:]
			class_current = np.argmax(scores)
			
			confidence_current = scores[class_current]
			
			if confidence_current > probability_minimum :
				
				box_current = detection[0:4] * np.array([w, h, w, h])
				x_center, y_center, box_width, box_height = box_current.astype('int')
				x_min = int(x_center - (box_width / 2))
				y_min = int(y_center - (box_height / 2))
				
				bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
				confidences.append(float(confidence_current))
				class_numbers.append(class_current)
	
	results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold) 
	
	if len(results) > 0 :
		for i in results.flatten() :
			x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
			box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
			
			colour_box_current = [int(j) for j in colours[class_numbers[i]]]
			
			cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 7)
			
			text_box_current = '%s' %labels[int(class_numbers[i])]
			cv2.putText(image_input, text_box_current, (x_min, y_min-7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box_current, 5)
	
	cv2.imshow("Frame", image_input)
	cv2.waitKey(1)
	frame_array.append(image_input)

	success, image_input = video_cap.read()


video_cap.release()
cv2.destroyAllWindows()

h, w, l = frame_array[0].shape	
size = (w, h)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), 14, size)
for i in range(len(frame_array)) :
	out.write(frame_array[i])

out.release()

