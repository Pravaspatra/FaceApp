from datetime import datetime, time
from io import BytesIO
from mmap import PAGESIZE
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import authentication, permissions
from django.contrib.auth.models import User

import employee
import random
import base64
from django.core.files.base import ContentFile 
import face_recognition
from PIL import Image, ImageOps
import requests
import numpy as np
# import cv2
#         # import numpy as np
import cv2
import os
# from darknet import darknet
import darknet

print('============================================cv2',cv2.__version__)

# import blur_detector

# from blur_detector import blur_detector.estimate_blur

class CheckFaceValid(APIView):
    # authentication_classes = [authentication.TokenAuthentication]
    # permission_classes = [permissions.IsAuthenticated]

    authentication_classes = []
    permission_classes = []


    def post(self, request, format=None):
        
        print("rrrrrrrrrrrrrrrrrr")
        data = request.data





        # return response
        face_b64 = data['face_b64']
        
        # print("=========================LLLLLLLLLLLLLLLLLLLLLll",request['header'])
        data1 = ContentFile(base64.b64decode(face_b64))
        img = Image.open(data1)
        img = ImageOps.exif_transpose(img)   
        img.thumbnail((400, 400))
        img = np.array(img)

        try:
            attachment_from_face_encoding = face_recognition.face_encodings(img)[0]
            return Response({'status':True, 'message': 'Face Validated Successfully'})
        except Exception as e :
            return Response({'status':False, 'message': 'User face not recognized5.'+ str(e)})





class CompareFaces(APIView):
    # authentication_classes = [authentication.TokenAuthentication]
    # permission_classes = [permissions.IsAuthenticated]

    authentication_classes = []
    permission_classes = []


    def post(self, request, format=None):

        data = request.data
        # print("attfrom", data['attachments_from'])
        # print("attto", len(data['attachments_to']))

        tolerance = 0.4

        # try:
        #     token = request.META.get('HTTP_AUTHORIZATION', '').replace('Token ', '')
        #     url = "https://reports.zenyq.com/employee/attachDocument"
            
        #     dataa = {}
        #     dataa['token'] =  token
        #     dataa['face_base64'] =  data['attachments_to']

        #     print("dfsss", len(data['attachments_to']))
        #     print("dfsss2", len(dataa['face_base64']))

        #     response = requests.request("POST", url, data=dataa)
        # except:
        #     pass
        if 'tolerance' in data:
            tolerance = data['tolerance']

        # print("tttttttttttttt", tolerance)

        if 'lux' in data:
            lux  = data['lux']
        else:
            lux = -1

        image_metrics = {'lux' : lux}

        # print("tttttttttttttt", len(data['attachments_from']), len(data['attachments_to']))

        if 'attachments_from' in data and 'attachments_to' in data:
            face_encodings_from = []
            face_encodings_to = []

            # print("c0001")
            try:
                data1 = ContentFile(base64.b64decode(data['face_initiated'][0]))
                img = Image.open(data1)
                #img.show()
                # print("c00013", is_digital_input(img))

                if is_digital_input(img,"01") == True:
                    #print("=======================face_initiated===0")
                    return Response({'status':False, 'message': 'Internal Digital Input detected!. Will be notified to Admin for approval.', "details": image_metrics })
                print("c00012")

                data1 = ContentFile(base64.b64decode(data['face_initiated_03'][0]))
                img = Image.open(data1)
                #img.show()

                if is_digital_input(img,"02") == True:
                    #print("=======================face_initiated===3")
                    return Response({'status':False, 'message': 'Internal Digital Input detected!. Will be notified to Admin for approval.', "details": image_metrics })
            except Exception as e:
                print("eeeeeeeeeeeeeim1", e)
                pass


            for e in data['attachments_to']:
                data1 = ContentFile(base64.b64decode(e))
                img = Image.open(data1)
                # img = Image.open(data1)
                #img.show()
                # print("c00014", is_digital_input(img))

                if is_digital_input(img,"03") == True:
                    #print("=====0003==================attachments_to")
                    return Response({'status':False, 'message': 'Internal Digital Input detected!. Will be notified to Admin for approval.', "details": image_metrics })

            



            for attachment_from_e in data['attachments_from']:
                #print("=========================attachments_fromattachments_from====")
                data1 = ContentFile(base64.b64decode(attachment_from_e))
                img = Image.open(data1)
                img = ImageOps.exif_transpose(img)   
                img.thumbnail((400, 400))
                img = np.array(img)

                try:
                    attachment_from_face_encoding = face_recognition.face_encodings(img)[0]
                    for attachments_to_e2 in data['attachments_to']:
                        data2 = ContentFile(base64.b64decode(attachments_to_e2))
                        img2 = Image.open(data2)
                        
                        '''blur image test'''
                        img2 = ImageOps.exif_transpose(img2)
                        
                        img2.thumbnail((400, 400))
                        img2 = np.array(img2)
                        try:
                            add = "000001"
                            face_locations = face_recognition.face_locations(img2)
                            attachment_to_face_encoding = face_recognition.face_encodings(img2, face_locations)[0]
                            add = "000002"
                            face_encodings = attachment_to_face_encoding
                            face_encoding = face_encodings[0]
                            top, right, bottom, left = face_locations[0]
                            face_image = img2[top:bottom, left:right]

                            add = "000003"
                            image = face_image
                            add = "000004"
                            gray_roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            
                            # Apply Laplacian filter to the face region
                            add = "000005"
                            sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                            
                            saturation = 120

                            # datetime_current = datetime.now()

                            if 'platform_os' in data and data['platform_os'] == 'ios':
                                saturation = 70
                            
                            image_metrics['saturation'] = saturation

                            image_metrics['sharpness'] = sharpness
                            print("sharpness====================", sharpness, saturation)


                            # print("=======================sharpnessless than saturation==========", sharpness)
                            if sharpness < saturation:
                                
                                return Response({'status':False, 'message': 'External Digital Input detected!. Will be notified to Admin for approval.', "details": image_metrics })

                            add = "000006"
                            results = face_recognition.compare_faces([attachment_from_face_encoding], attachment_to_face_encoding, tolerance=tolerance)   
                            if results[0] == True:
                                print("face success", tolerance)
                                return Response({'status':True, 'message': 'Face Validated Successfully', "details": image_metrics})
                            else:
                                print("face compare failed4444", results)
                        except Exception as e:
                            print("face compare failed222",e)
                            

                            print("exxxxxx3333", e)
                except Exception as e:
                    print("face compare failed1111",e)
                    pass

                    print("exxxxxx1111", e)
            print("facecccc callled2")
        return Response({'status':False, 'message': 'User face not recognized', "details": image_metrics})


# def is_digital_input(image):

#     import tempfile
#     numpy_array = np.array(image)
#     pillow_image = Image.fromarray(numpy_array)
#     temp_file = tempfile.NamedTemporaryFile(suffix='.jpg')
#     pillow_image.save(temp_file.name)
#     image = cv2.imread(temp_file.name)

#     try:
#         net = cv2.dnn.readNetFromDarknet('yolov3_001/yolov3_custom.cfg', 'yolov3_001/yolov3.weights')

#         # print("p001")
#         phone_detected = False

#         height, width = image.shape[:2]

#         desired_width = 416
#         desired_height = 416

#         aspect_ratio = width / height

#         # print("p0012")

#         if aspect_ratio > 1:
#             new_width = desired_width
#             new_height = int(desired_width / aspect_ratio)
#         else:
#             new_height = desired_height
#             new_width = int(desired_height * aspect_ratio)

#         resized_image = cv2.resize(image, (new_width, new_height))

#         canvas = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)

#         x = (desired_width - new_width) // 2
#         y = (desired_height - new_height) // 2

#         canvas[y:y+new_height, x:x+new_width] = resized_image

#         blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (416, 416), swapRB=True, crop=True)

#         net.setInput(blob)

#         output_layers = net.getUnconnectedOutLayersNames()

#         outs = net.forward(output_layers)

#         class_labels = ['Face','Mobile-Phone']  # Modify with desired class labels

#         conf_threshold = 0.3
#         nms_threshold = 0.3

#         (H, W) = image.shape[:2]

#         class_ids = []
#         confidences = []
#         boxes = []

#         phone_detected = False
#         for out in outs:
#             for detection in out:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]

#                 if confidence > conf_threshold:
#                     class_type = class_labels[class_id]  # Get the class type using class ID
#                     # print("Detected class type:",class_id, class_type)
#                     box = detection[0:4] * np.array([W, H, W, H])
#                     (center_x, center_y, box_width, box_height) = box.astype("int")
#                     x = int(center_x - (box_width / 2))
#                     y = int(center_y - (box_height / 2))

#                     class_ids.append(class_id)
#                     confidences.append(float(confidence))
#                     boxes.append([x, y, int(box_width), int(box_height)])
#                     phone_detected = True

#         indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
#         is_digital_input = False
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 x, y, w, h = boxes[i]
#                 class_id = class_ids[i]
#                 label = class_labels[class_id]
#                 confidence = confidences[i]

#                 cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 print('------------------------------000121', confidence,"----","==="+label+"===")
#                 print('------------------------------000121', confidence>.9 ,"----",label== 'Mobile-Phone')

#                 if label == 'Mobile-Phone' and confidence > .9:
#                     # print('------------------------------000123')
#                     is_digital_input = True

#         return is_digital_input        
#         # print("p0015")
#     except Exception as e:
#         return False
#         # print("eeeeaaa", e)    

#     # cv2.imshow("Object Detection", image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     # print('------------------------------000000',confidence)
#     # return Response({'status':False, 'message': 'Validating Photo'+ str(phone_detected)})



def is_digital_input(image, param):
    
    #print('======================0001=============================',param)
    import tempfile
    import numpy as np
    import cv2
    import darknet

    numpy_array = np.array(image)
    pillow_image = Image.fromarray(numpy_array)
    #temp_file = tempfile.NamedTemporaryFile(suffix='.jpg')
    #pillow_image.save(temp_file.name)
    #img = cv2.imread(temp_file.name)
    # img = Image.open(temp_file.name)
    img = np.array(pillow_image)
    #img.close()
    #cv2.waitKey(0)
# print("==================================09999")
    #cv2.destroyAllWindows()

    try:
        

        net = cv2.dnn.readNetFromDarknet('yolov3_final/yolov3_final/yolov3_custom.cfg' , 'yolov3_final/yolov3_final/yolov3_custom_4000.weights' )
        # net = ('yolov3_final/yolov3_final/yolov3_custom.cfg', 'yolov3_final/yolov3_final/yolov3_custom_4000.weights', batch_size = 1)
        # net.load_weights('yolov3_final/yolov3_final/yolov3_custom_4000.weights')
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        
        # /home/leora/Downloads/sample_photos/new/mob14.jpg
        
        classes = ['Face','Mobile-Phones']

        # img = cv2.imread('sample_photos/test/face.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height,width,_ = img.shape

        # print('==========size',height,width)
        

        blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)

        net.setInput(blob)

        output_layer_names = net.getUnconnectedOutLayersNames()
        layers_output = net.forward(output_layer_names)
        confidence_thresh = 0.5
        NMS_thresh = 0.45
        boxes =[]
        confidences = []
        class_ids = []

        for output in layers_output:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_thresh:
                    center_x = int(detection[0]*width)

                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)

                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    

        # print('================001',len(boxes))
        indexes = cv2.dnn.NMSBoxes(boxes,confidences,confidence_thresh,NMS_thresh)
        # print('------------------002',indexes.flatten())
        
        font = cv2.FONT_HERSHEY_PLAIN

        img2= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        is_digital_input = False
        # print("==================================0888")
        if len(indexes) > 0:
            for i in indexes.flatten():
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                
                cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(img2,label+' '+confidence,(x,y+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                print('==========================0003',float(confidence),'=============',label)    
                #cv2.imshow('image',img2)
                #cv2.waitKey(1)
                # print("==================================09999")
                #cv2.destroyAllWindows()
                # print("==================================destroyAllWindows")
                if label == 'Mobile-Phones' and float(confidence) > .9:
                    # print('------------------------------000123')
                    is_digital_input = True
        
                return is_digital_input
    except Exception as e:
        print("eeeeaaa==================exppppsssssssssss", e) 
        return False
           

    # cv2.imshow("Object Detection", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print('------------------------------000000',confidence)
    # return Response({'status':False, 'message': 'Validating Photo'+ str(phone_detected)})


class ValidatePhotoAuthenticity(APIView):
    # authentication_classes = [authentication.TokenAuthentication]
    # permission_classes = [permissions.IsAuthenticated]

    authentication_classes = []
    permission_classes = []

    def get(self, request):
        import cv2
        import numpy as np


        # define the minimum confidence (to filter weak detections), 
        # Non-Maximum Suppression (NMS) threshold, and the green color
        confidence_thresh = 0.5
        NMS_thresh = 0.3
        green = (0, 255, 0)

        # Load the image and get its dimensions
        image = cv2.imread("face/tmpteve9_fa.jpg")
        # output_jpg_filename = "face/tmpteve9_fa" + ".jpg"
        # image = cv2.imwrite(output_jpg_filename, image)
        # resize the image to 25% of its original size
        image = cv2.resize(image, 
                            (int(image.shape[0] * 0.25), 
                            int(image.shape[1] * 0.25)))

        # get the image dimensions
        h = image.shape[0]
        w = image.shape[1]

        # load the class labels the model was trained on
        classes = ['Face','Mobile-Phones']
            
        # load the configuration and weights from disk
        yolo_config = "yolov3_final2/yolov3_custom.cfg"
        yolo_weights = "yolov3_final2/yolov3_custom_4000.weights"

        # load YOLOv3 network pre-trained on the COCO dataset
        net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Get the name of all the layers in the network
        # layer_names = net.getLayerNames()
        output_layer_names = net.getUnconnectedOutLayersNames()
        
        # Get the names of the output layers
        # output_layers = [layers_output[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # create a blob from the image
        blob = cv2.dnn.blobFromImage(
            image, 1 /255, (416,416), swapRB=True, crop=False)
        # pass the blob through the network and get the output predictions
        net.setInput(blob)
        layers_output = net.forward(output_layer_names)
        # outputs = net.forward(output_layers)

        # create empty lists for storing the bounding boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []

        # loop over the output predictions
        for output in layers_output:
            # loop over the detections
            for detection in output:
                # get the class ID and confidence of the dected object
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence  = scores[class_id]

                # we keep the bounding boxes if the confidence (i.e. class probability) 
                # is greater than the minimum confidence 
                if confidence > confidence_thresh:
                    # perform element-wise multiplication to get
                    # the coordinates of the bounding box
                    box = [int(a * b) for a, b in zip(detection[0:4], [w, h, w, h])]
                    center_x, center_y, width, height = box
                    
                    # get the top-left corner of the bounding box
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    # append the bounding box, confidence, and class ID to their respective lists
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, width, height])

        # draw the bounding boxes on a copy of the original image 
        # before applying non-maxima suppression
        # image_copy = image.copy()
        # for box in boxes:
        #     x, y, width, height = box
        #     cv2.rectangle(image_copy, (x, y), (x + width, y + height), green, 2)
            
        # # show the output image
        # cv2.imshow("Before NMS", image_copy)
        # cv2.waitKey(0)

        # apply non-maximum suppression to remove weak bounding boxes that overlap with others.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, NMS_thresh)
        indices = indices.flatten()
        for i in indices:
            (x, y, w, h) = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            cv2.rectangle(image, (x, y), (x + w, y + h), green, 2)
            text = f"{classes[class_ids[i]]}: {confidences[i] * 100:.2f}%"
            cv2.putText(image, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
            print('------------------------------000000',confidences[i],'=====',classes[class_ids[i]])

        # show the output image
        #cv2.imshow("After NMS", image)
        #cv2.waitKey(0)
       
        return Response({'status':False, 'message': 'Validating Photo'})
    

#====================================================================================

# class ValidatePhotoAuthenticity(APIView):
#     # authentication_classes = [authentication.TokenAuthentication]
#     # permission_classes = [permissions.IsAuthenticated]

#     authentication_classes = []
#     permission_classes = []


#     def resize_and_pad_image(self, image, target_size):
#         # Get the original image dimensions
#         original_height, original_width = image.shape[:2]

#         # Calculate the aspect ratio of the target size
#         target_width, target_height = target_size
#         aspect_ratio = target_width / target_height

#         # Calculate the aspect ratio of the original image
#         original_aspect_ratio = original_width / original_height

#         # Determine if padding is needed on width or height
#         if aspect_ratio > original_aspect_ratio:
#             new_width = target_height * original_aspect_ratio
#             new_height = target_height
#         else:
#             new_width = target_width
#             new_height = target_width / original_aspect_ratio

#         # Resize the image while maintaining the aspect ratio
#         resized_image = cv2.resize(image, (int(new_width), int(new_height)))

#         # Create a blank image with dark pixels as background
#         padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

#         # Calculate the position to paste the resized image
#         x_offset = (target_width - resized_image.shape[1]) // 2
#         y_offset = (target_height - resized_image.shape[0]) // 2

#         # Paste the resized image onto the blank image
#         padded_image[y_offset:y_offset+resized_image.shape[0], x_offset:x_offset+resized_image.shape[1]] = resized_image

#         return padded_image


    # def get(self, request):

    #     import cv2
    #     import numpy as np

    #     net = cv2.dnn.readNetFromDarknet('yolov3_final2/yolov3_custom.cfg', 'yolov3_final2/yolov3_custom_4000.weights')

    #     image = cv2.imread('sample_photos/harikphone1.jpeg')
    #     max_pixel_value = np.max(image)
    #     print("============>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>=====================", max_pixel_value)


    #     phone_detected = False

    #     height, width = image.shape[:2]
    #     print("=============H and W, original=========================", width, height)
    #     desired_width = 416
    #     desired_height = 416

    #     aspect_ratio = width / height

    #     if aspect_ratio > 1:
    #         new_width = desired_width
    #         new_height = int(desired_width / aspect_ratio)
    #     else:
    #         new_height = desired_height
    #         new_width = int(desired_height * aspect_ratio)

    #     final_image = image

    #     print("ddddaa", new_width, new_height)
    #     resized_image = cv2.resize(image, (new_width, new_height), fx=desired_width/image.shape[1], fy=desired_height/image.shape[0])
    #     print("=============H and W, original=========================", width, height)
    #     target_size = (416, 416)
    #     final_image = resized_image
    #     final_image = self.resize_and_pad_image(final_image, target_size)



    #     final_height, final_width = resized_image.shape[:2]

    #     print("final_height, final_width ", final_height, final_width)

    #     # cv2.imshow("Object Detection2", final_image)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()

    #     # canvas = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)

    #     # x = (desired_width - new_width) // 2
    #     # y = (desired_height - new_height) // 2

    #     # canvas[y:y+new_height, x:x+new_width] = resized_image

    #     # blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (416, 416), swapRB=True, crop=False)


    #     blob = cv2.dnn.blobFromImage(final_image, scalefactor=1.0/225.0, size=(416, 416), swapRB=False, crop=False)
      
    #     net.setInput(blob)

    #     output_layers = net.getUnconnectedOutLayersNames()      

    #     outs = net.forward(output_layers)

    #     class_labels = ['Face','Mobile-Phone']  # Modify with desired class labels

    #     # with open('yolo/coco.names', 'r') as f:
    #     #     classes = [line.strip() for line in f.readlines()]
    #     # layer_names = net.getLayerNames110
    #     # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #     # colors = np.random.uniform(0, 255, size=(len(classes), 3))

    #     conf_threshold = 0.4
    #     nms_threshold = 0.4

    #     (H, W) = image.shape[:2]

    #     class_ids = []
    #     confidences = []
    #     boxes = []

    #     phone_detected = False
    #     for out in outs:
    #         for detection in out:
    #             scores = detection[5:]
    #             class_id = np.argmax(scores)
    #             confidence = scores[class_id]

    #             if confidence > conf_threshold:
    #                 class_type = class_labels[class_id]  # Get the class type using class ID
    #                 print("Detected class type:",class_id, class_type)
    #                 box = detection[0:4] * np.array([W, H, W, H])
    #                 (center_x, center_y, box_width, box_height) = box.astype("int")
    #                 x = int(center_x - (box_width / 2))
    #                 y = int(center_y - (box_height / 2))

    #                 class_ids.append(class_id)
    #                 confidences.append(float(confidence))
    #                 boxes.append([x, y, int(box_width), int(box_height)])
    #                 phone_detected = True

    #     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    #     if len(indices) > 0:
    #         for i in indices.flatten():
    #             x, y, w, h = boxes[i]
    #             class_id = class_ids[i]
    #             label = class_labels[class_id]
    #             confidence = confidences[i]

    #             cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #             cv2.putText(final_image, f"{label}: {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #             print('------------------------------0001',confidence,"----",label)
    #             print('------------------------------000000',confidence)

    #     # cv2.destroyAllWindows()
    #     print('------------------------------beforeshowi')
    #     cv2.imshow("Object Detection2", final_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     return Response({'status':False, 'message': 'Validating Photo'+ str(phone_detected)})
    # def get(self, request):
    #     import cv2
    #     import numpy as np

    #     net = cv2.dnn.readNetFromDarknet('yolov3_final/yolov3_custom.cfg','yolov3_final/yolov3_custom_4000.weights')
    #     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    #     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    #     # /home/leora/Downloads/sample_photos/new/mob14.jpg
    #     phone_detected = False
    #     classes = ['Face','Mobile-Phones']

    #     img = cv2.imread('sample_photos/test/face.jpg')
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     height,width,_ = img.shape

    #     print('==========size',height,width)
        

    #     blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)

    #     net.setInput(blob)

    #     output_layer_names = net.getUnconnectedOutLayersNames()
    #     layers_output = net.forward(output_layer_names)
    #     confidence_thresh = 0.5
    #     NMS_thresh = 0.45
    #     boxes =[]
    #     confidences = []
    #     class_ids = []

    #     for output in layers_output:
    #         for detection in output:
    #             scores = detection[5:]
    #             class_id = np.argmax(scores)
    #             confidence = scores[class_id]
    #             if confidence > confidence_thresh:
    #                 center_x = int(detection[0]*width)

    #                 center_y = int(detection[1]*height)
    #                 w = int(detection[2]*width)

    #                 h = int(detection[3]*height)

    #                 x = int(center_x - w/2)
    #                 y = int(center_y - h/2)

    #                 boxes.append([x,y,w,h])
    #                 confidences.append(float(confidence))
    #                 class_ids.append(class_id)
    #                 phone_detected = True

    #     print('================001',len(boxes))
    #     indexes = cv2.dnn.NMSBoxes(boxes,confidences,confidence_thresh,NMS_thresh)
    #     print('------------------002',indexes.flatten())
    #     font = cv2.FONT_HERSHEY_PLAIN

    #     img2= cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     for i in indexes.flatten():
    #         x,y,w,h = boxes[i]
    #         label = str(classes[class_ids[i]])
    #         confidence = str(round(confidences[i]*100,2))
            
    #         cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
    #         cv2.putText(img2,label+' '+confidence,(x,y+20),font,2,(255,255,255),2)
    #         print('==========================0003',confidence,'=============',label)
    #     cv2.imshow('image',img2)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
