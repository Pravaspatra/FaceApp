from datetime import datetime
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
import cv2
# import blur_detector

# from blur_detector import blur_detector.estimate_blur

class CheckFaceValid(APIView):
    # authentication_classes = [authentication.TokenAuthentication]
    # permission_classes = [permissions.IsAuthenticated]

    authentication_classes = []
    permission_classes = []


    def post(self, request, format=None):

        data = request.data

        face_b64 = data['face_b64']
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




5



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

        if 'tolerance' in data:
            tolerance = data['tolerance']

        print("tttttttttttttt", tolerance)

        # print("tttttttttttttt", len(data['attachments_from']), len(data['attachments_to']))

        if 'attachments_from' in data and 'attachments_to' in data:
            face_encodings_from = []
            face_encodings_to = []


            for attachment_from_e in data['attachments_from']:
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
                        # import cv2
                        # import numpy as np
                        # imgr = cv2.imread('sample_photos/blur1.png', 0)
                        # open_cv_image = numpy.array(img2) 
                        # Convert RGB to BGR 
                        # open_cv_image = open_cv_image[:, :, ::-1].copy() 

                        # array = np.fromstring(data2, dtype=np.uint8)
                        # imgr = cv2.imdecode(img2, cv2.IMREAD_COLOR)

                        # Convert the PIL image to a numpy array
                        # img_array = np.array(img2)

                        # Convert the numpy array to an OpenCV image
                        # imgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        


                        # data2 = ContentFile(base64.b64decode(attachments_to_e2))
                        # img2 = Image.open(data2)
                        # img_array = np.asarray(img2a)

                        # Convert the numpy array to an OpenCV image
                        # imgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        # imgr = cv2.cvtColor(img_array.reshape(-1, 3), cv2.COLOR_RGB2BGR)
                        


                        
                        # binary_data = base64.b64decode(attachments_to_e2)
                        # img_array = np.frombuffer(binary_data, dtype=np.uint8)
                        # imgr = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)


                        # face_cascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
                        # # img = cv2.imread('sample_photos/blur1.png', 0)

                        # decoded_data = base64.b64decode(attachments_to_e2)

                        # # Convert the binary data to a numpy array
                        # img_array = np.frombuffer(decoded_data, np.uint8)
                        # print("7770001")
                        # # Read the image with OpenCV
                        # imgr = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE) 
                        #                     # cv2.IMREAD_GRAYSCALE) 
                        # # cv2.IMREAD_COLOR
                        # print("77700012")
                        # blur_map1 = blur_detector.detectBlur(imgr, downsampling_factor=1, num_scales=3, scale_start=1)
                        # print("77700013")
                        # value = cv2.Laplacian(blur_map1, cv2.CV_64F).var()
                        # print("7770004")
                        # print("Dfdddddddddddddddddddd", value)

                        # cv2.imshow('c', blur_map1)
                        # cv2.waitKey(0)




                        # blur_map1 = blur_detector.detectBlur(imgr, downsampling_factor=1, num_scales=3, scale_start=1)
                        # print("0115")
                        # value = cv2.Laplacian(blur_map1, cv2.CV_64F).var()
                        # print("Dfdddddddddddddddddddd", value)

                        # decoded_data = ContentFile(base64.b64decode(attachments_to_e2))
                        # # base64.b64decode(attachments_to_e2)
                        # print("011")
                        # img2a = Image.open(decoded_data)
                        # print("0112")

                        # numpy_image=np.array(img2a)  
                        # print("0113")

                        # # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that 
                        # # the color is converted from RGB to BGR format
                        # imgr=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
                        # print("0114")

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
                            
                            # import numpy as np

                            # face_cascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")


                            # Detect faces in the input image
                            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        # Convert the face region to grayscale
                            add = "000004"
                            gray_roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            
                            # Apply Laplacian filter to the face region
                            add = "000005"
                            sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                            
                            print("dffff", sharpness)

                            # If the sharpness is below a threshold, mark the face region as blurred
                            # if sharpness < 100:
                            #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)


                            # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                            # # Apply Laplacian filter to each face region to calculate the sharpness
                            # for (x, y, w, h) in faces:
                            #     # Select the face region
                            #     roi = image[y:y+h, x:x+w]
                                
                            #     # Convert the face region to grayscale
                            #     gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                
                            #     # Apply Laplacian filter to the face region
                            #     sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                                
                            #     # If the sharpness is below a threshold, mark the face region as blurred
                            #     if sharpness < 100:
                            #         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
                            #     print("sharpnesssharpness", sharpness)



                            # Display the resulting image
                            # cv2.imshow("Blurred Face Detection", image)
                            # cv2.imshow("Blurred Face G", gray_roi)
                            
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()


                            # Create OpenCV image
                            # face_image_cv2 = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                            # cv2.imshow('c', face_image_cv2)
                            # cv2.waitKey(0)

                            saturation = 350

                            datetime_current = datetime.now()


                            
                            if sharpness < saturation:
                                print("=======================0444")
                                return Response({'status':False, 'message': 'Digital Input detected!. Will be notified to Admin for approval.'})
                            add = "000006"
                            results = face_recognition.compare_faces([attachment_from_face_encoding], attachment_to_face_encoding, tolerance=tolerance)   
                            if results[0] == True:
                                print("face success", tolerance)
                                return Response({'status':True, 'message': 'Face Validated Successfully'})
                            else:
                                print("face compare failed4444")
                        except Exception as e:
                            print("face compare failed222",e)
                            return Response({'status':False, 'message': 'User face not recognized3.'+str(e)+ str(add)})

                            print("exxxxxx3333", e)
                except Exception as e:
                    print("face compare failed1111",e)
                    return Response({'status':False, 'message': 'User face not recognized2.'+str(e)})

                    print("exxxxxx1111", e)
            print("facecccc callled2")

        return Response({'status':False, 'message': 'User face not recognized1.'+str(e)})


# class ValidatePhotoAuthenticity(APIView):
#     # authentication_classes = [authentication.TokenAuthentication]
#     # permission_classes = [permissions.IsAuthenticated]

#     authentication_classes = []
#     permission_classes = []


#     def getb(self, request, format=None):

#         data = request.data

#         import cv2
#         import numpy as np

#         # # Load the input image
#         # image = cv2.imread("sample_photos/blur1.png")

#         # # Load the face detection classifier
#         # # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#         face_cascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")



#         img = cv2.imread('sample_photos/blur1.png', 0)
#         blur_map1 = blur_detector.detectBlur(img, downsampling_factor=1, num_scales=3, scale_start=1)
#         faces = face_cascade.detectMultiScale(img, 1.1, 4)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(blur_map1, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # img = cv2.imread('nonblur1.png', 0)
#         # blur_map2 = blur_detector.detectBlur(img, downsampling_factor=1, num_scales=3, scale_start=1)
#         # faces = face_cascade.detectMultiScale(img, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     cv2.rectangle(blur_map2, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # img = cv2.imread('norold.png', 0)
#         # blur_map3 = blur_detector.detectBlur(img, downsampling_factor=1, num_scales=3, scale_start=1)
#         # faces = face_cascade.detectMultiScale(img, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     cv2.rectangle(blur_map3, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # cv2.imshow('a', blur_map1)
#         # cv2.imshow('b', blur_map2)
#         # cv2.imshow('c', blur_map3)
#         # cv2.waitKey(0)
#         return Response({'status':False, 'message': 'Validating Photo'})


#     def get(self, request, format=None):

#         data = request.data

#         import cv2
#         import numpy as np

#         # # Load the input image
#         # image = cv2.imread("sample_photos/blur1.png")

#         # # Load the face detection classifier
#         # # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#         face_cascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")



#         # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         # value = cv2.Laplacian(gray, cv2.CV_64F).var()

#         # # img = cv2.imread('sample_photos/blur1.png', 0)
#         # # blur_map1 = blur_detector.detectBlur(img, downsampling_factor=1, num_scales=3, scale_start=1)
#         # # faces = face_cascade.detectMultiScale(img, 1.1, 4)
#         # # for (x, y, w, h) in faces:
#         # #     cv2.rectangle(blur_map1, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # img = cv2.imread('sample_photos/nonblur1.png', 0)
#         # blur_map2 = blur_detector.detectBlur(img, downsampling_factor=1, num_scales=3, scale_start=1)
#         # faces = face_cascade.detectMultiScale(img, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     cv2.rectangle(blur_map2, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         #     # blur_percent += sharpness

#         # # img = cv2.imread('sample_photos/ori.png', 0)
#         # # blur_map3 = blur_detector.detectBlur(img, downsampling_factor=1, num_scales=3, scale_start=1)
#         # # faces = face_cascade.detectMultiScale(img, 1.1, 4)
#         # # for (x, y, w, h) in faces:
#         # #     cv2.rectangle(blur_map3, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # # cv2.imshow('a', blur_map2)
#         # # cv2.imshow('b', blur_map2)
#         # # cv2.imshow('c', blur_map3)
#         # # cv2.waitKey(0)

#         # img1 = cv2.imread('blur1.png', 0)
#         # img2 = cv2.imread('nonblur2.png', 0)
#         # img3 = cv2.imread('norold.png', 0)

#         # blur_level1 = blur_detector.estimate_blur(img1)
#         # faces = face_cascade.detectMultiScale(img1, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     roi = img1[y:y+h, x:x+w]
#         #     blur_level1_roi = blur_detector.estimate_blur(roi)
#         #     blur_level1 = max(blur_level1, blur_level1_roi)

#         # blur_level2 = blur_detector.estimate_blur(img2)
#         # faces = face_cascade.detectMultiScale(img2, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     roi = img2[y:y+h, x:x+w]
#         #     blur_level2_roi = blur_detector.estimate_blur(roi)
#         #     blur_level2 = max(blur_level2, blur_level2_roi)

#         # blur_level3 = blur_detector.estimate_blur(img3)
#         # faces = face_cascade.detectMultiScale(img3, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     roi = img3[y:y+h, x:x+w]
#         #     blur_level3_roi = blur_detector.estimate_blur(roi)
#         #     blur_level3 = max(blur_level3, blur_level3_roi)

#         # # Print the blur levels
#         # print("Blur Level of Image 1:", blur_level1)
#         # print("Blur Level of Image 2:", blur_level2)
#         # print("Blur Level of Image 3:", blur_level3)


#         # img1 = cv2.imread('sample_photos/nonblur1.png', 0)
#         # # blur_map1 = blur_detector.detectBlur(img1, downsampling_factor=1, num_scales=3, scale_start=1)
#         # blur_map1 = blur_detector.detectBlur(img1, downsampling_factor=1, num_scales=3, scale_start=1)
#         # # blur_value1 = cv2.mean(abs(blur_map1 - cv2.mean(blur_map1))[1])[0]
#         # blur_value1 = cv2.mean(abs(blur_map1 - cv2.mean(blur_map1))[0])[0]

#         # faces = face_cascade.detectMultiScale(img1, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # if blur_value1 > 30:
#         #     print("Image 1 is blurred.")
#         # else:
#         #     print("Image 1 is not blurred.")
#         # print("Blur value of image 1:", blur_value1)


#         # img1 = cv2.imread('sample_photos/nonblur1.png', 0)

#         img = cv2.imread('sample_photos/blur1.png', 0)
#         blur_map1 = blur_detector.detectBlur(img, downsampling_factor=1, num_scales=3, scale_start=1)
#         faces = face_cascade.detectMultiScale(img, 1.1, 4)
#         for (x, y, w, h) in faces:
#             cv2.rectangle(blur_map1, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         value = cv2.Laplacian(blur_map1, cv2.CV_64F).var()
#         print("Dfdddddddddddddddddddd", value)
#         cv2.imshow('a', blur_map1)
#         cv2.waitKey(0)

#         # blur_map1 = blur_detector.detectBlur(img1, downsampling_factor=1, num_scales=3, scale_start=1)
#         # blur_value1 = cv2.mean(abs(blur_map1 - cv2.mean(blur_map1))[0])[0]

#         # faces = face_cascade.detectMultiScale(img1, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # print("Blur value of image 1:", blur_value1)
#         # if blur_value1 > 30:
#         #     print("Image 1 is blurred.")
#         # else:
#         #     print("Image 1 is not blurred.")

#         # blur_map1 = blur_detector.detectBlur(img1, downsampling_factor=1, num_scales=3, scale_start=1)
#         # # blur_value1 = cv2.mean(abs(blur_map1 - cv2.mean(blur_map1))[0])[0]
#         # blur_value1 = cv2.mean(abs(blur_map1 - cv2.mean(blur_map1)[0]))[0]

#         # faces = face_cascade.detectMultiScale(img1, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # print("Blur value of image 1:", blur_value1)
#         # if blur_value1 > 30:
#         #     print("Image 1 is blurred.")
#         # else:
#         #     print("Image 1 is not blurred.")

#         # blur_map1 = blur_detector.apply(img1)
#         # blur_value1 = blur_map1.var()

#         # faces = face_cascade.detectMultiScale(img1, 1.1, 4)
#         # for (x, y, w, h) in faces:
#         #     cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # print("Blur value of image 1:", blur_value1)
#         # if blur_value1 > 30:
#         #     print("Image 1 is blurred.")
#         # else:
#         #     print("Image 1 is not blurred.")

#         return Response({'status':False, 'message': 'Validating Photo'})



#     # def get(self, request, format=None):

#     #     data = request.data

#     #     import cv2
#     #     import numpy as np

#     #     # Load the input image
#     #     image = cv2.imread("sample_photos/blur1.png")

#     #     # Load the face detection classifier
#     #     # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#     #     face_cascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
        
#     #     # Detect faces in the input image
#     #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     #     blur_percent = 0
        
#     #     # Apply Laplacian filter to each face region to calculate the sharpness
#     #     for (x, y, w, h) in faces:
#     #         # Select the face region
#     #         roi = image[y:y+h, x:x+w]
            
#     #         # Convert the face region to grayscale
#     #         gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
#     #         # Apply Laplacian filter to the face region
#     #         sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            
#     #         # If the sharpness is below a threshold, mark the face region as blurred
#     #         if sharpness < 100:
#     #             cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
#     #         blur_percent += sharpness

#     #     # Display the resulting image
#     #     # cv2.imshow("Blurred Face Detection", image)
#     #     # cv2.waitKey(0)
#     #     # cv2.destroyAllWindows()

#     #     # Calculate the average blur percentage of all face regions
#     #     if len(faces) > 0:
#     #         blur_percent /= len(faces)

#     #     # Determine if the photo is blurred or not
#     #     is_blurred = blur_percent < 100

#     #     # Print the blur percentage and the boolean value
#     #     print("Blur Percentage:", blur_percent)
#     #     print("Is Blurred:", is_blurred)
#     #     return Response({'status':False, 'message': 'Validating Photo'})



#     # def get(self, request, format=None):

#     #     data = request.data

#     #     import cv2
#     #     import numpy as np

#     #     # Load the input image
#     #     image = cv2.imread("sample_photos/ori.jpg")

#     #     # Load the face detection classifier
#     #     # face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#     #     face_cascade = cv2.CascadeClassifier("classifiers/haarcascade_frontalface_default.xml")
        
#     #     # Detect faces in the input image
#     #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     #     # Apply Laplacian filter to each face region to calculate the sharpness
#     #     for (x, y, w, h) in faces:
#     #         # Select the face region
#     #         roi = image[y:y+h, x:x+w]
            
#     #         # Convert the face region to grayscale
#     #         gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
#     #         # Apply Laplacian filter to the face region
#     #         sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            
#     #         # If the sharpness is below a threshold, mark the face region as blurred
#     #         if sharpness < 100:
#     #             cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

#     #     # Display the resulting image
#     #     cv2.imshow("Blurred Face Detection", image)
#     #     cv2.waitKey(0)
#     #     cv2.destroyAllWindows()


#     #     return Response({'status':False, 'message': 'Validating Photo'})



# class FaceDetectCheck(APIView):

#     authentication_classes = []
#     permission_classes = []

#     def get(self, request, format=None):

#         # not working
#         employeeDocument = EmployeeDocument.objects.get(id= '5e6e0a1d-5b51-4a07-a02e-72efc781e3a7')

#         # working
#         # employeeDocument = EmployeeDocument.objects.get(id= '4662e2b1-bd96-419b-bec9-4973a822b924')
        
#         print("==============01")
#         print("==============", employeeDocument.photo.url)


#         img = Image.open(requests.get(employeeDocument.photo.url, stream=True).raw)
#         img = ImageOps.exif_transpose(img)   
#         img.thumbnail((800, 800))
#         temp = BytesIO()
#         img.save(temp, format="png")
#         img = img.convert('RGB')
#         img  = np.array(img)

#         try:
#             unknown_face_encoding = face_recognition.face_encodings(img)[0]
#         except Exception as e:
#             print("exxxxxx", e)

#         return Response(get_validation_failure_response(None, "Checking Face1"))  


# class AttachUserFacePhoto(APIView):
#     authentication_classes = [authentication.TokenAuthentication]
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request, format=None):

#         data = request.data

#         print(data.keys())
#         request_info = get_user_company_from_request(request)
#         validation = UpdateEmployeeProfilePhotoSerializer(data=data)
#         if validation.is_valid() and request_info['company_info'] is not None:

#             is_face_valid = True
#             if 'attachments' in data:
#                 for attachment in data['attachments']:
#                     received_current_photo_base64 = attachment
#                     base_comparison_image = ContentFile(base64.b64decode(attachment)).open()
#                     try:
#                         unknown_picture = face_recognition.load_image_file(base_comparison_image)
#                         # unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]
#                         break
#                     except:
#                         is_face_valid = False
#                         break                                 
#                 ref_01 = None

#             if is_face_valid == False:
#                 return Response(get_validation_failure_response(None, 'Face not visible clearly. Please recapture and submit'))


#             employeeCompanyInfo = EmployeeCompanyInfo.objects.get(user = request_info['user'])
#             # employeeFacePhoto = employeeCompanyInfo.face_photo

#             # if employeeCompanyInfo.face_photo is None:
#             employeeFacePhoto = EmployeeFacePhoto.objects.create()
#             employeeCompanyInfo.face_photo = employeeFacePhoto
#             employeeCompanyInfo.save()

#             # employeeCompanyInfoPhoto = employeeCompanyInfo.face_photo
            
#             if 'attachments' in data:

#                 # attachment = data['attachment']
#                 # if attachment[-1] == '/':            
#                 #     file_name = get_file_name()+".png"
#                 # else:
#                 #     file_name = get_file_name()+".jpg"
#                 # data = ContentFile(base64.b64decode(attachment))  
#                 # employeePersonalInfo = EmployeePersonalInfo.objects.get(user = request_info['user'])
#                 # employeePersonalInfo.attachment_profile.save(file_name, data, save=True)

#                 for attachment in data['attachments']:                
#                     if attachment[-1] == '/':            
#                         file_name = get_file_name()+".png"
#                     else:
#                         file_name = get_file_name()+".jpg"
#                     data = ContentFile(base64.b64decode(attachment))

#                     companyDocument = EmployeeDocument.objects.create()
#                     companyDocument.save()
#                     companyDocument.photo.save(file_name, data, save=True)
#                     companyDocument.save()

#                     employeeFacePhoto.photos.add(companyDocument)
#                     employeeFacePhoto.recapture_enabled = False
#                     employeeFacePhoto.recapture_requested = False
#                     employeeFacePhoto.save()

#                 # employeePersonalInfo = EmployeePersonalInfo.objects.get(user = request_info['user'])
#                 # employeePersonalInfo.attachment_profile.save(file_name, data, save=True)

#                 return Response(get_success_response("Details updated successfully"))  
#             else:
#                  return Response(get_validation_failure_response(None, 'Invalid Request'))

#         else:
#             return Response(get_validation_failure_response(validation.errors))


# class ValidateUserByFace(APIView):
#     authentication_classes = [authentication.TokenAuthentication]
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request, format=None):

#         data = request.data

#         request_info = {}
#         request_info['user'] = get_user_from_token(request)
#         validation = ValidateUserByFaceSerializer(data=data)
#         if validation.is_valid():
#             #  and request_info['company_info'] is not None:
#             employeeCompanyInfo = EmployeeCompanyInfo.objects.get(user = request_info['user'])
#             employeeFacePhoto = employeeCompanyInfo.face_photo
#             base_comparison_image = None

#             if 'attachments' in data:
#                 for attachment in data['attachments']:
#                     base_comparison_image = ContentFile(base64.b64decode(attachment)).open()
                    
#                     # print("prerrrrrrrrrrrrrrrrrrrrrrrrrrrr=============")
#                     # print(attachment[0:20])
#                     break    

#                 unknown_picture = None
#                 can_read_face_from_bucket = False

#                 # # # if attachment[-1] == '/':            
#                 # file_name = get_file_name()+".png"
#                 # # else:
#                 # file_name = get_file_name()+".jpeg"
#                 data = ContentFile(base64.b64decode(attachment))

#                 # companyDocument = EmployeeDocument.objects.create()
#                 # companyDocument.save()
#                 # companyDocument.photo.save(file_name, data, save=True)
#                 # companyDocument.save()

#                 # print("receivedphoto===", companyDocument.photo.url)

#                 img = Image.open(data)
#                     # requests.get(companyDocument.photo.url, stream=True).raw)
#                 img = ImageOps.exif_transpose(img)   
#                 img.thumbnail((400, 400))
#                 # temp = BytesIO()
#                 # img.save(temp, format="png")
                
#                 # img = img.convert('RGB')
#                 img = np.array(img)


#                 try:
#                     unknown_face_encoding = face_recognition.face_encodings(img)[0]
#                 except Exception as e:
#                     print("exxxxxx", e)

#                     return Response(get_validation_failure_response(None, "Face not clear. Please recapture"))  

#                 employeeFacePhotos = employeeFacePhoto.photos.all()
#                 for employeeFacePhoto in employeeFacePhotos:

#                     try:
#                         imge = Image.open(requests.get(employeeFacePhoto.photo.url, stream=True).raw)
#                         imge = ImageOps.exif_transpose(imge)   

#                         imge.thumbnail((400, 400))
#                         # temp1 = BytesIO()
#                         # imge.save(temp1, format="png")                
#                         # imge = imge.convert('RGB')
#                         imge = np.array(imge)

#                         # face_locations = face_recognition.face_locations(imge)
#                         # print('face_locations', face_locations)
                        
#                         # my_face_encoding = face_recognition.face_encodings(imge, known_face_locations=[(345, 324, 531, 139)])[0]
#                         my_face_encoding = face_recognition.face_encodings(imge)[0]

#                         can_read_face_from_bucket = True
#                         break
#                     except:
#                         pass
                
#                 if can_read_face_from_bucket == False:
#                     return Response(get_validation_failure_response(None, 'User face not recognized.'))

#                 results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding, tolerance=0.6)   
#                 try:
#                     if results[0] == True:
#                         return Response(get_success_response("User Validated Suceessfully"))  
#                     else:
#                         return Response(get_validation_failure_response(None, 'User face not recognized.'))
#                 except:
#                         return Response(get_validation_failure_response(None, 'User face not recognized.'))
#             else:
#                  return Response(get_validation_failure_response(None, 'User face not recognized.'))
#         else:
#             return Response(get_validation_failure_response(validation.errors))

