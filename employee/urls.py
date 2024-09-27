from django.urls import path

from . import views

urlpatterns = [
    path('checkFaceValid', views.CheckFaceValid.as_view(), name='faceCheck'),    
    path('faceCheck', views.CompareFaces.as_view(), name='faceCheck'),
    path('validatePhotoAuthenticity', views.ValidatePhotoAuthenticity.as_view(), name='validatePhotoAuthenticity'),

]

