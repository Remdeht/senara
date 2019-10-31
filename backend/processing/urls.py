from . import views
from rest_framework import routers
from django.urls import path

router = routers.DefaultRouter()
# router.register(r'update/<image>', views.)

urlpatterns = [path('', views.start_image_processing),
               path('near/', views.near),
               path(r'update/<image>', views.update_values),]


