from . import views
from rest_framework import routers
from django.conf.urls.static import static
from django.conf import settings
from django.urls import path, include

router = routers.DefaultRouter()
router.register('', views.ipView)

urlpatterns = [path('', include(router.urls))] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)



