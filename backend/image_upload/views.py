from rest_framework import viewsets
from .models import ImageFile
from .serializers import ipSerializer
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse


class ipView(viewsets.ModelViewSet):

    queryset = ImageFile.objects.all()
    serializer_class = ipSerializer



