from rest_framework import serializers
from .models import ImageFile

class ipSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageFile
        fields = ('id', 'image', )
