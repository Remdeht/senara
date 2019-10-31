from django.db import models

# Create your models here.


class ImageFile(models.Model):
    image = models.ImageField(upload_to='images')
