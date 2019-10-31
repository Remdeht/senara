from django.contrib import admin
from .models import ImageProcessing
from .management.commands.models import Watermeter, Sector

# Register your models here.

admin.site.register(ImageProcessing)
admin.site.register(Watermeter)
admin.site.register(Sector)