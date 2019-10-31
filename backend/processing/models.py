from django.db import models
from django.utils import timezone
from .management.commands.models import Watermeter

# Create your models here.


class ImageProcessing(models.Model):
    name = models.CharField(max_length=50, primary_key=True)
    watermeter = models.ForeignKey(Watermeter, on_delete=models.CASCADE, null=True)
    date = models.DateTimeField(blank=True, default=timezone.now)
    pointer_value = models.IntegerField(null=True, blank=True, default=None)
    flag_pointer = models.BooleanField(default=False)
    tally_counter_1 = models.IntegerField(null=True, blank=True, default=None)
    flag_tally_counter_1 = models.BooleanField(default=False)
    tally_counter_2 = models.IntegerField(null=True, blank=True, default=None)
    flag_tally_counter_2 = models.BooleanField(default=False)
    tally_counter_3 = models.IntegerField(null=True, blank=True, default=None)
    flag_tally_counter_3 = models.BooleanField(default=False)
    tally_counter_4 = models.IntegerField(null=True, blank=True, default=None)
    flag_tally_counter_4 = models.BooleanField(default=False)
    tally_counter_5 = models.IntegerField(null=True, blank=True, default=None)
    flag_tally_counter_5 = models.BooleanField(default=False)
    tally_counter_6 = models.IntegerField(null=True, blank=True, default=None)
    flag_tally_counter_6 = models.BooleanField(default=False)

    def __str__(self):
        return self.name





