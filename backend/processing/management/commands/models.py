from django.contrib.gis.db import models as gisModels

# Create your models here.


class Sector(gisModels.Model):
    name = gisModels.TextField(max_length=150, primary_key=True)

    def __str__(self):
        return self.name


class Watermeter(gisModels.Model):
    location = gisModels.PointField(srid=4326, null=True)
    altitude = gisModels.IntegerField(null=True, blank=True, default=None)
    owner = gisModels.TextField(max_length=250, primary_key=True)
    sector = gisModels.ForeignKey(Sector, on_delete=gisModels.CASCADE)

    def __str__(self):
        return self.owner







