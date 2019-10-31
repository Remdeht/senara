from django.core.management.base import BaseCommand, CommandError
from .models import Watermeter, Sector
import csv
from pyproj import Proj, transform
import os


WATERMETERS = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + r'\\commands\\waterMeterLocations.csv'

class Command(BaseCommand):
    help = 'Loads in watermeter locations based on csv file of watermeters. In order to run command run "python ' \
           'manage.py add_watermeters" in terminal'

    def handle(self, *args, **options):
        with open(WATERMETERS) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:

                point = self.create_location_point(row[0], row[1])
                Sector(
                    name=row[4]
                ).save()

                Watermeter(
                    location=point,
                    altitude=int(float(row[2])),
                    owner=row[3],
                    sector=Sector.objects.filter(name=row[4])[0]
                ).save()

    def create_location_point(self, x, y):
        inProj = Proj(init='epsg:5456')
        outProj = Proj(init='epsg:4326')
        x_repr, y_repr = transform(inProj, outProj, x, y)
        point = 'POINT(' + str(x_repr) + ' ' + str(y_repr) + ')'
        return point
