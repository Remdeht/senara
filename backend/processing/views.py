from rest_framework import viewsets
import os
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import ImageProcessing
from .management.commands.models import Watermeter
from django.contrib.gis.measure import Distance
from django.contrib.gis.geos import Point
import json
from datetime import datetime
from .image_processing.main import detect

FOURIER_DESCRIPTORS = 30  # Pick from -> (10, 20, 30, 40, 50)
NUMBER_OF_NEIGHBOURS = 5  # Pick from -> (5, 7, 9, 11)
CLASSIFICATION_METHOD = 0  # 0 -> uniform weighted, 1 -> distance weighted
TEMPLATE_SET = 1  # 0 -> Structured Template Dataset, 1 -> Random Template Dataset


@csrf_exempt
def start_image_processing(request):
    """Starts image processing for uploaded image"""
    default = {"safely_executed": False}

    if request.method == "GET":
        image = request.GET.get('image', '')  # Select the parameters from the GET request: imagename, lat and long
        lat = request.GET.get('lat', '')
        long = request.GET.get('long', '')

        path_to_image = str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "\\images\\" + image
        # gets the nearest water meter object from the database based on the device location
        nearby_watermeter = check_for_nearby_watermeter(point=Point(float(long), float(lat)))

        try:
            # Start image Processing
            pointer_value, tally_values = detect(path_to_image,
                                                 fd=FOURIER_DESCRIPTORS, n=NUMBER_OF_NEIGHBOURS,
                                                 flag_knn=CLASSIFICATION_METHOD, flag_templates=TEMPLATE_SET)

            if pointer_value is not None:
                pointer_value_1, pointer_value_2, pointer_value_3 = split_pointer_value(pointer_value)  # split the pointer value to integers
            else:
                pointer_value_1 = 0
                pointer_value_2 = 0
                pointer_value_3 = 0

            tc_1, tc_2, tc_3, tc_4, tc_5, tc_6 = check_tally_value(tally_values)  # checks reliability of predictios

        except Exception as e:
            print(e)
            return JsonResponse(default)
        else:
            # os.remove(path_to_image)  # Remove the image after processing is done

            location = getattr(nearby_watermeter, "location")  # Gets the location of the nearest water meter
            owner = getattr(nearby_watermeter, "owner")  # Gets the name of the owner of the water meter

            default = {
                "safely_executed": True,
                "watermeter": str(owner),
                "latitude": location.x,
                "longitude": location.y,
                "pointer_value_1": pointer_value_1,
                "pointer_value_2": pointer_value_2,
                "pointer_value_3": pointer_value_3,
                "tally_counter_1": tc_1,
                "tally_counter_2": tc_2,
                "tally_counter_3": tc_3,
                "tally_counter_4": tc_4,
                "tally_counter_5": tc_5,
                "tally_counter_6": tc_6,
            }

            return JsonResponse(default)  # sends the results of the image processing to the application


def check_for_nearby_watermeter(point):
    """Selects the nearest water meter in the spatial database based on the location of the user's device"""

    nearest_watermeter = Watermeter.objects.raw(
        """SELECT owner, ST_AsText(location) 
        FROM processing_watermeter 
        ORDER BY location <-> ST_SetSRID(ST_Point(%s, %s), 4326) 
        LIMIT 1;""",
        [float(point.x), float(point.y)])

    if len(nearest_watermeter) is 0:
        raise ValueError("There are no watermeters in the vicinity of this location.")
    else:
        return Watermeter.objects.get(owner=nearest_watermeter[0])


def split_pointer_value(pointer_value):
    """Splits the pointer value into three integers for the number pickers of the application"""

    pointer_value_split = list(map(int, str(pointer_value)))
    if len(pointer_value_split) > 3:
        raise ValueError

    if len(pointer_value_split) == 1:
        pointer_value_1 = 0
        pointer_value_2 = 0
        pointer_value_3 = pointer_value_split[0]
    elif len(pointer_value_split) == 2:
        pointer_value_1 = 0
        pointer_value_2 = pointer_value_split[0]
        pointer_value_3 = pointer_value_split[1]
    else:
        pointer_value_1 = pointer_value_split[0]
        pointer_value_2 = pointer_value_split[1]
        pointer_value_3 = pointer_value_split[2]

    return pointer_value_1, pointer_value_2, pointer_value_3


def check_tally_value(tally_value):
    """Checks the reliability score and Avg Euclidean Distance for each prediction to determine which tally
    counter slots should be editable by the user"""

    if len(tally_value) is not 6:
        raise ValueError

    values_checked = []

    # If a prediction is deemed reliable, the digit 0 is added with the prediction. Unreliable predictions get a 1.
    for val in tally_value:
        if val[0] is None:  # If no prediction was made (Not able to extract a digit)
            values_checked.append([0, 1])
        elif val[2] < .95:  # If the reliability score is under .95 a prediction is editable for the user
            values_checked.append([val[0], 1])
        elif val[1] > .3:  # If the Avg. Euclidean Distance is under .3 a prediction is editable for the user
            values_checked.append([val[0], 1])
        else:
            values_checked.append([val[0], 0])

    return values_checked


@csrf_exempt
def update_values(request, image):
    """Stores the reviewed values into the database"""

    default = {"safely_executed": False}  # Default response

    if request.method == "POST":
        json_response = json.loads(request.body)

        pointer_value = int(str(json_response['pointer_value_1'])+
                            str(json_response['pointer_value_2'])+
                            str(json_response['pointer_value_3']))  # combine the three pointer values into one number

        try:
            ImageProcessing(
                name=image,
                watermeter=Watermeter.objects.get(owner=json_response['watermeter']),
                date=datetime.now(),
                pointer_value=pointer_value,
                flag_pointer=bool(json_response['flag_pointervalue']),
                tally_counter_1=int(json_response['tally_counter_1']),
                flag_tally_counter_1=bool(json_response['flag_tally_counter_1']),
                tally_counter_2=int(json_response['tally_counter_2']),
                flag_tally_counter_2=bool(json_response['flag_tally_counter_2']),
                tally_counter_3=int(json_response['tally_counter_3']),
                flag_tally_counter_3=bool(json_response['flag_tally_counter_3']),
                tally_counter_4=int(json_response['tally_counter_4']),
                flag_tally_counter_4=bool(json_response['flag_tally_counter_4']),
                tally_counter_5=int(json_response['tally_counter_5']),
                flag_tally_counter_5=bool(json_response['flag_tally_counter_5']),
                tally_counter_6=int(json_response['tally_counter_6']),
                flag_tally_counter_6=bool(json_response['flag_tally_counter_6']),
            ).save()  # Saves the new results to the database
        except Exception as e:
            default = {"safely_executed": False}
        else:
            default = {"safely_executed": True}
            return JsonResponse(default)
    return JsonResponse(default)


@csrf_exempt
def near(request):
    """Gets the nearby water meters and returns them to the application"""
    if request.method == "GET":
        lat = request.GET.get('lat', '')
        long = request.GET.get('long', '')
        default = {"safely_executed": False}

        try:
            result = get_nearby_watermeters(radius=1000, point=Point(float(long), float(lat)))
            default.update({"safely_executed": True})
            default.update(result)
        except ValueError:
            return JsonResponse(default.update({"safely_executed": False}))
        finally:
            return JsonResponse(default)


def get_nearby_watermeters(radius, point):
    """Gets the 5 nearest watermeters """
    watermeters = Watermeter.objects.filter(location__distance_lt=(point, Distance(m=radius)))
    coords_nearby_watermeters = {}

    if len(watermeters) is 0:
        raise ValueError("There are no watermeters in the vicinity of this location.")
    elif len(watermeters) > 5:
        for index, watermeter in enumerate(Watermeter.objects.raw(
                """SELECT owner, ST_AsText(location) 
                FROM processing_watermeter 
                ORDER BY location <-> ST_SetSRID(ST_Point(%s, %s), 4326) 
                LIMIT 5;""",
                [float(point.x), float(point.y)])):
            field_value = (getattr(watermeter, "location"))
            coords_nearby_watermeters["watermeter_%d" % (index + 1)] = [watermeter.__str__(), str(field_value.x),
                                                                        str(field_value.y)]
        return coords_nearby_watermeters
    else:
        for index in range(5):
            try:
                watermeter = watermeters[index]
                test = Watermeter.objects.get(owner=watermeter)
                field_value = getattr(test, "location")
                coords_nearby_watermeters["watermeter_%d" % (index + 1)] = [watermeter.__str__(), str(field_value.x),
                                                                            str(field_value.y)]
            except IndexError:
                # If there are less than 5 watermeters within a radius of 1000m, None values are assigned.
                coords_nearby_watermeters["watermeter_%d" % (index + 1)] = [None, None,
                                                                            None]
        return coords_nearby_watermeters
