from rest_framework import serializers
from .models import Drink
from .models import Translation

class DrinkSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Drink
        fields = ['id', 'name', 'description']
    

class TranslationSerializer(serializers.ModelSerializer) :
    class Meta:
        model = Translation
        fields = ['id', 'src', 'tgt', 'srcText', 'tgtText', 'timeStamp']
    