from django.http import JsonResponse
from .models import Drink, Translation
from .serializer import DrinkSerializer, TranslationSerializer
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime

#from drinks import OWLTrans
from drinks import HFTrans
#from OwlTranslation import OwlTranslation

@api_view(['GET', 'POST'])
def drink_list(request):

    if request.method=='GET':
        drinks  = Drink.objects.all()
        serializer = DrinkSerializer(drinks, many=True)
        return JsonResponse({'drinks': serializer.data})

    if request.method=='POST':
        serializer = DrinkSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)

@api_view(['GET', 'PUT', "DELETE"])
def drink_detail(request, id):
    try:
        drink = Drink.objects.get(pk=id)
    except Drink.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method=='GET':
        serializer = DrinkSerializer(drink)
        return Response(serializer.data)
    elif request.method == 'PUT':
        serializer = DrinkSerializer(drink, data=request.data)
        if (serializer.is_valid()):
            serializer.save()
            return Response(serializer.data)
    elif request.method == 'DELETE':
        drink.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['POST'])
def doTranslation(request) :
    print("LOG doTranslation is called with ", request.data)
    #yoonFuncs.printFunc()
    #result = onmtModel.translate(request.data['src'], request.data['srcText'])
    request.data['timeStamp'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    result = ""
    if request.data['src']=='ko' and request.data['tgt']=='en':
        result = HFTrans.translateKoEn([request.data['srcText']])
        #result = OWLTrans.translateKoEn([request.data['srcText']])

    elif request.data['src'] == 'en' and request.data['tgt']=='ko':
        #print("[LOG] if en and ko")
        result = HFTrans.translateEnKo([request.data['srcText']])
        #result = OWLTrans.translateEnKo([request.data['srcText']])

    print("[LOG result]", result)    
    request.data['tgtText'] = result[0]
    print(request.data)

    if request.method=='POST':
        serializer = TranslationSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            print("LOG is not valid!! something error ...")
            return Response(serializer.errors)