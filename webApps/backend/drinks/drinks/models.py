from django.db import models

class Drink(models.Model) :
    name = models.CharField(max_length=200)
    description = models.CharField(max_length=500)

    def __str__(self):
        return self.name + ' ' + self.description

class Translation(models.Model) :
    src = models.CharField(max_length=10)
    tgt = models.CharField(max_length=10)
    srcText =  models.CharField(max_length=500)
    tgtText =  models.CharField(max_length=500)
    timeStamp =  models.CharField(max_length=150, default="")

    def __str__(self):
        return self.srcText + ' : ' + self.tgtText