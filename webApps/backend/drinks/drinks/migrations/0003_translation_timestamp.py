# Generated by Django 3.2.20 on 2023-08-02 22:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('drinks', '0002_translation'),
    ]

    operations = [
        migrations.AddField(
            model_name='translation',
            name='timeStamp',
            field=models.CharField(default='', max_length=15),
        ),
    ]
