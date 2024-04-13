# search/models.py
from django.db import models

class Document(models.Model):
    topic = models.CharField(max_length=200)
    title = models.CharField(max_length=200)
    content = models.TextField()
    url = models.URLField()
    
    # def __str__(self):
    #     return f'{self.title}, {self.content}, {self.url}'
    
