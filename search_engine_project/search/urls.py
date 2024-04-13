from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('result', views.result, name='result'),
    path('search', views.search, name='search'),
    path('search', views.search_with_rocchio, name='search2'),
    path('testing', views.testing, name='testing'),
]