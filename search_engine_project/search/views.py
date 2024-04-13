from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import Document
import handle_query 


def main(request):
  template = loader.get_template('index.html')
  return HttpResponse(template.render())


def result(request):
  template = loader.get_template('search_results.html')
  results = Document.objects.all().values()
  context = {
    'results': results[:30]
  }
  return HttpResponse(template.render(context=context, request=request))


def testing(request):
  template = loader.get_template('test.html')
  query = request.GET.get('q')
  print(query)
  query_tfidf_vector = handle_query.parse_query(query_text='bác sĩ')
  res = handle_query.rocchio(query_tfidf_vector)
  
  results = Document.objects.filter(id__in=res).values()

  context = {
    'results': results
  }
  return HttpResponse(template.render(context=context, request=request))


@csrf_exempt
def search(request):
  # template = loader.get_template('search_results.html')
  template = loader.get_template('test.html')

  if request.method == 'POST':
    searched = request.POST['searched']
    query_tfidf_vector = handle_query.parse_query(query_text=searched)
    res = handle_query.search(query_tfidf_vector)
    results = Document.objects.filter(id__in=res).values()
  else:
    searched = ""
    results = None

  context = {
    'searched': searched,
    'results': results
  }
  return HttpResponse(template.render(context=context, request=request))

@csrf_exempt
def search_with_rocchio(request):
  # template = loader.get_template('search_results.html')
  template = loader.get_template('test.html')

  if request.method == 'POST':
    searched = request.POST['searched']
    query_tfidf_vector = handle_query.parse_query(query_text=searched)
    D_rel = request.POST.getlist('Drel')
    res = handle_query.rocchio(query_tfidf_vector, D_rel=D_rel)
    results = Document.objects.filter(id__in=res).values()
  else:
    searched = ""
    results = None

  context = {
    'searched': searched,
    'results': results
  }
  return HttpResponse(template.render(context=context, request=request))