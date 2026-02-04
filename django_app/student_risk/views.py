from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .services import ModelWrapper

def index(request):
    return render(request, 'index.html')

def dashboard(request):
    # Initial load is empty, data comes from CSV upload
    return render(request, 'dashboard.html', {})

def search_student(request):
    result = None
    error = None
    if request.method == 'GET' and 'student_id' in request.GET:
        student_id_str = request.GET.get('student_id')
        if student_id_str:
            try:
                student_id = int(student_id_str)
                wrapper = ModelWrapper()
                result = wrapper.get_student_risk(student_id)
                if 'error' in result:
                    error = result['error']
                    result = None
            except ValueError:
                error = "Invalid Student ID format"
                
    return render(request, 'student_search.html', {'result': result, 'error': error})

@csrf_exempt
def predict_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            # Instantiate model wrapper (singleton)
            wrapper = ModelWrapper()
            
            # Predict
            result = wrapper.predict(data)
            
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid method'}, status=405)

@csrf_exempt
def analyze_csv(request):
    if request.method == 'POST' and request.FILES.get('file'):
        csv_file = request.FILES['file']
        wrapper = ModelWrapper()
        result = wrapper.process_bulk_predictions(csv_file)
        if 'error' in result:
             return JsonResponse(result, status=400)
        return JsonResponse(result)
    return JsonResponse({'error': 'Invalid request'}, status=400)
