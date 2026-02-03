from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('search/', views.search_student, name='search_student'),
    path('api/predict/', views.predict_api, name='predict_api'),
    path('api/analyze_csv/', views.analyze_csv, name='analyze_csv'),
]
# Forced reload
