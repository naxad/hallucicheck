from django.urls import path
from . import views

app_name = 'verifier'

urlpatterns = [
    path('', views.home, name='home'),
    path('run/', views.run_check, name='run_check'),
    path('dashboard/', views.dashboard, name='dashboard'),
]