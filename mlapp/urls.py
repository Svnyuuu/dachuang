from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.model_training, name='model_training'),
]
