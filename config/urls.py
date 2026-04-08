from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.contrib.auth import views as auth_views
from django.http import HttpResponse

def health(request):
    return HttpResponse("OK", content_type="text/plain")

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("verifier.urls")),
    path("login/", auth_views.LoginView.as_view(template_name="verifier/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("health/", health)
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)