from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^hillary_trump/', include('hillary_trump.urls')),
    url(r'^admin/', admin.site.urls),
]