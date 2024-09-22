from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User
from django.contrib.auth.admin import UserAdmin as DefaultUserAdmin
from django.contrib.auth.models import User
from django.contrib import admin

# Register our custom User model with the admin site
# This allows us to manage users through Django's built-in admin interface


from django.contrib.auth.models import User
from django.contrib import admin

# Check if the User model is registered
try:
    admin.site.unregister(User)
except admin.sites.NotRegistered:
    pass

# Register the User model with custom UserAdmin
from django.contrib.auth.admin import UserAdmin as DefaultUserAdmin

class UserAdmin(DefaultUserAdmin):
    model = User
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff')

admin.site.register(User, UserAdmin)

# Register any other models here
# The admin site is like a control panel for our app's data