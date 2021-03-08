from django.urls import path

from reid import views

urlpatterns = [
	path("login/", views.LoginView.as_view()),
	path("register/", views.RegisterView.as_view()),
	path("home/", views.HomeView.as_view()),
	path("session/", views.get_session_data),
	path("upload/", views.Dataupload.as_view()),
	path("preview/", views.DataPreview.as_view()),
	path("getajax/", views.get_view),
	path("fastreid/", views.FastReID.as_view()),
	path("usercenter/",views.UserCenter.as_view()),
	path("usercenter/changepwd/",views.ChangePasswd.as_view()),
	path("usercenter/managedata/",views.ManageData.as_view()),
	path("usercenter/performance/", views.Performance.as_view())
]