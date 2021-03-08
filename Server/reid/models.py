from django.db import models

# Create your models here.
class User(models.Model):

	name = models.CharField(max_length = 20, unique = True)
	password = models.CharField(max_length = 12)

	def __str__(self):
		return  'user: {}'.format(self.name)