from django.shortcuts import render
from .forms import ImageForm

from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np



# Create your views here.


model_path = 'media/vgg19.h5'
# load the model:
model = load_model(model_path)
# model._make_predict_function

# fxn for img preprocessing:
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))

    # preprocessing the img:
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds



def index(request):
    """ Process the images uploaded by user"""
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()

            # get the current instance object to display in template
            img_obj = form.instance

            pred = model_predict(img_obj.image.path, model)
            pred_class = decode_predictions(pred, top=1)
            result = str(pred_class[0][0][1])


            return render(request, 'ImgClassifier/homePg.html',context={
                'form':form,
                'img_obj':img_obj,
                'prediction_label': result
            })

    else: # on GET Request
        form = ImageForm()
    return render(request, 'ImgClassifier/homePg.html',context={
            'form':form,
        })

