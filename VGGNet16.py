from keras.applications.vgg16 import VGG16

def VGGNet16Model():
    # Load the model
    vgg16_model = VGG16(weights='imagenet')
    return vgg16_model
   
    


