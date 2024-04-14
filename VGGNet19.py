from keras.applications.vgg19 import VGG19

def VGGNet19Model():
    # Load the model
    vgg19_model = VGG19(weights='imagenet')
    return vgg19_model
    

    
