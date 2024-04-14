from keras.applications.resnet50 import ResNet50

def ResNet50Model():
    # Load the model
    resnet50_model = ResNet50(weights='imagenet')
    return resnet50_model
