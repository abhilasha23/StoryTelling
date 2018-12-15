"""
Configuration for the generate module
"""

#-----------------------------------------------------------------------------#
# Flags for running on CPU
#-----------------------------------------------------------------------------#
FLAG_CPU_MODE = True

#-----------------------------------------------------------------------------#
# Paths to models and biases
#-----------------------------------------------------------------------------#
paths = dict()

# Skip-thoughts
paths['skmodels'] = './models/'
paths['sktables'] = './models/'

# Decoder
paths['decmodel'] = './../storyteller/romance.npz'
paths['dictionary'] = './../storyteller/romance_dictionary.pkl'

paths['decmodel1'] = './../storyteller/fairytales.npz'
paths['dictionary1'] = './../storyteller/fairytales.pkl'


# Image-sentence embedding
paths['vsemodel'] = './../storyteller/coco_embedding.npz'

# VGG-19 convnet
paths['vgg'] = './vgg19.pkl'
paths['pycaffe'] = './../caffe/python'
paths['vgg_proto_caffe'] = './VGG_ILSVRC_19_layers_deploy.prototxt'
paths['vgg_model_caffe'] = './VGG_ILSVRC_19_layers.caffemodel'


# COCO training captions
paths['captions'] = './../storyteller/coco_train_caps.txt'

# Biases
paths['negbias'] = './../storyteller/caption_style.npy'
paths['posbias'] = './../storyteller/romance_style.npy'
