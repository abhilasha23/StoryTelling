"""
Story generation
"""
import _pickle as pkl
import numpy
import copy
import sys
import skimage.transform

import skipthoughts
import decoder
import embedding

import config
import requests
import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
# if not config.FLAG_CPU_MODE:
#     from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer

from scipy import optimize, stats
from collections import OrderedDict, defaultdict, Counter
from numpy.random import RandomState
from scipy.linalg import norm

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def story(z, image_loc, k=100, bw=50, lyric=False):
    """
    Generate a story for an image at location image_loc
    """
    # Load the image
    rawim, im = load_image(image_loc)

    # Run image through convnet
    feats = compute_features(z['net'], im).flatten()
    feats /= norm(feats)

    # Embed image into joint space
    feats = embedding.encode_images(z['vse'], feats[None,:])

    # Compute the nearest neighbours
    scores = numpy.dot(feats, z['cvec'].T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [z['cap'][a] for a in sorted_args[:k]]

    print ('NEAREST-CAPTIONS: ')
    for s in sentences[:5]:
        print (s)
    print ('')


   ## Commenting below code to output only image captions

    # # Compute skip-thought vectors for sentences
    # svecs = skipthoughts.encode(z['stv'], sentences, verbose=False)
    #
    # # Style shifting
    # shift = svecs.mean(0) - z['bneg'] + z['bpos']
    #
    # # Generate story conditioned on shift
    # passage = decoder.run_sampler(z['dec'], shift, beam_width=bw)
    # print ('OUTPUT: ')
    # if lyric:
    #     for line in passage.split(','):
    #         if line[0] != ' ':
    #             print (line)
    #         else:
    #             print (line[1:])
    # else:
    #     print (passage)

sentences = ['The salad has many different types of vegetables in it .',
'A salad is concocted with broccoli , potatoes and tomatoes .',
'A table filled with greens and lots of vegetables .',
'Pasta salad with tomatoes , broccoli , beans and sausage in a bowl..',
'There is a lot if veggies that are in the tray']

def get_captions(imageurl):
    print('generating captions for image. . . .')
    vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"
    analyze_url = vision_base_url + "analyze"
    subscription_key = "< - -  ENTER YOU KEY - - - >"
    image_path = './../test_image/'+imageurl
    image_data = open(image_path, "rb").read()
    headers = {'Ocp-Apim-Subscription-Key': subscription_key,'Content-Type': 'application/octet-stream'}
    params = {'visualFeatures': 'Categories,Description,Color'}
    response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
    analysis = response.json()
    image_captions = analysis["description"]["tags"]
    image_captions = [' '.join(image_captions)]
    image_captions.append(analysis["description"]["captions"][0]["text"])
    print('CAPTIONS ARE: \n')
    print(image_captions)
    return image_captions

def get_story(z,imageurl, bw=50, lyric=False):

    sentences = get_captions(imageurl)

    svecs = skipthoughts.encode(z['stv'], sentences, verbose=False)

    # Style shifting
    shift = svecs.mean(0) - z['bneg'] + z['bpos']

    # Generate story conditioned on shift
    passage = decoder.run_sampler(z['dec'], shift, beam_width=bw)
    print ('OUTPUT: ')
    if lyric:
        for line in passage.split(','):
            if line[0] != ' ':
                print (line)
            else:
                print (line[1:])
    else:
        print (passage)



def load_all(style):


    """
    Load everything we need for generating
    """
    #print (config.paths['decmodel'])

    # Skip-thoughts
    print ('Loading skip-thoughts...')
    stv = skipthoughts.load_model(config.paths['skmodels'],
                                  config.paths['sktables'])
    print("type of stv" , type(stv))

    # Decoder
    if (style == 'romance'):
        print('Loading romance decoder...')
        dec = decoder.load_model(config.paths['decmodel'],
                                 config.paths['dictionary'])
    elif(style == 'fairytale'):
        print('Loading children decoder...')
        dec = decoder.load_model(config.paths['decmodel1'],
                                 config.paths['dictionary1'])

    print("type of dec", type(dec))
    # Image-sentence embedding
    print ('Loading image-sentence embedding...')
    vse = embedding.load_model(config.paths['vsemodel'])

    # VGG-19
    print ('Loading and initializing ConvNet...')

    # if config.FLAG_CPU_MODE:
    #     sys.path.insert(0, config.paths['pycaffe'])
    #     import caffe
    #     caffe.set_mode_cpu()
    #     net = caffe.Net(config.paths['vgg_proto_caffe'],
    #                     config.paths['vgg_model_caffe'],
    #                     caffe.TEST)
    # else:
    #     net = build_convnet(config.paths['vgg'])

    # Captions
    print ('Loading captions...')
    cap = []
    with open(config.paths['captions'], 'rb') as f:
        for line in f:
            cap.append(line.strip())

    # Caption embeddings
    print ('Embedding captions...')
    cvec = embedding.encode_sentences(vse, cap, verbose=False)

    # Biases
    print ('Loading biases...')
    bneg = numpy.load(config.paths['negbias'])
    bpos = numpy.load(config.paths['posbias'])

    print("type of bneg", type(bneg))
    print("type of bpos", type(bpos))

    # Pack up
    z = {}
    z['stv'] = stv
    z['dec'] = dec
    z['vse'] = vse
    #z['net'] = net
    z['cap'] = cap
    z['cvec'] = cvec
    z['bneg'] = bneg
    z['bpos'] = bpos

    return z

def load_image(file_name):
    """
    Load and preprocess an image
    """
    MEAN_VALUE = numpy.array([103.939, 116.779, 123.68]).reshape((3,1,1))
    image = Image.open(file_name)
    im = numpy.array(image)

    # Resize so smallest dim = 256, preserving aspect ratio
    if len(im.shape) == 2:
        im = im[:, :, numpy.newaxis]
        im = numpy.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = numpy.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = numpy.swapaxes(numpy.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUE
    return rawim, floatX(im[numpy.newaxis])

def compute_features(net, im):
    """
    Compute fc7 features for im
    """
    if config.FLAG_CPU_MODE:
        net.blobs['data'].reshape(* im.shape)
        net.blobs['data'].data[...] = im
        net.forward()
        fc7 = net.blobs['fc7'].data
    else:
        fc7 = numpy.array(lasagne.layers.get_output(net['fc7'], im,
                                                    deterministic=True).eval())
    return fc7

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pkl.dump(obj, output, -1)

# def build_convnet(path_to_vgg):
#     """
#     Construct VGG-19 convnet
#     """
#     net = {}
#     net['input'] = InputLayer((None, 3, 224, 224))
#     net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
#     net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
#     net['pool1'] = PoolLayer(net['conv1_2'], 2)
#     net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
#     net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
#     net['pool2'] = PoolLayer(net['conv2_2'], 2)
#     net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
#     net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
#     net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
#     net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
#     net['pool3'] = PoolLayer(net['conv3_4'], 2)
#     net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
#     net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
#     net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
#     net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
#     net['pool4'] = PoolLayer(net['conv4_4'], 2)
#     net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
#     net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
#     net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
#     net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
#     net['pool5'] = PoolLayer(net['conv5_4'], 2)
#     net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
#     net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
#     net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
#     net['prob'] = NonlinearityLayer(net['fc8'], softmax)
#
#     print ('Loading parameters...')
#     output_layer = net['prob']
#     model = pkl.load(open(path_to_vgg))
#     lasagne.layers.set_all_param_values(output_layer, model['param values'])
#
#     return net


