import os,sys
import numpy as np
import torch
# import utils
from torchvision import datasets,transforms
from sklearn.utils import shuffle

import scipy.io as sio
import pdb
import pickle
import random
import matplotlib.pyplot as plt

def cifar100_superclass_python(task_order, group=5, validation=False, val_ratio=0.05, flat=False, one_hot=True, seed = 0 ):
    CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

    sclass = []
    sclass.append(' beaver, dolphin, otter, seal, whale,')                      #aquatic mammals
    sclass.append(' aquarium_fish, flatfish, ray, shark, trout,')               #fish
    sclass.append(' orchid, poppy, rose, sunflower, tulip,')                    #flowers
    sclass.append(' bottle, bowl, can, cup, plate,')                            #food
    sclass.append(' apple, mushroom, orange, pear, sweet_pepper,')              #fruit and vegetables
    sclass.append(' clock, computer keyboard, lamp, telephone, television,')    #household electrical devices
    sclass.append(' bed, chair, couch, table, wardrobe,')                       #household furniture
    sclass.append(' bee, beetle, butterfly, caterpillar, cockroach,')           #insects
    sclass.append(' bear, leopard, lion, tiger, wolf,')                         #large carnivores
    sclass.append(' bridge, castle, house, road, skyscraper,')                  #large man-made outdoor things
    sclass.append(' cloud, forest, mountain, plain, sea,')                      #large natural outdoor scenes
    sclass.append(' camel, cattle, chimpanzee, elephant, kangaroo,')            #large omnivores and herbivores
    sclass.append(' fox, porcupine, possum, raccoon, skunk,')                   #medium-sized mammals
    sclass.append(' crab, lobster, snail, spider, worm,')                       #non-insect invertebrates
    sclass.append(' baby, boy, girl, man, woman,')                              #people
    sclass.append(' crocodile, dinosaur, lizard, snake, turtle,')               #reptiles
    sclass.append(' hamster, mouse, rabbit, shrew, squirrel,')                  #small mammals
    sclass.append(' maple_tree, oak_tree, palm_tree, pine_tree, willow_tree,')  #trees
    sclass.append(' bicycle, bus, motorcycle, pickup_truck, train,')            #vehicles 1
    sclass.append(' lawn_mower, rocket, streetcar, tank, tractor,')             #vehicles 2

    # sclass.append(' apple, aquarium_fish, baby, bear, beaver,')                      #aquatic mammals
    # sclass.append(' bed, bee, beetle, bicycle, bottle,')               #fish
    # sclass.append(' bowl, boy, bridge, bus, butterfly,')                    #flowers
    # sclass.append(' camel, can, castle, caterpillar, cattle,')                            #food
    # sclass.append(' chair, chimpanzee, clock, cloud, cockroach,')              #fruit and vegetables
    # sclass.append(' couch, crab, crocodile, cup, dinosaur,')                       #household furniture
    # sclass.append(' dolphin, elephant, flatfish, forest, fox,')    #household electrical devices
    # sclass.append(' girl, hamster, house, kangaroo, keyboard,')           #insects
    # sclass.append(' lamp, lawn_mower, leopard, lion, lizard,')                         #large carnivores
    # sclass.append(' lobster, man, maple_tree, motorcycle, mountain,')                  #large man-made outdoor things   
    # sclass.append(' mouse, mushroom, oak_tree, orange, orchid,')                      #large natural outdoor scenes    
    # sclass.append(' otter, palm_tree, pear, pickup_truck, pine_tree,')            #large omnivores and herbivores    
    # sclass.append(' plain, plate, poppy, porcupine, possum,')                   #medium-sized mammals    
    # sclass.append(' rabbit, raccoon, ray, road, rocket,')                       #non-insect invertebrates    
    # sclass.append(' rose, sea, seal, shark, shrew,')                              #people    
    # sclass.append(' skunk, skyscraper, snail, snake, spider,')               #reptiles
    # sclass.append(' squirrel, streetcar, sunflower, sweet_pepper, table,')                  #small mammals    
    # sclass.append(' tank, telephone, television, tiger, tractor,')  #trees
    # sclass.append(' train, trout, tulip, turtle, wardrobe,')            #vehicles 1    
    # sclass.append(' whale, willow_tree, wolf, woman, worm,')             #vehicles 2

    # download CIFAR100
    dataset_train = datasets.CIFAR100('./data/',train=True, download=True)
    dataset_test  = datasets.CIFAR100('./data/',train=False,download=True)
    
    if validation == True:
        data_path = './data/cifar-100-python/train'
    else:
        data_path = './data/cifar-100-python/test'
    
    n_classes = 100
    size=[3,32,32]
    data={}
    taskcla =[]
    mean=np.array([x/255 for x in [125.3,123.0,113.9]])
    std=np.array([x/255 for x in [63.0,62.1,66.7]])

    files = open(data_path, 'rb')
    dict = pickle.load(files, encoding='bytes')

    # NOTE Image Standardization
    images = (dict[b'data'])
    images = np.float32(images)/255
    labels = dict[b'fine_labels']
    labels_pair = [[jj for jj in range(100) if ' %s,'%CIFAR100_LABELS_LIST[jj] in sclass[kk]] for kk in range(20)]

    #flat_pair = np.concatenate(labels_pair)

    argsort_sup = [[] for _ in range(20)]
    for _i in range(len(images)):
        for _j in range(20):
            if labels[_i] in labels_pair[_j]:
                argsort_sup[_j].append(_i)

    argsort_sup_c = np.concatenate(argsort_sup)


    train_split = []
    val_split = []
    position = [_k for _k in range(0,len(images)+1,int(len(images)/20))]

    
    if validation == True:
        s_train = 'train'
        s_valid = 'valid'
    else:
        s_train = 'test'


    for idx in task_order:
        data[idx]={}
        data[idx]['name']='cifar100'
        data[idx]['ncla']=5
        data[idx][s_train]={'x': [],'y': []}
        # print('range : [%d,%d]'%(position[idx], position[idx+1]))
        gimages = np.take(images,argsort_sup_c[position[idx]:position[idx+1]], axis=0)

        if not flat:
            gimages = gimages.reshape([gimages.shape[0], 32, 32, 3])

            # gimages = (gimages-mean)/std # mean,std normalization
            gimages = gimages.swapaxes(2,3).swapaxes(1,2)
             #gimages = tf.image.per_image_standardization(gimages)

        glabels = np.take(labels,argsort_sup_c[position[idx]:position[idx+1]])
        for _si, swap in enumerate(labels_pair[idx]):
            glabels = ['%d'%_si if x==swap else x for x in glabels]
        # if idx <2:    
        #     imshow(gimages[0])

        data[idx][s_train]['x']=torch.FloatTensor(gimages)

        data[idx][s_train]['y']=torch.LongTensor(np.array([np.int32(glabels)],dtype=int)).view(-1)
        # print(data[idx][s_train]['x'].max(), data[idx][s_train]['x'].min())


        if validation==True:
            r=np.arange(data[idx][s_train]['x'].size(0))
            r=np.array(shuffle(r,random_state=seed),dtype=int)
            nvalid=int(val_ratio*len(r))
            ivalid=torch.LongTensor(r[:nvalid])
            itrain=torch.LongTensor(r[nvalid:])
            data[idx]['valid']={}
            data[idx]['valid']['x']=data[idx]['train']['x'][ivalid].clone()
            data[idx]['valid']['y']=data[idx]['train']['y'][ivalid].clone()
            data[idx]['train']['x']=data[idx]['train']['x'][itrain].clone()
            data[idx]['train']['y']=data[idx]['train']['y'][itrain].clone()
    # pdb.set_trace()
        # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n
    
    return data, taskcla


def imshow(img):
    # img = img /2 +0.5 # unnormalize
    npimg = img
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()