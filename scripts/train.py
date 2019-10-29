"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from PIL import Image
import argparse
import os
import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model

import tensorflow as tf

import keras_retinanet.losses
import keras_retinanet.layers
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.preprocessing.pascal_voc import PascalVocGenerator
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.models.resnet import ResNet18RetinaNet
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.transform import random_transform_generator

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_models(num_classes, weights='imagenet', multi_gpu=0, checkpoint=False, num_channels=3):
    # create "base" model (no NMS)
    image = keras.layers.Input((None, None, num_channels))
    if (weights == 'None') or (weights == None):
        weights = None
    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            model = ResNet18RetinaNet(image, num_classes=num_classes, weights=weights, nms=False)
        training_model = multi_gpu_model(model, gpus=multi_gpu)
    else:
        model = ResNet18RetinaNet(image, num_classes=num_classes, weights=weights, nms=False)
        training_model = model

    # append NMS for prediction only
    classification   = model.outputs[1]
    detections       = model.outputs[2]
    boxes            = keras.layers.Lambda(lambda x: x[:, :, :4])(detections)
    detections       = keras_retinanet.layers.NonMaximumSuppression(name='nms')([boxes, classification, detections]) #max boxes and nms threshold set here
    prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])

    # compile model
    if checkpoint:
        learning_rate = 1e-7
    else:
        learning_rate = 1e-5
    if weights == None:
        learning_rate = 1e-3

    training_model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=learning_rate, clipnorm=0.001, decay=1e-3/25)
    )

    return model, training_model, prediction_model


def create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        dataset_type,
        snapshot_path,
        args):
    callbacks = []

    # save the prediction model
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            snapshot_path,
            'resnet50_{dataset_type}_{{epoch:02d}}.h5'.format(dataset_type=dataset_type)
        ),
        verbose=1
    )
    checkpoint = RedirectModel(checkpoint, prediction_model)
    callbacks.append(checkpoint)

    if args.log_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                 = args.log_dir,
            histogram_freq          = 0,
            batch_size              = args.batch_size,
            write_graph             = True,
            write_grads             = False,
            write_images            = False,
            embeddings_freq         = 0,
            embeddings_layer_names  = None,
            embeddings_metadata     = None
        )
        callbacks.append(tensorboard_callback)

    if dataset_type == 'coco':
        import keras_retinanet.callbacks.coco

        # use prediction model for evaluation
        evaluation = keras_retinanet.callbacks.coco.CocoEval(validation_generator)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=10,
        verbose=1,
        mode='auto',
        epsilon=0.0001,
        cooldown=0,
        min_lr=0)
    callbacks.append(lr_scheduler)

    return callbacks


def create_generators(args):
    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        rotation_range=25
    )
    val_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        rotation_range=25
    )

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from keras_retinanet.preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            train_image_data_generator,
            batch_size=args.batch_size
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            val_image_data_generator,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'trainval',
            train_image_data_generator,
            batch_size=args.batch_size
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            val_image_data_generator,
            batch_size=args.batch_size
        )
    elif args.dataset_type == 'csv':
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            args.mean_image,
            train_image_data_generator,
            batch_size=args.batch_size,
            image_min_side=int(args.image_min_side),
            image_max_side=int(args.image_max_side),
            num_channels=args.num_channels
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                args.mean_image,
                val_image_data_generator,
                batch_size=args.batch_size,
                image_min_side=int(args.image_min_side),
                image_max_side=int(args.image_max_side),
                num_channels=args.num_channels
            )
        else:
            validation_generator = None
    else:
        raise ValueError(
            'Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    :param parsed_args: parser.parse_args()
    :return: parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size, parsed_args.multi_gpu))

    return parsed_args


def parse_args():
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--mean_image', default=None, help='Path to mean image of data set to subtract (optional).')
    csv_parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')
    csv_parser.add_argument('--image_min_side', default=1080, help='Length of minimum image side. Image will be scaled to this')
    csv_parser.add_argument('--image_max_side', default=1920, help='Length of maximum image side. Image will be scaled to this')

    parser.add_argument('--weights', help='Weights to use for initialization (defaults to ImageNet).', default='imagenet')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu', help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--snapshot-path', help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--log-dir', default=None, help='path to store tensorboard logs')
    parser.add_argument('--num_processors', type=int, default=8, help='Number of image preprocessing objects')
    parser.add_argument('--resume', action='store_true', help='Adjust learning parameters for resume or transfer learning')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of channels in input images')

    return check_args(parser.parse_args())

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(args)

    # create the model
    print('Creating model, this may take a second...')
    model, training_model, prediction_model = create_models(
            num_classes=train_generator.num_classes(), 
            weights=args.weights, 
            multi_gpu=args.multi_gpu, 
            checkpoint=args.resume,
            num_channels=args.num_channels)

    # print model summary
    #print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(model, training_model, prediction_model, validation_generator, args.dataset_type, args.snapshot_path, args)

    # start training
    training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=1000,
        epochs=100,
        verbose=1,
        callbacks=callbacks,
        use_multiprocessing=False,
        workers=args.num_processors,
        max_queue_size = 20,
        validation_data=validation_generator
    )

