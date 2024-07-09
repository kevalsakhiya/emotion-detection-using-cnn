from src.data.clean_dataset import remove_invalid_images
from src.models.train_model import create_data_generators, create_resnet50v2_model, compute_class_weights, create_callbacks
import tensorflow as tf

# Constants
TRAIN_DIR = 'data/raw/train'
TEST_DIR = 'data/raw/test'
BATCH_SIZE = 64
INPUT_SHAPE = (224, 224, 3)
CNN_PATH = 'models'
MODEL_NAME = 'ResNet50_Transfer_Learning.keras'
DATA_DIR = 'data/raw'

def main():
    print('*** Image Processing and Model Training ***')

    # Step 1: Remove invalid image types
    print('Removing invalid image types...')
    # remove_invalid_images(DATA_DIR)

    # Step 2: Create data generators for training and testing data
    print('Creating data generators for training and testing data...')
    train_generator, test_generator = create_data_generators(TRAIN_DIR, TEST_DIR, BATCH_SIZE)

    # Step 3: Compute class weights to handle imbalanced data
    print('Computing class weights to handle imbalanced data...')
    class_weights_dict = compute_class_weights(train_generator)

    # Step 4: Create the ResNet50V2 model
    print('Creating the ResNet50V2 model...')
    model = create_resnet50v2_model(INPUT_SHAPE)
    model.summary()  # Print the model summary

    # Step 5: Compile the model
    print('Compiling the model...')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Step 6: Create callbacks for training
    print('Creating callbacks for training...')
    callbacks = create_callbacks(CNN_PATH, MODEL_NAME)

    # Step 7: Calculate steps per epoch for training and testing
    train_steps_per_epoch = train_generator.samples // train_generator.batch_size + 1
    test_steps_epoch = test_generator.samples // test_generator.batch_size + 1

    # Step 8: Train the model
    print('Starting model training...')
    with tf.device('/GPU:0'):
        model.fit(
            train_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=30,
            validation_data=test_generator,
            validation_steps=test_steps_epoch,
            class_weight=class_weights_dict,
            callbacks=callbacks
        )
    
    print('Model training completed.')

    # Step 9: Save the trained model
    print('Saving the model...')
    model.save("models/final_ResNet50.keras")
    print('Model saved as final_ResNet50.keras.')

if __name__ == '__main__':
    main()
