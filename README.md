# Generator-Classifier Networks

### Training Specifications

1. BATCH_SIZE
2. NUMBER_OF_EPOCHS
3. TRAIN_RATIO
4. RATIO_OF_MASKS
5. NUMBER_OF_WORKERS (for data loader)

### GenClassifier Configuration

1. MODEL_CODE (genclassifier)
2. NUMBER_OF_MASK_SAMPLES
3. PATH_TO_GENERATOR
4. PATH_TO_CLASSIFIER

### Generator Configuration

1. MODEL_CODE (generator)
2. IMAGE_CHANNELS IMAGE_HEIGHT IMAGE_WIDTH
3. ARCHITECTURE (one line for each module)

### Classifier Configuration

1. MODEL_CODE (classifier)
2. IMAGE_CHANNELS IMAGE_HEIGHT IMAGE_WIDTH
3. USE_MASKS CROP_SIZE
4. ARCHITECTURE (one line for each module)
