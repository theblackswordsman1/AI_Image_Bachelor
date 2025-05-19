import tensorflow as tf
import json
from typing import Dict, Any, List, Union

class CNNCodeGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.preprocessing = config.get('preprocessing', {})
        self.model_config = config.get('model_config', {})
        self.training_config = config.get('training_config', {})
        self.session_data = config

        if 'num_classes' not in self.preprocessing:
            if 'dataset_info' in config and 'classes' in config['dataset_info']:
                self.preprocessing['num_classes'] = len(config['dataset_info']['classes'])
            else:
                self.preprocessing['num_classes'] = len(config.get('uploaded_classes', [])) or 2

    def generate_imports(self) -> str:
        return (
            "import tensorflow as tf\n"
            "from tensorflow.keras import layers, models\n"
            "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n"
            "import numpy as np\n"
            "import os\n"
            "import json\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.metrics import confusion_matrix, classification_report\n"
            "np.random.seed(42)\n"
            "tf.random.set_seed(42)\n"
        )
    
    #Dataset prep
    def generate_dataset_preparation(self) -> str:
        image_size = int(self.preprocessing.get('image_size', 224))
        
        base_code = f"""
import os
import shutil
from PIL import Image

data_dir = os.path.join(os.path.dirname(os.path.abspath('')), 'training_data')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

print("Setting up directories...")
for directory in [data_dir, train_dir, val_dir, test_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    print(f"Created directory: {{directory}}")
"""

        dataset_code = ""
        if 'selected_dataset' in self.session_data:
            selected_dataset = self.session_data['selected_dataset']
            
            if selected_dataset == 'mnist':
                dataset_code = f"""
from tensorflow.keras.datasets import mnist
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Loaded MNIST dataset with {{len(x_train)}} training samples")

for class_idx in range(10):
    os.makedirs(os.path.join(train_dir, str(class_idx)), exist_ok=True)
    os.makedirs(os.path.join(val_dir, str(class_idx)), exist_ok=True)
    os.makedirs(os.path.join(test_dir, str(class_idx)), exist_ok=True)
    print(f"Created directories for class {{class_idx}}")

split_idx = int(0.8 * len(x_train))
x_val = x_train[split_idx:]
y_val = y_train[split_idx:]
x_train = x_train[:split_idx]
y_train = y_train[:split_idx]

def save_image_set(images, labels, output_dir, set_name):
    print(f"Processing {{len(images)}} {{set_name}} images...")
    for idx in range(len(images)):
        class_idx = labels[idx]
        img = Image.fromarray(images[idx])
        img = img.convert('RGB')
        img = img.resize(({image_size}, {image_size}))
        save_path = os.path.join(output_dir, str(class_idx), f'{{set_name}}_{{idx}}.png')
        img.save(save_path)
        if idx % 1000 == 0:
            print(f"Saved {{idx}} {{set_name}} images...")

save_image_set(x_train, y_train, train_dir, "train")
save_image_set(x_val, y_val, val_dir, "val")
save_image_set(x_test, y_test, test_dir, "test")
"""
            elif selected_dataset == 'cifar10':
                dataset_code = f"""
from tensorflow.keras.datasets import cifar10
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
print(f"Loaded CIFAR-10 dataset with {{len(x_train)}} training samples")

for class_idx in range(10):
    os.makedirs(os.path.join(train_dir, str(class_idx)), exist_ok=True)
    os.makedirs(os.path.join(val_dir, str(class_idx)), exist_ok=True)
    os.makedirs(os.path.join(test_dir, str(class_idx)), exist_ok=True)
    print(f"Created directories for class {{class_idx}}")

split_idx = int(0.8 * len(x_train))
x_val = x_train[split_idx:]
y_val = y_train[split_idx:]
x_train = x_train[:split_idx]
y_train = y_train[:split_idx]

def save_image_set(images, labels, output_dir, set_name):
    print(f"Processing {{len(images)}} {{set_name}} images...")
    for idx in range(len(images)):
        class_idx = labels[idx]
        img = Image.fromarray(images[idx])
        img = img.resize(({image_size}, {image_size}))
        save_path = os.path.join(output_dir, str(class_idx), f'{{set_name}}_{{idx}}.png')
        img.save(save_path)
        if idx % 1000 == 0:
            print(f"Saved {{idx}} {{set_name}} images...")

save_image_set(x_train, y_train, train_dir, "train")
save_image_set(x_val, y_val, val_dir, "val")
save_image_set(x_test, y_test, test_dir, "test")
"""

        return base_code + dataset_code + "\nprint('Dataset preparation completed')" 

    # Preprocessing
    def generate_preprocessing(self) -> str:
        # Pre-created datasets
        if 'selected_dataset' in self.session_data:
            ds = self.session_data['selected_dataset']
            image_size = int(self.preprocessing.get('image_size', 224))
            batch_size = int(self.training_config.get('batch_size', 32))
            train_split = float(self.preprocessing.get('train_split', 80)) / 100
            code = ""
            # MNIST
            if ds == 'mnist':
                code += (
                    f"image_size = ({image_size}, {image_size})\n"
                    "from tensorflow.keras.datasets import mnist\n"
                    "import tensorflow as tf\n"
                    "import numpy as np\n"
                    "(x_train, y_train), (x_test, y_test) = mnist.load_data()\n"
                    "x_train = x_train.astype('float32') / 255.0\n"
                    "x_test = x_test.astype('float32') / 255.0\n"
                    f"num_classes = {self.preprocessing.get('num_classes', 10)}\n"
                    "y_train_cat = y_train\n"
                    "y_test_cat = y_test\n"
                    f"split_idx = int(len(x_train) * {train_split})\n"
                    "x_val, y_val_cat = x_train[split_idx:], y_train[split_idx:]\n"
                    "x_train, y_train_cat = x_train[:split_idx], y_train[:split_idx]\n"
                    "def process_mnist(image, label):\n"
                    "    image = tf.expand_dims(image, -1)\n"
                    "    image = tf.image.grayscale_to_rgb(image)\n"
                    f"    image = tf.image.resize(image, ({image_size}, {image_size}))\n"
                    "    return image, label\n"
                    "\n"
                    f"train_generator = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat)).map(process_mnist).shuffle(1000).batch({batch_size})\n"
                    f"validation_generator = tf.data.Dataset.from_tensor_slices((x_val, y_val_cat)).map(process_mnist).batch({batch_size})\n"
                    f"test_generator = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).map(process_mnist).batch({batch_size})\n"
                    "has_test_set = True\n"
                    "print('Dataset processing completed')\n"
                )
                return code
            # CIFAR
            elif ds == 'cifar10':
                code += (
                    f"image_size = ({image_size}, {image_size})\n"
                    "from tensorflow.keras.datasets import cifar10\n"
                    "import tensorflow as tf\n"
                    "import numpy as np\n"
                    "(x_train, y_train), (x_test, y_test) = cifar10.load_data()\n"
                    "y_train = y_train.flatten()\n"
                    "y_test = y_test.flatten()\n"
                    "x_train = x_train.astype('float32') / 255.0\n"
                    "x_test = x_test.astype('float32') / 255.0\n"
                    f"num_classes = {self.preprocessing.get('num_classes', 10)}\n"
                    "y_train_cat = y_train\n"
                    "y_test_cat = y_test\n"
                    f"split_idx = int(len(x_train) * {train_split})\n"
                    "x_val, y_val_cat = x_train[split_idx:], y_train[split_idx:]\n"
                    "x_train, y_train_cat = x_train[:split_idx], y_train[:split_idx]\n"
                    "def process_cifar10(image, label):\n"
                    f"    image = tf.image.resize(image, ({image_size}, {image_size}))\n"
                    "    return image, label\n"
                    "\n"
                    f"train_generator = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat)).map(process_cifar10).shuffle(1000).batch({batch_size})\n"
                    f"validation_generator = tf.data.Dataset.from_tensor_slices((x_val, y_val_cat)).map(process_cifar10).batch({batch_size})\n"
                    f"test_generator = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).map(process_cifar10).batch({batch_size})\n"
                    "has_test_set = True\n"
                    "print('Dataset processing completed')\n"
                )
                return code

        # Uploaded datasets
        image_size = int(self.preprocessing.get('image_size', 224))
        batch_size = int(self.training_config.get('batch_size', 32))
        train_split = float(self.preprocessing.get('train_split', 80)) / 100
        aug = self.preprocessing.get('augmentation', [])
        aug_params = []
        if 'horizontal_flip' in aug: aug_params.append('horizontal_flip=True')
        if 'vertical_flip' in aug: aug_params.append('vertical_flip=True')
        if 'rotation' in aug: aug_params.append('rotation_range=20')
        if 'zoom' in aug: aug_params.append('zoom_range=0.2')
        if 'shear' in aug: aug_params.append('shear_range=0.2')
        if 'brightness' in aug: aug_params.append('brightness_range=[0.8, 1.2]')
        if 'width_shift' in aug: aug_params.append('width_shift_range=0.2')
        if 'height_shift' in aug: aug_params.append('height_shift_range=0.2')
        aug_str = ',\n    '.join(aug_params)
        val_split = 1 - train_split

        num_classes = int(self.preprocessing.get('num_classes', 2))

        return (
            f"num_classes = {num_classes}\n"
            "train_datagen = ImageDataGenerator(\n"
            "    rescale=1./255,\n"
            f"    validation_split={val_split:.2f}" +
            (",\n    " + aug_str if aug_str else "") +
            "\n)\n"
            "test_datagen = ImageDataGenerator(rescale=1./255)\n"
            f"image_size = ({image_size}, {image_size})\n"
            f"batch_size = {batch_size}\n"
            "train_generator = train_datagen.flow_from_directory(\n"
            "    train_dir,\n"
            "    target_size=image_size,\n"
            "    batch_size=batch_size,\n"
            "    class_mode='sparse',\n"
            "    subset='training'\n"
            ")\n"
            "validation_generator = train_datagen.flow_from_directory(\n"
            "    val_dir,\n"
            "    target_size=image_size,\n"
            "    batch_size=batch_size,\n"
            "    class_mode='sparse',\n"
            "    subset='validation'\n"
            ")\n"
            "has_test_set = os.path.exists(test_dir)\n"
            "test_generator = None\n"
            "if has_test_set:\n"
            "    test_generator = test_datagen.flow_from_directory(\n"
            "        test_dir,\n"
            "        target_size=image_size,\n"
            "        batch_size=batch_size,\n"
            "        class_mode='sparse'\n"
            "    )\n"
            "num_classes = max(num_classes, len(train_generator.class_indices))\n"
        )

    # Model generation
    def generate_model(self) -> str:
        filters = self.model_config.get('filters', [32])
        kernel_sizes = self.model_config.get('kernel_size', [3])
        dense_units = self.model_config.get('dense_units', [128])
        
        activations = self.model_config.get('activation', [])
        if isinstance(activations, str):
            activations = [activations] 
            
        paddings = self.model_config.get('padding', [])
        if isinstance(paddings, str):
            paddings = [paddings] 
            
        default_activation = 'relu'
        default_padding = 'same'
        
        dropout_rate = float(self.model_config.get('dropout_rate', 0.5))
        pool_size = int(self.model_config.get('pool_size', 2))
        pool_type = self.model_config.get('pool_type', 'max')
        
        if 'selected_dataset' in self.session_data:
            ds = self.session_data['selected_dataset']
            if ds in ['mnist', 'cifar10']:
                num_classes = 10  
            else:
                num_classes = int(self.preprocessing.get('num_classes', 2))
        else:
            num_classes = int(self.preprocessing.get('num_classes', 2))
        
        use_batch_norm = self.model_config.get('batch_norm', False)

        code = [
            "inputs = layers.Input(shape=(*image_size, 3))",
        ]
        
        first_activation = activations[0] if activations else default_activation
        first_padding = paddings[0] if paddings else default_padding
        
        code.append(f"x = layers.Conv2D({filters[0]}, {kernel_sizes[0]}, padding='{first_padding}')(inputs)")
        if use_batch_norm:
            code.append("x = layers.BatchNormalization()(x)")
        code.append(f"x = layers.Activation('{first_activation}')(x)")
        
        if pool_type == 'max':
            code.append(f"x = layers.MaxPooling2D(pool_size=({pool_size}, {pool_size}))(x)")
        else:
            code.append(f"x = layers.AveragePooling2D(pool_size=({pool_size}, {pool_size}))(x)")
            
        for i, (f, k) in enumerate(zip(filters[1:], kernel_sizes[1:]), 1):
            layer_activation = activations[i] if i < len(activations) else first_activation
            layer_padding = paddings[i] if i < len(paddings) else first_padding
            
            code.append(f"x = layers.Conv2D({f}, {k}, padding='{layer_padding}')(x)")
            if use_batch_norm:
                code.append("x = layers.BatchNormalization()(x)")
            code.append(f"x = layers.Activation('{layer_activation}')(x)")
            
            if pool_type == 'max':
                code.append(f"x = layers.MaxPooling2D(pool_size=({pool_size}, {pool_size}))(x)")
            else:
                code.append(f"x = layers.AveragePooling2D(pool_size=({pool_size}, {pool_size}))(x)")
                
        code.append("x = layers.Flatten()(x)")
        
        for units in dense_units:
            code.append(f"x = layers.Dense({units})(x)")
            code.append(f"x = layers.Activation('{first_activation}')(x)") 
            code.append(f"x = layers.Dropout({dropout_rate})(x)")
            
        code.append(f"outputs = layers.Dense({num_classes}, activation='softmax')(x)")
        code.append(f"print(f'Model output shape: {{num_classes}} classes')")
            
        code.append("model = models.Model(inputs=inputs, outputs=outputs)")
        code.append("model.summary()")
        
        return '\n'.join(code)

    def generate_training(self) -> str:
        epochs = int(self.training_config.get('epochs', 10))
        batch_size = int(self.training_config.get('batch_size', 32))
        learning_rate = float(self.training_config.get('learning_rate', 0.001))
        patience = int(self.training_config.get('patience', 5))
        
        optimizer = 'Adam'
        loss_function = 'sparse_categorical_crossentropy'

        return (
            f"patience = {patience}\n"
            f"model.compile(\n"
            f"    optimizer=tf.keras.optimizers.{optimizer}(learning_rate={learning_rate}),\n"
            f"    loss='{loss_function}',\n"
            "    metrics=['accuracy']\n"
            ")\n"
            "callbacks = [\n"
            "    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),\n"
            "    tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),\n"
            "    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(2, patience // 2), min_lr=0.00001)\n"
            "]\n"
            "history = model.fit(\n"
            "    train_generator,\n"
            f"    epochs={epochs},\n"
            "    validation_data=validation_generator,\n"
            "    callbacks=callbacks,\n"
            "    verbose=1\n"
            ")\n"
            "model.save('final_model.h5')\n"
            "with open('training_history.json', 'w') as f:\n"
            "    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)\n"
            "print('Training complete. Model and history saved.')\n"
        )

    def generate_complete_code(self) -> str:
        data_paths = self.session_data.get('data_paths', {})
        train_dir = data_paths.get('train_dir', 'train_dir')
        val_dir = data_paths.get('val_dir', 'val_dir')
        test_dir = data_paths.get('test_dir', 'test_dir')
        
        return (
            self.generate_imports() +
            f"\ntrain_dir = '{train_dir}'\nval_dir = '{val_dir}'\ntest_dir = '{test_dir}'\n\n" +
            self.generate_dataset_preparation() + "\n\n" + 
            self.generate_preprocessing() + "\n\n" +
            self.generate_model() + "\n\n" +
            self.generate_training()
        )