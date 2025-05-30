# Built-in libraries
import os
import sys
import time
import random
import csv
import gc

# Scientific computing and data handling
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from collections import Counter, defaultdict

# Scikit-learn
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight, compute_class_weight
from sklearn.preprocessing import StandardScaler

# TensorFlow and Keras
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import L2
from keras.layers import (
    Layer, Dense, Activation, Concatenate, Conv1D, Conv2DTranspose, Embedding,
    GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalMaxPooling2D, 
    LayerNormalization, LeakyReLU, MultiHeadAttention, Multiply, Reshape, 
    TimeDistributed, UpSampling2D, ZeroPadding2D, SeparableConv2D, 
    DepthwiseConv2D, SpatialDropout2D
)

# TensorFlow Keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Lambda, Add, Activation, Permute, 
    Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D, DepthwiseConv2D,
    BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D, Multiply, 
    Reshape, concatenate, dot, Softmax, LayerNormalization
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
)

# Add-ons and external libraries
import tensorflow_addons as tfa
import keras_nlp
from keras_nlp.layers import TransformerEncoder
from keras_cv_attention_models import swin_transformer_v2

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')

gc.collect()
tf.keras.backend.clear_session()


tf.config.optimizer.set_experimental_options({"layout_optimizer": False})# because i had issue with the spatial dropout

tf.random.set_seed(42)    # TensorFlow operations
np.random.seed(42)        # NumPy operations
random.seed(42) 
# Augmentation functions






class TrainableAlphaConcat(Layer):
    def __init__(self, num_blocks=6, initial_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.alphas = self.add_weight(
            name="alpha_weights",
            shape=(1, 1, 1, num_blocks),  # Shape compatible for broadcasting
            initializer=tf.keras.initializers.Constant(initial_value),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0) 
        )

    def call(self, inputs):
        # Stack inputs along the last axis to create a single tensor
        inputs_tensor = tf.stack(inputs, axis=-1)  # Shape: (batch, 32, 1000, num_blocks, 8)
        print(f" alpfa 6 inputs_tensor shape: {inputs_tensor.shape}")
        
        # Multiply each block with its corresponding alpha
        weighted_blocks = inputs_tensor * self.alphas  # Broadcasts correctly
        
        # **Concatenate along feature dimension**
        return tf.concat(tf.unstack(weighted_blocks, axis=-1), axis=-1)  # (batch, 32, 1000, num_blocks * 8)


class TrainableAlpha(Layer):
    def __init__(self, initial_value1=0.8,initial_value2=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = self.add_weight(
            name="alpha_weight",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_value1),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0) 
        )
        self.beta = self.add_weight(
            name="beta_weight",
            shape=(),
            initializer=tf.keras.initializers.Constant(initial_value2),
            trainable=True,
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0) 
        )

    def call(self, inputs):
        temporal_features, transformer_output = inputs
        return concatenate([
            temporal_features * self.alpha,
            transformer_output * (self.beta)
        ], axis=3)






def best_sofardil(nb_classes=2, Chans=32, Samples=1000,
                            dropoutRate=0.3, F1=8, D=2, F2=16, 
                            norm_rate=0.25, dropoutType='Dropout'):
            if dropoutType == 'SpatialDropout2D':
                dropoutType = SpatialDropout2D
            elif dropoutType == 'Dropout':
                dropoutType = Dropout
            else:
                raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout')
            #lana rhoades
            # Input layer
            input1 = Input(shape=(Chans, Samples, 1), name='input1')
            
            # Path 1: Multi-scale Temporal Convolutions
            kernel_sizes = [5, 9, 13, 29, 61, 125]#[3, 5, 7, 15, 31, 63]#[15,25,31, 63, 125, 250]
            temporal_blocks = []
            N=1
            
            for i, k_size in enumerate(kernel_sizes):
                temp_conv = Conv2D(F1, (N, k_size), padding='same',
                                use_bias=False, name=f'temporal_conv_{i}')(input1)
                temp_conv = BatchNormalization(name=f'temporal_bn_{i}')(temp_conv)
                temp_conv = Activation('elu', name=f'temporal_act_{i}')(temp_conv)
                temp_conv=SpatialDropout2D(0.50)(temp_conv) #0.582 with this spatial dropout of 0.2
                temporal_blocks.append(temp_conv)
                
            temporal_features = TrainableAlphaConcat(num_blocks=len(temporal_blocks))(temporal_blocks)
           
            
            # Simplified transformer with residual
            transformer_input = Permute((2, 1, 3))(input1)
            shape = transformer_input.shape
            transformer_input = Reshape((shape[1], shape[2]))(transformer_input)
            original_input = transformer_input
            
            transformer_output = keras_nlp.layers.TransformerEncoder(
                intermediate_dim=32,  
                num_heads=2,  
                dropout=0.2 ,
                activation='gelu' 
            )(transformer_input)
            
            # Add residual connection
            """
            Input → Transform → Output
                ↘_________________↗
                Direct connection (residual)
            """
            transformer_output = Add()([transformer_output, original_input])#-new
        
            transformer_output = Reshape((Chans, Samples, 1))(transformer_output)

            # Balance features with scaling
                    
            merged = TrainableAlpha()( [temporal_features, transformer_output] )
            
            merged = LayerNormalization()(merged)
        
            # Spatial Processing
            spatial = DepthwiseConv2D((Chans, 1), use_bias=False,
                                    depth_multiplier=D,
                                    depthwise_constraint=max_norm(1.),
                                    name='spatial_conv')(merged)
            spatial = BatchNormalization(name='spatial_bn')(spatial)
            
            spatial = Activation('elu', name='spatial_act')(spatial)
        
            spatial = AveragePooling2D((1, 4), name='spatial_pool')(spatial)
            spatial = dropoutType(dropoutRate, name='spatial_drop')(spatial)

            
            # Separable Convolutions
            separator = SeparableConv2D(F2, (1, 16), use_bias=False,
                                    padding='same', name='separator')(spatial)
            separator = BatchNormalization(name='separator_bn')(separator)
            separator = Activation('elu', name='separator_act')(separator)
            separator = AveragePooling2D((1, 8), name='separator_pool')(separator)
            separator = dropoutType(dropoutRate, name='separator_drop')(separator)
    
            # Classification Head
            flatten = Flatten(name='flatten')(separator)
            dense = Dense(nb_classes, name='dense1',
                        kernel_constraint=max_norm(norm_rate))(flatten)
            softmax = Activation('softmax', name='softmax')(dense)
            
            return Model(inputs=input1, outputs=softmax)

def create_lr_scheduler():
    return ReduceLROnPlateau(
        monitor='loss',  # Monitor training loss. It can be updated to validation loss when using validation data
        factor=0.5,      # Reduce learning rate by a factor of 0.5
        patience=10,      # Number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-6      # Minimum learning rate
    )

def create_lr_scheduler_ft():
    # For fine-tuning phases
    return ReduceLROnPlateau(
        monitor='val_loss',  # Monitor validation loss
        mode='min',          # wasnt existing before
        factor=0.5,#was 0.5
        patience=5,#was5
        verbose=1,
        min_lr=1e-4 #was 1e-4
    )


def create_early_stopping():
    return EarlyStopping(
        monitor='val_accuracy',  # Monitors validation loss
        mode='max',          # Mode: minimize the loss
        patience=200,  #was 15       # Stops training after 5 epochs with no improvement
        restore_best_weights=True,  # Restores weights from the best epoch
        min_delta=0.0001,
        verbose=1
    )


class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, **kwargs):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.kwargs = kwargs
        self.best_val_accuracy = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_loss')
        if current_val_accuracy < self.best_val_accuracy:
            # Delete old file if exists
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
            # Save new weights
            self.model.save_weights(self.filepath)
            self.best_val_accuracy = current_val_accuracy

def create_best_model_checkpoint():
    checkpoint_path = 'checkpoints/downstream_model/best_model_weights_downstream.h5'
    
    # Ensure directory exists
    os.makedirs('checkpoints/downstream_model', exist_ok=True)
    
    return CustomModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )



def create_best_model_checkpoint_first():

    return ModelCheckpoint( #best so far with accuracy
        'best_model_weights.h5',  # Path to save the weights
        monitor='val_accuracy',  # Metric to monitor
        mode='max',               # Maximize the validation accuracy
        save_best_only=True,      # Save only the best model weights
        restore_best_weights=True,  # Automatically restore best weights after training
        verbose=1
    )
def create_check_point2():

    
    
    
        return ModelCheckpoint(
        'best_model_weights.h5', 
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        restore_best_weights=True,
        
        verbose=1
    )




def print_class_distribution(y, dataset_name="Dataset"):
    """
    Prints the distribution of classes in the given labels.

    Args:
        y (numpy.ndarray or tf.Tensor): One-hot encoded labels.
        dataset_name (str): Name of the dataset (e.g., "Training", "Validation").
    """
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    labels = np.argmax(y, axis=1)  # Convert one-hot to single labels
    counter = Counter(labels)
    print(f"{dataset_name} Class Distribution:")
    for cls, count in counter.items():
        print(f"  Class {cls}: {count} samples")
    print()


def cross_session_split(subject_sessions):
    # For simplicity, assume the first session is training, remaining sessions are testing
    train_sessions = []
    test_sessions = []
    for sessions in subject_sessions:
        if len(sessions) > 0:
            train_sessions.append(sessions[0])    # Use first session for training
            test_sessions.append(sessions[1:])    # Remaining sessions for testing
        else:
            train_sessions.append(None)
            test_sessions.append([])
    return train_sessions, test_sessions






def get_next_versioned_filename(base_name, extension):
    version = 1
    while True:
        filename = f"{base_name}_v{version}.{extension}"
        if not os.path.exists(filename):
            return filename
        version += 1


def epoch_data(eeg_data, window_size, stride):
    num_trials, Chans, Samples = eeg_data.shape
    num_windows = (Samples - window_size) // stride + 1
    windows = np.zeros((num_trials, num_windows, Chans, window_size, 1))
    for i in range(num_trials):
        for j in range(num_windows):
            start = j * stride
            end = start + window_size
            windows[i, j, :, :, 0] = eeg_data[i, :, start:end]
    return windows


if __name__ == "__main__":
    


    
    tf.random.set_seed(42)    # TensorFlow operations
    np.random.seed(42)        # NumPy operations
    random.seed(42) 

    
    # Load your subject files and data
    subject_files = [
        # Session 1
        *[f"/gpfs/home5/pandrikopoulos/.local/sub-{i:03d}_ses-01_task_motorimagery_eeg.mat" for i in range(1, 26)],
        # Session 2
        *[f"/gpfs/home5/pandrikopoulos/.local/sub-{i:03d}_ses-02_task_motorimagery_eeg.mat" for i in range(1, 26)],
        # Session 3
        *[f"/gpfs/home5/pandrikopoulos/.local/sub-{i:03d}_ses-03_task_motorimagery_eeg.mat" for i in range(1, 26)],
        # Session 4
        *[f"/gpfs/home5/pandrikopoulos/.local/sub-{i:03d}_ses-04_task_motorimagery_eeg.mat" for i in range(1, 26)],
        # Session 5
        *[f"/gpfs/home5/pandrikopoulos/.local/sub-{i:03d}_ses-05_task_motorimagery_eeg.mat" for i in range(1, 26)],
    ]

    print("Number of subject files:", len(subject_files))
    all_eeg = []
    all_labels = []
    max_trials = 100

    # Load data
    for file in subject_files:
        mat_data = scipy.io.loadmat(file)
        eeg_data = mat_data["data"]
        print("EEG Data Shape:", eeg_data.shape)
        labels = mat_data["labels"]
        if eeg_data.shape[0] < max_trials:
            pad_width = max_trials - eeg_data.shape[0]
            eeg_data =np.pad(eeg_data, ((0, pad_width), (0, 0), (0, 0)), mode='constant') #np.pad(eeg_data, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
            labels = np.pad(labels, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1)
         # Standardize the EEG data
        #eeg_data = standardize_eeg_data(eeg_data)
        all_eeg.append(eeg_data)
        all_labels.append(labels)

    all_eeg = np.array(all_eeg)
    print("EEG Data Shape:", all_eeg.shape)
    all_labels = np.array(all_labels)

    # Prepare session grouping by subject
    # There are 25 subjects, repeated across session files
    subject_sessions = [[] for _ in range(25)]
    for i in range(len(all_eeg)):
        subject_idx = i % 25
        subject_sessions[subject_idx].append(i)

    # Cross-session split (first session = train, rest = test)
    print("Subject Sessions:", subject_sessions)
    train_list, test_list = cross_session_split(subject_sessions)


   

    session_accuracies_dict = defaultdict(list)
    class_accuracies_dict = defaultdict(lambda: defaultdict(list))
    subject_accuracies_dict = defaultdict(list)  # Store accuracies for each subject
    confusion_matrices_dict = defaultdict(lambda: defaultdict(list))

    for subject_idx in range(25):
        
       
        if train_list[subject_idx] is None:
            continue

        train_eeg = all_eeg[train_list[subject_idx]]
        print("train_eeg type",type(train_eeg))
        print("shape of labels before mask: ",all_labels[train_list[subject_idx]].shape)
        
        train_labels = all_labels[train_list[subject_idx]].squeeze()
        print("shape of train_eeg before mask: ",train_labels.shape)
        #train_mask1
        train_mask = train_labels != -1
        #print("shape of train_eeg before mask: ",train_eeg.shape)
        train_eeg = train_eeg[train_mask]
        print("train_eeg shape",train_eeg.shape)
        
        train_labels = train_labels[train_mask]

        
        train_eeg_new, val_eeg, train_labels_new, val_labels = train_test_split(
            train_eeg,
            train_labels,
            test_size=0.2,
            random_state=42,
            shuffle=False
        )
        vall_eeg_phase2=val_eeg

    
        print("train_eeg shape after split",train_eeg_new.shape)
        print("val_eeg shape after split",val_eeg.shape)

        
       
        
        train_labels_new = (train_labels_new - 1).clip(0)
        train_labels_new = tf.keras.utils.to_categorical(train_labels_new, num_classes=2)
        train_labels = (train_labels - 1).clip(0)
        train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)
        val_labels = (val_labels - 1).clip(0)
        val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=2)
        
        print("Training data shape:", train_eeg_new.shape)
        #create Adjacency matrix Version 1 54.8 with k=2
        train_eeg_forADJ=train_eeg_new
        Adj = np.corrcoef(train_eeg_forADJ.reshape(-1, 32).T, ddof=1)
        Adj = tf.convert_to_tensor(Adj, dtype=tf.float32)
        
        model = best_sofardil()

        
        best_model_callback = create_best_model_checkpoint_first()

        print("Train EEG shape befor epoching:", train_eeg_new.shape)
       
        
        optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-6,)#weight_decay was 1e-4

        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        count = 0
        for i, sample in enumerate(train_eeg_new):
            if sample.shape[1] != 1000:
                print(f"Sample {i} has a time dimension of {sample.shape[1]}")
                count += 1

        print(f"Total samples with incorrect time dimension: {count}")
        

        y_integers = np.argmax(train_labels_new, axis=1)
       


        model.fit(train_eeg_new, train_labels_new, epochs=100, batch_size=16,validation_data=(val_eeg,val_labels),callbacks=[create_lr_scheduler_ft(),best_model_callback] ,verbose=1)
        
        model.load_weights('best_model_weights.h5')
        
        
        
        
        latency_list = []  # Store individual latencies
        
        for i, s_id in enumerate(test_list[subject_idx], start=1):
            eeg_s = all_eeg[s_id]
            labels_s = all_labels[s_id].squeeze()
            mask = labels_s != -1
            eeg_s = eeg_s[mask]
            labels_s = labels_s[mask]
            
            
            #scalers=refited_scalers
            labels_s = (labels_s - 1).clip(0)
            labels_s = tf.keras.utils.to_categorical(labels_s, num_classes=2)


            start_time = time.time()
            preds = model.predict(eeg_s, batch_size=16, verbose=0)
            end_time = time.time()

            # Compute and store latency
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latency_list.append(latency)
         
            predicted_labels = tf.argmax(preds, axis=1).numpy()
            true_labels = tf.argmax(labels_s, axis=1).numpy()
            # print(f"true label: {true_labels}")
            acc = np.mean(predicted_labels == true_labels)

            # Compute the confusion matrix for the current session
            cm = confusion_matrix(true_labels, predicted_labels)

            # Store the confusion matrix in the dictionary (session x subject)
            confusion_matrices_dict[subject_idx][i] = cm


            # acc = np.mean(np.argmax(preds, axis=1) == np.argmax(labels_s, axis=1))
            print(f"Subject {subject_idx+1:02d}, Session {i:02d} Accuracy: {acc:.3f}")

            session_accuracies_dict[i].append(acc)
            subject_accuracies_dict[subject_idx].append(acc)
        avg_acc_subject = np.mean(subject_accuracies_dict[subject_idx])
        print(f"Subject {subject_idx+1:02d} - Average Accuracy Over All Sessions: {avg_acc_subject:.3f}")
        overall_mean_acc = np.mean([np.mean(acc_list) for acc_list in subject_accuracies_dict.values()])
        print(f"\nOverall Mean Accuracy SO far: {overall_mean_acc:.3f}")
       
        # Compute and print the mean latency
        mean_latency = np.mean(latency_list)
        print(f"\nMean Inference Latency: {mean_latency:.2f} ms")
            

    # Print mean accuracy per session across all subjects
    accurac=0
    counter=0
    for session_idx, acc_list in session_accuracies_dict.items():
        session_mean_acc = np.mean(acc_list) #if acc_list else 0
        accurac+=session_mean_acc
        counter+=1
        print("final counter",counter)
        print(f"Session {session_idx:02d} Mean Accuracy Across Subjects: {session_mean_acc:.3f}")
    print(f"Mean Accuracy Across all Sessions: {accurac/counter:.3f}")

    # Convert dict to DataFrame
    df = pd.DataFrame.from_dict(subject_accuracies_dict, orient='index')

    # Rename columns to session names
    df.columns = ['Session 2', 'Session 3', 'Session 4', 'Session 5']

    # Rename index to Subject number (1-based indexing)
    df.index = [f'Subject {i+1}' for i in df.index]

    # Save to CSV
    df.to_csv("subject_accuracies_MyModel_dil.csv")

    # After collecting confusion matrices, store them into a CSV file

    # Prepare a list to store rows for the CSV
    rows = []

    # Iterate through the confusion_matrices_dict and add data to rows
    for subject_idx, sessions in confusion_matrices_dict.items():
        for session_idx, cm in sessions.items():
            # Flatten the confusion matrix to make it easy to store in a CSV
            cm_flat = cm.flatten()  # Flatten the matrix (e.g., 2x2 -> 4 elements)
            rows.append([subject_idx + 1, session_idx, *cm_flat])

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(rows, columns=['Subject', 'Session', 'TP', 'FN', 'FP', 'TN'])
    df.to_csv('aggregated_confusion_matrices_dil.csv', index=False)

    print("Confusion matrices saved to 'aggregated_confusion_matrices.csv'")

    # After the loop over subjects and sessions
    subject_mean_accuracies = []

    for subject_idx in range(25):
        accs = subject_accuracies_dict[subject_idx]
        if len(accs) > 0:
            mean_acc = np.mean(accs)
            subject_mean_accuracies.append(mean_acc)

    subject_mean_accuracies = np.array(subject_mean_accuracies)
    std_across_subjects = np.std(subject_mean_accuracies)

    print("Mean accuracy for each subject:", subject_mean_accuracies)
    print("Standard deviation across subjects:", std_across_subjects) 