from tensorflow.keras.layers import Conv2D, Flatten, Reshape, GlobalAveragePooling1D

def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN Feature Extractor
    x = Reshape((input_shape[0], input_shape[1], 1))(inputs)  # Reshape for convolution
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)  # Flatten for transformer
    
    # Vision Transformer Layer
    patch_size = 16
    num_patches = input_shape[0] // patch_size
    x = Dense(128, activation='relu')(x)  # Patch embedding
    position_embeddings = Dense(128, activation='relu')(tf.range(num_patches, dtype=tf.float32))
    x = x + position_embeddings

    for _ in range(4):  # 4 transformer layers
        x1 = LayerNormalization()(x)
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)(x1, x1)
        x2 = Dropout(0.1)(attention_output + x)
        x3 = LayerNormalization()(x2)
        x = Dense(128, activation='relu')(x3)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

hybrid_model = build_hybrid_model((X_train.shape[1],))
hybrid_model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Hybrid Model
history_hybrid = hybrid_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks)

# Save Hybrid Model
hybrid_model_path = os.path.join(base_dir, 'Model_4 Using_Hybrid_Approach__CNN__ViT__.h5')
hybrid_model.save(hybrid_model_path)
print(f"Hybrid model saved at: {hybrid_model_path}")

# Evaluate and plot
metrics = hybrid_model.evaluate(X_test, y_test)
print(f"Test Accuracy (Hybrid): {metrics[1] * 100:.2f}%")
