import models
import train_model
import tensorflow as tf
import images_to_dataset
if __name__=='__main__':
    #get the generator and discriminator models
    generator=models.generator()
    discriminator=models.discriminator()

    #create object for GAN 
    gan=train_model.GAN(generator,discriminator)

    #define the optimiser and loss function
    #here iam using single optimiser for both models
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    #compile the model
    gan.compile(g_optimizer=optimizer, d_optimizer=optimizer, loss_fn=loss_fn)

    #get the dataset 
    dataset=images_to_dataset.images_to_dataset('path to image directory')
    #train the model
    gan.fit(dataset)



