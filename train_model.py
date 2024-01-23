import time
import sys
import pickle
import os
import tensorflow as tf
class GAN():
    def __init__(self, generator, discriminator,old_images_path=None):
        """
        imputs: 
        generator is the generator model
        discriminator is the discriminator model
        old_image_path is the path for the pickle file that contains the images of old epoches  
        """
        self.generator = generator
        self.discriminator = discriminator
        self.old_images_path=old_images_path
        #images generated at each epoch
        self.images=pickle.load(open(self.old_images_path,'rb')) if (self.old_images_path) else []
        self.g_optimizer = None
        self.d_optimizer = None
        self.loss_fn = None
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn


    #function to update the weights of generator and discriminator
    def train_step(self, real_images):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
              generated_images = self.generator(tf.random.normal([32, 100]), training=True)

              real_output = self.discriminator(real_images, training=True)
              fake_output = self.discriminator(generated_images, training=True)

              gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
              disc_loss = self.loss_fn(tf.ones_like(real_output), real_output) + self.loss_fn(tf.zeros_like(fake_output), fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss,disc_loss
    
    #function to display the loss and accuracy of each batch 
    def display(self,batches,n,ti,d):
        
        if n%int(batches//20)==0:
            t=f"{n+1}/{batches}[{'='*(self.c+1)}{' '*(20-(self.c+1))}] {ti} {d}"
            self.c+=1
        else:
            t=f"{n+1}/{batches}[{'='*(self.c+1)}{' '*(20-(self.c+1))}] {ti} {d}"
        
        sys.stdout.write('\r' + t)
        sys.stdout.flush()

    def fit(self, train_dataset, epochs=1, batch_size=32):
        #raising exception if the 
        if not (self.g_optimizer) or not(self.d_optimizer) or not(self.loss_fn):
            raise Exception("first compile the model by passing optimizers and loss function")
        s=train_dataset.shape[0]
        batches=int(s/batch_size)

        for epoch in range(epochs):
            
            it=time.time()
            print(f'Epoch:{epoch+1}/{epochs}')
            self.c=0            
            
            for i in range(batches):
                real_images=train_dataset[i*batch_size:i*batch_size+batch_size]
                gl,dl=self.train_step(real_images)
                
                d=f"Gen Loss={gl:.4f} Dis Loss={dl:.4f}"
                self.display(batches,i,int(time.time()-it),d)
                 

            
            
            self.images.append(((self.generator(tf.random.normal([1, 100]))[0]*127.5)+127.5).numpy().astype('uint8')[:,:,::-1])
            print()

    #saving the models and images of epoches
    def save(self, gen_path=None,disc_path=None,images_path=None):
        if images_path:
            pickle.dump(self.images,open(images_path,'wb')) 
        else:
            pickle.dump(self.images,open('images.pkl','wb'))
        if gen_path:
             self.generator.save(gen_path)
        else:
            self.generator.save('new_generator.h5')
        if disc_path:
            self.discriminator.save(disc_path)
        else:
            self.discriminator.save('new_discriminator  .h5')
        

# Example usage

