from data.datagen import *
from data.loader import *
from models.custom import *



print("Loading data\n")
load_data()
print("Data loaded\n")

print("getting ids and labels\n")
#preprocess the data, impute
train, valid, labels = get_split_data()
print("Done\n")

print("Training Generator\n")
training_generator = DataGenerator(train, "dataset/data/images", labels)
print("Validation Generator\n")
validation_generator = DataGenerator(valid, "dataset/data/images", labels)

print("getting model\n")
model = get_model()
print("starting training\n")
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=5)

#model.summary()
