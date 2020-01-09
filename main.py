from functions.Database import Database
from A1.gender_classification import model_A1
from A2.is_smiling_classification import model_A2
from B1.shape_classification import model_B1
from B2.eye_colour_classification import model_B2
# ======================================================================================================================
# Task A1
print('------- Task A1 -------')
db_celeba_gender = Database('celeba', 'gender')
db_celeba_gender.preprocess_images()
db_celeba_gender.extract_features()
db_celeba_gender.read_labels()
all_features = db_celeba_gender.read_database()
task_A1 = model_A1(all_features, db_celeba_gender.labels)
task_A1.train_model(task_A1.train_set, task_A1.train_lbls)
task_A1.test_model(task_A1.test_set, task_A1.test_lbls)

del db_celeba_gender, all_features

# =============================================================================
# Task A2
print('------- Task A2 -------')
db_celeba_smiling = Database('celeba', 'smiling', area_extract='mouth')
db_celeba_smiling.preprocess_images()
db_celeba_smiling.extract_features()
db_celeba_smiling.read_labels()
all_features = db_celeba_smiling.read_database()
task_A2 = model_A2(all_features, db_celeba_smiling.labels)
task_A2.train_model(task_A2.train_set, task_A2.train_lbls)
task_A2.test_model(task_A2.test_set, task_A2.test_lbls)

del db_celeba_smiling, all_features

# =============================================================================
# Task B1
print('------- Task B1 -------')
db_cartoon_shape = Database('cartoon_set', 'face_shape')
db_cartoon_shape.preprocess_images()
db_cartoon_shape.extract_features()
db_cartoon_shape.read_labels()
all_features = db_cartoon_shape.read_database()
task_B1 = model_B1(all_features, db_cartoon_shape.labels)
task_B1.train_model(task_B1.train_set, task_B1.train_lbls)
task_B1.test_model(task_B1.test_set, task_B1.test_lbls)

del db_cartoon_shape, all_features

# =============================================================================
# Task B2
print('------- Task B2 -------')
db_cartoon_eye_colour = Database('cartoon_set', 'eye_colour',
                                 area_extract='eyes', feature_extract='rgb')
db_cartoon_eye_colour.preprocess_images()
db_cartoon_eye_colour.extract_features()
db_cartoon_eye_colour.read_labels()
all_features = db_cartoon_eye_colour.read_database()
task_B2 = model_B2(all_features, db_cartoon_eye_colour.labels)
task_B2.train_model(task_B2.train_set, task_B2.train_lbls)
task_B2.test_model(task_B2.test_set, task_B2.test_lbls)

del db_cartoon_eye_colour, all_features

# =============================================================================

print('TA1: train {}, test {}; TA2: train {}, test {}; TB1: train {}, '
      'test {}; TB2: train {}, test {};'.format(task_A1.train_accuracy,
                                                task_A1.test_accuracy,
                                                task_A2.train_accuracy,
                                                task_A2.test_accuracy,
                                                task_B1.train_accuracy,
                                                task_B1.test_accuracy,
                                                task_B2.train_accuracy,
                                                task_B2.test_accuracy))
