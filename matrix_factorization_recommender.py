import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split
pd.set_option('display.max_columns', None)

# -----------------------------
# Data Preparation
# -----------------------------
movie = pd.read_csv('datasets\movie.csv')
rating = pd.read_csv('\datasets\ rating.csv')
df = movie.merge(rating, how="left", on="movieId")


df.head()
# Sample movies for demonstration
movie_ids = [130219, 356, 4422, 541]
sample_df = df[df.movieId.isin(movie_ids)]


# Surprise requires (user, item, rating) format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(sample_df[['userId', 'movieId', 'rating']], reader)


# -----------------------------
# Train/Test Split
# -----------------------------
trainset, testset = train_test_split(data, test_size=0.25)

# -----------------------------
# Model Training
# -----------------------------
svd_model = SVD()
svd_model.fit(trainset)

# Predictions
predictions = svd_model.test(testset)
rmse_score = accuracy.rmse(predictions)
# RMSE: 0.9348



# -----------------------------
# Model Tuning with GridSearch
# -----------------------------
param_grid = {
    'n_epochs': [5, 10, 20],
    'lr_all': [0.002, 0.005, 0.007]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
gs.fit(data)


print("Best RMSE score:", gs.best_score['rmse'])
# Best RMSE score: 0.93031919767182
print("Best parameters:", gs.best_params['rmse'])
# Best parameters: {'n_epochs': 5, 'lr_all': 0.002}


# -----------------------------
# Final Model Training
# -----------------------------
svd_model = SVD(**gs.best_params['rmse'])
full_trainset = data.build_full_trainset()
svd_model.fit(full_trainset)

# Example prediction
userId = 10
movieId = 1
df.loc[(df["userId"] == 10) & (df["movieId"]==1)]["rating"]
# movieId   title               genres                                        userId  rating    timestamp
#     1     Toy Story (1995)    Adventure|Animation|Children|Comedy|Fantasy   10.0     4.0      1999-11-25 02:44:47



pred = svd_model.predict(uid=user_id, iid=movie_id, verbose=True)
# user: 10         item: 1          r_ui = None   est = 4.05   {'was_impossible': False}
