import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split


# -----------------------------
# Data Preparation
# -----------------------------
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")


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
print(f"RMSE: {rmse_score}")

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
print("Best parameters:", gs.best_params['rmse'])

# -----------------------------
# Final Model Training
# -----------------------------
svd_model = SVD(**gs.best_params['rmse'])
full_trainset = data.build_full_trainset()
svd_model.fit(full_trainset)

# Example prediction
user_id = 1.0
movie_id = 541
pred = svd_model.predict(uid=user_id, iid=movie_id, verbose=True)
