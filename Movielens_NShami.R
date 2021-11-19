if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(knitr)
library(ggpubr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)


ratings <- fread(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines( "ml-10M100K/movies.dat"), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


head(movies)
head(ratings)

# Joining movies and ratings togother

movielens <- left_join(ratings, movies, by = "movieId")
head(movielens)

# extract release year from title field, and split it to *title* that contains movie's name and *year* that contains release year
movies_df <- movielens%>%
  extract(title, c("title", "year"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", remove = T)
head(movies_df)


#Split data to training set and validation set, Validation set will be 10% of MovieLens data.
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movies_df[-test_index,]
temp <- movies_df[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
nrow(removed)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#to see how many unique user and movie we have, we use the below code:
number_of_users <- edx %>% distinct(userId) %>% summarise(n())
number_of_movies <- edx %>% distinct(movieId)%>% summarise(n())

kable(tibble(
  Users = number_of_users,
  Movies = number_of_movies
))

#To check if there is any missing values
NA_count <- t(data.frame(lapply(edx, function(x) sum(is.na(x)))))
colnames(NA_count) <- c("NA_count")
kable(NA_count)

#some users have rated more than others, as shown in the two histograms below
movies_hist <- edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 50, color = "black", fill = "steelblue1") + 
  labs(title = "Ratings per movie",
       x = "# ratings per movie", y = "Count") 

users_hist<-edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 50, color = "black",fill = "hotpink1") + 
  labs(title = "Ratings per user",
       x = "# ratings per user", y = "Count") +
  theme_classic()

ggarrange(movies_hist, users_hist,
          ncol = 2, nrow = 1)

#distribution of the ratings
ratings_count <- edx %>%
  mutate(rating = as.factor(rating)) %>%
  group_by(rating) %>%
  summarise(number_of_ratings = n()) %>%
  mutate(prop = (number_of_ratings / sum(number_of_ratings)) * 100) %>%
  select(rating, number_of_ratings, prop)

ggplot(ratings_count, aes(rating, number_of_ratings)) +
  geom_bar(stat = "identity", fill="steelblue", color="black") +
  xlab("Rating") + ylab("Number of Ratings") +
  ggtitle("Frequency of Ratings Per Rating Value") 


# Separating genres for edx and validation sets
edx_seperated_genres <-edx %>% separate_rows(genres, sep = "\\|")
head(edx_seperated_genres)

validation_seperated_genres <-validation %>% separate_rows(genres, sep = "\\|")


#Analyze data based on genres
genres_count <- edx_seperated_genres %>% 
  group_by(genres) %>%
  summarize(count = n())%>%
  arrange(desc(count))
kable(genres_count)
genres_count %>% ggplot(aes(x= reorder(genres, count),y=count))+
  geom_bar(stat="identity" , fill ="steelblue")+
  labs(title = "Ratings per genre",x = "Genre", y = "Counts")+
  scale_y_continuous(labels = paste0(1:4, "M"), breaks = 10^6 * 1:4)+
  coord_flip()

#convert time stamp to date format
edx_seperated_genres$date <- as.POSIXct(edx_seperated_genres$timestamp, origin = "1970-01-01")  # as.POSIXct function
head(edx_seperated_genres)
validation_seperated_genres$date <- as.POSIXct(validation_seperated_genres$timestamp, origin = "1970-01-01")  # as.POSIXct function
head(validation_seperated_genres)

#create features for rate year and rate month for both test and validation sets
edx_seperated_genres$rate_year <- format(edx_seperated_genres$date,"%Y")
edx_seperated_genres$rate_month <- format(edx_seperated_genres$date,"%m")

validation_seperated_genres$rate_year <- format(validation_seperated_genres$date,"%Y")
validation_seperated_genres$rate_month <- format(validation_seperated_genres$date,"%m")

head(edx_seperated_genres)

######################################################################
#Model Creation
######################################################################

#calculating average
mu <- mean(edx_seperated_genres$rating)
mu
#Just average model (Naive model)
just_avg = RMSE(validation_seperated_genres$rating, mu)
rmse_results <- tibble(Method = "Just the average", RMSE = just_avg)
kable(rmse_results)
#######################################################################
##Average + movie bias mode
b_i <- edx_seperated_genres %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

#predict all unknown ratings with mu and b_i
predicted_ratings_movie <- validation_seperated_genres %>% 
  left_join(b_i, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

avg_movie = RMSE(validation_seperated_genres$rating, predicted_ratings_movie)
rmse_results <- bind_rows(rmse_results,tibble(Method = "avg + movie bias", RMSE = avg_movie))
kable(rmse_results)
########################################################################
# Average + Movie and user effect method

b_u <- edx_seperated_genres %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings_movie_user <- validation_seperated_genres %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
# calculate RMSE of movie ranking effect
avg_movie_user <- RMSE(predicted_ratings_movie_user, validation_seperated_genres$rating)
rmse_results <- bind_rows(rmse_results,tibble(Method = "avg + movie bias + user bias", RMSE = avg_movie_user))

print(avg_movie_user)

#######################################################################
# Regularization

# determine best lambda from a sequence
lambdas <- seq(from=0, to=10, by=0.25)

# output RMSE of each lambda, repeat earlier steps (with regularization)
rmses <- sapply(lambdas, function(l){
  # calculate average rating across training data
  mu <- mean(edx_seperated_genres$rating)
  # compute regularized movie bias term
  b_i <- edx_seperated_genres %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  # compute regularize user bias term
  b_u <- edx_seperated_genres %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  # compute predictions on validation set based on these above terms
  predicted_ratings <- validation_seperated_genres %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  # output RMSE of these predictions
  return(RMSE(predicted_ratings, validation_seperated_genres$rating))
})

# quick plot of RMSE vs lambdas
qplot(lambdas, rmses)
# print minimum RMSE 
min(rmses)

rmse_results <- bind_rows(rmse_results,tibble(Method = "Regularized avg + movie bias + user", RMSE = min(rmses)))
#kable(rmse_results)

######################################################
# Final model with regularized movie and user effects
######################################################

# The final linear model with the minimizing lambda
lam <- lambdas[which.min(rmses)]

b_i <- edx_seperated_genres %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lam))
# compute regularize user bias term
b_u <- edx_seperated_genres %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lam))
# compute predictions on validation set based on these above terms
predicted_ratings <- validation_seperated_genres %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)


# output RMSE of these predictions
RMSE(predicted_ratings, validation_seperated_genres$rating)

################################################################
# Final model with regularized movie and user and genres effects
################################################################

# The final linear model with the minimizing lambda
lam <- lambdas[which.min(rmses)]

b_i <- edx_seperated_genres %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lam))
# compute regularize user bias term
b_u <- edx_seperated_genres %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lam))
# compute regularize genres bias term
b_g <- edx_seperated_genres %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lam))
# compute predictions on validation set based on these above terms
predicted_ratings <- validation_seperated_genres %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

reg_movie_user_genre <- RMSE(predicted_ratings, validation_seperated_genres$rating)

rmse_results <- bind_rows(rmse_results,tibble(Method = "Regularized avg + movie bias + user + genre", RMSE = reg_movie_user_genre))


##############################################################
# Final Results
##############################################################
kable(rmse_results)
