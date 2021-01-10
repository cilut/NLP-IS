library(tidyverse)
library(tidytext)
library(naivebayes)
library(tm)
library(caret)
library(tibble)
library(RCurl)
library(prob)
library(randomForest)
library(naniar)

# FIRST STAGE: TRAIN AND TEST A MODEL

# AXILIAR FUNCTION TO GET THE DATA
# Get the data https://stackoverflow.com/questions/9548630/read-gzipped-csv-directly-from-a-url-in-r
get_data <- function(file_url) {
  con <- gzcon(url(file_url))
  txt <- readLines(con)
  return(jsonlite::stream_in(textConnection(txt)))
}


# 1st DOWNLOAD THE DATA AND CREATE A DATA FRAME
df_known <- get_data('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz')
# df_known <- get_data('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/All_Beauty_5.json.gz')


# 2nd EXTRACT THE VARIABLES REQUIERED AND CONVERT TO FACTOR THE OVERALL FOR THE CLASIFICATION
df_known <- df_known[,c("overall", "reviewText")]
df_known <- mutate(df_known, overall = as.factor(overall))
class(df_known)
class(df_known$overall)

gc()

# 3rd GENERATE THE TDM WITH ALL THE AVAILABLE TEXT
# https://stackoverflow.com/questions/44680696/how-to-cast-a-dataframe-into-a-dtm
corpus <- Corpus(VectorSource(df_known[,2])) 

# EXTRACT STOPWORDS, PUNTUATUATION, NUMERS AND APPLY STEAMMING
tdm = TermDocumentMatrix(corpus,
                         control=list(stopwords = T,
                                      removePunctuation = T, 
                                      removeNumbers = T,
                                      stemming = T))
tdm
inspect(tdm[100:103,10:40])

# 4th OBTAIN A DATA FRAME WITH THE TDM RELATED TO EACH OVERALL
mdisp_known <- t(as.matrix(tdm[]))
gc()
mdisp_df_known <- mdisp_known %>% data.frame
mdisp_df_known <- data.frame(overall=df_known[,1], mdisp_df_known)%>% tbl_df
class(mdisp_df_known)

# 5th DIVIDE THE DATA ON TRAINING AND TESTING
# https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function

smp_size <- floor(0.7 * nrow(mdisp_df_known))

## set the seed to make your partition reproducible
set.seed(2000)
train_index <- sample(seq_len(nrow(mdisp_df_known)), size = smp_size)

train <- mdisp_df_known[train_index, ]
test <- mdisp_df_known[-train_index, ]


# 6th TRY TWO MODELS:
  # THE NAIVE BAYES MODEL
model1 <- naive_bayes(formula = overall ~ .,  data = train)
model1_prediction <- predict(model1, test)
head(model1_prediction, 25)
confusionMatrix(model1_prediction, test[["overall"]])
class(test)
  # THE RANDOM TREE MODEL
model2 <- randomForest(formula = overall ~ ., data = train)
model2_prediction <- predict(model2, test)

model2_prediction
confusionMatrix(model2_prediction, test[["overall"]])

head(model2_prediction, 25)



# SECOND STAGE: CLASSIFIE UNKNOWN REVIEWS

################################################################################
#                               INSERT THE NEW REVIEW                         #
################################################################################

review <- "which is thin with oily roots and dry ends) way stronger and shinny without being oily! It's really fantastic. I took a star away for 2 reasons, one is because the price is a bit outrages, and two is because if I use it every day my hair starts to get buildup so I have to find a clean"

################################################################################


# PREPROCESS THE NEW REVIEW
corpus_review <- Corpus(VectorSource(review)) 
# EXTRACT STOPWORDS, PUNTUATUATION, NUMERS AND APPLY STEAMMING
tdm_review = TermDocumentMatrix(corpus_review,
                         control=list(stopwords = T,
                                      removePunctuation = T, 
                                      removeNumbers = T,
                                      stemming = T))
review <- as.matrix(tdm_review[])%>% as.data.frame
review <-row.names(review) %>% as.vector
review

# CREATE A MATRIX WITH THE VARIABLES AVAILABLES ON THE TDM AND MATCH WORKS OF THE INPUT
new_review_tdm <- replace_with_na(test[1,], test[1,])
new_review_tdm$overall <- as.numeric(new_review_tdm$overall)
new_review_tdm[is.na(new_review_tdm)] <- 0
for (word in review){
  if(colnames(new_review_tdm) %>% isin(word)){
    index<-which(colnames(new_review_tdm) ==word)
    ifelse(is_na(new_review_tdm[index]), 
           new_review_tdm[1, word]<-1, 
           new_review_tdm[1, word]<-new_review_tdm[1, word]+1)
  }
}

# CLASIFFIE THE REVIEW
model2_prediction_review <- predict(model2, new_review_tdm)
model2_prediction_review

# https://rpubs.com/rgcmme/IS-HO1
# https://rpubs.com/rgcmme/IS-HO2
# https://rpubs.com/rgcmme/IS-HO3
# https://www.rpubs.com/jboscomendoza/naive_nayes_con_r_clasificacion_texto
# https://www.r-bloggers.com/2018/01/how-to-implement-random-forests-in-r/
# https://www.r-bloggers.com/2015/12/how-to-write-the-first-for-loop-in-r/
# https://stackoverflow.com/questions/61333592/problem-when-training-naive-bayes-model-in-r
# https://datarac.blogspot.com/2020/08/text-classification-using-naive-bayes-r.html


