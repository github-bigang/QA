export PYTHONPATH=src/:$PYTHONPATH
DATA=$1
RUN_ID=$2

DATA_DIR=data/unit/
MODEL_DIR=model/unit/$RUN_ID_$3
TRAIN_FILE=$DATA_DIR/qa.txt 
DEV_FILE=$DATA_DIR/qa.txt
TEST_FILE=$DATA_DIR/qa.txt
W2V_PATH=resource/glove.6B.300d.txt
INTERMEDIATE_TEST_OUTPUT_FILE=$MODEL_DIR/test.prediction.intermediate

LEARNING_RATE=0.0001
BATCH_SIZE=100
MP_DIM=20
MAX_EPOCHS=20
DROPOUT_RATE=0.2
WITH_HIGHWAY="" #"--with_highway True"

#cd /u/yyu/wksp1/BiMPM-modified

mkdir -p $MODEL_DIR
echo "" > $INTERMEDIATE_TEST_OUTPUT_FILE.scores

# train and test
python src/pair/SentenceMatchTrainer.py --batch_size $BATCH_SIZE --train_path $TRAIN_FILE --dev_path $DEV_FILE --test_path $TEST_FILE --word_vec_path $W2V_PATH --suffix sample --fix_word_vec --model_dir $MODEL_DIR --MP_dim $MP_DIM --learning_rate $LEARNING_RATE --max_epochs $MAX_EPOCHS --dropout_rate $DROPOUT_RATE $WITH_HIGHWAY --out_path $INTERMEDIATE_TEST_OUTPUT_FILE --model_prefix $MODEL_DIR/SentenceMatch.sample --wo_char

# test
python src/pair/SentenceMatchDecoder.py --in_path $TEST_FILE --word_vec_path $W2V_PATH --mode prediction --model_prefix $MODEL_DIR/SentenceMatch.sample --out_path $MODEL_DIR/test.prediction

# score
python scripts/score_prediction.py $MODEL_DIR/test.prediction > $MODEL_DIR/test.prediction.out
