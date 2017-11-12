export PYTHONPATH=./src/:$PYTHONPATH
DATA=$1
RUN_ID=$2

MODEL_DIR=~/stor/webqsp/models/$DATA/${RUN_ID}_$3
mkdir -p $MODEL_DIR
#DATA_DIR=data/$DATA-data/
DATA_DIR=/u/kshasan/tensorflow_workspace/BiMPM/data/webqsp-data/
DATA_DIR=~/stor/webqsp/simplequestions4/
TRAIN_FILE=$DATA_DIR/sq.train #$DATA.train.bimpm.$RUN_ID
DEV_FILE=$DATA_DIR/sq.valid #$DATA.valid.bimpm.$RUN_ID
TEST_FILE=$DATA_DIR/sq.test #$DATA.test.bimpm.$RUN_ID
W2V_PATH=/u/zhangwei/stor/data/glove/glove.6B.300d.txt
INTERMEDIATE_TEST_OUTPUT_FILE=$MODEL_DIR/test.prediction.intermediate

LEARNING_RATE=0.0001
BATCH_SIZE=100
MP_DIM=20
MAX_EPOCHS=20
DROPOUT_RATE=0.2
WITH_HIGHWAY="" #"--with_highway True"

if [ $DATA == "webqsp" ]
then
	DEV_FILE=$TEST_FILE
else
	#TEST_FILE=$DATA_DIR/$DATA.test.bimpm.linker_output
	DEV_FILE=$TEST_FILE
fi

cd /u/yyu/wksp1/BiMPM-modified

mkdir -p $MODEL_DIR
echo "" > $INTERMEDIATE_TEST_OUTPUT_FILE.scores

# train and test
python src/triple/TripleMatchTrainer.py --batch_size $BATCH_SIZE --train_path $TRAIN_FILE --dev_path $DEV_FILE --test_path $TEST_FILE --word_vec_path $W2V_PATH --suffix sample --fix_word_vec --model_dir $MODEL_DIR --MP_dim $MP_DIM --learning_rate $LEARNING_RATE --max_epochs $MAX_EPOCHS --dropout_rate $DROPOUT_RATE $WITH_HIGHWAY --out_path $INTERMEDIATE_TEST_OUTPUT_FILE --model_prefix $MODEL_DIR/TripleMatch.sample --wo_char

# test
python src/triple/TripleMatchDecoder.py --in_path $TEST_FILE --word_vec_path $W2V_PATH --mode prediction --model_prefix $MODEL_DIR/TripleMatch.sample --out_path $MODEL_DIR/test.prediction

# score
python score_prediction.py $MODEL_DIR/test.prediction > $MODEL_DIR/test.prediction.out
