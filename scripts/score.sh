TEST_OUTPUT=$1

# score
python scripts/score_prediction.py $TEST_OUTPUT > $TEST_OUTPUT.out
echo "$(tail -1 $TEST_OUTPUT.out)" >> $TEST_OUTPUT.scores